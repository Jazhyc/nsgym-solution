"""Measure actual PPO update sizes on locally collected static Ant transitions."""

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from tensordict import TensorDict
from torchrl.envs.libs.gym import GymWrapper

from AAMAS_Comp.agents.ppo import make_ppo_models
from AAMAS_Comp.agents.policy_checks import nonnegative_sample_kl, guarded_optimizer_step
from AAMAS_Comp.agents.recovery import restore_learner
from AAMAS_Comp.envs.domain_randomization import DynamicsEnv
from AAMAS_Comp.envs.wrappers import RewardNormalizer, SeededResetWrapper


def audit(checkpoint_path, pool, cfg, guarded=False, levels=None):
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
    env_spec = GymWrapper(gym.make('Ant-v5'))
    models = make_ppo_models(env_spec, cfg, device=torch.device('cpu'))
    env_spec.close()
    actor, critic = models['actor'], models['critic']
    optimizer, scheduler = models['optimizer'], models['scheduler']
    restore_learner(checkpoint, actor, critic, optimizer, scheduler,
                    checkpoint['training_state']['collect_iter'] + 1)
    rms = checkpoint['obs_rms']
    mean, std = rms['mean'].float(), rms['std'].float()
    normalizer = RewardNormalizer()
    normalizer.reward_rms.load_state_dict(checkpoint['training_state']['reward_rms'])
    # Same batch geometry as the cluster: 24 workers x 128 transitions.
    if levels:
        from AAMAS_Comp.curriculum.plr_env import _build_ns_env
        from AAMAS_Comp.evaluation.paper import decode_config
        selected = [level for origin in ('fresh', 'mutated')
                    for level in [r for r in levels if r['origin'] == origin][:12]]
        if len(selected) != 24:
            raise ValueError('Need at least 12 fresh and 12 mutated recorded levels')
        envs = [_build_ns_env(decode_config(row['config'])) for row in selected]
    else:
        envs = [DynamicsEnv('ant', {}, samples=pool, seed=1000+i) for i in range(24)]
    envs = [SeededResetWrapper(env, 3000+i) for i, env in enumerate(envs)]
    observations = [env.reset(seed=2000+i)[0] for i, env in enumerate(envs)]
    transitions = []
    maximum_location = 0.
    minimum_scale = float('inf')
    saturated_actions = 0
    action_count = 0
    torch.manual_seed(12345)
    try:
        for _ in range(128):
            obs = (torch.as_tensor(np.asarray(observations)).float()-mean)/std
            with torch.no_grad():
                dist = actor.get_dist(TensorDict({'observation': obs}, [24]))
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                maximum_location = max(maximum_location, float(dist.loc.abs().max()))
                minimum_scale = min(minimum_scale, float(dist.scale.min()))
                saturated_actions += int((actions.abs() >= 1-1e-6).sum())
                action_count += actions.numel()
            outcomes = [env.step(action.numpy()) for env, action in zip(envs, actions)]
            next_obs = (torch.as_tensor(np.asarray([r[0] for r in outcomes])).float()-mean)/std
            terminal = torch.tensor([r[2] for r in outcomes]).unsqueeze(-1)
            truncated = torch.tensor([r[3] for r in outcomes]).unsqueeze(-1)
            transitions.append(TensorDict({
                'observation': obs, 'action': actions,
                models['loss_module'].tensor_keys.sample_log_prob: log_probs,
                'next': TensorDict({'observation': next_obs,
                    'reward': torch.tensor([r[1] for r in outcomes]).unsqueeze(-1),
                    'terminated': terminal, 'truncated': truncated,
                    'done': terminal | truncated}, [24])}, [24]))
            observations = [env.reset()[0] if r[2] or r[3] else r[0]
                            for env, r in zip(envs, outcomes)]
    finally:
        for env in envs:
            env.close()
    batch = torch.stack(transitions, dim=1)
    normalizer.normalize_batch(batch)
    with torch.no_grad():
        models['advantage'](batch)
    batch = batch.reshape(-1)
    log_key = models['loss_module'].tensor_keys.sample_log_prob
    fingerprint = hashlib.sha256()
    for key in ['observation', 'action', log_key, ('next', 'observation'), ('next', 'reward')]:
        fingerprint.update(batch[key].detach().contiguous().numpy().tobytes())

    def divergence(td):
        with torch.no_grad():
            log_prob = actor.get_dist(td).log_prob(td['action'])
            return float(nonnegative_sample_kl(log_prob, td[log_key]))

    rows = []
    permutation = torch.randperm(batch.numel())
    for i in range(12):
        minibatch = batch[permutation[i*256:(i+1)*256]]
        before = divergence(minibatch)
        if before > .03:
            rows.append({'minibatch': i, 'pre_kl': before, 'guard_stopped': True})
            break
        losses = models['loss_module'](minibatch)
        objective = sum(losses[k] for k in ('loss_objective', 'loss_critic'))
        objective.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(models['optim_params'], cfg.training.max_grad_norm)
        update = None
        if guarded:
            update = guarded_optimizer_step(optimizer, models['optim_params'],
                                            lambda: divergence(batch), .03)
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        rows.append({'minibatch': i, 'pre_kl': before,
                     'post_minibatch_kl': divergence(minibatch),
                     'post_batch_kl': divergence(batch), 'gradient_norm': float(grad_norm),
                     'accepted': update.accepted if update else True,
                     'attempts': update.attempts if update else 1,
                     'step_scale': update.lr_scale if update else 1.})
        if update and not update.accepted:
            break
    return {'checkpoint': str(checkpoint_path), 'lr': optimizer.param_groups[0]['lr'],
            'frames': checkpoint['training_state']['global_step'], 'guarded': guarded,
            'maximum_gaussian_location': maximum_location, 'minimum_gaussian_scale': minimum_scale,
            'saturated_action_fraction': saturated_actions/action_count,
            'minimum_behavior_log_prob': float(batch[log_key].min()),
            'probe_sha256': fingerprint.hexdigest(),
            'environment_probe': 'recorded temporal fresh/mutated levels' if levels else 'static snapshot pool',
            'updates': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--guarded', action='store_true')
    parser.add_argument('--levels', type=Path, help='JSON containing recorded temporal configs')
    args = parser.parse_args()
    torch.set_num_threads(1)
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2]/'config'), version_base=None):
        cfg = compose(config_name='config_ant_v2')
    pool = json.loads(args.manifest.read_text())['training']['tasks']['ant']['dr_samples']
    results = []
    levels = json.loads(args.levels.read_text())['configs'] if args.levels else None
    for checkpoint in args.checkpoints:
        result = audit(checkpoint, pool, cfg, guarded=args.guarded, levels=levels)
        results.append(result)
        print(json.dumps(result), flush=True)
        args.output.write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
