import json

import pytest
import torch

from AAMAS_Comp.agents.recovery import check_finite, learner_state, restore_learner
from AAMAS_Comp.envs.wrappers import RunningMeanStd, RewardNormalizer


def models():
    actor, critic = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW([*actor.parameters(), *critic.parameters()], lr=.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=.1, total_iters=100)
    return actor, critic, optimizer, scheduler


def test_nonfinite_is_rejected_and_logged_before_use(tmp_path):
    with pytest.raises(FloatingPointError, match="PPO gradients"):
        check_finite({'gradient': torch.tensor(float('nan'))}, stage='PPO gradients',
                     frames=100, directory=tmp_path)
    assert json.loads((tmp_path/'nonfinite_failure.json').read_text())['nonfinite_keys'] == ['gradient']
    check_finite({'reward': torch.tensor(2.)}, stage='reward', frames=101, directory=tmp_path)


def test_legacy_resume_restores_weights_and_original_lr_clock():
    actor, critic, _, _ = models()
    new_actor, new_critic, opt, scheduler = models()
    restore_learner({'actor': actor, 'critic': critic}, new_actor, new_critic, opt, scheduler, 40)
    assert torch.equal(actor.weight, new_actor.weight)
    assert torch.equal(critic.weight, new_critic.weight)
    assert opt.param_groups[0]['lr'] == pytest.approx(.01*(1-.9*.4))
    opt.step(); scheduler.step()
    assert opt.param_groups[0]['lr'] == pytest.approx(.01*(1-.9*.41))


def test_nonfinite_checkpoint_cannot_be_resumed():
    actor, critic, opt, scheduler = models()
    with torch.no_grad(): actor.weight.fill_(float('nan'))
    with pytest.raises(ValueError, match='non-finite actor'):
        restore_learner({'actor': actor, 'critic': critic}, actor, critic, opt, scheduler, 40)


def test_full_learner_checkpoint_preserves_optimizer_scheduler_and_normalizers():
    actor, critic, opt, scheduler = models()
    (actor(torch.ones(1, 2)).sum() + critic(torch.ones(1, 2)).sum()).backward()
    opt.step(); scheduler.step()
    obs, reward = RunningMeanStd((2,)), RewardNormalizer()
    obs.update(torch.ones(3, 2)); reward.reward_rms.update(torch.tensor([2., 3., 4.]))
    state = learner_state(opt, scheduler, obs, reward, 200, 1)
    new_actor, new_critic, new_opt, new_scheduler = models()
    restore_learner({'actor': actor, 'critic': critic, 'training_state': state},
                    new_actor, new_critic, new_opt, new_scheduler, 2)
    assert new_opt.state_dict()['state']
    assert new_scheduler.state_dict() == scheduler.state_dict()
    assert state['reward_rms']['count'] == reward.reward_rms.count
    assert torch.equal(state['obs_rms']['mean'], obs.mean)
    assert state['worker_state_saved'] is False


def test_training_checkpoint_is_atomic_and_still_loads_for_inference(tmp_path):
    from AAMAS_Comp.agents.ppo import PPOAgent
    actor, critic, opt, scheduler = models()
    path = tmp_path/'checkpoint.pt'
    state = learner_state(opt, scheduler, None, None, 200, 1)
    PPOAgent(actor, critic).save(str(path), training_state=state)
    loaded = torch.load(path, weights_only=False)
    assert loaded['training_state']['global_step'] == 200
    assert not path.with_name(path.name+'.tmp').exists()
    inference = PPOAgent.load(str(path))
    assert torch.equal(inference.actor.weight, actor.weight)
