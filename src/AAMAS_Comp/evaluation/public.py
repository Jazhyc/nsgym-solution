"""Original registered public examples, through the submitted stochastic agent."""
import json
from pathlib import Path
import random
import gymnasium as gym
import numpy as np
import torch
import AAMAS_Comp  # environment registration
from AAMAS_Comp.agent import MyModelFreeAgent
from AAMAS_Comp.evaluation.evaluate_agent import run_single_episode, _default_serializer
from AAMAS_Comp.evaluation.statistics import summarize

IDS = {'cartpole': ('ExampleNSCartPole-v0', 'CartPole-v1', 500),
       'frozenlake': ('ExampleNSFrozenLake-v0', 'FrozenLake-v1', 100),
       'ant': ('ExampleNSAnt-v0', 'Ant-v5', 1000)}

def evaluate_public_checkpoint(checkpoint, task, output, metadata=None, episodes_per_level=100,
                               *, environment_id=None, notification_modes=('full', 'change', 'none')):
    output = Path(output)
    records = []
    env_id, base_id, horizon = IDS[task]
    if environment_id is not None:
        allowed = {env_id}
        if task == 'ant':
            allowed.add('ExampleNSAnt-v1')
            allowed.add('ExampleNSAnt-v2')
        if environment_id not in allowed:
            raise ValueError(f'Unsupported evaluation environment for {task}: {environment_id}')
        env_id = environment_id
    target = output / 'public_episodes.jsonl'
    modes = {'full': (True, True), 'change': (True, False), 'none': (False, False)}
    if not notification_modes or len(set(notification_modes)) != len(notification_modes) or any(
            label not in modes for label in notification_modes):
        raise ValueError('notification_modes must contain distinct full/change/none modes')
    if target.exists():
        raise FileExistsError(target)
    with target.open('w') as stream:
        for label in notification_modes:
            notify, delta = modes[label]
            agent = MyModelFreeAgent(str(checkpoint), env_id=base_id,
                                    online_learning=False, deterministic=False)
            for i in range(episodes_per_level):
                seed = 42 + i
                agent._rng = np.random.default_rng(seed)
                torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
                # Fresh instances prevent parameter/update state leaking across episodes.
                env = gym.make(env_id, change_notification=notify,
                               delta_change_notification=delta,
                               disable_env_checker=True, order_enforce=False)
                try:
                    episode = run_single_episode(env, agent, seed)
                finally:
                    env.close()
                steps = len(episode['rewards'])
                assert steps <= horizon, (task, steps, horizon)
                record = {'notification': label, 'rollout_seed': seed,
                          'return': float(sum(episode['rewards'])), 'steps': steps,
                          'trajectory': episode}
                stream.write(json.dumps(record, default=_default_serializer) + '\n')
                records.append(record)
    metrics = {'public/pooled/' + k: v for k, v in summarize([r['return'] for r in records]).items()}
    for label in notification_modes:
        metrics.update({'public/' + label + '/' + k: v for k, v in
                        summarize([r['return'] for r in records if r['notification'] == label]).items()})
    (output / 'public_summary.json').write_text(json.dumps({
        'metadata': metadata or {}, 'environment': env_id, 'horizon': horizon,
        'action_mode': 'stochastic', 'episodes_per_level': episodes_per_level,
        'notification_modes': list(notification_modes),
        'metrics': metrics}, indent=2))
    return metrics
