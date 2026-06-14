"""Random-policy baseline for the report.

Evaluates a uniformly-random agent on each competition environment to provide a
lower-bound (floor) return against which the trained agent's numbers can be
read. The random policy ignores the observation entirely, so the notification
level is irrelevant; we run each environment once under no notification.

Usage:
    python scripts/eval_random_baseline.py --num-episodes 100 --start-seed 42
"""

import argparse
import gymnasium as gym
import AAMAS_Comp  # noqa: F401 -- triggers environment registration
from AAMAS_Comp.base_agent import ModelFreeAgent
from AAMAS_Comp.evaluation import run_complete_evaluation


# Same competition environments as evaluator.py.
ENVIRONMENTS = {
    "ExampleNSFrozenLake-v0": "FrozenLake-v1",
    "ExampleNSCartPole-v0": "CartPole-v1",
    "ExampleNSAnt-v0": "Ant-v5",
}


class RandomAgent(ModelFreeAgent):
    """Samples a uniformly-random valid action, ignoring the observation."""

    def __init__(self, action_space: gym.Space, seed: int = 0) -> None:
        super().__init__()
        self.action_space = action_space
        self.action_space.seed(seed)

    def get_action(self, obs):  # noqa: ARG002 -- obs ignored by design
        return self.action_space.sample()


def evaluate_random(num_episodes=100, start_seed=42):
    for env_id in ENVIRONMENTS:
        # Random policy is notification-agnostic; one no-notification run suffices.
        env = gym.make(
            env_id,
            change_notification=False,
            delta_change_notification=False,
            disable_env_checker=True,
            order_enforce=False,
        )

        agent = RandomAgent(env.action_space, seed=start_seed)
        name_prefix = f"Random__{env_id}"

        run_complete_evaluation(
            env=env,
            agent=agent,
            start_seed=start_seed,
            num_episodes=num_episodes,
            name_prefix=name_prefix,
        )

        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=42)
    args = parser.parse_args()

    evaluate_random(num_episodes=args.num_episodes, start_seed=args.start_seed)
