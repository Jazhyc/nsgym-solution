"""Competition evaluator. Runs the submitted agent against competition environments."""

import argparse
import gymnasium as gym
import AAMAS_Comp  # noqa: F401 -- triggers environment registration
from AAMAS_Comp.evaluation import run_complete_evaluation

from collections import namedtuple

from submission import get_agent

#Uncomment this line to run baseline models from example submission. 
# from example_submission import get_agent

#Add additional envionments here or comment out the ones you do not want to evaluate. 
ENVIRONMENTS = {
    "ExampleNSFrozenLake-v0": "FrozenLake-v1",
    "ExampleNSCartPole-v0": "CartPole-v1",
    "ExampleNSAnt-v0": "Ant-v5",
}



NotificationSetting = namedtuple("Notification",["change_notification", "delta_change_notification", "label"])

# This sets the notifiation levels. Comment out the levels you do not want to evaluate
NOTIFICATIONS = [NotificationSetting(True,  True,  "notify-full"),
                 NotificationSetting(True,  False, "notify-change"),
                 NotificationSetting(False, False, "notify-none")]



def evaluate_local(num_episodes=10, start_seed=42):
    for env_id, base_env_id in ENVIRONMENTS.items():
        for change_notification, delta_change_notification, notify_label in NOTIFICATIONS:

            agent = get_agent(base_env_id)
            env = gym.make(
                env_id,
                change_notification=change_notification,
                delta_change_notification=delta_change_notification,
                disable_env_checker=True,
                order_enforce=False,
            )

            name_prefix = f"{env_id}__{notify_label}"

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
    parser.add_argument("--mode", default="local", choices=["local"])
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "local":
        evaluate_local(num_episodes=args.num_episodes, start_seed=args.start_seed)
