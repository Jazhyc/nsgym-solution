"""Example: Baseline MCTS on a preconfigured non-stationary FrozenLake environment."""

from AAMAS_Comp.examples.agents import AAMASCompBaselineMCTS
from AAMAS_Comp.evaluation import run_complete_evaluation
import gymnasium as gym


def main():

    START_SEED = 42

    # Notification levels:
    #   CHANGE_NOTIFICATION  - agent is told a change occurred (but not the magnitude)
    #   DELTA_CHANGE_NOTIFICATION - agent is told a change occurred AND its magnitude
    CHANGE_NOTIFICATION = True
    DELTA_CHANGE_NOTIFICATION = True

    # Pre-configured non-stationary FrozenLake.
    # Starts fully deterministic; at each step the transition function becomes
    # incrementally more slippery (P(intended) -= 0.025, P(perpendicular) += 0.0125).
    ns_env = gym.make(
        "ExampleNSFrozenLake-v0",
        change_notification=CHANGE_NOTIFICATION,
        delta_change_notification=DELTA_CHANGE_NOTIFICATION,
        disable_env_checker=True,
        order_enforce=False,
    )

    agent = AAMASCompBaselineMCTS(d=50, m=100, c=1.4, gamma=0.99)

    run_complete_evaluation(
        env=ns_env,
        agent=agent,
        start_seed=START_SEED,
        num_episodes=10,
        name_prefix="MCTS_Example",
    )


if __name__ == "__main__":
    main()
