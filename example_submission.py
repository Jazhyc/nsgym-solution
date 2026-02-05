"""Sample submission using baseline agents. Fill `submission.py` in with your own 


- Ant-v5: Pre-trained PPO with VecNormalize observation normalization.
- FrozenLake-v1: MCTS with chance nodes (model-based, uses planning env).
- CartPole-v1: MCTS with chance nodes (model-based, uses planning env).
"""

from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from AAMAS_Comp.examples.agents import AAMASCompBaselinePPO, AAMASCompBaselineMCTS
import gymnasium as gym


def get_agent(env_id: str):
    """Return an agent instance configured for the given environment.

    Args:
        env_id: The base environment being evaluated. One of:
            - "FrozenLake-v1"
            - "CartPole-v1"
            - "Ant-v5"
    """
    if env_id == "Ant-v5":
        model_path = Path("models/ppo_ant/ppo_ant.zip")
        vec_norm_path = Path("models/ppo_ant/ppo_ant_vecnormalize.pkl")

        if not model_path.exists() or not vec_norm_path.exists():
            raise FileNotFoundError(
                f"Pre-trained PPO model not found at {model_path}. "
                "Train it first with: python examples/ppo_example.py"
            )

        dummy_env = DummyVecEnv([lambda: gym.make("Ant-v5")])
        vec_normalize = VecNormalize.load(str(vec_norm_path), dummy_env)
        vec_normalize.training = False
        vec_normalize.norm_reward = False

        return AAMASCompBaselinePPO(model_path=str(model_path), vec_normalize=vec_normalize)

    elif env_id == "FrozenLake-v1":
        return AAMASCompBaselineMCTS(d=50, m=100, c=1.4, gamma=0.99)

    elif env_id == "CartPole-v1":
        return AAMASCompBaselineMCTS(d=5, m=20, c=1.4, gamma=0.99)

    else:
        raise ValueError(f"Unknown environment: {env_id}")
