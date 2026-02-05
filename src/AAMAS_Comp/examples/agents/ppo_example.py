"""Example: Stable Baselines 3 PPO agent wrapped for the AAMAS competition interface.

The SB3Agent wrapper works with any Stable Baselines 3 algorithm (PPO, A2C, DQN, etc.)
since they all expose the same `predict()` API.
"""

from typing import Dict
from stable_baselines3 import PPO
from AAMAS_Comp.base_agent import SB3Agent

class AAMASCompBaselinePPO(SB3Agent):
    """AAMAS Competition Baseline PPO.

    Wraps a Stable Baselines 3 PPO model for the competition interface.
    Provide either a pre-trained model instance or a path to a saved model.

    Args:
        model_path (str): Path to a saved PPO model (loaded via PPO.load).
        model (PPO): An already-trained PPO instance. Takes precedence over model_path.
        deterministic (bool): Use deterministic actions. Defaults to True.
        vec_normalize: A VecNormalize instance for observation normalization.
    """

    def __init__(self, model_path=None, model=None, deterministic=True, vec_normalize=None) -> None:
        if model is None and model_path is None:
            raise ValueError("Provide either model or model_path")
        if model is None:
            model = PPO.load(model_path)
        super().__init__(model=model, deterministic=deterministic, vec_normalize=vec_normalize)
