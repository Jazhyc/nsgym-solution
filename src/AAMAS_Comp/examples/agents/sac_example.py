"""Example: Stable Baselines 3 SAC agent wrapped for the AAMAS competition interface."""

from stable_baselines3 import SAC
from AAMAS_Comp.base_agent import SB3Agent


class AAMASCompBaselineSAC(SB3Agent):
    """AAMAS Competition Baseline SAC.

    Wraps a Stable Baselines 3 SAC model for the competition interface.
    Provide either a pre-trained model instance or a path to a saved model.

    Args:
        model_path (str): Path to a saved SAC model (loaded via SAC.load).
        model (SAC): An already-trained SAC instance. Takes precedence over model_path.
        deterministic (bool): Use deterministic actions. Defaults to True.
    """

    def __init__(self, model_path=None, model=None, deterministic=True) -> None:
        if model is None and model_path is None:
            raise ValueError("Provide either model or model_path")
        if model is None:
            model = SAC.load(model_path)
        super().__init__(model=model, deterministic=deterministic)
