
from typing import Dict
from AAMAS_Comp.base_agent import ModelBasedAgent, ModelFreeAgent
from typing import Dict
import numpy as np
import torch
from AAMAS_Comp.base_agent import ModelFreeAgent
from AAMAS_Comp.agents.ppo import PPOAgent as TorchRLPPOAgent

"""Implement your code here. 
"""
class MyModelFreeAgent(ModelFreeAgent):
    def __init__(self, model_path: str, device: str = "cpu"):
        super().__init__()
        # Load your existing TorchRL checkpoint
        self._inner = TorchRLPPOAgent.load(model_path, device=device, deterministic=True)

    def get_action(self, obs: Dict) -> np.ndarray:
        return self._inner.get_action(obs)

    def set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)


class MyModelBasedAgent(ModelBasedAgent):

    def __init__(self):
        """YOUR CODE HERE
        """
        raise NotImplementedError


    def get_action(self, obs: Dict, planning_env):
        """YOUR CODE HERE
        """
        raise NotImplementedError
    
    def set_seed(self, seed):
        raise NotImplementedError



