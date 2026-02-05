
from typing import Dict
from AAMAS_Comp.base_agent import ModelBasedAgent, ModelFreeAgent


"""Implement your code here. 
"""

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
    

class MyModelFreeAgent(ModelFreeAgent):

    def __init__(self):
        """YOUR CODE HERE
        """
        raise NotImplementedError

    def get_action(self, obs):
        """YOUR CODE HERE
        """
        return None
    
    def set_seed(self, seed):
        raise NotImplementedError

    




