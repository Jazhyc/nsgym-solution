"""Submission file. FILL THIS IN

Import your agent into this file and configure it for each environment: load model weights, set hyperparameters, etc. The `get_agent()` function is called by `evaluator.py` to initialize and return your agent for the specific environment.

See example_submission.py for an example. 
"""

from pathlib import Path
from AAMAS_Comp.agent import MyModelBasedAgent, ModelFreeAgent
import gymnasium as gym


def get_agent(env_id: str):
    """Return an agent instance configured for the given environment.

    Args:
        env_id: The base environment being evaluated. One of:
            - "FrozenLake-v1"
            - "CartPole-v1"
            - "Ant-v5"

    Returns: 
        Agent: Your initialized agent object.
    """
    if env_id == "Ant-v5":

        ####################
        ## YOUR CODE HERE ##
        ####################
        raise NotImplementedError(f"Sumbission not implemented for {env_id}")

    elif env_id == "FrozenLake-v1":
        ####################
        ## YOUR CODE HERE ##
        ####################
        raise NotImplementedError(f"Sumbission not implemented for {env_id}")

    elif env_id == "CartPole-v1":
        ####################
        ## YOUR CODE HERE ##
        ####################
        raise NotImplementedError(f"Sumbission not implemented for {env_id}")

    else:
        raise ValueError(f"{env_id} not in: Ant-v5, FrozenLake-v1, CartPole-v1")
