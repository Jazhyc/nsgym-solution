"""Submission file. FILL THIS IN

Import your agent into this file and configure it for each environment: load model weights, set hyperparameters, etc. The `get_agent()` function is called by `evaluator.py` to initialize and return your agent for the specific environment.

See example_submission.py for an example. 
"""

from pathlib import Path
from AAMAS_Comp.agent import MyModelBasedAgent, ModelFreeAgent, MyModelFreeAgent
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
        ant_model_path = Path("models/ppo_ant/ppo_final.pt")
        return MyModelFreeAgent(str(ant_model_path), env_id=env_id, device="cpu")

    elif env_id == "FrozenLake-v1":
        ####################
        ## YOUR CODE HERE ##
        ####################
        return MyModelFreeAgent("models/ppo_frozenlake/ppo_final.pt", 
                                device="cpu")

    elif env_id == "CartPole-v1":
        ####################
        ## YOUR CODE HERE ##
        ####################
        return MyModelFreeAgent("models/ppo_cartpole/ppo_final.pt", device="cpu")

    else:
        raise ValueError(f"{env_id} not in: Ant-v5, FrozenLake-v1, CartPole-v1")
