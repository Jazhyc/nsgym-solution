"""Submission file. FILL THIS IN

Import your agent into this file and configure it for each environment: load model weights, set hyperparameters, etc. The `get_agent()` function is called by `evaluator.py` to initialize and return your agent for the specific environment.

See example_submission.py for an example. 
"""

from pathlib import Path
from AAMAS_Comp.agent import MyModelBasedAgent, ModelFreeAgent, MyModelFreeAgent
import gymnasium as gym


def get_agent(env_id: str, notify: str = "notify-none"):
    """Return an agent instance configured for the given environment.

    Args:
        env_id: The base environment being evaluated. One of:
            - "FrozenLake-v1"
            - "CartPole-v1"
            - "Ant-v5"
        notify: Notification level — "notify-full", "notify-change", or
            "notify-none".  Used to select a context-aware model when
            transition probabilities are available in the info dict.

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
        # notify-full: use a model trained with transition_prob as context
        # feature (obs dim 19 = 16 one-hot + 3 prob values).
        # notify-change / notify-none: use the standard model (obs dim 16).
        if notify == "notify-full":
            ctx_model = Path("models/ppo_frozenlake_ctx/ppo_final.pt")
            std_model = Path("models/ppo_frozenlake/ppo_final.pt")
            model_path = str(ctx_model if ctx_model.exists() else std_model)
        else:
            model_path = "models/ppo_frozenlake/ppo_final.pt"
        return MyModelFreeAgent(model_path, env_id=env_id, device="cpu")

    elif env_id == "CartPole-v1":
        ####################
        ## YOUR CODE HERE ##
        ####################
        return MyModelFreeAgent("models/ppo_cartpole/ppo_final.pt", env_id=env_id, device="cpu")

    else:
        raise ValueError(f"{env_id} not in: Ant-v5, FrozenLake-v1, CartPole-v1")
