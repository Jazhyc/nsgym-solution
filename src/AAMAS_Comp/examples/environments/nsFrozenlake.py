from ns_gym.wrappers import NSFrozenLakeWrapper
from ns_gym.update_functions import DistributionDecrementUpdate
from ns_gym.schedulers import ContinuousScheduler
import gymnasium as gym


def get_ns_wrapper(env):
    curr = env
    while hasattr(curr, "env"):
        if isinstance(curr, NSFrozenLakeWrapper):
            return curr
        curr = curr.env
    return None


def make_env(**kwargs):

    change_notification = kwargs.get("change_notification", False)
    delta_change_notification = kwargs.get("delta_change_notification", False)

    base_env = gym.make("FrozenLake-v1", disable_env_checker=True)
    scheduler = ContinuousScheduler()

    k = 0.025
    update_fn  = DistributionDecrementUpdate(scheduler=scheduler, k=k)

    tunable_params = {"P": update_fn}

    ns_env = NSFrozenLakeWrapper(base_env,
                                tunable_params=tunable_params,
                                change_notification=change_notification,
                                delta_change_notification=delta_change_notification,
                                initial_prob_dist=[1,0,0])
    
    return ns_env
    

if __name__ == "__main__":
    env = make_env()

    