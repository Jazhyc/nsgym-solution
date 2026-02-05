import gymnasium as gym
from ns_gym.wrappers import MujocoWrapper
from ns_gym.schedulers import PeriodicScheduler
from ns_gym.update_functions import ExponentialDecay
import numpy as np
import mujoco


def make_env(**kwargs):

    change_notification = kwargs.get("change_notification", False)
    delta_change_notification = kwargs.get("delta_change_notification", False)

    env = gym.make("Ant-v5")
    
    # Define a real update function to make the Ant "floatier" over time
    scheduler = PeriodicScheduler(period=500, start=100, end=500)
    
    # The step size will reduce gravity's pull each step
    updateFn = ExponentialDecay(scheduler=scheduler, decay_rate=0.9)

    tunable_params = {"torso_mass": updateFn}


    ns_env = MujocoWrapper(env,
                           tunable_params,
                           change_notification=change_notification,
                           delta_change_notification=delta_change_notification)
    
    return ns_env


if __name__ == "__main__":
    ns_env = make_env()