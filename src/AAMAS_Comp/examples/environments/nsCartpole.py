import gymnasium as gym
from ns_gym.wrappers import NSClassicControlWrapper
from ns_gym.schedulers import ContinuousScheduler, PeriodicScheduler
from ns_gym.update_functions import RandomWalk, IncrementUpdate

def make_env(**kwargs):

    
    change_notification = kwargs.get("change_notification", False)
    delta_change_notification = kwargs.get("delta_change_notification", False)

    base_env = gym.make("CartPole-v1")

    scheduler_1 = ContinuousScheduler()
    scheduler_2 = PeriodicScheduler(period=3)


    update_function1= IncrementUpdate(scheduler_1, k=0.1)
    update_function2 = RandomWalk(scheduler_2)

    tunable_params = {"masspole":update_function1, "gravity": update_function2}

    ns_env = NSClassicControlWrapper(base_env, 
                                     tunable_params,
                                     change_notification=change_notification,
                                     delta_change_notification= delta_change_notification)
    
    return ns_env
    

if __name__ == "__main__":
    env = make_env()

    