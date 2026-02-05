import numpy as np 
import torch
import pandas as pd


def summarize_metrics(episode_metrics, verbose=True):
    """Make episode metric dataframe. Print metrics if vebose. 

    episode_metrics (dict): Episode metrics dict
    verbose (bool): Print episdoe metrics to std out. Defaults to true.
    """

    episode_df = pd.DataFrame(episode_metrics)

    if verbose: 
        print("Episode Results")
        print(f"Total Reward: {episode_df["rewards"].sum()}")
        print(f"Total Episode Steps: {episode_df["actions"].count()}")
        print(f"Mean Decision Time: {episode_df["decision_time"].mean()} +/- {episode_df["decision_time"].std()}")


    return episode_df



    
