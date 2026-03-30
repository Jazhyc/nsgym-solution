from __future__ import annotations

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from torchrl.envs import ParallelEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp


def run_eval_shards(
    actor,
    eval_shard_factories: list[list],
    eval_rollout_steps: int,
    device: torch.device,
) -> dict:
    """Run evaluation across all shards and return aggregated metrics.

    Builds each shard's env on demand, runs a deterministic rollout, then
    tears it down before moving to the next shard.  At most one shard's worth
    of subprocesses is alive at once.

    Args:
        actor: Policy module (callable, TorchRL actor).
        eval_shard_factories: List of shards, each a list of env factory callables.
        eval_rollout_steps: Max steps per episode.
        device: Torch device for tensor allocation.

    Returns:
        dict with eval/reward_mean, eval/reward_iqm, eval/reward_per_step,
        eval/step_count, n_shards, n_configs.
    """
    all_returns: list[torch.Tensor] = []
    all_lengths: list[torch.Tensor] = []
    n_shards = len(eval_shard_factories)

    eval_pbar = tqdm(eval_shard_factories, desc="Eval", leave=False, total=n_shards, unit="shard")
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for factories in eval_pbar:
            n_envs = len(factories)
            if n_envs == 1:
                eval_env = factories[0]()
            else:
                eval_env = ParallelEnv(
                    num_workers=n_envs,
                    create_env_fn=factories,
                    serial_for_single=True,
                )

            shard_returns = torch.zeros(n_envs, device=device)
            shard_lengths = torch.zeros(n_envs, dtype=torch.long, device=device)
            shard_finished = torch.zeros(n_envs, dtype=torch.bool, device=device)

            td = eval_env.reset()
            for _ in range(eval_rollout_steps):
                td = actor(td)
                td = eval_env.step(td)

                reward = td["next", "reward"].squeeze(-1)
                done = (td["next", "terminated"] | td["next", "truncated"]).squeeze(-1)

                shard_returns += reward * ~shard_finished
                shard_lengths += (~shard_finished).long()
                shard_finished = shard_finished | done

                if shard_finished.all():
                    break

                td = step_mdp(td)

            all_returns.append(shard_returns)
            all_lengths.append(shard_lengths)
            eval_env.close()
    eval_pbar.close()

    ep_returns = torch.cat(all_returns)
    ep_lengths = torch.cat(all_lengths)

    reward_mean = ep_returns.mean().item()
    steps = ep_lengths.float().mean().item()

    q1, q3 = torch.quantile(ep_returns, torch.tensor([0.25, 0.75], device=device))
    iqm_mask = (ep_returns >= q1) & (ep_returns <= q3)
    reward_iqm = ep_returns[iqm_mask].mean().item() if iqm_mask.any() else reward_mean

    return {
        "eval/reward_mean": reward_mean,
        "eval/reward_iqm": reward_iqm,
        "eval/reward_per_step": reward_mean / max(steps, 1),
        "eval/step_count": steps,
        "n_shards": n_shards,
        "n_configs": ep_returns.shape[0],
    }


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



    
