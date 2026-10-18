"""Completed-episode metrics and latest per-worker curriculum observations."""

import numpy as np


class TrainingEpisodes:
    """Accumulate raw rewards across collector batches, without pooling steps."""

    def __init__(self, frozenlake=False):
        self.frozenlake = frozenlake
        self.pending = {}
        self.completed = 0
        self.pending_steps = {}
        self.incomplete = 0
        self.abandoned = 0

    def update(self, batch):
        trajectories = batch["collector", "traj_ids"].detach().cpu().numpy().reshape(-1)
        rewards = batch["next", "reward"].detach().cpu().numpy().reshape(-1)
        done = batch["next", "done"].detach().cpu().numpy().reshape(-1)
        terminated = batch["next", "terminated"].detach().cpu().numpy().reshape(-1)
        step_counts = (batch["step_count"].detach().cpu().numpy().reshape(-1)
                       if "step_count" in batch.keys() else None)
        if "plr_worker" in batch.keys():
            workers = batch["plr_worker"].detach().cpu().numpy().reshape(-1)
        elif "env_worker" in batch.keys():
            workers = batch["env_worker"].detach().cpu().numpy().reshape(-1)
        elif batch.ndim > 2:
            # Stacked synchronous collector groups can share initial traj IDs.
            workers = np.broadcast_to(np.arange(np.prod(batch.shape[:-1])).reshape(
                *batch.shape[:-1], 1), tuple(batch.shape)).reshape(-1)
        else:
            workers = np.zeros_like(trajectories)
        identifiers = np.stack([workers, trajectories], axis=1)
        finished = []
        for worker, trajectory in np.unique(identifiers, axis=0):
            if trajectory < 0:  # padded collector frames
                continue
            key = (int(worker), int(trajectory))
            mask = (workers == worker) & (trajectories == trajectory)
            complete = True
            if step_counts is not None:
                abandoned = [old for old in self.pending if old[0] == key[0] and old[1] < key[1]]
                for old in abandoned:
                    self.pending.pop(old)
                    self.pending_steps.pop(old, None)
                    self.abandoned += 1
                expected, complete = self.pending_steps.get(key, (0, True))
                observed = step_counts[mask]
                complete = complete and np.array_equal(observed, np.arange(expected, expected + len(observed)))
                self.pending_steps[key] = (int(observed[-1]) + 1, bool(complete))
            total, steps = self.pending.get(key, (0., 0))
            total += float(rewards[mask].sum())
            steps += int(mask.sum())
            if done[mask].any():
                if complete:
                    finished.append((total, steps, bool(terminated[mask].any())))
                else:
                    self.incomplete += 1
                self.pending.pop(key, None)
                self.pending_steps.pop(key, None)
            else:
                self.pending[key] = (total, steps)
        self.completed += len(finished)
        metrics = {"train/episodes_completed_total": self.completed,
                   "train/episodes_completed_batch": len(finished)}
        if step_counts is not None:
            metrics.update({"train/incomplete_episode_metrics_total": self.incomplete,
                            "train/abandoned_episode_metrics_total": self.abandoned,
                            "train/pending_episode_metrics": len(self.pending)})
        if finished:
            returns, lengths, terminal = np.asarray(finished).T
            metrics.update({"train/episode_return_mean": float(returns.mean()),
                            "train/episode_length_mean": float(lengths.mean()),
                            "train/timeout_rate": float((terminal == 0).mean())})
            if self.frozenlake:
                success = returns > 0
                metrics.update({"train/success_rate": float(success.mean()),
                                "train/hole_rate": float(((terminal == 1) & ~success).mean())})
        return metrics


class WorkerCurriculumMetrics:
    """Use one latest snapshot per worker, with explicit reporting coverage.

    Sums are observed cumulative worker counters, not exact synchronized
    global counts; collectors can run ahead of the learner.
    """

    COUNTERS = ("plr/applied_scores", "plr/discarded_scores", "plr/episode_count",
                "plr/replay_count", "accel/mutation_count", "plr/forced_replay_count",
                "plr/levels_created", "plr/levels_first_scored", "plr/unscored_evictions")

    def __init__(self, n_workers):
        self.n_workers = n_workers
        self.latest = {}

    def update(self, samples):
        for sample in samples:
            worker = int(sample["plr/worker_id"])
            if not 0 <= worker < self.n_workers:
                raise ValueError(f"Invalid curriculum worker ID: {worker}")
            self.latest[worker] = sample
        if not self.latest:
            return {}
        rows = list(self.latest.values())
        keys = set().union(*(row.keys() for row in rows)) - {"plr/worker_id"}
        metrics = {key: float(np.mean([row[key] for row in rows if key in row])) for key in keys}
        metrics["plr/workers_reporting"] = len(rows)
        metrics["plr/worker_coverage"] = len(rows) / self.n_workers
        for key in self.COUNTERS:
            metrics[f"{key}_observed_total"] = sum(row.get(key, 0) for row in rows)
        for origin in ('fresh', 'mutated'):
            totals = {}
            for counter in ('selection_count', 'replay_count', 'completed_count', 'return_sum',
                            'steps_sum', 'partial_count'):
                key = f'plr/{origin}_{counter}'
                totals[counter] = sum(row.get(key, 0) for row in rows)
                metrics[key+'_observed_total'] = totals[counter]
            if totals['completed_count']:
                metrics[f'plr/{origin}_return_mean_observed'] = totals['return_sum']/totals['completed_count']
                metrics[f'plr/{origin}_episode_length_mean_observed'] = totals['steps_sum']/totals['completed_count']
        replays = metrics['plr/replay_count_observed_total']
        if replays:
            metrics['plr/mutated_replay_share_observed'] = metrics['plr/mutated_replay_count_observed_total']/replays
        applied = metrics["plr/applied_scores_observed_total"]
        discarded = metrics["plr/discarded_scores_observed_total"]
        if applied + discarded:
            metrics["plr/feedback_discard_fraction_observed"] = discarded / (applied + discarded)
        episodes = metrics["plr/episode_count_observed_total"]
        if episodes:
            for metric, counter in (("plr/replay_fraction", "plr/replay_count"),
                                    ("accel/mutation_fraction", "accel/mutation_count"),
                                    ("plr/forced_replay_fraction", "plr/forced_replay_count")):
                metrics[metric] = metrics[f"{counter}_observed_total"] / episodes
        created = metrics["plr/levels_created_observed_total"]
        if created:
            metrics["plr/new_levels_first_scored_fraction_observed"] = (
                metrics["plr/levels_first_scored_observed_total"] / created)
        return metrics
