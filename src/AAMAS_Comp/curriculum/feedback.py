"""Episode-specific learner feedback for asynchronous, per-worker PLR buffers."""

import numpy as np

PLR_KEYS = ("plr_worker", "plr_level", "plr_episode")


class EpisodeScores:
    """Accumulate advantages and reject episodes with missing transitions.

    Completeness uses the environment step counter across batches. A skipped
    async batch must never turn an episode fragment into a whole-episode score.
    Set require_complete=False only for explicit legacy diagnostic controls.
    """

    def __init__(self, require_complete=True):
        self.pending = {}
        self.require_complete = require_complete
        self.pending_steps = {}
        self.incomplete_episodes = 0
        self.abandoned_episodes = 0

    def update(self, batch, queues):
        identifiers = np.stack(
            [batch[key].detach().cpu().numpy().reshape(-1) for key in PLR_KEYS], axis=1
        )
        advantages = batch["advantage"].detach().abs().cpu().numpy().reshape(-1)
        done = batch["next", "done"].detach().cpu().numpy().reshape(-1)
        steps = (batch["step_count"].detach().cpu().numpy().reshape(-1)
                 if self.require_complete else None)
        sent = 0
        unique = sorted(np.unique(identifiers, axis=0), key=lambda x: (x[0], x[2]))
        for identifier in unique:
            key = tuple(int(v) for v in identifier)
            mask = (identifiers == identifier).all(axis=1)
            if self.require_complete:
                # Each worker's stream is ordered. A later episode means an
                # earlier unfinished episode lost its terminal batch.
                abandoned = [old for old in self.pending
                             if old[0] == key[0] and old[2] < key[2]]
                for old in abandoned:
                    self.pending.pop(old)
                    self.pending_steps.pop(old, None)
                    self.abandoned_episodes += 1
                expected, complete = self.pending_steps.get(key, (0, True))
                observed = steps[mask]
                complete = complete and np.array_equal(
                    observed, np.arange(expected, expected + len(observed)))
                self.pending_steps[key] = (int(observed[-1]) + 1, bool(complete))
            total, count = self.pending.get(key, (0.0, 0))
            total += float(advantages[mask].sum())
            count += int(mask.sum())
            if done[mask].any():
                worker, level, episode = key
                complete = self.pending_steps.pop(key, (0, True))[1]
                if complete:
                    queues[worker].put((level, episode, total / count))
                    sent += 1
                else:
                    self.incomplete_episodes += 1
                self.pending.pop(key, None)
            else:
                self.pending[key] = (total, count)
        return sent
