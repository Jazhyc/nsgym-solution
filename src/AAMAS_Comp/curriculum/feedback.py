"""Episode-specific learner feedback for asynchronous, per-worker PLR buffers."""

import numpy as np

PLR_KEYS = ("plr_worker", "plr_level", "plr_episode")


class EpisodeScores:
    """Accumulate raw absolute advantages across batches until an episode ends."""

    def __init__(self):
        self.pending = {}

    def update(self, batch, queues):
        identifiers = np.stack(
            [batch[key].detach().cpu().numpy().reshape(-1) for key in PLR_KEYS], axis=1
        )
        advantages = batch["advantage"].detach().abs().cpu().numpy().reshape(-1)
        done = batch["next", "done"].detach().cpu().numpy().reshape(-1)
        sent = 0
        for identifier in np.unique(identifiers, axis=0):
            key = tuple(int(v) for v in identifier)
            mask = (identifiers == identifier).all(axis=1)
            total, count = self.pending.get(key, (0.0, 0))
            total += float(advantages[mask].sum())
            count += int(mask.sum())
            if done[mask].any():
                worker, level, episode = key
                queues[worker].put((level, episode, total / count))
                self.pending.pop(key, None)
                sent += 1
            else:
                self.pending[key] = (total, count)
        return sent
