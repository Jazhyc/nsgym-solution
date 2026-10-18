"""Optional complete-episode PPO batches for finite-horizon diagnostics."""

import torch


class CompleteEpisodes:
    """Carry unfinished trajectories forward; never train on a fragment.

    Requires ordered synchronous collection with stable environment rows.
    Collector groups may reuse trajectory IDs; each row has its own namespace.
    Retained tensors are cloned because collectors may reuse their storage.
    """

    def __init__(self):
        self.pending = {}
        self.frames_emitted = 0

    @property
    def pending_frames(self):
        return sum(part.numel() for part in self.pending.values())

    def update(self, batch):
        completed = []
        rows = batch.reshape(-1, batch.shape[-1])
        for worker, row in enumerate(rows.unbind(0)):
            ids = row['collector', 'traj_ids'].reshape(-1)
            for trajectory in ids.unique(sorted=True).tolist():
                if trajectory < 0:
                    raise ValueError('Padded collector batches are unsupported')
                key = (worker, trajectory)
                part = row[ids == trajectory].clone()
                previous = self.pending.pop(key, None)
                expected = 0 if previous is None else previous.numel()
                steps = part['step_count'].reshape(-1)
                if not torch.equal(steps, torch.arange(expected, expected + part.numel(),
                                                       device=steps.device)):
                    raise ValueError(f'Incomplete trajectory {key}: steps={steps.tolist()}, expected={expected}')
                if previous is not None:
                    part = torch.cat([previous, part], dim=0)
                done = part['next', 'done'].reshape(-1)
                if done.any():
                    if not done[-1] or done[:-1].any():
                        raise ValueError(f'Multiple episode endings in {key}')
                    completed.append(part)
                else:
                    self.pending[key] = part
        if not completed:
            return None
        result = torch.cat(completed, dim=0)
        self.frames_emitted += result.numel()
        return result


def estimate_advantages(module, batch, *, bootstrap_timeouts=True):
    """Apply the configured estimator without changing recorded termination."""
    if bootstrap_timeouts:
        module(batch)
        return
    targets = batch.clone()
    targets['next', 'terminated'] = targets['next', 'done'].clone()
    module(targets)
    for key in ['advantage', 'value_target', 'state_value', ('next', 'state_value')]:
        if key in targets.keys(include_nested=True):
            batch[key] = targets[key]
