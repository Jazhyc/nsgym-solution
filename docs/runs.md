# Run instructions

Authoritative output directory:
`results/20261018-static-mutation-v2-local-r2`.
W&B project `ml_exp/nsgym-paper`, group
`20261018-static-mutation-v2-local-r2-final`.

Only **static PLR with revised mutations** is being trained: CartPole and
FrozenLake, seeds 0–4, ten full runs. There are no full-budget Ant runs or baseline reruns in this queue. A separate reduced-worker Ant validation is a test only.
Existing CartPole jobs on Habrok remain untouched. The validated scientific
release is frozen locally at `ROOT/release`, source SHA256
`eea392a5f1c2b1dd5862fbd97f0d90e4be6157194a581fd6fe4258a2ce8aa7f7`.
The new orchestrator and runtime startup files live outside this release and
have their own digests in `launch_record.json` and `manifest.json`.

Budgets remain 10M CartPole / 4M FrozenLake transitions, 24 environments in two
asynchronous groups of 12, four learner threads, and one collector thread.
PPO settings, calibration pools, test cases, stochastic final actions, and
horizons are identical to the prepared Habrok ablation. Sigma remains 0.2 and
mutation probability 0.25 conditional on exploration.

## Runtime and concurrency

The local machine has eight physical cores / 16 logical threads and 23 GiB RAM.
Default independent worker startup used approximately 21 GiB proportional set
size (PSS) and caused swapping. That initial run was stopped early, its local
queue paused, and its diagnostics preserved in
`results/20261018-static-mutation-v2-local`; it is not part of the comparison.

The active release uses Linux CPU `fork` startup via an isolated
`runtime/sitecustomize.py`. This shares imported Python/PyTorch memory between
workers. The actual CPU CartPole pilot peaked at **2.25 GiB PSS** (RSS sums
overcount the shared pages), completed normally, reported all 24 workers and
recorded 160 created mutants plus 27 mutant-origin replays. A full 24-worker
FrozenLake pilot also passed delayed-feedback checks, stochastic evaluation,
and online W&B artifact/log verification. Its W&B run is
[hiejlluc](https://wandb.ai/ml_exp/nsgym-paper/runs/hiejlluc).

This is a recorded runtime change, not a reduction in environment count or a
change to PPO/curriculum settings. As with all asynchronous runs, process
scheduling does not promise identical trajectories across machines.

The persistent supervisor starts one full run, observes 90 seconds of resource
usage, then admits at most **two concurrent full runs** when:

* Available RAM exceeds `max(6 GiB, 1.35 × observed peak run PSS) + 4 GiB`.
* Recent CPU consumption leaves headroom and the machine is not actively
  swapping. Cold swapped-out pages from earlier processes do not alone block
  admission.

Order is CartPole seed 0, FrozenLake seed 0, then the corresponding pair for
each subsequent seed. The supervisor fills a free slot subject to the same
checks. It retains per-run logs, resource measurements, exit state, and PIDs.

## Monitoring and stopping

Live state: `ROOT/status.json`. Queue controls: `ROOT/control.json`.
Supervisor log: `ROOT/supervisor.log`. Individual logs: `ROOT/logs/`.
The [launch record](../dump/20261018-cleanup/docs/results/20261018-static-mutation-v2-local-launch.json)
records the complete run matrix, runtime, source, and W&B group.

```bash
tmux attach -t nsgym-static-mut-v2-r2
```

Setting `pause_new_runs` to `true` in `control.json` pauses future starts while
letting active runs finish. `max_parallel: 1` prevents another concurrent run;
it does not terminate an active run. Existing completed checkpoints are never
overwritten. On restart, the supervisor refuses to duplicate live recorded
processes or restart a checkpoint with incomplete evaluation.

Training metrics, complete/partial episode outcomes, ancestry, replay by origin,
final broad trajectories and public evaluation are saved and uploaded to W&B.
If final W&B shutdown stalls for five minutes **after** the checkpoint and all
4,000 broad / 300 public episodes are verified present, the supervisor stops
that process tree and records `compute_complete_upload_pending`. It does not
pretend that an unfinished upload succeeded or interrupt training/evaluation
on this timeout. Local outputs and W&B sync data remain available for recovery.

After all ten scientific outputs are complete, the supervisor builds a local
`comparison.json` using the frozen aggregation implementation. Incomplete runs
are marked failed and are not silently retrained or included as complete.
