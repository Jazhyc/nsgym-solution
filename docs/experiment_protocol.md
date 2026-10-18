# Paper experiment protocol

The paper compares **public NS-Gym benchmark results**. Historical broader
robustness suites are not the paper comparison. Existing frozen releases still
compute their configured diagnostics; their outputs are preserved separately.
Use this specification together with each run's saved manifest/resolved config.
Operations are in [runs.md](runs.md); corrections are in [fix_history.md](fix_history.md).

## Methods and matched conditions

| Training dynamics | Fresh configurations only | Prioritized replay | Replay with mutations |
|---|---|---|---|
| Fixed within each episode | Static DR | Static PLR | Static PLR + mutations |
| Can change within each episode | Temporal DR | Temporal PLR | Temporal PLR + mutations |

Stationary training is the seventh main method. Uniform replay controls are
optional ablations, not additional primary DR baselines.

Static methods use the same saved joint training snapshot pool, sampled
uniformly for fresh levels. Values stay fixed within an episode. Replay restores
physical parameters, not initial states or action randomness. Temporal methods
share the original scheduler/update-function level generator. Temporal FrozenLake
starts at uniform slip; its dynamics can subsequently change. Static FrozenLake
uses the sampled pool probabilities. Fixed-uniform temporal starts are an
accepted change from the random-start pre-DR experiment, not an exact reproduction.

Five training seeds (0–4) per method/task are intended. Compare matched seeds
and protocol versions; do not pool old and revised mutation runs. Ant full runs
were cancelled; its reduced-worker validation is not a paper result.

## Training settings

| Setting | CartPole | FrozenLake | Ant |
|---|---:|---:|---:|
| Learner transitions | 10M | 4M | 500M |
| Training horizon | 500 | 200 | 1000 |
| Actor/critic hidden layers | 64, 64 | 64, 64 | 256, 256 |
| Initial → final learning rate | 3e-4 → 3e-5 | 5e-4 → 1e-5 | 3e-4 → 3e-5 |
| PPO epochs | 10 | 5 | 5 |
| Gradient norm cap | .2 | .5 | 1 |
| Entropy coefficient | 0 | .01 | 0 |
| Normalize observations | yes | no; one-hot | yes |
| Normalize rewards | no | no | yes |
| PLR capacity per worker | 100 | 50 | 200 |
| Replay probability | .3 | .5 | .3 |
| Rank temperature | .9 | .1 | .9 |
| Staleness mixture | .25 | .1 | .25 |

Shared: AdamW, separate tanh actor/critic networks, gamma .99, GAE lambda .95,
PPO clip .2, target KL .02, value coefficient .5, smooth-L1 critic loss,
target batch 2560, minibatch 256. Actual batches round to environment multiples;
budgets can overshoot by a batch and actual frames are logged.

Collection uses **24 environments, two asynchronous groups of 12**, CPU float32,
four learner threads and one collector thread. The separate Ant test uses two
environments only. Local full runs retain all 24 environments and use recorded
Linux fork startup to share import memory.

Hyperparameters are retained from the task recipes, with the explicit protocol
corrections below. No new test-score tuning or best-test checkpoint selection:
evaluate the final-budget checkpoint. During-training deterministic monitoring
uses separate validation seed 72301 and does not select the final model.
Saved manifests retain exact calibration pools, RNG seeds and source digests.

## PLR and mutations

Scores are mean absolute GAE over an identified **complete episode**; fragments
are rejected. The latest score replaces the previous one (alpha 1). Staleness
advances at selection. Only scored levels receive priority; pending levels are
protected from eviction, with an explicit uniform recheck fallback. Warm-up is
10% of per-worker capacity. Equal scores have equal rank priority.

Mutation probability is .25 conditional on exploration; sigma is .2. Temporal
mutations retain the existing scheduler/update-function mutation operator.
Revised static mutations use `relative-reflect-symmetric-v2`:

- Scalar standard deviation is
  `.2 * max(abs(parent), .01 * median_nonzero_absolute_pool_value)`.
- Proposals reflect into calibrated bounds; constant parameters remain constant.
- FrozenLake mutates straight-motion probability and assigns half the remaining
  mass to each lateral slip, preserving symmetry and the probability simplex.

Historical `range-clip-v1` static mutations used the full marginal range and
clipping; these results are a separate cohort. Manifests lacking the new rule
explicitly resolve to legacy behavior. No frozen release is silently rewritten.

## Public evaluation and reporting

Use `ExampleNSCartPole-v0`, `ExampleNSFrozenLake-v0`, and `ExampleNSAnt-v0`, with
episode limits **500 / 100 / 1000**. Final evaluation uses **stochastic actions**,
a frozen policy, and matched environment/policy RNG seeds across methods.
There is no online parameter update or recurrent adaptation during evaluation.

Evaluate 100 episodes at each notification mode (full/change/none), seeds 42–141.
The modes reuse seeds and are not 300 independent dynamics trials. Report mean
return for CartPole/Ant and goal success for FrozenLake (equal to undiscounted
binary return here). IQM is supplementary. Aggregate uncertainty over independent
training seeds; label partial seed sets explicitly. The public Ant example has
effectively stationary realized dynamics, limiting nonstationarity claims.

W&B project: `ml_exp/nsgym-paper`. Runs retain resolved configuration, source
digest, checkpoint, public episode records and training metrics. New PLR runs
also retain worker-scoped level genealogy, selections, feedback, complete and
partial outcomes in `curriculum/worker*.jsonl`, uploaded in the run artifact.
Worker-generated transitions can exceed learner-delivered transitions.
Historical logs cannot retrospectively provide missing ancestry.

## Maintained tools and archived material

`scripts/train.py`, `paper_campaign.py`, and `reevaluate_paper_campaign.py`
provide training/campaign/evaluation entry points. Helpers under
`scripts/diagnostics/prelaunch`, `cartpole_constraints`, `static_plr`, and
`mutation_audit` retain launch, validation and analysis dependencies. Relevant
checks remain under `tests/`. Existing benchmark/profiling utilities are retained.

Superseded investigation scripts, full historical reports and dated audit JSON
are preserved in `dump/20261018-cleanup/`, which is gitignored. Its `manifest.json`
lists original paths. Those scripts are historical evidence, not maintained
entry points. The pre-DR configuration reference remains in `docs/pre_dr_configuration_reference.json`
because launch validation reads it. Checkpoints, experiment outputs and frozen
releases remain in their original locations.
