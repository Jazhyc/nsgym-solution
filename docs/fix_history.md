# Experiment corrections

This is the maintained correction record. Historical reports and detailed audit
outputs are preserved locally under `dump/20261018-cleanup/docs/`, with their
original paths recorded in the archive manifest. Frozen releases/results retain
their original semantics; fixes are not applied retroactively.

| Area | Correction or explicit protocol change | Regression coverage |
|---|---|---|
| PLR feedback identity | Distinguish worker, level generation and episode; reject obsolete feedback | `test_plr_collector_feedback.py` |
| PLR episode scoring | Complete contiguous episodes across batches; reject incomplete fragments instead of worker-batch proxy scores | `test_complete_episode_training.py`, `test_plr_collector_feedback.py` |
| PLR score refresh | Latest completed score (alpha 1), replacing the report's intentional lifetime mean; a reference-alignment change | `test_plr_sampling_integrity.py` |
| PLR selection | Selection-based staleness, scored eligibility/warm-up, protected pending levels, explicit fallback, equal tied ranks and stable weights | `test_plr_sampling_integrity.py` |
| Uniform controls | Score-independent selection/eligible eviction | `test_plr_sampling_integrity.py` |
| FrozenLake simulator | Transition ordering and reset/scheduler RNG consistency with native behavior | `test_paper_protocol.py`, `test_corrected_protocol.py` |
| FrozenLake starts | Fixed uniform temporal starts; accepted distribution change from randomized initial slip | `test_async_prelaunch_protocol.py` |
| Collection | Restore two asynchronous groups of 12; preserve fragment rejection | `test_async_prelaunch_protocol.py` |
| Time limits and evaluation | Explicit task horizons, seeded stochastic final actions, separate public evaluator | `test_corrected_protocol.py`, `test_paper_protocol.py` |
| CartPole physics | Match native finite/positive mass and length constraints and nonnegative gravity; corrected training calibration | `test_cartpole_constraints.py` |
| Static controls | Share exact DR fresh pool; hold parameters fixed for fresh/replayed/mutated episodes | `test_static_plr.py` |
| Static mutation scale | Relative scalar steps and reflection replace full-range clipping; symmetric FrozenLake mutation | `test_mutation_v2.py` |
| Metrics | Unique trajectory namespaces, contiguous episode counters, complete/partial distinction and weighted worker aggregation | `test_training_metrics.py` |
| Curriculum audit | Genealogy, feedback/outcomes and probabilities before staleness reset; audit logging preserves seeded choices | `test_mutation_v2.py` |
| Local execution | Shared-memory CPU startup, bounded concurrency; only terminate stalled uploads after verifying complete scientific outputs | `test_local_mutation_supervisor.py` |

W&B upload stalls have occurred after completed computation. Preserve local
outputs and sync data and label upload-pending states; do not count a shutdown
timeout as proof of failed training or successful upload. Recovery tooling remains
in `scripts/diagnostics/corrected_protocol/recover_finalization.py`.

The pre-DR configuration reference is retained in
`docs/pre_dr_configuration_reference.json`. Main PPO recipes and async
collection are retained; the PLR score refresh, FrozenLake initial condition,
explicit time limits and integrity fixes are documented changes. Do not describe
the entire new study as exact reproduction of the submitted checkpoint training.

The paper uses the public benchmark. Do not quote superseded broader-test results
as the current comparison, mix mutation versions, or treat matched notification
modes as independent training replications. Manuscript claims still need to match
the completed seed sets and public evaluation limitations. Correct implementation
does not imply a guaranteed ranking between methods.
