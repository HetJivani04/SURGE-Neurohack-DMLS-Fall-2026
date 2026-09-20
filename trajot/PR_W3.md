# W3: Baselines + Evaluation Harness (Part of #2)

## Summary
This PR implements the W3 scope under the `trajot/` workspace:
- Baseline interface and registry with concrete method modules.
- Evaluation core for identification, permutation nulls, alignment gain, controls, and metrics contract validation.
- End-to-end evaluation CLI (`scripts/evaluate.py`) with synthetic mode for development-time verification.
- W3-focused test coverage for baseline behavior, evaluation invariants, and script wiring.

Linked issue: #2 (W3 scope)

## Scope and Structure
All additions are inside `trajot/`:
- `trajot/src/trajot/baselines/`
- `trajot/src/trajot/eval/`
- `trajot/scripts/evaluate.py`
- `trajot/tests/baselines/`
- `trajot/tests/eval/`
- `trajot/tests/scripts/test_evaluate_script.py`

## What was added
### Baseline interface and methods
- `baseline` protocol with `fit` / `transform` contract.
- Registry helpers (`register_baseline`, `get_baseline`) and synthetic sanity baseline (`FakeBaseline`).
- Implemented baseline modules:
  - `NoAlign`
  - `BrainSync`
  - `FUGW` (CPU-safe adapter placeholder path)
  - `ConnSRM`
  - `AblatedModel` (bridge placeholder while W2 training artifacts are pending)

### Evaluation core
- Fold and pair helpers:
  - `sample_pairs`
  - `make_two_run_splits`
- Identification:
  - `identify_subjects` with cosine scoring over vectorized connectomes.
- Permutation testing:
  - `permutation_p` with exact finite-sample formula `(count_ge + 1) / (B + 1)`.
- Summary metrics:
  - `alignment_gain`
  - `nonidentifiable_pairs`
  - control outputs via `random_permutation_control`
- Metrics payload contract checks:
  - top-level and per-method key validation
  - writer utility

### Evaluate script
- `scripts/evaluate.py` now supports:
  - `--config`
  - `--methods`
  - `--n-jobs`
  - `--pairs`
  - `--seed`
  - `--synthetic`
  - `--output`
- Uses one shared harness path for all methods, including `noalign` and `fake`.
- Respects CPU-first policy for float64-heavy alignment workloads.

## Tests
Added/updated tests:
- `trajot/tests/baselines/test_w3_baselines.py`
- `trajot/tests/eval/test_w3_eval_core.py`
- `trajot/tests/scripts/test_evaluate_script.py`

Current result:
- `python -m pytest -q` => 109 passed

## Checklist vs W3 expectations
### Completed
- Baseline package implemented with common interface and registry.
- Evaluation core implemented (identification, permutation null, gain, controls, identifiability count).
- Evaluate CLI implemented and tested on synthetic path.
- Contract-level tests added for edge cases and reproducibility behavior.

### Known boundaries (intentional at this stage)
- `scripts/run_experiment.py` experiment dispatch remains the W0 stub path and is not yet wired to execute W3 methods directly.
- Real-data experiment table generation (Phase 2 outputs) is not part of this PR.
- W2 model/inference artifacts are not consumed yet beyond the ablated baseline bridge placeholder.

## Commit stack
- `3e2dba0` Feat: Add W3 baseline registry and methods
- `9fbe27b` Feat: Add W3 evaluation core modules
- `73c3566` Feat: Implement W3 evaluate CLI harness
- `3a9a884` Test: Add W3 baseline and eval core tests
- `d9b27a1` Test: Add W3 evaluate script synthetic test

## Notes for reviewers
- This PR is intentionally scoped to W3 module and harness delivery so W2 and W4 can proceed in parallel without interface churn.
- The synthetic execution path is deliberate for Phase 1 development speed and deterministic validation.
