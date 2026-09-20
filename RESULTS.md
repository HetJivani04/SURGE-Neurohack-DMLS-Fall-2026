# SURGE Neurohack 2026 - TrajOT Honest Results

## Track A: Identification Accuracy (N=49, same data_hash, B=200)

| Method | Ident | Gain | NonIdent | Status |
|--------|-------|------|----------|--------|
| noalign | **0.959** | 0.000 | 500 | ok |
| brainsync | **0.959** | -0.00005 | 469 | ok (near no-op on connectomes) |
| fugw | 0.918 | -0.005 | 466 | ok |
| conn_srm | 0.082 | **+0.395** | 467 | ok (ident collapses to chance) |
| ours_full | 0.857 | +0.018 | 472 | ok (K=100, beta=29.19, tau_phi=0.075) |

**Track A verdict: TrajOT does NOT win.** noalign/brainsync score 0.959 vs ours 0.857.

## Track B: Alignment Gain

- conn_srm: +0.395 (highest gain but ident collapses to 0.082 - not usable)
- ours_full: +0.018 (only identity-preserving method with positive gain)
- noalign/brainsync: 0.0
- fugw: -0.005

**Track B verdict: Partial win.** Ours has best positive gain without ident collapse.

## Unique Contribution: Per-Pair Uncertainty

Only ours_full reports `per_pair_uncertainty` (tau_phi=0.075 full, 0.103 ablated).
Baselines cannot produce this column. This matches PLAN.md Section 7.5 guaranteed result.

## Mathematical Diagnosis

Root cause: barycentric projection toward template reduces between-subject variance needed for identification. Track A (fingerprinting) needs BETWEEN-subject differences; Track B gain needs WITHIN-subject commonality. These are mathematically opposed.

Ceiling effect: raw Schaefer-100 connectomes already ident=0.94-0.96 at N=30-49.

Full analysis: docs/compose/reports/surge-mathematical-analysis.md
Primary table: trajot/results/tables/comparison_pilot_primary.csv
Table write-up: trajot/results/tables/RESULTS.md

## Code Fixes Landed

Pushed as `24bff67`: evaluate.py diagnostics, run_experiment.py fit_error, ours.py C_pop path, configs K=100, compare.py CLI, ALLOWED_METHOD_KEYS import.

Tests: 114 passed / 0 failed.

## Honest Conclusion

TrajOT's defensible claim is calibrated posterior uncertainty + positive gain without ident collapse + non-identifiability reporting. It does NOT beat baselines on identification accuracy. This matches PLAN.md's stated guaranteed result (Section 7.5), not a claim of superior fingerprinting.

Note: `trajot/reports/RESULTS.md` remains the pre-experiment protocol document (claim discipline / empty-by-design template). Pilot results above come from the frozen N=49 comparison runs.
