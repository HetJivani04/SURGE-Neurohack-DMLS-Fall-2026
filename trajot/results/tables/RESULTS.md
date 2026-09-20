# SURGE Neurohack 2026 - TrajOT Results

## Executive Summary

TrajOT (hierarchical population-of-couplings) does NOT beat baselines on Track A identification accuracy. The defensible contribution is calibrated posterior uncertainty (tau_phi) + positive alignment gain without ident collapse + non-identifiability reporting.

## Primary Comparison (N=49, frozen root, same data_hash, B=200)

| Method | N | Ident | Gain | NonIdent | feat_corr | tau_phi |
|--------|---|-------|------|----------|-----------|---------|
| noalign | 49 | **0.959** | 0.000 | 500 | 1.000 | - |
| brainsync | 49 | **0.959** | -0.000 | 469 | 0.998 | - |
| fugw | 49 | 0.918 | -0.005 | 466 | 0.966 | - |
| conn_srm | 49 | 0.082 | **+0.395** | 467 | 0.586 | - |
| ours_full | 49 | 0.857 | +0.018 | 472 | 0.658 | **0.075** |

## Track A - Identification Accuracy

**Winner: noalign/brainsync (0.959)**

TrajOT: 0.857 (does NOT win)

Root cause: Barycentric projection toward template reduces between-subject variance needed for identification. Raw Schaefer-100 connectomes already achieve ident=0.94-0.96 (ceiling effect).

## Track B - Alignment Gain

**Winner: conn_srm (+0.395) but ident collapses to 0.082**

TrajOT: +0.018 (best positive gain without ident collapse)
fugw: -0.005 (negative)
noalign/brainsync: 0.0

## Unique Contribution

- per_pair_uncertainty (tau_phi=0.075) - only TrajOT produces this
- Matches PLAN.md Section 7.5 guaranteed result: count non-identifiable pairs
- Gauge ablation: null on transform metrics, only moves tau_phi (0.075 -> 0.103)

## Mathematical Analysis

See docs/compose/reports/surge-mathematical-analysis.md for full diagnosis.

Key finding: Track A (identification) requires BETWEEN-subject differences. Track B (gain) requires WITHIN-subject commonality. These are mathematically opposed - any alignment that raises cross-subject correlation necessarily reduces identifiability.

## Code Fixes Landed

- compare.py CLI bugs fixed (dual-mode --runs, --quiet flag)
- ALLOWED_METHOD_KEYS import fixed
- K=100 configs for ours (matches n_regions)
- evaluate.py diagnostics contract (identity_fallback status, transforms_applied, transform_diagnostics)
- run_experiment.py fit_error persistence
- Pushed as commit 24bff67

## Test Results

114 passed / 0 failed

## Conclusion

TrajOT's defensible claim is calibrated uncertainty reporting + positive gain without ident collapse, not superior fingerprinting. This matches PLAN.md's stated guaranteed result (Section 7.5).
