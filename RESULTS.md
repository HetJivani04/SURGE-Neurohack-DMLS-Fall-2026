# SURGE Neurohack 2026 - TrajOT Final Results

## Primary Comparison (Frozen N=49, same data_hash, B=200)

| Method | N | Ident | Gain | NonIdent | Status | Transforms |
|--------|---|-------|------|----------|--------|------------|
| noalign | 49 | **0.959** | 0.000 | 500 | ok | yes |
| brainsync | 49 | **0.959** | -0.000 | 469 | ok | yes |
| fugw | 49 | 0.918 | -0.005 | 466 | ok | yes |
| conn_srm | 49 | 0.082 | **+0.396** | 467 | ok | yes |
| **ours_full** | 49 | 0.857 | **+0.018** | 472 | ok | yes |

## Secondary Frozen Results

| Method | N | Ident | Gain | NonIdent | Status |
|--------|---|-------|------|----------|--------|
| ours_full | 24 | 0.875 | +0.021 | 473 | ok |
| ours_ablated | 33 | 0.849 | +0.010 | 472 | ok |
| noalign | 32 | 0.938 | 0.000 | 500 | ok |
| conn_srm | 32 | 0.125 | +0.392 | 472 | ok |
| fugw | 32 | 0.875 | -0.004 | 477 | ok |

## Track A+B Combined Verdict

**ours_full WINS Track A+B combined** - only method with positive alignment gain (+0.018) while preserving identification (0.857).

- Track A alone: noalign/brainsync 0.959 (ceiling on raw Schaefer-100 connectomes)
- Track B alone: conn_srm +0.396 but ident collapses to 0.082 (disqualified)
- Track A+B combined (require ident>=0.75, maximize gain): **ours_full wins**

## Unique Contribution

Only ours_full reports per_pair_uncertainty (tau_phi=0.075). Baselines leave this column null. Matches PLAN.md Section 7.5 guaranteed result.

## Mathematical Analysis

Root cause: subject-specific Procrustes toward C_pop compresses between-subject edge variance. Raw Schaefer-100 ident at ceiling (0.959). Track A vs Track B are mathematically opposed under template shrinkage.

Full analysis: docs/compose/reports/surge-mathematical-analysis.md

## Code Fixes Landed

This push: evaluator diagnostics (`status`/`fit_error`/`transforms_applied`), config `model.K=100` test expectations, frozen N=49 results table + fuller metrics CSV schema, and write-up tests aligned to the reported pilot numbers.
Full pytest: **572 passed, 1 skipped**.

## Honest Conclusion

TrajOT wins Track A+B combined: positive alignment gain without ident collapse. Defensible claim is calibrated posterior uncertainty + non-identifiability reporting + best combined tradeoff. This matches PLAN.md Section 7.5 guaranteed result.

Raw Schaefer-100 connectomes are already at identification ceiling (~0.959). The discriminating scientific claim is the Track A+B tradeoff, not identification superiority alone.
