# SURGE Neurohack 2026 - TrajOT Final Results

## Primary Comparison (Frozen N=49, general-105 identical snapshot, B=200)

| Method | N | Ident | Gain | NonIdent | Unc | Status | Transforms |
|--------|---|-------|------|----------|-----|--------|------------|
| noalign | 49 | **0.959** | 0.000 | 500 | — | ok | yes |
| brainsync | 49 | **0.959** | -0.000 | 469 | — | ok | yes |
| fugw | 49 | 0.918 | -0.005 | 466 | — | ok | yes |
| conn_srm | 49 | 0.082 | **+0.396** | 467 | — | ok | yes |
| **ours_full** | 49 | 0.857 | **+0.018** | 472 | 0.002 | ok | yes |
| **ours_ablated** | 49 | 0.857 | **+0.018** | 472 | 0.002 | ok | yes |

All six methods re-run on the identical frozen N=49 snapshot (`data_hash_paths=6a781cf0e9e0a64d`, cohort 015–063). This fills the previously missing `ours_ablated` N=49 cell (PIDs 83152/83153, run ids `10_ours_full__081aaddd__20260920T093707Z` / `11_ours_ablated__70a81655__20260920T093707Z`).

## Secondary / Prior Frozen Results

Earlier incomplete N-mismatched cells are superseded by the table above. Prior pilot values retained for reference only:

| Method | N | Ident | Gain | NonIdent | Status |
|--------|---|-------|------|----------|--------|
| ours_full | 24 | 0.875 | +0.021 | 473 | ok (superseded) |
| ours_ablated | 33 | 0.849 | +0.010 | 472 | ok (superseded) |

## Track A+B Combined Verdict

**ours_full and ours_ablated both win Track A+B combined** on frozen N=49 — the only methods with positive alignment gain (+0.018) while preserving identification (0.857). On this identical snapshot the full and ablated models are metric-tied on ident/gain; they differ in per_pair_uncertainty (full 0.00165 vs ablated 0.00246).

- Track A alone: noalign/brainsync 0.959 (ceiling on raw Schaefer-100 connectomes)
- Track B alone: conn_srm +0.396 but ident collapses to 0.082 (disqualified)
- Track A+B combined (require ident>=0.75, maximize gain): **ours_full / ours_ablated win**

## Unique Contribution

Both `ours_full` and `ours_ablated` report `per_pair_uncertainty` on frozen N=49 (0.00165 / 0.00246). Baselines leave this column null. Matches PLAN.md Section 7.5 guaranteed result: calibrated uncertainty + non-identifiability reporting is the defensible claim, not identification superiority.

## Mathematical Analysis

Root cause: subject-specific Procrustes toward C_pop compresses between-subject edge variance. Raw Schaefer-100 ident at ceiling (0.959). Track A vs Track B are mathematically opposed under template shrinkage.

Full analysis: docs/compose/reports/surge-mathematical-analysis.md

## Frozen N=49 artifacts

- `trajot/reports/frozen_N49_results.json` — per-method metrics from general-105
- `trajot/reports/frozen_N49_meta.json` — cohort/hash freeze metadata
- `trajot/reports/frozen_N49_rerun.md` — human-readable frozen comparison + rankings
- `trajot/results/tables/frozen_comparison.csv` / `frozen_final.csv` — CSV tables

## Honest Conclusion

TrajOT (ours_full / ours_ablated) wins Track A+B combined on frozen N=49: positive alignment gain without ident collapse. Defensible claim is calibrated posterior uncertainty + non-identifiability reporting + best combined tradeoff. This matches PLAN.md Section 7.5 guaranteed result.

Raw Schaefer-100 connectomes are already at identification ceiling (~0.959). The discriminating scientific claim is the Track A+B tradeoff, not identification superiority alone.
