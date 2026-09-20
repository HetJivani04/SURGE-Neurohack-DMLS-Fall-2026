# SURGE Neurohack 2026 — TrajOT Final Results

**Official write-up:** [`trajot/reports/RESULTS.md`](trajot/reports/RESULTS.md)
**Statistical tables:** [`trajot/results/tables/REAL_sota_stats.md`](trajot/results/tables/REAL_sota_stats.md) · [`REAL_sota_stats.json`](trajot/results/tables/REAL_sota_stats.json) · [`REAL_n49_gap_sota.json`](trajot/results/tables/REAL_n49_gap_sota.json)

## Task SOTA (REAL N=49, ds000243, posterior_shrink, same Q_s on both runs)

| Method | Ident | Scan-rescan after | Gain | τ_φ | Group n_eff | Status |
|--------|------:|------------------:|-----:|----:|------------:|--------|
| noalign | 0.959 | 0.642 | 0.000 | — | — | valid baseline |
| BrainSync | 0.959 | 0.643 | +0.0007 | — | — | near no-op on connectomes |
| FUGW | 0.918 | 0.620 | −0.005 | — | — | valid baseline |
| conn_srm | 0.082 | 0.850 | +0.395 | — | — | **INVALID — identity collapse** |
| **ours_full_posterior_shrink** | **0.980** | **0.685** | −0.035 | **0.0092** | **7.45 < 12** | **primary** |

## Statistical verdict (paired bootstrap B=10,000)

| Comparison | Ident Δ [95% CI] | Reliability Δ [95% CI] |
|---|---|---|
| vs noalign | +0.020 **[0.000, 0.041]** (trend) | **+0.043 [0.038, 0.049]** |
| vs BrainSync | +0.020 **[0.000, 0.041]** (trend) | **+0.043 [0.037, 0.049]** |
| vs FUGW | +0.061 **[0.000, 0.061]** (trend) | **+0.065 [0.058, 0.072]** |

**Claim: task_sota_reliability.** Posterior-gated hierarchical alignment improves scan-rescan reliability vs noalign, BrainSync, and FUGW — all three reliability CIs exclude 0. Identification point estimate is higher but CI touches 0 → **trend only, not ID superiority.**

## Gap columns (PLAN §1 unclaimed object)

Only the hierarchical model emits **τ_φ** (mean 0.0092) and **group REML n_eff = 7.45 < S = 12** (ci_ratio 1.28) on real posteriors. Baselines return an alignment for every subject with **no** uncertainty flag. This fills the Thual 2025 / BrainSync 2018 / Takeda 2025 documented gap: point estimates only; individual-level reliability unreported.

## Synthetic coupling recovery (alignment SOTA sidecar)

- Coupling recovery: **ours 0.70** vs point/EMD 0.03, FUGW 0.028, random 0.02
- Identity is Bayes-optimal for ‖C−C_true‖ under C=C_true+E → planted assignment recovery is the fair aligner metric
- Coverage@0.9 / AUROC-τ **FAIL** — not headlined

## Do not claim

- Identification superiority (bootstrap CI includes 0)
- Calibrated posterior coverage on real rest
- Gain-null nonident counts as scientific rates
- conn_srm as a competitor on alignment gain

## Point-estimate path (superseded narrative)

Earlier frozen point-Procrustes rows (ours_full ident 0.857, gain +0.018) remain in `trajot/reports/results_table.md` for audit. The hierarchical posterior_shrink path is the primary result; the point path is the ablation that shows the posterior drives the map.

---

Full claim-discipline appendix (attributions, finitely many optima, Marek, band prior): `trajot/reports/RESULTS.md`.
