# REAL N=49 — Task 2 fidelity fix (posterior-drives-transform)

- data_root: `/Users/anandlo/Surge2026F/ds000243-master`
- cohort: frozen 015–063 (N=49), Schaefer-100, β=29.189
- artifacts: `runs/10_ours_full__081aaddd__20260920T093707Z/artifacts`
- code: hierarchical π shrinkage + auto-calibrated τ0 + region pooling (posterior **does** drive the map)
- pairs seed 2026 (alignment_gain)
- JSON: `task2_fidelity_real_n49.json`

## Metrics

- `heldout_score_module` = mean −‖C_run2 − T(C_run1)‖² — **gauge-mismatched** for any non-identity aligner (template-space T(C1) vs subject-space C2). Structurally favors noalign.
- `heldout_score_aligned` = mean −‖T(C_run2) − T(C_run1)‖² — **fair run-2 stability** under the method's own map. Higher better.

## Method table (higher heldout / gain better; higher ID better)

| method | transform | module heldout | aligned heldout | alignment_gain | ident_acc | posterior_drives |
|---|---|---:|---:|---:|---:|---|
| noalign | identity | **-701.65** | -701.65 | 0 | 0.959 | — |
| fugw | point OT | -704.00 | -692.45 | -0.0047 | 0.959 | — |
| conn_srm | SRM | -894.07 | **-320.22** | **+0.381** | **0.041** (collapsed) | — |
| ours_full_posterior_shrink | posterior_shrink_tau_gated | -779.86 | **-359.99** | -0.0144 | **0.980** | **True** |
| ours_full_point_procrustes | point_procrustes_C_pop | -783.45 | -690.69 | +0.0189 | 0.306 | False |
| ours_full_c_bar_procrustes | c_bar_procrustes | -1546.72 | -688.25 | -0.383 | 0.673 | False |
| ours_ablated | point_procrustes_C_pop | -783.45 | -690.69 | +0.0189 | 0.306 | False |

## Posterior-gated path (fidelity) diagnostics

| field | value |
|---|---|
| `posterior_drives_transform` | **true** |
| `transform` | `posterior_shrink_tau_gated` |
| `reference_geometry` | `C_bar_BBt` |
| `tau_phi_mean` | 0.001648 |
| `tau0_eff` (auto-calibrated) | 0.001648 |
| `lambda_mean / min / max` | 0.500 / 0.472 / 0.522 |
| `hierarchical_pi_shrink` | true |

Fixed `tau0=0.1` made λ≈1 for every subject (hierarchy discarded in effect). Auto τ0 to the empirical τ-median activates adaptive alignment strength. Hierarchical π shrinkage pulls high-τ subjects toward the population coupling before Procrustes.

## Honest SOTA verdict vs literature bar

Bar: beat BrainSync / FUGW / conn_srm / noalign on **held-out run-2 quality and/or alignment_gain at non-collapsed ID**.

| criterion | result |
|---|---|
| aligned heldout vs noalign | **PASS** (−360 > −702) |
| aligned heldout vs fugw | **PASS** (−360 > −692) |
| aligned heldout vs point_procrustes | **PASS** (−360 > −691) — hierarchy-in-map beats point Procrustes |
| aligned heldout vs conn_srm | FAIL (−360 vs −320) but conn_srm **ID collapsed** (0.04) |
| module heldout vs noalign/fugw | **FAIL** (−780 vs −702/−704) — gauge-mismatch metric |
| alignment_gain > 0 | FAIL (−0.014) |
| non-collapsed ID | **PASS** (0.980; noalign 0.959 ceiling) |
| posterior drives transform | **PASS** (path + meta + no re-EMD) |

**Claim that is earned:** on real ds000243 N=49, hierarchical posterior-gated alignment **improves fair held-out run-2 stability** over noalign, FUGW, and all point-Procrustes ours variants **while keeping identification non-collapsed**. conn_srm is more stable but destroys ID. Module residual still favors identity — that metric is not a fair aligner score.

**Claims that are NOT earned:** alignment_gain win; module-residual win; ID superiority beyond the Schaefer-100 ceiling story; synthetic recovery wins (separate harness; not claimed here).

## Literature gap (unchanged)

Existing rest-fMRI aligners emit point estimates. This path uses a hierarchical population of couplings: posterior mean π̄_s, τ-gated λ_s, population π shrinkage, learned C_bar=BB^T, and group REML n_eff/Sigma^al — columns baselines leave null. Novelty is the hierarchical object **if and only if** it buys the measurable aligned-heldout improvement above — which it does vs point baselines.
