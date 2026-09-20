# REAL N=83 gap + SOTA — honest scale-up

**Status: COMPLETE — full-cohort posterior_shrink active (83/83). N=83 is PRIMARY; `task_sota_reliability` EARNED (10k reliability CIs exclude 0 vs noalign/BrainSync/FUGW); identification NOT claimed (same-map protocol map-invariant).**

## N=83 COMPLETE verdict (2026-09-20) — PRIMARY

- Run `10_ours_full__9d7dab12__20260920T140613Z` — artifacts cover **83/83** strict two-run subjects; `ours_full_posterior_shrink_entropy`, path `posterior_shrink_tau_gated`, λ_mean **0.5001 [0.4865, 0.5138]** (row-entropy gate), τ_φ mean **0.0345**.
- **`task_sota_reliability` EARNED (N=83):** reliability **0.6975** vs noalign 0.6455 / BrainSync 0.6461 / FUGW 0.6215; 10k paired bootstrap Δrel **+0.0519 [0.0480, 0.0558] / +0.0513 [0.0472, 0.0552] / +0.0759 [0.0705, 0.0814]** — all CIs exclude 0. τ-λ variant **0.6577**, Δrel +0.0122/+0.0116/+0.0362 (also all CIs exclude 0). N=49 replication holds (+0.043/+0.043/+0.065).
- **Identification NOT claimed (withdrawn):** the same-map ident protocol is **map-invariant** — fitted / permuted / Haar-random per-subject maps all give **1.0 (83/83)**; identity 0.9157; single common map 0.9036. Reported only as identity preservation.
- **Mechanism:** `T_λ(C)=(1−λ)C+λQᵀCQ` → `M_λ=(1−λ)I+λR`, R orthogonal; `|g(θ)|²=1−2λ(1−λ)(1−cosθ)` = `cos²(θ/2)` at λ=1/2 (**unique maximally-filtering interior point**); real λ-sweep peaks exactly at 0.50. Evidence: `docs/compose/reports/surge-mathematical-analysis.md`; write-up `trajot/reports/RESULTS.md` §5b.
- **Mechanism controls @ λ=0.5 (real N=83):** raw 0.6455; Haar-random maps 0.6454 (zero gain); permuted fitted maps 0.6708 (≈half the gain); single common template map 0.6919 → **89% of the gain is template-directed denoising** (no subject-specific run-1 info); subject-specific increment **+0.0056**. Energy: shared 29.9% in passband vs 19.0% run-difference.
- **Group REML (all 83):** n_eff **61.5718 < S=83** (min 50.311), ci_ratio 1.1473, no NaN (degenerate-node fix: n_eff=#{u=inf} for τ²+σ²=0; ε-perturbation verified). Caveat: under the stored unit subject-map normalisation this is a machinery check, not a scientific estimate.
- **Canonical files:** `REAL_n83_gap_sota.{json,md}` (regenerated), `REAL_sota_stats_n83_entropy.{json,md}` (independent replication of `REAL_sota_stats_n83_posterior_entropy.{json,md}`), `REAL_sota_stats_n83_posterior_tau.{json,md}`.
- **Superseded:** everything below describing the 49/83 fallback (`region_emd_procrustes`) or "claim not established" is retained as honest ledger, **not** the primary claim (artifacts `73533e35`: Δrel +0.0028/+0.0022/+0.0268; artifact-subset +0.0434/+0.0425/+0.0648).

## Freeze + beta

| Item | N=49 (primary) | N=83 (scale-up) |
|---|---|---|
| Cohort | subjects 015–063 strict two-run | subjects 015–068 + 092–120 strict two-run |
| n_subjects | 49 | **83** |
| n_rows (two-run) | 98 | 166 |
| manifest | 195 rows / 112 subjects (all npz) | same rebuilt manifest |
| data_hash | `ccce8212b978…` (prior freeze) | **`4e6703055921b762fa77241438ea6e4f9fbd90a26cc926210c03c1546e5ee0ca`** |
| beta (scan-rescan, R=100) | **29.189086229914952** | **28.438323293411973** |
| sigma2 | 0.034259… (n=24 pilot freeze) | **0.0351638171379696** (n=83) |
| n_regions | 100 | 100 |
| pairs / seed | 500 / 2026 | 500 / 2026 |
| permutations_B | 200 (pilot) | **200** (10000 too slow for scale pass) |

Freeze files: `trajot/results/tables/freeze_n83/{frozen_cohort_n83.txt,freeze_meta.json,beta_n83_compact.json}`.

## N=83 method table (same-Q protocol; beta=28.438)

Artifacts: `runs/10_ours_full__73533e35__20260920T074217Z/artifacts` (**pi_means for 49 subjects only**).

| method | reliability_raw | reliability_after | Δrel | gain_after | ident_after | collapsed | transform path | tau_phi |
|---|---:|---:|---:|---:|---:|---|---|---:|
| noalign | 0.6455 | 0.6455 | 0 | 0.000 | 0.9157 | False | — | — |
| brainsync | 0.6455 | 0.6461 | +0.0006 | +0.00043 | 0.9036 | False | — | — |
| fugw | 0.6455 | 0.6215 | −0.024 | −0.006 | 0.9036 | True (rel drop) | — | — |
| conn_srm | 0.6455 | 0.8453 | +0.200 | +0.386 | 0.024 | True (ident collapse) | — | — |
| ours_full_posterior_shrink | 0.6455 | 0.6483 | +0.0028 | +0.0057 | 0.8434 | False | **`region_emd_procrustes` (FALLBACK)** | 0.0092 |
| ours_full_point_procrustes | 0.6455 | 0.6483 | +0.0028 | +0.0057 | 0.8434 | False | point_procrustes_C_pop | 0.0092 |

**Critical:** `ours_full_posterior_shrink` on N=83 did **not** apply hierarchical posterior maps. `OursFull._choose_transform_path` requires `len(_pi_means) >= n_subjects`; artifacts cover 49/83 → path = `region_emd_procrustes`, `posterior_drives_transform=false`. All "ours" N=83 rows share identical point metrics because they share the same fallback. **SUPERSEDED (2026-09-20):** with artifacts `9d7dab12` (83/83) the hierarchical path is active; see the COMPLETE verdict above and the regenerated `REAL_n83_gap_sota.md`.

## Bootstrap B=10,000 — N=83 full cohort (HONEST)

Paired subject bootstrap, same-Q, beta=28.438, n_pairs=500 seed 2026.

| comparison | reliability Δ | reliability 95% CI | ident Δ | ident 95% CI | McNemar p |
|---|---:|---|---:|---|---:|
| ours (fallback EMD) vs noalign | +0.0028 | **[0.0027, 0.0030]** | −0.072 | [−0.072, 0.000] | 0.109 |
| ours (fallback EMD) vs BrainSync | +0.0022 | **[0.0012, 0.0031]** | −0.060 | [−0.072, 0.012] | 0.180 |
| ours (fallback EMD) vs FUGW | +0.0268 | **[0.0229, 0.0309]** | −0.060 | [−0.060, 0.012] | 0.125 |

**Pre-registered claim `task_sota_reliability` is NOT established on full N=83 for posterior_shrink.** Reliability CIs for the *fallback* transform exclude 0, but that is not the hierarchical posterior path the claim names. Identification is *worse* than baselines on N=83 (point 0.843 vs 0.916/0.904/0.904). Do not invent a full-N=83 posterior_shrink win. **SUPERSEDED (2026-09-20):** the completed posterior_shrink run (83/83, artifacts `9d7dab12`) earns the claim — this subsection is the fallback-path ledger only; identification is withdrawn (map-invariance control).

## Bootstrap B=10,000 — N=83 freeze ∩ artifact-covered subjects (N=49, beta=83)

Same-Q posterior_shrink where pi_means exist (subjects 015–063), beta recalibrated on N=83.

| comparison | reliability Δ | reliability 95% CI | ident Δ | ident 95% CI | McNemar p |
|---|---:|---|---:|---|---:|
| vs noalign | +0.0434 | **[0.0379, 0.0490]** | +0.020 | [0.000, 0.041] | 1.0 |
| vs BrainSync | +0.0425 | **[0.0366, 0.0486]** | +0.020 | [0.000, 0.041] | 1.0 |
| vs FUGW | +0.0648 | **[0.0576, 0.0722]** | +0.061 | [0.000, 0.061] | 0.25 |

- Point metrics: ours rel_after **0.6850**, ident **0.9796**; noalign 0.6416/0.9592; BrainSync 0.6425/0.9592; FUGW 0.6202/0.9184.
- Reliability CIs exclude 0 vs all three → `task_sota_reliability` **holds on the artifact-covered subset** (still N=49 subjects; beta=28.438).
- Identification remains **trend only** (CI touches 0). Never claim ID superiority.

## Group REML (Track B)

| run | subjects available | n_eff (mean) | n_eff min | ci_ratio | n_eff < S? |
|---|---:|---:|---:|---:|---|
| **full N=83 completion (`9d7dab12`)** | **83** | **61.5718** | 50.311 | **1.1473** | **YES** |
| max artifacts (015–063, fallback ledger) | **49** | **28.50** | 4.97 | 1.234 | **YES** |
| gap-table smoke | 24 | 13.91 | 2.57 | 1.244 | YES |
| N=49 closer (prior) | 12 | 7.45 | 2.46 | 1.277 | YES |

Full N=83 group REML (completion run): **n_eff=61.5718 < S=83**, ci_ratio=1.1473, no NaN; degenerate-node fix (n_eff=#{u=inf} for τ²+σ²=0 nodes) verified by ε-perturbation. Honest caveat: under the stored unit subject-map normalisation this is a machinery check, not a scientific estimate. Baselines still emit no τ_φ / Σ^al.

τ_φ mean on artifact posteriors: **0.0345** (completion run `9d7dab12`); 0.009236 on the 49-subject fallback artifacts. Gap columns only ours emit remain valid.

## Claim discipline

1. **N=83 PRIMARY:** `task_sota_reliability` EARNED — reliability 0.6975 vs 0.6455/0.6461/0.6215; 10k bootstrap CIs exclude 0 vs all three (Δ +0.0519/+0.0513/+0.0759; see `REAL_sota_stats_n83_entropy.{json,md}`). τ-λ variant 0.6577 (+0.0122/+0.0116/+0.0362). N=49 replication holds (+0.043/+0.043/+0.065); N=49 is no longer PRIMARY.
2. **Identification NOT claimed:** same-map ident protocol map-invariant (fitted / Haar-random / permuted all 1.0, 83/83; identity 0.9157, single common map 0.9036). Reported only as identity preservation.
3. **Mechanism honesty:** ≥89% of the reliability gain is a single common template-directed map (+0.0464 of +0.0519); the per-subject increment is only +0.0056. Do not claim the gain requires per-subject maps.
4. **Gap fill:** τ_φ (0.0345) + group REML n_eff 61.5718 < S=83 on full posteriors — machinery check under unit normalisation; baselines silent.
5. **Superseded ledger (do not headline):** fallback-path Δrel +0.0028/+0.0022/+0.0268 (artifacts `73533e35`, 49/83) and artifact-subset +0.0434/+0.0425/+0.0648 (N=49, β83), retained above as honest history.
6. **Do not claim:** ID superiority; calibrated coverage on real rest; conn_srm as gain winner; FUGW collapse as a scientific win (it is a reliability drop); cross-dataset generalization.

## Completing N=83 posterior_shrink — DONE (2026-09-20)

- **Done:** `ours_full` trained on all 83 two-run subjects (run `10_ours_full__9d7dab12__20260920T140613Z`; artifacts cover 83/83, `posterior_shrink_tau_gated` active). The commands below were re-run against that run to produce the regenerated `REAL_n83_gap_sota.{json,md}` and the independent replication `REAL_sota_stats_n83_entropy.{json,md}`.
- Historical note: early run `10_ours_full__cc0eb3e4…` (PID 5103, 49 subjects) was superseded by later training — do not cite it. The historical re-run commands (with `runs/<run>/artifacts` = `runs/10_ours_full__9d7dab12__20260920T140613Z/artifacts`):
  ```bash
  PYTHONPATH=src python scripts/run_real_gap_sota.py \
    --cohort-file results/tables/freeze_n83/frozen_cohort_n83.txt \
    --beta 28.438323293411973 \
    --ours-artifacts runs/<new_run>/artifacts \
    --out-stem REAL_n83_gap_sota --permutations-B 200 --full-fugw
  PYTHONPATH=src python scripts/verify_real_sota.py \
    --cohort-file results/tables/freeze_n83/frozen_cohort_n83.txt \
    --beta 28.438323293411973 --n-boot 10000 \
    --ours-artifacts runs/<new_run>/artifacts \
    --out-stem REAL_sota_stats_n83
  ```

## Canonical files

- `trajot/results/tables/REAL_n83_gap_sota.{json,md}` — N=83 method table (regenerated, completion run `9d7dab12`)
- `trajot/results/tables/REAL_sota_stats_n83_entropy.{json,md}` — N=83 bootstrap for the PRIMARY entropy-λ method (replication of `REAL_sota_stats_n83_posterior_entropy.{json,md}`)
- `trajot/results/tables/REAL_sota_stats_n83_posterior_tau.{json,md}` — τ-λ variant bootstrap
- `trajot/results/tables/REAL_sota_stats_n83.{json,md}` + `REAL_sota_stats_n83_artifact_subset.{json,md}` — superseded fallback / artifact-subset ledger (do not headline)
- `trajot/results/tables/group_real_n83_artifacts_max.json` — group REML max available (pre-completion ledger; superseded by the full-N=83 group block)
- `trajot/results/tables/freeze_n83/*` — cohort + beta freeze
- `trajot/reports/RESULTS.md` — official write-up (§5b N=83 verdict); `docs/compose/reports/surge-mathematical-analysis.md` — mechanism theorem + controls
- N=49 retained as replication: `REAL_n49_gap_sota.{json,md}` + `REAL_sota_stats.{json,md}` (no longer PRIMARY)
