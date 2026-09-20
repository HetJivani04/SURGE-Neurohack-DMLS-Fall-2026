# REAL N=83 gap + SOTA — honest scale-up

**Status: INCOMPLETE for full-cohort posterior_shrink. N=49 remains PRIMARY.**

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

**Critical:** `ours_full_posterior_shrink` on N=83 did **not** apply hierarchical posterior maps. `OursFull._choose_transform_path` requires `len(_pi_means) >= n_subjects`; artifacts cover 49/83 → path = `region_emd_procrustes`, `posterior_drives_transform=false`. All "ours" N=83 rows share identical point metrics because they share the same fallback.

## Bootstrap B=10,000 — N=83 full cohort (HONEST)

Paired subject bootstrap, same-Q, beta=28.438, n_pairs=500 seed 2026.

| comparison | reliability Δ | reliability 95% CI | ident Δ | ident 95% CI | McNemar p |
|---|---:|---|---:|---|---:|
| ours (fallback EMD) vs noalign | +0.0028 | **[0.0027, 0.0030]** | −0.072 | [−0.072, 0.000] | 0.109 |
| ours (fallback EMD) vs BrainSync | +0.0022 | **[0.0012, 0.0031]** | −0.060 | [−0.072, 0.012] | 0.180 |
| ours (fallback EMD) vs FUGW | +0.0268 | **[0.0229, 0.0309]** | −0.060 | [−0.060, 0.012] | 0.125 |

**Pre-registered claim `task_sota_reliability` is NOT established on full N=83 for posterior_shrink.** Reliability CIs for the *fallback* transform exclude 0, but that is not the hierarchical posterior path the claim names. Identification is *worse* than baselines on N=83 (point 0.843 vs 0.916/0.904/0.904). Do not invent a full-N=83 posterior_shrink win.

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
| max artifacts (015–063) | **49** | **28.50** | 4.97 | 1.234 | **YES** |
| gap-table smoke | 24 | 13.91 | 2.57 | 1.244 | YES |
| N=49 closer (prior) | 12 | 7.45 | 2.46 | 1.277 | YES |

Full N=83 group REML cannot run: posterior artifacts cover only 49 subjects. Report **n_eff=28.50 < S=49** on max available; baselines still emit no τ_φ / Σ^al.

τ_φ mean on artifact posteriors: **0.009236** (same as N=49 closer). Gap columns only ours emit remain valid.

## Claim discipline

1. **N=49 PRIMARY:** `task_sota_reliability` — reliability 0.685 vs 0.642/0.643/0.620; bootstrap CIs exclude 0 vs all three (see `REAL_sota_stats.json`).
2. **N=83 full cohort:** posterior_shrink **INCOMPLETE** (artifacts 49/83). Fallback EMD reliability CIs exclude 0 but are **not** the pre-registered claim. Ident degrades on N=83.
3. **Artifact-covered subset within N=83 freeze (beta=28.438):** claim holds with the same reliability pattern; still N=49 subjects.
4. **Gap fill:** τ_φ + group REML n_eff < S on max available posteriors — baselines silent.
5. **Do not claim:** full-N=83 posterior_shrink SOTA; ID superiority; calibrated coverage; conn_srm as gain winner; FUGW collapse as a scientific win (it is a reliability drop).

## What is needed to complete N=83 posterior_shrink

- Train `ours_full` on all 83 two-run subjects (K=100, beta=28.438) so `posterior_samples.npz` + `tau_phi.npz` cover every subject.
- PID 5103 (`10_ours_full__cc0eb3e4…`) is still training (log shows 49 subjects, epochs≈18/50) — **not killed**. When complete artifacts with pi_means for a larger cohort exist, re-run:
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

- `trajot/results/tables/REAL_n83_gap_sota.{json,md}` — N=83 method table
- `trajot/results/tables/REAL_sota_stats_n83.{json,md}` — N=83 bootstrap (fallback path)
- `trajot/results/tables/REAL_sota_stats_n83_artifact_subset.{json,md}` — subset bootstrap (posterior_shrink)
- `trajot/results/tables/group_real_n83_artifacts_max.json` — group REML max available
- `trajot/results/tables/freeze_n83/*` — cohort + beta freeze
- N=49 primary remains `REAL_n49_gap_sota.{json,md}` + `REAL_sota_stats.{json,md}`
