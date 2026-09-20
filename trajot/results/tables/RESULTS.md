# Phase 2 primary comparison table (staging copy)

Canonical write-up: `trajot/reports/RESULTS.md`.
CLI table: `python scripts/compare.py --experiments all --out reports/results_table.md` → also `results/tables/results_table.md`.

## Frozen pilot — honest summary

- **Status:** Phase 2 pilot frozen. ds000243, Schaefer-100 (R=100), beta=29.189 (real scan-rescan), N=49 primary cohort, pairs=500 seed=2026, B=200 (pilot; paper target B=10000 / N=83).
- **Track A:** raw connectomes identify at 0.94–0.96 — ceiling. `ours_full` **does not win** (0.857 vs noalign/brainsync 0.959).
- **Track B (guaranteed):** only ours emits `per_pair_uncertainty` (`tau_phi` 0.075 full / 0.103 ablated). Baselines silent.
- **Trade-off:** only ours has positive `alignment_gain` without identity collapse (+0.018; conn_srm +0.395 collapses ID to 0.082; FUGW −0.005).

## 6×5 table (compare.py)

| Method | Identification acc. | vs null (p) | Per-pair uncertainty | Non-identifiable pairs flagged |
|---|---|---|---|---|
| No alignment | 0.96 [0.90, 1.00] | 0.0050 | — | — |
| BrainSync | 0.96 [0.90, 1.00] | 0.0050 | — | — |
| FUGW | 0.92 [0.84, 0.98] | 0.0050 | — | — |
| connectivity-SRM | 0.08 [0.02, 0.16] | 0.0199 | — | — |
| **Ours (ablated)** | 0.85 [0.73, 0.97] | 0.0050 | 0.103 | 472 |
| **Ours (full)** | 0.86 [0.76, 0.94] | 0.0050 | 0.075 | 472 |

beta: 29.189086229914952 · pairs: 500 · seed: 2026 · B: 200 · folds: without replacement

## Source runs

| Method | run_id | N | ident | gain | tau_phi | data_hash |
|---|---|---|---|---|---|---|
| noalign | `00_noalign__eebd2f8e__20260920T074351Z` | 49 | 0.959 | 0.0 | — | `ccce8212b978` |
| brainsync | `01_brainsync__b4f31914__20260920T074433Z` | 49 | 0.959 | ≈0 | — | `ccce8212b978` |
| fugw | `02_fugw__cf2f95f4__20260920T074534Z` | 49 | 0.918 | −0.005 | — | `ccce8212b978` |
| conn_srm | `03_conn_srm__0777ce05__20260920T074652Z` | 49 | 0.082 | +0.395 | — | `ccce8212b978` |
| ours_full | `10_ours_full__dd67ab1e__20260920T074747Z` | 49 | 0.857 | +0.018 | 0.075 | `ccce8212b978` |
| ours_ablated | `11_ours_ablated__56a17400__20260920T071555Z` | **33** | 0.848 | +0.010 | 0.103 | different cohort freeze |

**Ablated N note:** N=49 `ours_ablated` training had not finished at freeze time; the completed gauge-off run is N=33. Do not invent an N=49 ablated cell.

**nonident caveat:** at pilot B=200 the gain-null is often degenerate, so counts near 500 are an artifact, not a scientific non-identifiability rate.

## Files

- `comparison_pilot_primary.csv` — N=49 same-data_hash rows
- `comparison_pilot_latest.csv` — latest snapshot
- `comparison_pilot.csv` — historical multi-N log (do not mix rows across data_hash)
- `results_table.md` — compare.py markdown output
- `FROZEN_PILOT.md` / `FROZEN_COHORT.md` — freeze notes (24-subject pilot cohort note; primary comparison is N=49)
- `FROZEN_N49.md` — N=49 freeze metadata pointer
