# SURGE Neurohack 2026 — TrajOT results (canonical location)

Full honest write-up: **[`trajot/reports/RESULTS.md`](trajot/reports/RESULTS.md)**.

CLI table: `trajot/reports/results_table.md` (from `scripts/compare.py --experiments all`).

## Frozen N=49 verdict (one screen)

Phase 2 pilot frozen · ds000243 · Schaefer-100 · beta=29.189 · N=49 · pairs=500 seed=2026 · B=200 pilot.

| Method | ID acc | alignment_gain | per_pair_uncertainty | run_id (abbrev) |
|---|---|---|---|---|
| noalign | 0.959 | 0.0 | — | `00_noalign__eebd2f8e__…` |
| brainsync | 0.959 | ≈0 | — | `01_brainsync__b4f31914__…` |
| fugw | 0.918 | −0.005 | — | `02_fugw__cf2f95f4__…` |
| conn_srm | 0.082 | +0.395 | — | `03_conn_srm__0777ce05__…` |
| **ours_full** | 0.857 | **+0.018** | **0.075** | `10_ours_full__dd67ab1e__…` |
| **ours_ablated** | 0.85 (N=33) | ~0 | **0.103** | `11_ours_ablated__56a17400__…` |

- **Track A:** ceiling — raw connectomes ID at 0.94–0.96; TrajOT does **not** win identification.
- **Track B:** only ours produces per-pair uncertainty (tau_phi). That is the contribution (PLAN §7.5).
- **Trade-off:** only ours has positive gain without identity collapse.
- Do not invent wins; do not fill unfillable cells with 0; do not claim unique minimizer or dynamics for the band prior.
