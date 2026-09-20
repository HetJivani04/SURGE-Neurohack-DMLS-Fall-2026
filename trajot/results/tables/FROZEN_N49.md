# N=49 primary freeze

See canonical write-up: `trajot/reports/RESULTS.md`.

- frozen_root: `/tmp/surge-coord/frozen_ds000243`
- n_subjects: 49 (subjects 015–063 two-run subset used for primary comparison)
- beta: 29.189086229914952 (real scan-rescan; R=100 connectome side)
- n_regions: 100 (Schaefer-100; K must equal 100)
- pairs: 500, seed 2026
- permutations_B: 200 (pilot; final paper B=10000 after N=83)
- data_hash_primary: `ccce8212b978`
- freeze_meta: `frozen_ds000243/derivatives/trajot/freeze_meta.json`

## Verdict (do not invent wins)

1. **Track A ceiling:** raw/noalign/brainsync ID acc 0.959 at Schaefer-100 N=49. Identification cannot discriminate alignment methods at this parcellation.
2. **ours_full does not win ID** (0.857). Only method with positive alignment_gain without identity collapse (+0.018).
3. **Track B unique contribution:** `per_pair_uncertainty` / tau_phi (0.075 full, 0.103 ablated). Baselines report null for that column.
4. **BrainSync** is a mathematical no-op on spatial connectomes (`XQQ^T X^T = XX^T`); feat_corr≈0.998.
5. **Gauge ablation** moves tau_phi, not point transform metrics.
6. **nonident counts** near n_pairs are null-degeneracy artifacts at pilot B — caveated, not headlined.
