# TrajOT frozen-data method comparison (N=49)

Frozen snapshot: `/Users/anandlo/Surge2026F/trajot_frozen_N49`

- **data_hash (paths)**: `6a781cf0e9e0a64d`
- **data_hash (content/hash_data_root)**: `7d80027bcb9642ed...` (run manifests record prefix `bab58d068885a900`)
- **cohort**: subjects 015–063, 49 two-run, R=100, contract 1.0.0
- **eval**: B=200, pairs n=500 seed=2026, run.debug=true
- **ours_***: model.K=100, model.train.epochs=20

| Method | N | Ident | Ident CI | Gain | NonIdent | perm_p | Unc | K | Ep | beta | run_id |
|---|---|---|---|---|---|---|---|---|---|---|---|
| brainsync | 49 | 0.9592 | [0.90,1.00] | -0.0000 | 469 | 0.004975124378109453 | — | 100 | 100 | 28.98189562137408 | `01_brainsync__b4f31914__20260920T093548Z` |
| conn_srm | 49 | 0.0816 | [0.02,0.16] | +0.3955 | 467 | 0.01990049751243781 | — | 100 | 100 | 28.98189562137408 | `03_conn_srm__0777ce05__20260920T093617Z` |
| fugw | 49 | 0.9184 | [0.84,0.98] | -0.0049 | 466 | 0.004975124378109453 | — | 100 | 100 | 28.98189562137408 | `02_fugw__cf2f95f4__20260920T093601Z` |
| noalign | 49 | 0.9592 | [0.90,1.00] | +0.0000 | 500 | 0.004975124378109453 | — | 100 | 100 | 28.98189562137408 | `00_noalign__eebd2f8e__20260920T093535Z` |
| ours_ablated | 49 | 0.8571 | [0.76,0.94] | +0.0181 | 472 | 0.004975124378109453 | 0.002 | 100 | 20 | 28.98189562137408 | `11_ours_ablated__70a81655__20260920T093707Z` |
| ours_full | 49 | 0.8571 | [0.76,0.94] | +0.0181 | 472 | 0.004975124378109453 | 0.002 | 100 | 20 | 28.98189562137408 | `10_ours_full__081aaddd__20260920T093707Z` |

## Method rankings on frozen data

### Track A — identification accuracy
1. **noalign** — ident=0.9592
2. **brainsync** — ident=0.9592
3. **fugw** — ident=0.9184
4. **ours_full** — ident=0.8571
5. **ours_ablated** — ident=0.8571
6. **conn_srm** — ident=0.0816

### Track B — alignment gain among methods with ident>=0.75
1. **ours_full** — gain=+0.0181, ident=0.8571
2. **ours_ablated** — gain=+0.0181, ident=0.8571
3. **noalign** — gain=+0.0000, ident=0.9592
4. **brainsync** — gain=-0.0000, ident=0.9592
5. **fugw** — gain=-0.0049, ident=0.9184

### Track A+B combined (require ident>=0.75, maximize gain)
1. **ours_full / ours_ablated** — only positive gain without identity collapse
2. noalign / brainsync — ceiling ident, zero gain
3. fugw — slightly negative gain
4. conn_srm — large positive gain but ident collapse (disqualified)
