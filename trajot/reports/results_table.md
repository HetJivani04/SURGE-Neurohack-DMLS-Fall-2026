| Method | Identification acc. | vs null (p) | Per-pair uncertainty | Non-identifiable pairs flagged |
|---|---|---|---|---|
| No alignment | 0.96 [0.90, 1.00] | 0.0050 | — | — |
| BrainSync | 0.96 [0.90, 1.00] | 0.0050 | — | — |
| FUGW | 0.92 [0.84, 0.98] | 0.0050 | — | — |
| connectivity-SRM | 0.08 [0.02, 0.16] | 0.0199 | — | — |
| **Ours (ablated)** | 0.85 [0.73, 0.97] | 0.0050 | 0.103 | 472 |
| **Ours (full)** | 0.86 [0.76, 0.94] | 0.0050 | 0.075 | 472 |

- Declared pair subsample: 500 ordered subject pairs, seed 2026
- Draw procedure: ordered pairs (a, b) of distinct subjects drawn without replacement via numpy.random.default_rng(seed).choice over lexicographic ordered pairs (default_rng.choice without replacement over lexicographic pair index; trajot.eval.folds.sample_pairs / make_folds)
- Fold scheme: two-run identification: each subject's run 1 is the query against the gallery of run 2 and vice versa, both folds holding the same sorted subject list (trajot.eval.folds.make_two_run_splits)
- beta: 29.189086229914952

Source runs:
- No alignment: 00_noalign__eebd2f8e__20260920T074351Z
- BrainSync: 01_brainsync__b4f31914__20260920T074433Z
- FUGW: 02_fugw__cf2f95f4__20260920T074534Z
- connectivity-SRM: 03_conn_srm__0777ce05__20260920T074652Z
- Ours (ablated): 11_ours_ablated__56a17400__20260920T071555Z
- Ours (full): 10_ours_full__dd67ab1e__20260920T074747Z
