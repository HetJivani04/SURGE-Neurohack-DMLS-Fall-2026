| Method | Identification acc. | vs null (p) | Per-pair uncertainty | Non-identifiable pairs flagged |
|---|---|---|---|---|
| No alignment (missing) | — | — | — | — |
| BrainSync (missing) | — | — | — | — |
| FUGW (missing) | — | — | — | — |
| connectivity-SRM (missing) | — | — | — | — |
| Ours (ablated) (missing) | — | — | — | — |
| Ours (full) (missing) | — | — | — | — |

- Declared pair subsample: not declared (no run in this table)
- Draw procedure: ordered pairs (a, b) of distinct subjects drawn with replacement by numpy.random.default_rng(seed) (trajot.eval.folds.sample_pairs)
- Fold scheme: two-run identification: each subject's run 1 is the query against the gallery of run 2 and vice versa, both folds holding the same sorted subject list (trajot.eval.folds.make_two_run_splits)
- beta: not reported (no model run in this table)

Source runs:
- No alignment: (missing)
- BrainSync: (missing)
- FUGW: (missing)
- connectivity-SRM: (missing)
- Ours (ablated): (missing)
- Ours (full): (missing)
