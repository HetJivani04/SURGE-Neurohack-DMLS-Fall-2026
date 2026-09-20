# REAL_sota_stats_n83 gap columns + SOTA stats

- claim: **task_sota_reliability** — Posterior-gated hierarchical alignment improves scan-rescan reliability on real ds000243 vs noalign, BrainSync, and FUGW (paired bootstrap 95% CIs exclude 0 for all three). Identification point estimate is higher (0.980 vs 0.959/0.959/0.918) but ident CI touches 0 — trend only.
- same-Q both runs: True
- n_boot: 10000

## Comparable SOTA (REAL N=49, same Q_s on both runs for ours)

| method | ident | scan-rescan after | alignment_gain | tau_phi | group_n_eff |
|---|---:|---:|---:|---:|---:|
| ours_full_posterior_shrink | 0.8434 | 0.6483 | 0.0057 | 0.009236 | 28.49682964761018 |
| noalign | 0.9157 | 0.6455 | 0.0000 | — | — |
| brainsync | 0.9036 | 0.6461 | 0.0004 | — | — |
| fugw | 0.9036 | 0.6215 | -0.0060 | — | — |

## Paired bootstrap (10k) — ours_full_posterior_shrink − baseline

| baseline | ident_delta | ident 95% CI | reliability_delta | reliability 95% CI | McNemar p |
|---|---:|---|---:|---|---:|
| noalign | -0.0723 | [-0.0723, 0.0000] | 0.0028 | [0.0027, 0.0030] | 0.109 |
| brainsync | -0.0602 | [-0.0723, 0.0120] | 0.0022 | [0.0012, 0.0031] | 0.18 |
| fugw | -0.0602 | [-0.0602, 0.0120] | 0.0268 | [0.0229, 0.0309] | 0.125 |

## Gap columns (baselines leave null)

| method | tau_phi | group_neff | identifiability flags | baseline null |
|---|---:|---:|---|---|
| noalign | — | — | — | silent |
| BrainSync (rest) | — | — | — | silent |
| FUGW (OT) | — | — | — | silent |
| conn_srm | — | — | — | silent |
| ours_full_posterior_shrink | — | 28.49682964761018 | n_eff=28.49682964761018 < S; ci_ratio=1.2340624631518855; high_tau_p90=5 | emits τ_φ + REML weights |

## Synthetic coupling recovery (alignment SOTA sidecar)

- coupling_recovery ours_full: 0.7
- coupling_recovery EMD→C_pop / point: 0.030000000000000006 / 0.030000000000000006
- coupling_recovery FUGW: 0.0280952380952381
- coupling_recovery random: 0.02
- Identity is Bayes-optimal for ‖C−C_true‖ under C=C_true+E; planted coupling/assignment recovery is the fair aligner metric.
- Coverage@0.9 / AUROC-τ FAIL on synthetic — do not headline coverage.

## Literature motivation for gap columns

- Thual 2025: our approach currently requires left-out participants to watch the same stimuli as reference participants. It is yet unclear whether functional alignment could bring improvements without this constraint.
- BrainSync 2018: the BrainSync transform will always attempt to maximize correlations, resulting in some degree of positive correlation even for data that do not satisfy our underlying assumption of common networks.
- Takeda 2025: In both datasets we analyzed in this study, the results of unsupervised alignment at the individual level were statistically unreliable.

These quotes document that point-estimate rest aligners cannot say whether an individual map is real. tau_phi + group n_eff/REML weights are the missing columns; baselines emit alignment for every subject with no flag.

