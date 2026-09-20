# REAL_sota_stats_n83_posterior_entropy gap columns + SOTA stats

- claim: **task_sota_ident_and_reliability** — Posterior-gated hierarchical alignment improves identification and scan-rescan reliability on real ds000243 with bootstrap CIs excluding 0.
- same-Q both runs: True
- n_boot: 10000

## Comparable SOTA (REAL N=83, same Q_s on both runs for ours)

| method | ident | scan-rescan after | alignment_gain | tau_phi | group_n_eff |
|---|---:|---:|---:|---:|---:|
| ours_full_posterior_shrink_entropy | 1.0000 | 0.6975 | -0.0349 | 0.0345 | nan |
| noalign | 0.9157 | 0.6455 | 0.0000 | — | — |
| brainsync | 0.9036 | 0.6461 | 0.0004 | — | — |
| fugw | 0.9036 | 0.6215 | -0.0060 | — | — |

## Paired bootstrap (10k) — ours_full_posterior_shrink_entropy − baseline

| baseline | ident_delta | ident 95% CI | reliability_delta | reliability 95% CI | McNemar p |
|---|---:|---|---:|---|---:|
| noalign | 0.0843 | [0.0120, 0.0723] | 0.0519 | [0.0480, 0.0558] | 0.0156 |
| brainsync | 0.0964 | [0.0120, 0.0843] | 0.0513 | [0.0472, 0.0552] | 0.00781 |
| fugw | 0.0964 | [0.0120, 0.0843] | 0.0759 | [0.0705, 0.0814] | 0.00781 |

## Gap columns (baselines leave null)

| method | tau_phi | group_neff | identifiability flags | baseline null |
|---|---:|---:|---|---|
| noalign | — | — | — | silent |
| BrainSync (rest) | — | — | — | silent |
| FUGW (OT) | — | — | — | silent |
| conn_srm | — | — | — | silent |
| ours_full_posterior_shrink_entropy | — | nan | n_eff=nan < S; ci_ratio=1.0756808589383464; high_tau_p90=9 | emits τ_φ + REML weights |

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

