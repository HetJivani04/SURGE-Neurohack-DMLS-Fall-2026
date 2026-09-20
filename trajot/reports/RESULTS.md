# Results: a hierarchical population of couplings for rest-fMRI alignment

**Status: Phase 2 pilot frozen N=49, ds000243, Schaefer-100, beta = 29.189 (real scan-rescan, R = 100). This document reports what the frozen comparison actually produced. It does not invent wins.**

Production protocol (PLAN §7.4) declares 500 ordered pairs, draw seed 2026, and **10,000 permutations**. The frozen pilot ran **B = 200** (debug-scale; the smallest attainable p-value is 1/201 ≈ 0.005). Every p-value below is therefore a pilot p-value. Chance at N=49 is 1/49 ≈ 0.020.

The results table is produced by:

```bash
cd trajot
python scripts/compare.py --experiments all --out reports/results_table.md
```

which reads only `runs/index.csv` and each selected run's `metrics.json`. A copy is also written to `results/tables/results_table.md`. Unfilled baseline cells are em-dashes (`—`), never `0`.

## 1. The results table

The six-row five-column table below is **exactly what `compare.py` rendered** at freeze time (`trajot/reports/results_table.md`). The two right-hand columns are quantities no baseline produces.

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

### Diagnostic tradeoff table (alignment_gain + transform)

Same runs, additional columns that make the Track A / Track B tension visible. Baseline uncertainty cells stay `—`. The ablated row is the **completed** run at **N=33** (see caveat); all other rows are N=49 on frozen data_hash `ccce8212b978`.

| Method | N | ID acc | alignment_gain | per_pair_uncertainty | feat_corr | run_id |
|---|---|---|---|---|---|---|
| noalign | 49 | **0.959** | 0.000 | — | 1.000 | 00_noalign__eebd2f8e__20260920T074351Z |
| brainsync | 49 | **0.959** | −0.00005 | — | 0.998 | 01_brainsync__b4f31914__20260920T074433Z |
| fugw | 49 | 0.918 | −0.005 | — | 0.966 | 02_fugw__cf2f95f4__20260920T074534Z |
| conn_srm | 49 | 0.082 | **+0.395** | — | 0.586 | 03_conn_srm__0777ce05__20260920T074652Z |
| ours_ablated | 33† | 0.848 | +0.010 | **0.103** | 0.667 | 11_ours_ablated__56a17400__20260920T071555Z |
| ours_full | 49 | 0.857 | +0.018 | **0.075** | 0.658 | 10_ours_full__dd67ab1e__20260920T074747Z |

† No completed N=49 `ours_ablated` `metrics.json` existed at freeze time (`11_ours_ablated__70a81655__*` and `2455c6a6__*` logs show training started, K=100, gauge off, no finished metrics). The ablated row is therefore the latest successful ablated run (N=33). Do not read it as an N=49 estimate.

Cohort metadata: N=49 two-run subjects 015–063 from frozen root `frozen_ds000243`; beta 29.189 frozen for all methods; pairs=500 seed=2026; B=200 pilot (not 10,000). `ours_full` transform is a real region-level map (K=100): OT coupling to C_pop then orthogonal Procrustes; `transforms_applied=true`, max|aligned−raw| ≈ 2.58.

### Declared subsample

The protocol declares the subsample before any run: 500 ordered subject pairs, draw seed 2026, and 10,000 permutations (`configs/eval/default.yaml`). A run that does not declare `n_pairs`, `pairs_seed` and `permutations_B` is rejected by the `metrics.json` validator. **This pilot ran B=200, not 10,000.** The values shown in the metadata above are read from the runs, not invented in this document.

## 2. Track A: cross-run identification — ceiling diagnosis

**What Track A reports.** For each method, accuracy of identifying a subject's run-2 connectome among the two-run subjects from run 1, with a binomial CI and a permutation p-value against the label-shuffled null. Chance at N=49 is ≈ 0.020. Under the production protocol the null uses a 10,000-permutation draw; this pilot used B=200 (minimum p ≈ 0.0050).

**Outcome: raw connectomes are already at ceiling.** No-alignment and BrainSync both score **0.959** (≈ 47/49; self-pair correlation ≈ 0.67 vs cross-subject ≈ 0.46 on raw Schaefer-100 features). FUGW is slightly below at 0.918. `ours_full` is 0.857. **TrajOT does not win Track A.** There is no headroom for any alignment method to *improve* identification at this parcellation and cohort size: any nontrivial template-directed transform can only spend fingerprint variance.

**What this is not.** It is not evidence that the hierarchical model is "wrong." PLAN §7.3 already stated that Track A may fail and that a negative Track A is why Track B exists. The scientific reading is narrower and stronger: **identification on Schaefer-100 rest connectomes at N=49 cannot discriminate alignment methods**, because the unaligned ceiling leaves no room.

**Group contrast.** Whether one method beats another over the declared pairs is tested by a sign-flip permutation (Winkler et al. 2014) over exactly the 500 declared pairs (`trajot.report.group.sign_flip_test`). Not run in this pilot table; left unfilled rather than invented.

## 3. Track B: the guaranteed result — uncertainty the baselines cannot produce

**What PLAN §7.5 guarantees.** Run every existing method on the same data and report how many subject pairs are not identifiable. Every existing method returns an alignment for every pair **with no uncertainty**. That gap is the result.

**What the frozen table shows.**

1. **Unique column.** Only the hierarchical population-of-couplings framework produces `per_pair_uncertainty` from posterior widths `tau_phi`: **0.075 (ours_full)** and **0.103 (ours_ablated)**. Every baseline row has `—` in that column — not zero, not a missing run, a structural absence. Point-estimate methods (noalign, BrainSync, FUGW, connectivity-SRM) have no posterior and therefore no calibrated width to report.

2. **alignment_gain tradeoff (honest).** Methods that force cross-subject correlation destroy individual identity. connectivity-SRM reaches the highest gain (**+0.395**) but identification collapses to **0.082** (near chance; shared loadings erase fingerprints). FUGW gain is slightly **negative** (−0.005). `ours_full` is the **only method with positive alignment_gain without identity collapse** (+0.018 at ident 0.857). That is a tradeoff statement, **not** an accuracy win.

3. **Gauge ablation moves uncertainty, not the point transform.** Full vs ablated point metrics are nearly identical (ident 0.857 vs 0.848; gain +0.018 vs +0.010). The ablation **does** move posterior width: `tau_phi` 0.075 → 0.103. The gauge channel is therefore an **uncertainty** channel under this implementation, not a point-accuracy lever.

4. **BrainSync is a structural no-op on spatial connectomes.** A time-domain orthogonal map Q leaves the spatial connectome invariant: `X Q Q^T X^T = X X^T`. Observed `feat_corr ≈ 0.998` and gain ≈ 0. This is **not** a failed code path — it is what BrainSync can and cannot change when the evaluation object is a post-hoc connectome rather than raw time series. BrainSync's own paper warns against syncing short time courses; median run length here is ~132 volumes.

5. **`nonidentifiable_pairs` counts must be caveated.** The gain-null test flags a pair when its alignment gain is indistinguishable from the method's own permuted null at α=0.05. When that null is degenerate (noalign has zero gain by definition; every pair is "non-identifiable"), the count saturates at the declared pair count (500). Counts in the 466–500 range across methods are **not** a scientific ranking of uncertainty — they are partly an artifact of a degenerate null. The scientifically unique output is the **posterior-width column** (`per_pair_uncertainty`), not the gain-null saturation.

**Track B verdict.** The contribution is **calibrated per-pair uncertainty / identifiability statements** that no baseline in this comparison produces — the PLAN §7.5 guaranteed result — **not** a magic accuracy boost on identification.

## 4. The inverse temperature beta

**Calibration.** `beta` is fixed from the data and never tuned on test labels:

```
sigma_C^2 = 1/2 * mean_s ||C_s(1) - C_s(2)||_F^2 / R^2
beta = 1 / sigma_C^2
```

computed in `trajot.inference.beta.calibrate_beta` over two-run subjects, with **R = connectome side (parcels) = 100** (Schaefer-100), not the synthetic default. Scan-rescan, not synthetic.

**Frozen pilot value.** **beta = 29.189086229914952** (sigma_hat_C^2 ≈ 0.03426). It is recorded in each model run's `metrics.json` / `artifacts/beta.json` and printed by `compare.py`. Early synthetic fits used beta ≈ 50; those are **not** the production values reported here.

## 5. Per-subject posterior widths

**What is reported.** `tau_phi` — mean posterior width under the Sinkhorn-parametrized variational posterior — is the quantity loaded into `per_pair_uncertainty` for model rows. Full: 0.075. Ablated (gauge off): 0.103. A diffuse posterior is a reportable outcome: the data do not determine that subject's alignment.

**Entropy wiring caveat.** Early development notes correctly warned that dropping the entropy term understates uncertainty. Current configs wire entropy (`model.entropy.weight = 1.0`; estimator `shannon_pi+hutchinson_slq`). The Jacobian logdet is **detached**; full pushforward calibration is technical debt (PLAN §8.3). Widths are reported as produced by the frozen runs — **not** claimed as fully calibrated posterior-contraction statements. Calibration against planted correspondence on synthetic data and a shuffled-time (N2) diffuse-posterior control remain open work.

## 6. Group-level random-effects analysis

Built and verified on simulated subject maps (tests assert collapse to the one-sample t-test when alignment variance → 0, delta-method covariance agreement, REML/MoM consistency, Satterthwaite size). **Not applied** to any frozen real posterior in this pilot. A subject whose alignment the data do not determine is designed to be down-weighted, not silently averaged in.

## 7. Scan-length sensitivity

**Not yet run** on the long-run subset. The long-run subset is the **26 subjects** whose run has at least 300 volumes (runs of **360**, **480**, and **724** volumes in the dataset; `trajot.report.sensitivity.long_run_subset`). All 26 are one-run subjects: none of the two-run identification subjects has a run that long, so cross-run identification cannot be computed on this subset without a separately defined evaluation (e.g. halves of a long run). When run, it is registered under `<experiment>_long` and printed by `compare.py --sensitivity` under its own heading — never mixed into the headline table.

## 8. Limitations

- **Pilot permutations.** B=200, not the protocol 10,000. p-values are floored at 1/201 ≈ 0.0050 and must not be read as production significance.
- **N=49 of 83 two-run subjects.** Frozen primary comparison cohort is 49 two-run subjects (015–063). The full ds000243 two-run set is 83; scaling is incomplete. `ours_ablated` in the paper table is N=33 (no completed N=49 ablated metrics at freeze).
- **CPU-only budget.** Transport / Procrustes arithmetic is float64 on CPU (Apple's GPU backend has no float64) on a ~16 GB machine, serially during development. Vertex count, template size, posterior draws and epochs are bounded by that budget.
- **Short runs.** Median run ≈ 130 volumes. Fisher-z correlations carry substantial standard error; BrainSync's own paper warns that syncing shorter time courses degrades rapidly.
- **No ground-truth correspondence at rest.** Rest has no shared time axis and no known vertex correspondence between subjects, so nothing here can be scored against a true alignment. Identification accuracy is a proxy. Recovery of a planted correspondence is checked on synthetic data only.
- **Registration and geometry.** Preprocessing registers mean EPI to MNI152 with one affine step (no nonlinear warp, no T1w). All subjects are sampled on the same mesh, so geometric features are template-identical; only signal differs.
- **`nonidentifiable_pairs` caveat.** Gain-null counts saturate when the null is degenerate (see §3). They are not a calibrated uncertainty ranking.
- **Entropy / width calibration.** Entropy is wired but the Jacobian logdet is detached; posterior widths are reported, not claimed fully calibrated.
- **Transform vs PLAN object.** The applied `ours_full` map at freeze is region-level OT to C_pop + orthogonal Procrustes, with hierarchical artifacts (beta, tau_phi) reported alongside. Posterior samples do **not** currently drive the applied transform. Presenting the transform as "the full hierarchical posterior object" would overclaim; the uncertainty column is the honest surface of that object in this table.
- **Empty is not zero.** Missing runs stay missing. Unfillable cells stay `—`. Nothing here is filled by estimate.

## 9. Reproducing a row

Every row names its source `run_id`. A row is reproduced from that run's `runs/<run_id>/manifest.json` (config, config hash, git commit, data hash, seed) by re-running the experiment at that commit with that config. `compare.py` never re-runs anything. Frozen cohort notes: `trajot/results/tables/FROZEN_N49.md`. Primary same-hash CSV: `trajot/results/tables/comparison_pilot_primary.csv`. Mathematical root-cause analysis: `docs/compose/reports/surge-mathematical-analysis.md`.

## Appendix: claim discipline

- **Finitely many optima; single-point minimizers are not claimed.** Gromov-Wasserstein is invariant to isometries (Memoli 2011), so a pairwise coupling is defined only up to a symmetry group. The feature term, linear in the coupling, reduces the isometry orbit to **finitely many optima**, following Demetci et al. (2024). The objective still contains the quadratic GW term. No single optimizer is asserted.
- **Inherited components are attributed.** Subject-to-template plans with a barycenter template: **FUGW** (**Thual** et al., NeurIPS 2022). The amortized encoder over subjects: **ULOT** (**Mazelet**, Flamary, Thirion, NeurIPS 2025, arXiv:2506.12025). A distribution over transport plans: **Mallasto**, Gerolin, Minh (ACML 2021, PMLR v157) and De et al. (ICML 2026). Alignment variance in the group model: **Keller**, Roche, Tucholka, Thirion (*Statistica Sinica* 2008), with Hu et al. (ICLR 2025) for learned registration. The Gromov-Wasserstein objective: **Memoli** (*FoCM* 11:417-487, 2011) and **Demetci** et al. (PMLR 238:298-306, AISTATS 2024). Posterior width read as evidence about identifiability rests on standard Bayesian posterior-contraction theory, cited and not re-derived. The nearest neighbour, **OTTER** (bioRxiv 2026, doi:10.64898/2026.08.24.746652), has no posterior, no hierarchy and no uncertainty propagation. The contribution is the assembled object, not the parts.
- **Absences are search-based.** The hierarchy as a modeling idea is textbook; as an object, a population distribution over latent alignment couplings with shrinkage on the transport polytope for cross-subject rest-fMRI, it is unclaimed in the sources we checked. That absence is stated as a search result: **no such work was found**.
- **The band prior is a band prior and not a dynamics model.** It constrains the frequency content of a coupling (it charges the coupling of a vertex to a template node the gap between their dominant frequencies, unless both lie in the 0.01 to 0.1 Hz band) and says nothing about trajectories or a shared time axis. Its weight is 0 in the default configuration, so it is not in the objective unless switched on. Frozen runs used band prior weight 0.0.
- **No behavioural or cognitive prediction is reported as a metric.** **Marek et al. 2022** report a median brain-behaviour correlation of **|r| = 0.01 at N = 3,928**; ds000243 has 120 subjects; any such number here would be noise presented as a result.
- **Random-effects collapse identity.** The group model reduces to the **one-sample t-test** when alignment variance is zero; that property is tested, not asserted.
