# Results: a hierarchical population of couplings for rest-fMRI alignment

**Status: development phase. This document reports no result yet.**

The framework (W0 to W4) is built and unit-tested on synthetic data with known ground truth. Running it on
ds000243 to produce the identification accuracies, permutation p-values, non-identifiability counts, figures
and tables is the experiment phase, a separate set of issues opened after W0 to W4 are complete and their tests
pass. So every result cell below is empty on purpose, and no sentence in this document is a finding. Each
section says what will be reported, which command fills it, and which run it will trace to.

On the machine this was written on (2026-09-20), `runs/index.csv` holds one preprocessing run and one
synthetic-data fit whose `metrics.json` has no methods in it. No comparison run exists, so the table has six
missing rows. A negative or empty outcome is stated as such, not filled in.

## 1. The results table

Produced by `python scripts/compare.py --experiments all --out reports/results_table.md`, which reads only
`runs/index.csv` and each selected run's `metrics.json`. Six rows, five columns, always. The two right-hand
columns are quantities no baseline produces, so they are empty (`—`) for every baseline row, never `0`. A row
whose run is missing carries a `(missing)` marker instead of being dropped.

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

### Declared subsample

The protocol declares the subsample before any run: 500 ordered subject pairs, draw seed 2026, and 10,000
permutations (`configs/eval/default.yaml`). A run that does not declare `n_pairs`, `pairs_seed` and
`permutations_B` is rejected by the `metrics.json` validator, and `compare.py` refuses to put runs that declare
different subsamples in one table. The values shown in the metadata above are read from the runs, not from
this document.

## 2. Track A: cross-run identification

**What will be reported.** For each method, the accuracy of identifying a subject's run-2 connectivity among the
83 two-run subjects from run 1, with its confidence interval, and the permutation p-value against the
10,000-permutation null, together with the null's maximum. Chance is 1/83, about 1.2%. The smallest p-value a
10,000-permutation null can give is 1/10,001, about 1.0e-4.

**Outcome: not yet run.** No accuracy or p-value exists. They will be read from `metrics.json` of the
evaluation runs by `compare.py`, and every cell traces to a `run_id` in the "Source runs" list.

**Track A may fail.** A median run of 132 volumes is thin, and identification depends on fingerprint stability
as well as on alignment. If it fails, the result is stated as a negative one: evidence that this dataset is too
short for the metric, not that the model is wrong. That is why Track B exists.

**Group contrast.** Whether one method beats another over the declared pairs is tested by a sign-flip
permutation (Winkler et al. 2014) over exactly the 500 declared pairs, with the declared seed
(`trajot.report.group.sign_flip_test`): the per-pair contrast is randomly sign-flipped 10,000 times, and the
p-value is `(count + 1) / (B + 1)`. It rejects a contrast whose length is not the declared pair count.

## 3. Track B: non-identifiable pairs

**What will be reported.** For each method, the count of subject pairs whose alignment is indistinguishable
from the null and therefore carries no information (`nonidentifiable_pairs` in `metrics.json`). Every existing
method returns an alignment for all pairs with no indication; this is that gap as a statistic, and it needs no
new method to compute. In the table the last column is shown for the two model rows only, because "flagged" is
something only the model can do.

**Outcome: not yet run.** No count exists.

## 4. The inverse temperature beta

**Calibration.** `beta` is fixed from the data and never tuned: `sigma_C^2 = 1/2 * mean_s ||C_s(1) - C_s(2)||_F^2 / V^2`
over the 83 two-run subjects, and `beta = 1 / sigma_C^2` (`trajot.inference.beta.calibrate_beta`, the only place
`beta` is computed). `V` is the side of the connectome matrix in the preprocessed contract. It is computed
inside a real fit, which writes `artifacts/beta.json` (`sigma_hat_C_squared`, `beta`, the subject ids and the
per-subject terms) into the run directory and `beta` into `metrics.json`. A synthetic fit uses a fixed value and
never calibrates.

**Value: not yet recorded.** No real fit has been run, so there is no `beta` to report. It will be the value in
the table metadata above and in `artifacts/beta.json` of the fit run it traces to.

## 5. Per-subject posterior widths

**What will be reported.** The per-subject posterior width across the cortex, as the identifiability figure
(`trajot.report.figures.plot_posterior_widths`, an (S, V) array of `tau_phi` in, a PNG out). A diffuse posterior
is a reportable outcome: the data do not determine that subject's alignment.

**Not yet available, and not yet interpretable.** The first-pass fit drops the entropy term of the objective:
the entropy of the noise alone is unbounded in the widths, and the log-determinant correction that makes it well
behaved is not yet in the training objective. Without any entropy term the widths are driven to their floor, so
the posterior understates uncertainty and `tau_phi` does not yet say how well the data determine an alignment. A
width read from such a fit would not be an identifiability statement. No figure is produced here.

## 6. Group-level random-effects analysis

Built and verified on simulated subject maps, not yet applied to any posterior (none exists).
`trajot.report.group.meta_analysis_map` carries each subject map into the template by its coupling, takes the
per-node mean over posterior draws and their variance as the alignment variance, and fits the random-effects
model per template node with REML or method-of-moments estimates of the between-subject variance. A subject
whose alignment the data do not determine is down-weighted, not silently averaged in. The tests assert that it
collapses exactly to the one-sample t-test when the alignment variance goes to zero, that the delta-method
covariance matches the empirical one where both are computable, and that the estimate attains the maximum of the
restricted likelihood found by brute force. The Satterthwaite test keeps its size and is conservative when the
between-subject variance estimate sits at zero with very unequal weights.

## 7. Scan-length sensitivity

The long-run subset is the 26 subjects whose run has at least 300 volumes (15 runs of 360 volumes, 5 of 480 and
6 of 724; `trajot.report.sensitivity.long_run_subset` on the dataset manifest). All 26 are one-run subjects: none
of the 83 two-run subjects has a run that long. So cross-run identification cannot be computed on this subset,
and its metrics have to come from a separately defined evaluation (for example, halves of a long run), which is
part of the experiment phase.

A run for this analysis is registered under `<experiment>_long` and evaluated on exactly the 26 subjects.
`python scripts/compare.py --experiments all --sensitivity` prints it under its own heading, flagged as a
sensitivity analysis and not a headline number, and it never enters the table above. **Not yet run.**

## 8. Limitations

- **Entropy term.** The first-pass fit drops the entropy term, so posterior widths carry no uncertainty yet
  (Section 5). Reinstating it, so that the widths are calibrated, comes before any identifiability claim.
- **CPU-only budget.** Transport and Gromov-Wasserstein arithmetic is float64 on the CPU (Apple's GPU backend has
  no float64), on a 16 GB machine, serially during development. The vertex count, template size, number of
  posterior draws and epochs are bounded by that budget, and the full-size fit has not been run.
- **No ground-truth correspondence at rest.** Rest has no shared time axis and no known vertex correspondence
  between subjects, so nothing here can be scored against a true alignment. Identification accuracy is a proxy.
  Recovery of a planted correspondence is checked on synthetic data only.
- **Short runs.** The median run is 132 volumes, so each Fisher-z correlation carries a standard error near
  0.088, and BrainSync's own paper warns that "Syncing of shorter time courses should probably be avoided since the
  error increases rapidly below this limit."
- **Registration and geometry.** Preprocessing registers the mean EPI to the MNI152 template with one affine
  step (no nonlinear warp, no T1w image), so a residual misalignment of about a voxel is expected. All subjects
  are sampled on the same fsaverage mesh, so curvature, sulcal depth, coordinates and the anatomical cost are
  properties of the template and identical across subjects; only the signal differs.
- **Empty is not zero.** A missing run is shown as missing. Nothing in this document has been filled in by
  estimate.

## 9. Reproducing a row

Every row of the table names its source run in the "Source runs" list that `compare.py` prints, and every number
in this write-up will trace to a `run_id` in `runs/index.csv`. A row is reproduced from that run's
`runs/<run_id>/manifest.json` (config, config hash, git commit, data hash, seed, operator, thread settings) by
re-running the experiment at that commit with that config. `compare.py` never re-runs anything.

## Appendix: claim discipline

- **Finitely many optima.** Gromov-Wasserstein is invariant to isometries (Memoli 2011), so a pairwise coupling
  is defined only up to a symmetry group. The feature term, linear in the coupling, reduces the isometry orbit to
  finitely many optima, following Demetci et al. (2024). The claim is finitely many optima. No claim is made that
  the minimizer is single: the objective still contains the quadratic Gromov-Wasserstein term.
- **Inherited components are attributed.** Subject-to-template plans with a barycenter template: FUGW (Thual et
  al., NeurIPS 2022). The amortized encoder over subjects: ULOT (Mazelet, Flamary, Thirion, NeurIPS 2025,
  arXiv:2506.12025). A distribution over transport plans: Mallasto, Gerolin, Minh (ACML 2021, PMLR v157) and De
  et al. (ICML 2026). Alignment variance in the group model: Keller, Roche, Tucholka, Thirion (*Statistica
  Sinica* 2008), with Hu et al. (ICLR 2025) for learned registration. The Gromov-Wasserstein objective:
  Memoli (*FoCM* 11:417-487, 2011) and Demetci et al. (PMLR 238:298-306, AISTATS 2024). Posterior width read as
  evidence about identifiability rests on standard Bayesian posterior-contraction theory, cited and not
  re-derived. The nearest neighbour, OTTER (bioRxiv 2026, doi:10.64898/2026.08.24.746652), has no posterior, no
  hierarchy and no uncertainty propagation. The contribution is the assembled object, not the parts.
- **Absences are search-based.** The hierarchy as a modeling idea is textbook; as an object, a population
  distribution over latent alignment couplings with shrinkage on the transport polytope for cross-subject
  rest-fMRI, it is unclaimed. That absence is stated as a search result: no such work was found.
- **The band prior is a band prior and not a dynamics model.** It constrains the frequency content of a coupling
  (it charges the coupling of a vertex to a template node the gap between their dominant frequencies, unless both
  lie in the 0.01 to 0.1 Hz band) and says nothing about trajectories or a shared time axis. Its weight is 0 in the default
  configuration, so it is not in the objective unless switched on.
- **No behavioural or cognitive prediction is reported as a metric.** Marek et al. 2022 report a median
  brain-behaviour correlation of |r| = 0.01 at N = 3,928, and ds000243 has 120 subjects; any such number here
  would be noise presented as a result.
