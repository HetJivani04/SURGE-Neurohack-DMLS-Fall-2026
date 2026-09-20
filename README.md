# TrajOT - knowing when a brain alignment is real

**SURGE Neurohack, Fall 2026** · AI & Machine Learning stream

**Team TrajOT** - Anand Lo · Het Jivani · Nafisah Nubah · Rafat Hossain · Zawad Atif · Sophie Eruokwu

---

## Overview

Comparing brain activity across people usually needs a shared timeline — everyone watching the same
movie, so second 30 means the same thing in every scan. Resting-state fMRI has no such clock, and the
methods that align it return **one map per subject with no indication of whether that map captured
anything**. The field documents this about itself:

> "In both datasets we analyzed in this study, the results of unsupervised alignment at the individual
> level were statistically unreliable." — Takeda et al. 2025, *iScience*

TrajOT is a **hierarchical population-of-couplings model**: a posterior over subject-to-template
transport plans, with shrinkage toward a population coupling, an inverse temperature calibrated from
the data's own scan–rescan reliability rather than tuned, and group random-effects inference that
down-weights subjects whose alignment the data do not determine.

The posterior is not a side diagnostic — it drives the map. The applied transform is gated by
posterior row-entropy:

```
C̃ = (1 − λ) C + λ Qᵀ C Q        λ_s = 1 / (1 + (H_s / H₀)²)
```

where `H_s` is the row-entropy of subject s's posterior coupling, so a subject whose alignment the
data do not determine is moved less.

## Results

**N = 83** two-run subjects from ds000243, Schaefer-100, β = 28.438, same map on both runs,
λ from posterior row-entropy (λ_mean = 0.500). Source tables:
[`REAL_sota_stats_n83_posterior_entropy.md`](trajot/results/tables/REAL_sota_stats_n83_posterior_entropy.md)
and [`REAL_n83_gap_sota.md`](trajot/results/tables/REAL_n83_gap_sota.md).

| Method | Identification&nbsp;† | Scan–rescan reliability | Δ reliability (ours − baseline), 95% CI | τ_φ |
|---|---:|---:|---|---:|
| No alignment | 0.9157 | 0.6455 | +0.0519 [0.0480, 0.0558] | — |
| BrainSync | 0.9036 | 0.6461 | +0.0513 [0.0472, 0.0552] | — |
| FUGW | 0.9036 | 0.6215 | +0.0759 [0.0705, 0.0814] | — |
| **TrajOT (posterior-gated)** | **1.0000**&nbsp;† | **0.6975** | — | **0.0345** |

† Under a per-subject same-map protocol, identification is map-invariant: fitted, permuted and
Haar-random maps all score 83/83 (identity 0.9157, single common map 0.9036). The column measures the
protocol rather than the alignment, so leak-free comparison requires cross-fitted maps.

**`task_sota_reliability`.** Posterior-gated hierarchical alignment improves scan–rescan reliability
over no-alignment, BrainSync and FUGW; a 10,000-sample paired bootstrap puts all three CIs clear of
zero. N = 49 (β = 29.189) replicates the same verdict.

**Why λ = 1/2.** The transform acts on `vec(C)` as `M_λ = (1−λ)I + λR` with `R = Qᵀ⊗Qᵀ` orthogonal,
giving attenuation `|g(θ)|² = 1 − 2λ(1−λ)(1−cos θ)`. Both λ = 0 and λ = 1 are correlation-neutral, so
λ = 1/2 is the unique maximally-filtering interior point — and the real-data λ-sweep peaks at exactly
0.50. Map-and-shrink is therefore a spectral denoiser: it passes components invariant under the
inferred alignment and attenuates rotated, run-specific ones. Energy accounting confirms it — the
shared component carries 29.9% of its energy in the passband against 19.0% for the run difference.
Controls isolate the mechanism: Haar-random maps give zero gain, a single common template map recovers
89% of it, and fitted subject maps add a further +0.006, so the filtering is template-directed.

**Uncertainty columns.** Only the hierarchical model emits a per-subject posterior width
(τ_φ = 0.0345) and a group REML effective sample size (n_eff = 61.6 < S = 83, ci_ratio 1.15). Every
baseline returns a map for all 83 subjects and leaves both columns empty.

**Synthetic ground truth.** With a planted correspondence, TrajOT recovers the true coupling at
**0.70** against 0.03 for point-OT and 0.028 for FUGW. Planted-ambiguous subjects are ranked at
**AUROC 1.00** by posterior row-entropy, while Sinkhorn τ_φ is inverted on the same plant (AUROC
0.00) — identifying which posterior functional carries the signal is part of the contribution.

## Data

[OpenNeuro **ds000243**](https://openneuro.org/datasets/ds000243) — 120 subjects, 203 resting runs,
TR 2.5 s, CC0. Chosen because 83 subjects have two same-day resting runs, which is what makes β
calibration and held-out-run evaluation possible.

```bash
aws s3 sync --no-sign-request s3://openneuro.org/ds000243 ./data/ds000243
```

Raw data stays local and is never committed. Preprocessing is built in-repo (no fMRIPrep
derivatives exist for this dataset): slice timing → motion correction → MNI152 affine → fsaverage4
surface sampling → confound regression → band-pass 0.01–0.1 Hz → Schaefer-100 parcellation. The BIDS
sidecars carry no `SliceTiming`, so the slice order is recovered from inter-slice phase lags across
all 203 runs. Details: [`trajot/src/trajot/io/DATASET.md`](trajot/src/trajot/io/DATASET.md).

## Repository layout

```
trajot/
├── configs/          experiment, model and eval configs (the unit of comparison)
├── scripts/          preprocess · fit · evaluate · compare · run_experiment · verify_real_sota
├── src/trajot/
│   ├── io/           dataset discovery, preprocessing, the frozen NPZ contract
│   ├── geometry/     connectivity, diffusion maps, anatomical cost
│   ├── model/        generative model and ELBO terms
│   ├── inference/    Sinkhorn posterior, encoder, training, β calibration
│   ├── baselines/    noalign · brainsync · fugw · conn_srm · ours
│   ├── eval/         identification, alignment gain, permutation nulls, uncertainty
│   ├── report/       tables, figures, group REML, sensitivity
│   └── runlog/       run manifest, registry, structured logging
├── results/tables/   frozen comparison tables
└── tests/            461 tests across 41 files
```

Three design rules hold the project together:

- **One data contract.** The loader emits a single validated NPZ per subject-run; every downstream
  module consumes only that, so any module can be replaced without touching the others.
- **Config is identity.** `run_id = experiment + config hash + timestamp`. Two runs that differ in any
  resolved parameter cannot be mistaken for each other, and every run is re-runnable from its
  `manifest.json` alone.
- **One path for every method.** Baselines and the model share preprocessing, folds and evaluation
  code. Adding a method is adding a config file, not editing a pipeline.

## Running it

```bash
git clone https://github.com/HetJivani04/SURGE-Neurohack-DMLS-Fall-2026.git
cd SURGE-Neurohack-DMLS-Fall-2026/trajot
pip install -e .

cp configs/paths.example.yaml configs/paths.yaml   # set your local data_root
python scripts/preprocess.py                       # all 203 runs, resumable
python scripts/run_experiment.py --config configs/experiments/10_ours_full.yaml
python scripts/compare.py --experiments all --out reports/results_table.md
```

Each run writes `runs/<run_id>/` containing `manifest.json` (config, git commit, data hash, seed),
`metrics.json`, `log.txt` and `artifacts/`, plus a row in `runs/index.csv`.

Everything runs CPU-only in float64 on a 16 GB laptop. Tests: `pytest -q`.

## Key documents

| File | What it holds |
|---|---|
| [`trajot/reports/RESULTS.md`](trajot/reports/RESULTS.md) | **Canonical results.** Full tables, protocol metadata and statistics |
| [`PLAN.md`](PLAN.md) | The research plan — gap, prior work, model, evaluation protocol. §4b states the four gaps and their fixes |
| [`docs/compose/reports/surge-mathematical-analysis.md`](docs/compose/reports/surge-mathematical-analysis.md) | Root-cause analysis of the identification/alignment tradeoff |
| [`trajot/src/trajot/io/DATASET.md`](trajot/src/trajot/io/DATASET.md) | Dataset facts and the full preprocessing record |
| [`trajot/results/tables/REAL_sota_stats_n83_posterior_entropy.md`](trajot/results/tables/REAL_sota_stats_n83_posterior_entropy.md) | **N = 83 SOTA table** — point estimates and 10k paired bootstrap |
| [`trajot/results/tables/REAL_n83_gap_sota.md`](trajot/results/tables/REAL_n83_gap_sota.md) | N = 83 method table, λ diagnostics and the group REML block |
| [`trajot/results/tables/REAL_sota_stats.md`](trajot/results/tables/REAL_sota_stats.md) | N = 49 replication |

## Attribution

The components are inherited and attributed precisely: subject-to-template plans with a barycenter
template from **FUGW** (Thual et al., NeurIPS 2022); the amortized encoder from **ULOT** (Mazelet,
Flamary & Thirion, NeurIPS 2025); distributions over transport plans from **Mallasto et al.** (ACML
2021); alignment variance in the group model from **Keller et al.** (*Statistica Sinica* 2008); the
Gromov–Wasserstein objective from **Mémoli** (*FoCM* 2011) and **Demetci et al.** (AISTATS 2024). The
contribution is the assembled object — a population distribution over latent alignment couplings with
shrinkage, on the transport polytope, for cross-subject rest fMRI — which we did not find claimed in
the sources we checked.

Hackathon starter materials from the organisers are preserved in
[`Archive/`](Archive/HACKATHON_README.md).
