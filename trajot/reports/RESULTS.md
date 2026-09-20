# Results: hierarchical population-of-couplings for rest-fMRI alignment

**Status: Phase 2 REAL N=83 statistical closer COMPLETE and PRIMARY. Cohort 015–068 + 092–120 (83 strict two-run subjects), ds000243, Schaefer-100, beta = 28.438 (scan-rescan). Same subject-level map Q_s on both runs (Q_s fit on run-1, applied to both). Claim `task_sota_reliability` EARNED at N=83 (reliability 95% CIs exclude 0 vs noalign/BrainSync/FUGW, 10k paired bootstrap); identification NOT claimed — the same-map protocol is map-invariant (Haar-random and permuted maps also score 83/83); see §5b. N=49 is retained as replication. This document does not invent wins.**

Canonical tables (N=83 PRIMARY):
- `trajot/results/tables/REAL_sota_stats_n83_posterior_entropy.{json,md}` — **read this first**: N=83 same-Q SOTA stats, entropy-λ (headline)
- `trajot/results/tables/REAL_sota_stats_n83_entropy.{json,md}` — independent replication of the same N=83 entropy-λ stats (same numbers; identification not claimed — §5b)
- `trajot/results/tables/REAL_sota_stats_n83_posterior_tau.{json,md}` — N=83 same-Q SOTA stats, τ-λ variant (same verdict, smaller Δrel)
- `trajot/results/tables/REAL_n83_gap_sota_summary.md` — N=83 method-table summary
- `trajot/results/tables/REAL_n83_gap_sota.{json,md}` — N=83 method table
- `trajot/results/tables/freeze_n83/` — N=83 cohort + beta freeze

N=49 replication tables:
- `trajot/results/tables/REAL_sota_stats.{json,md}` — N=49 same-Q verification + bootstrap + gap columns
- `trajot/results/tables/REAL_n49_gap_sota.json` — N=49 method-level REAL harness
- `trajot/results/tables/group_real_n49.json` — group REML on real posteriors
- `trajot/results/tables/synthetic_gap_results.json` — planted-GT coupling recovery sidecar

Superseded N=83 fallback tables (posterior artifacts 73533e35 covered 49/83 subjects; kept for the honest ledger):
- `trajot/results/tables/REAL_sota_stats_n83.{json,md}` — N=83 bootstrap (fallback path)
- `trajot/results/tables/REAL_sota_stats_n83_artifact_subset.{json,md}` — posterior_shrink on freeze∩artifacts
- `trajot/results/tables/group_real_n83_artifacts_max.json` — group REML max available (S=49)

Reproduce the N=83 primary statistical verification (entropy-λ headline):

```bash
cd trajot
PYTHONPATH=src python scripts/verify_real_sota.py \
  --cohort-file results/tables/freeze_n83/frozen_cohort_n83.txt \
  --beta 28.438323293411973 \
  --data-root /Users/anandlo/Surge2026F/ds000243-master \
  --ours-artifacts runs/10_ours_full__9d7dab12__20260920T140613Z/artifacts \
  --methods noalign,brainsync,fugw,ours_full_posterior_shrink_entropy \
  --ours-lambda-source row_entropy \
  --out-stem REAL_sota_stats_n83_posterior_entropy \
  --n-boot 10000
```

τ-λ variant: same command with `--ours-lambda-source tau` (default), `--methods noalign,brainsync,fugw,ours_full_posterior_shrink`, `--out-stem REAL_sota_stats_n83_posterior_tau`.

N=49 replication: `--ours-artifacts runs/10_ours_full__73533e35__20260920T074217Z/artifacts` (default N=49 cohort) → `REAL_sota_stats`. Superseded N=83 fallback: same 73533e35 artifacts → `REAL_sota_stats_n83` / `REAL_sota_stats_n83_artifact_subset`.

---

## 1. Literature gap (why these columns exist)

**PLAN §1, in one sentence:** for resting-state fMRI you cannot tell whether a cross-subject functional alignment is real, because every method returns a map and none reports whether that map captured anything.

Verbatim class quotes that motivate the gap columns:

- **Thual et al. 2025, TMLR** (FUGW follow-up): *"our approach currently requires left-out participants to watch the same stimuli as reference participants. It is yet unclear whether functional alignment could bring improvements without this constraint."*
- **BrainSync, Joshi et al. 2018, NeuroImage**: *"the BrainSync transform will always attempt to maximize correlations, resulting in some degree of positive correlation even for data that do not satisfy our underlying assumption of common networks."*
- **Takeda et al. 2025, iScience**: *"In both datasets we analyzed in this study, the results of unsupervised alignment at the individual level were statistically unreliable."*
- **Haxby et al. 2020, eLife**: the general validity of a common rs-fMRI alignment model *"has not yet been established."*
- **Bazeille et al. 2021**: it *"remains unclear how researchers should choose among the available functional alignment methods."*

**Unclaimed object (PLAN §3–§4):** a hierarchical population-of-couplings with group Σ^al / τ_φ — a posterior over subject-to-template maps, shrinkage toward a population coupling, and group REML that down-weights subjects whose alignment the data do not determine. Point-estimate SOTA baselines (noalign, BrainSync, FUGW, conn_srm) emit a map for every subject and leave uncertainty columns null.

---

## 2. Comparable SOTA table — REAL N=49 (replication; posterior_shrink path)

N=49 replication of the earlier closer; the primary claim is now N=83 (§5b).

Artifacts: `runs/10_ours_full__73533e35__20260920T074217Z` (gauge on, hierarchical π shrink, tau0_auto). Transform: posterior-gated `C̃ = (1−λ)C + λ Q_s^T C Q_s` with the **same** `(Q_s, λ_s)` for both runs of subject s. λ from τ_φ via `λ = 1/(1+(τ/τ0)²)`, `tau0_eff ≈ median(τ)`.

| Method | Ident | Scan-rescan after | Alignment gain | τ_φ | Group n_eff | Valid SOTA competitor? |
|---|---:|---:|---:|---:|---:|---|
| noalign | 0.959 | 0.642 | 0.000 | — | — | yes |
| BrainSync (rest) | 0.959 | 0.643 | +0.0007 | — | — | yes (near no-op on connectomes) |
| FUGW (OT) | 0.918 | 0.620 | −0.005 | — | — | yes |
| conn_srm | 0.082 | 0.850 | +0.395 | — | — | **NO — identity collapse** |
| **ours_full_posterior_shrink** | **0.980** | **0.685** | −0.035 | **0.0092** | **7.45 < 12** | replication row |
| ours_full_point_procrustes | 0.857 | 0.644 | +0.018 | 0.0092 | — | ablation (hierarchy off the map) |
| ours_full_c_bar_procrustes | 1.000 | 0.643 | −0.400 | 0.0092 | — | failed retry (BB^T poor EMD target) |

Notes:
- **conn_srm is invalid as a gain competitor**: +0.395 gain with ident 0.082 is identity collapse, not alignment.
- **BrainSync is a structural near-no-op** on spatial connectomes: time-domain `Q` leaves `XQQ^T X^T = XX^T` unchanged. Observed gain ≈ +0.0007.
- **point_procrustes** (hierarchy not driving the map) loses reliability (0.644) and ident (0.857) relative to posterior_shrink — the hierarchical posterior is the map, not a side diagnostic.
- Native-gauge `heldout_score_module = −||C_run2 − T(C_run1)||²` is **not** used for the SOTA claim: any nontrivial reindexing Q≠I inflates this residual even when reliability improves. The correct held-out is same-Q scan-rescan reliability `corr(vec(T(C1)), vec(T(C2)))`.

---

## 3. Task SOTA claim — N=49 (replication; statistical verification)

**Claim rule (pre-declared):** claim task SOTA only if the paired-bootstrap 95% CI for `(ident_ours − ident_baseline)` excludes 0 in the positive direction, **or** reliability_delta is positive with CI excluding 0 vs noalign (and non-negative / significant vs BrainSync/FUGW). Identification superiority is **not** claimed when the CI includes 0. **Post-hoc override (2026-09-20):** the identification branch was overridden by the map-invariance control — under per-subject same-map protocols the ident metric is trivially perfect for any map family (fitted / Haar-random / permuted all 83/83), so the N=83 claim rests on the reliability branch alone (§5b).

### Paired bootstrap, B=10,000 (subject-level; ID bootstrap recomputes top-1 on resampled galleries)

| Baseline | Ident Δ | Ident 95% CI | Reliability Δ | Reliability 95% CI | McNemar exact p |
|---|---:|---|---:|---|---:|
| noalign | +0.0204 | **[0.0000, 0.0408]** | **+0.0434** | **[0.0379, 0.0490]** | 1.0 (discordant 1 vs 0) |
| BrainSync | +0.0204 | **[0.0000, 0.0408]** | **+0.0425** | **[0.0366, 0.0486]** | 1.0 |
| FUGW | +0.0612 | **[0.0000, 0.0612]** | **+0.0648** | **[0.0576, 0.0722]** | 0.25 |

### Verdict (N=49, replication): **task_sota_reliability** — claim holds at N=49

Superseded by the N=83 claim (§5b) — **reliability only**; identification was withdrawn there as a protocol artifact (map-invariance control).

> **Posterior-gated hierarchical alignment improves scan-rescan reliability on real ds000243 vs noalign, BrainSync, and FUGW** (paired bootstrap 95% CIs exclude 0 for all three: Δ_rel = +0.043 / +0.043 / +0.065). **Identification point estimate is higher** (0.980 vs 0.959 / 0.959 / 0.918) **but ident bootstrap CIs touch 0** — reported as a **trend only**, not as ID superiority.

Same-Q verification: `transform_all(run2)` reuses the run-1 maps (`max_abs_diff = 0.0`). Cross-subject pair gain on transformed features does **not** improve (ours −0.035); the scientific win is **within-subject reliability under a shared subject map**, not forced cross-subject correlation (which is what collapses conn_srm).

---

## 4. Gap fill — columns baselines cannot produce

From real posteriors (`group_real_n49.json`, `REAL_sota_stats.json` gap block):

| Method | τ_φ | Group n_eff | Identifiability flags | Baseline null |
|---|---:|---:|---|---|
| noalign | — | — | — | silent: map for every subject |
| BrainSync | — | — | — | silent |
| FUGW | — | — | — | silent |
| conn_srm | — | — | — | silent (and collapses identity) |
| **ours_full_posterior_shrink** | **0.00924** (median 0.00925; p90 subject-mean 0.00974) | **7.45 < S=12** (min node n_eff 2.46; ci_ratio 1.28) | REML weights; 0 subjects with τ > 2×median; **5/49** subjects in the top τ decile (diffuse alignments) | **emits τ_φ + Σ^al via REML** |

**Why these columns matter (literature):** Thual 2025 documents that stimulus-free alignment is unproven; BrainSync 2018 documents that the rest aligner maximizes correlation even when the common-network assumption fails; Takeda 2025 documents that individual-level unsupervised alignment is statistically unreliable. None of those methods reports *whether a given subject's map is determined*. The hierarchical model does: group REML on real posteriors yields **n_eff = 7.45 < 12**, i.e. alignment uncertainty down-weights subjects before group inference. Baselines report an alignment for all 49 subjects with no flag.

**High-τ flag counts (honest):** under τ > 2× cohort median, **0/49** subjects are flagged on this cohort (τ is tightly concentrated ≈ 0.0086–0.0098). Under a top-decile criterion, **5/49** subjects carry the diffuse tail. The structural gap metric that PASSes is **n_eff < S** under REML — that is the group-level statement baselines cannot make.

**Unclaimed object filled:** hierarchical population-of-couplings + group Σ^al / τ_φ on REAL rest, next to a reliability SOTA win vs noalign/BrainSync/FUGW.

---

## 5. Coupling recovery + identifiability ranking (synthetic planted-GT sidecar)

Planted-GT synthetic (`synthetic_gap_results.json`, v2; N=60, R=K=50, β=29.189, M=20, epochs=10):

| Metric | ours_full | point / EMD→C_pop | FUGW | random |
|---|---:|---:|---:|---:|
| **coupling_recovery** (assignment of planted π*) | **0.70** | 0.03 | 0.028 | 0.02 |
| **AUROC(ambiguity)** via posterior row-entropy of π̄ | **1.00** | — (no posterior) | — | 0.5 |
| AUROC via Sinkhorn τ_φ (legacy) | 0.00 | — | — | 0.5 |

**Why coupling recovery, not reconstruction error:** under the generative model `C_s = C_true + E`, the Bayes-optimal estimator of `C_true` from `C_s` without a perfect map is the identity — reconstruction error to noisy connectomes rewards doing nothing (`recovery_error_noalign` = 6.49 is the noise floor; every aligner is ≈20 because imperfect maps move away from the identity). The fair alignment metric is recovery of the **map** (posterior coupling / assignment vs planted P*). On that metric ours_full beats point-OT/EMD/FUGW by a wide margin (0.70 vs ≈0.03).

**Identifiability ranking works — through entropy, not τ.** Planted-ambiguous subjects (0.5 P₁ + 0.5 P₂) are ranked perfectly by the **posterior row-entropy of π̄** (AUROC = 1.00; ambiguous mean entropy 0.976 vs sharp 0.204). Sinkhorn τ_φ is *inverted* on this plant (AUROC = 0.00) because it is a noise scale on encoder scores, not an ambiguity measure — the uncertainty column that carries identifiability is the assignment entropy.

**Coverage: reported, not sold.** Raw quantile coverage of planted P* is 0.004 — the detached entropy Jacobian makes posterior draws overconfident. A held-out temperature-calibrated interval reaches 0.973, which is a calibration device, **not** Bayes coverage; `coverage_is_bayes=false` in the artifact. Do not cite coverage as a calibrated result until the Jacobian term is wired.

**No ground-truth correspondence at rest.** Rest has no shared time axis and no known vertex correspondence between subjects, so nothing on real data can be scored against a true alignment; the planted correspondence above is the only ground truth in this document.


---

## 5b. N=83 scale-up — COMPLETE; PRIMARY reliability claim earned (2026-09-20)

Manifest rebuilt from all contract npz: **195 rows / 112 subjects / 83 strict two-run** (015–068 + 092–120; equal n_volumes per subject). Freeze: `results/tables/freeze_n83/`. `data_hash=4e6703055921b762fa77241438ea6e4f9fbd90a26cc926210c03c1546e5ee0ca`.

The earlier fallback state (posterior artifacts `73533e35` covered 49/83 subjects; transform fell back to `region_emd_procrustes`) is **superseded**: new artifacts `runs/10_ours_full__9d7dab12__20260920T140613Z/artifacts` carry posteriors for all 83 subjects, and the hierarchical path is active on full N=83.

**Beta calibration (scan-rescan, R=100):**
- N=83 (primary): **β = 28.438323293411973**, σ̂² = 0.0351638171379696, n_regions=100
- N=49 (replication): **β = 29.189086229914952**
- N=83 runs use β=28.438; N=49 tables keep 29.189. Do not mix.

### N=83 point estimates — PRIMARY (same-Q; pairs=500 seed 2026)

Protocol: same-map — `Q_s` fit on run-1 and applied to **both** runs, `T(C) = (1−λ)C + λ Q_s^T C Q_s`; λ from posterior row-entropy (`λ_mean = 0.500`, `λ_source = row_entropy`); 83 two-run subjects (`frozen_cohort_n83`); 10,000 paired bootstrap; McNemar exact on identification discordance. Baselines (noalign / BrainSync / FUGW) are evaluated under the **same-map protocol** as ours.

| Method | Ident | Ident correct | Scan-rescan after |
|---|---:|---:|---:|
| **ours_full_posterior_shrink_entropy** | **1.0000** | **83/83** | **0.6975** |
| noalign | 0.9157 | 76/83 | 0.6455 |
| BrainSync | 0.9036 | 75/83 | 0.6461 |
| FUGW | 0.9036 | 75/83 | 0.6215 |

**Ident column caveat:** under this same-map protocol the Ident column is **protocol-invariant and not interpretable for per-subject maps** — controls on the real N=83 cache score fitted Q = 1.0, Haar-random Q = 1.0, permuted fitted Q = 1.0 (identity 0.9157; single common map 0.9036). Identification is therefore not claimed; the Ident column and the McNemar stats below are retained for the ledger only (withdrawal note in the verdict).

conn_srm on N=83 remains **INVALID** (identity collapse: ident 0.024, gain +0.386) and is not a gain competitor.

### Paired bootstrap B=10,000 — ours_full_posterior_shrink_entropy − baseline (reliability CIs exclude 0 positively; ident Δ / McNemar retained for the ledger only — protocol-invariant, not a claim)

| Baseline | Ident Δ | Ident 95% CI | Reliability Δ | Reliability 95% CI | McNemar (ours-correct / base-wrong) | McNemar exact p |
|---|---:|---|---:|---|---|---:|
| noalign | +0.0843 | **[0.0120, 0.0723]** | +0.0519 | **[0.0480, 0.0558]** | 7 / 0 | **0.0156** |
| BrainSync | +0.0964 | **[0.0120, 0.0843]** | +0.0513 | **[0.0472, 0.0552]** | 8 / 0 | **0.0078** |
| FUGW | +0.0964 | **[0.0120, 0.0843]** | +0.0759 | **[0.0705, 0.0814]** | 8 / 0 | **0.0078** |

### Verdict: **task_sota_reliability** — claim EARNED at N=83; identification WITHDRAWN (protocol artifact)

> **Posterior-gated hierarchical alignment improves scan-rescan reliability on real ds000243 vs noalign, BrainSync, and FUGW** — Δrel **+0.0519** [0.0480, 0.0558] / **+0.0513** [0.0472, 0.0552] / **+0.0759** [0.0705, 0.0814] (10k paired bootstrap; point estimates ours 0.6975 / noalign 0.6455 / BrainSync 0.6461 / FUGW 0.6215). **Identification is not claimed** — the same-map protocol is map-invariant.

- **Spectral mechanism (derived and numerically verified).** `T_λ(C) = (1−λ)C + λ Qᵀ C Q` acts on `vec(C)` as `M_λ = (1−λ)I + λR`, `R = Qᵀ ⊗ Qᵀ` orthogonal with eigenvalues `e^{iθ}` (θ = φ_i − φ_j); attenuation `|g(θ)|² = 1 − 2λ(1−λ)(1−cos θ)`, which at λ = 1/2 is `cos²(θ/2)`. λ = 0 and λ = 1 are both correlation-neutral; **λ = 1/2 is the unique maximally-filtering interior point**, and the real-data λ-sweep peaks exactly at 0.50 (0.6975; 0.4→0.6942, 0.6→0.6939). Energy accounting: the shared component carries 29.9% of its energy in the passband vs 19.0% for the run-difference (bottom quartile 11.7% vs 18.5%) — the filter removes run noise preferentially. The posterior row-entropy gate `λ_s = 1/(1+(H_s/H₀)²)` makes λ adaptive per subject.
- **Controls isolate template-directed denoising.** Reliability at λ = 0.5 on the real N=83 cache: raw 0.6455; Haar-random maps 0.6454 (zero gain — not generic smoothing); permuted fitted maps 0.6708 (≈half the gain); single common template map 0.6919 (89% of the gain); fitted subject maps 0.6975. So **≥89% of the reliability gain is template-directed denoising using no subject-specific run-1 information**; the subject-specific increment is **+0.006**.
- **Identification withdrawal.** Under the per-subject same-map protocol (map fitted on run-1, applied to both runs), identification accuracy is fitted Q = 1.0, Haar-random Q = 1.0, permuted fitted Q = 1.0, single common map = 0.9036, identity/no-map = 0.9157. Any per-subject map family makes the protocol trivially perfect (the correct pair shares its map; wrong pairs are compared across mismatched maps), so identification is uninterpretable for per-subject maps. The earlier draft claim `task_sota_ident_and_reliability` (commit ed7b984, not pushed) is **retracted**.
- **τ-λ variant** (`REAL_sota_stats_n83_posterior_tau.{json,md}`, `λ_mean=0.893`): reliability Δrel +0.0122 / +0.0116 / +0.0362, all CIs exclude 0 — same reliability verdict, smaller Δrel. No cross-dataset claim: this is ds000243 rest, same-map protocol only.

### Superseded: fallback-path bootstrap (artifacts 73533e35; honest ledger)

The earlier 49/83-posterior run produced fallback-path numbers (full N=83 fallback: Δrel +0.0028 / +0.0022 / +0.0268 vs noalign / BrainSync / FUGW; ident −0.072 / −0.060 / −0.060 — identification *worse* than baselines) and a 49-subject subset (Δrel +0.0434 / +0.0425 / +0.0648; ident +0.020 / +0.020 / +0.061 trend-only). Those are **superseded** by the primary N=83 table above; they remain documented here because this ledger does not rewrite history.

### Group REML on max available posteriors (superseded artifacts)

| Artifacts subjects | n_eff (mean) | n_eff min | ci_ratio | n_eff < S |
|---:|---:|---:|---:|---|
| **49** (015–063, max available) | **28.50** | 4.97 | 1.234 | **YES** |
| 24 (smoke) | 13.91 | 2.57 | 1.244 | YES |
| 12 (N=49 closer) | 7.45 | 2.46 | 1.277 | YES |

The regenerated N=83 gap table now carries a full-cohort group block — n_subjects = 83, n_eff = **61.5718** (min 50.311), ci_ratio = 1.1473, **no NaN** (degenerate-node limit fix: n_eff = #{u=inf} for τ²+σ²=0 nodes; verified by ε-perturbation). n_eff < S = 83 honest; caveat: under the stored unit subject-map normalisation this is a machinery check, not a scientific estimate. Baselines still leave τ_φ / Σ^al null.

### N=83 claim verdict (honest)

1. **`task_sota_reliability` on full N=83: EARNED** — hierarchical maps applied to all 83 subjects (same-Q); reliability 95% CIs exclude 0 vs all three baselines (Δrel **+0.0519** [+0.0480, +0.0558] / **+0.0513** [+0.0472, +0.0552] / **+0.0759** [+0.0705, +0.0814]; 10k paired bootstrap).
2. **Identification is NOT claimed** — the same-map protocol is map-invariant: fitted, Haar-random, and permuted maps all score 83/83 (identity 0.9157; single common map 0.9036); the ident CIs and McNemar stats in §5b are not a scientific result.
3. **N=49 retained as replication** (`REAL_sota_stats.json`; verdict was `task_sota_reliability`, ident trend-only at that N — identification remains unclaimed at every N).
4. **Superseded history:** the first N=83 attempt (artifacts 73533e35, 49/83 posteriors, EMD-Procrustes fallback) is documented above — it is not the primary claim.
5. **Gap columns:** τ_φ + group REML n_eff < S on full N=83 — regenerated group block: n_subjects=83, **n_eff=61.5718** (min 50.311), ci_ratio=1.1473, **no NaN** (degenerate-node limit fix: n_eff = #{u=inf} for τ²+σ²=0 nodes; verified by ε-perturbation). n_eff < S = 83 honest. Caveat: under the stored unit subject-map normalisation this is a machinery check, not a scientific estimate.
6. **Do not claim:** ID superiority where a CI includes 0; **identification superiority under per-subject-map protocols (map-invariant artifact — Haar control = 83/83)**; calibrated coverage on real rest; gain-null nonident counts; conn_srm on gain; cross-dataset generalization.

---

## 6. What failed (honest ledger)

1. **Coverage@0.9 / AUROC-τ on synthetic** — FAIL (overconfident posterior on mixed maps; C_bar scale mismatch ‖C_bar‖≈1.4 vs ‖C_pop_true‖≈24.7). Not claimed.
2. **c_bar Procrustes retry** — heldout residual −1617, gain −0.400. Learned BB^T is a poor EMD target vs empirical C_pop. One retry; no further invented wins.
3. **Native-gauge heldout_score_module** — ours does not beat noalign/FUGW on this residual (−775 vs −702/−704). Metric caveat: Q≠I inflates residual to native-gauge run2. SOTA claim uses same-Q reliability instead.
4. **conn_srm** — high gain via identity collapse (ident 0.082). Disqualified as a gain competitor.
5. **Identification superiority at N=49** — point estimate won (0.980) but bootstrap CI touched 0; McNemar p=1.0 vs noalign (one discordant subject). **Trend only at N=49.** At N=83 the same test appeared decisive (7–8 discordant pairs, p < 0.02; §5b) but was subsequently shown to be a protocol artifact and **withdrawn** — identification is unclaimed at every N under per-subject-map protocols.
6. **Cross-subject alignment_gain** — ours_full_posterior_shrink is negative (−0.035). Positive gain without collapse is **not** the headline; reliability under a shared subject map is.
7. **Gain-null nonident counts** — saturate under degenerate nulls; not a scientific uncertainty rate. Posterior τ / REML n_eff are the uncertainty surface.
8. **Long-run scan-length sensitivity** — not run on the frozen two-run cohort (long runs are one-run subjects).
9. **N=83 hierarchical posterior_shrink (first attempt, artifacts 73533e35)** — was INCOMPLETE: posteriors covered 49/83 subjects and the transform fell back to `region_emd_procrustes` (ident degraded, 0.843 vs ~0.90–0.92 baselines). **Superseded**: artifacts `9d7dab12` cover 83/83 with the hierarchical path active; the N=83 **reliability** claim is earned (§5b).

---

## 7. Claim discipline appendix

- **Finitely many optima; single-point minimizers are not claimed.** Gromov–Wasserstein is invariant to isometries (Mémoli 2011); the feature term reduces the isometry orbit to **finitely many optima** (Demetci et al., AISTATS 2024). No single optimizer is asserted.
- **Inherited components are attributed.** Subject-to-template plans with a barycenter: **FUGW** (Thual et al., NeurIPS 2022). Amortized encoder: **ULOT** (Mazelet, Flamary, Thirion, NeurIPS 2025). Distributions over transport plans: **Mallasto, Gerolin, Minh** (ACML 2021) and **De et al.** (ICML 2026). Alignment variance in the group model: **Keller, Roche, Tucholka, Thirion** (*Statistica Sinica* 2008); Hu et al. (ICLR 2025) for learned registration. GW objective: **Mémoli** (*FoCM* 2011) and **Demetci et al.** (PMLR 238, 2024). Nearest neighbour **OTTER** (bioRxiv 2026) has soft mass, not a posterior/hierarchy.
- **Absences are search-based.** The assembled object — population distribution over latent alignment couplings with shrinkage on the transport polytope for cross-subject rest-fMRI — is unclaimed in the sources we checked: **no such work was found**. Stated as a search result, not as a metaphysical claim.
- **The band prior is a band prior and not a dynamics model.** It constrains frequency content of a coupling; weight 0 in frozen runs.
- **No behavioural or cognitive prediction is reported as a metric.** **Marek et al. 2022** report a median brain–behaviour correlation of **|r| = 0.01 at N = 3,928**; ds000243 has 120 subjects; any such number here would be noise presented as a result.
- **Random-effects collapse identity.** Group model reduces to the one-sample t-test when alignment variance → 0; tested, not asserted.
- **Do not claim:** ID superiority where the CI includes 0; **no identification claim at all under per-subject-map protocols (map-invariant artifact — Haar control = 83/83)**; calibrated coverage on real rest; gain-null nonident counts as scientific rates; conn_srm as a gain competitor. (The ident branch of the pre-declared rule was overridden by the map-invariance control; the N=83 claim is reliability-only — no cross-dataset generalization is claimed.)

---

## 8. Protocol metadata

### N=83 (PRIMARY; hierarchical path complete)

- Cohort: 83 strict two-run (**015–068 + 092–120**); freeze `results/tables/freeze_n83/` (`frozen_cohort_n83`)
- Manifest: 195 rows / 112 subjects rebuilt via `phase2_rebuild_manifest.py`
- data_hash: `4e6703055921b762fa77241438ea6e4f9fbd90a26cc926210c03c1546e5ee0ca`
- beta: **28.438323293411973** (N=83 scan-rescan; σ̂²=0.0351638171379696; R=100)
- pairs **500** seed **2026**; bootstrap **B=10,000**; McNemar exact on identification discordance
- Ours artifacts: `runs/10_ours_full__9d7dab12__20260920T140613Z/artifacts` — **posteriors for all 83 subjects**
- Transform: `posterior_shrink_tau_gated`, `posterior_drives_transform=true`, `lambda_source=row_entropy` (headline; λ_mean=0.500); τ-λ variant λ_mean=0.893 — same verdict
- Claim: **task_sota_reliability EARNED** — reliability 95% CIs exclude 0 vs noalign / BrainSync / FUGW (Δrel +0.052 / +0.051 / +0.076); identification **withdrawn** (same-map protocol map-invariant — fitted / Haar-random / permuted all 83/83)
- Python: `/Users/anandlo/.central_venv/bin/python3`

### N=49 (replication)

- Cohort: 49 two-run subjects **015–063**, frozen data root ds000243-master
- beta: **29.189086229914952** (scan-rescan, R=100 Schaefer parcels)
- Declared pairs: **500**, seed **2026**
- Bootstrap: **B=10,000** paired (verification script); pilot permutation table remains B=200
- Ours artifacts: `runs/10_ours_full__73533e35__20260920T074217Z/artifacts`
- Transform path: `posterior_shrink_tau_gated`, `posterior_drives_transform=true`, `tau0_auto=true`, λ_mean≈0.50
- Group REML smoke: 12 subjects present in artifacts, n_eff=7.45, ci_ratio=1.28, PASS
- Claim: `task_sota_reliability` holds at N=49; ident trend-only at N=49; identification withdrawn at N=83 (protocol artifact — §5b)

## 9. Reproducing a row

Every harness row is regenerable from `scripts/run_real_gap_sota.py` + `scripts/verify_real_sota.py` against the named artifacts and data root. `compare.py` reads only `runs/index.csv` + `metrics.json` and never re-runs training. Long train PIDs on the machine were **not** killed for this write-up.

---

## 10. The results table — frozen N=49 pilot comparison (superseded; exactly as `compare.py` rendered it)

The six-row five-column table below is **exactly what `compare.py` rendered** at freeze time (`reports/results_table.md`). It is the superseded frozen N=49 pilot comparison, kept as the honest ledger entry (the primary N=83 claim table is §5b; the N=49 replication table is §2). The ablated row is the latest successful ablated run at freeze time (N=33), not an N=49 estimate.

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
