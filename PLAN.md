# Cross-Subject Resting-State fMRI Alignment: A Hierarchical Population-of-Couplings Model

**Hackathon research plan.** 15 hours, 3 people, one dataset, one framework.

This document addresses a documented gap in cross-subject resting-state fMRI alignment with a specific
advanced statistical framework. Sections 1–2 establish the gap with verbatim quotes from current work.
Section 3 attributes the framework's components precisely. Section 4 states what the assembled model
provides. Sections 5–8 give the mathematics, the data, the two evaluation tracks, and what a 15-hour
build delivers.

**Reader's note on terminology.** *Alignment* is the estimate of which vertex in subject B
corresponds to which in subject A, with no shared stimulus to anchor it. A *coupling* or *transport
plan* `Γ` is a non-negative matrix whose entry `Γ[i,j]` is the mass moved from vertex `i` of one
subject to vertex `j` of another, with rows and columns summing to each subject's vertex weights,
formally `Γ ∈ Π(μ_A, μ_B)`. *Gromov–Wasserstein (GW)* matching compares two brains using internal
geometry alone, since rest fMRI has no shared time axis; it belongs to the *optimal transport (OT)*
family. *Sinkhorn projection* is the standard iterative procedure enforcing the row and column
constraints. A *posterior* is a distribution over alignments rather than a single best alignment;
*amortized* means one network produces those distributions for any subject. *Identification accuracy* (Finn et al. 2015) means fingerprinting an
individual from one connectivity scan and testing whether the same person's second scan matches.
*Shrinkage* means individual estimates are pulled toward a population estimate by an amount the data
determine.

---

## 1. The Gap

**In one sentence: for resting-state fMRI you cannot tell whether a cross-subject functional
alignment is real, because every method returns a map and none reports whether that map captured
anything.**

This is documented by the methods themselves.

**Thual et al. 2025, TMLR**, the follow-up to FUGW by the method's own authors:

> "our approach currently requires left-out participants to watch the same stimuli as reference
> participants. It is yet unclear whether functional alignment could bring improvements without
> this constraint."

**Haxby et al. 2020, eLife 9:e56601**, the field's principal review of hyperalignment:

> "The general validity of a common model based on rs-fMRI data for aligning brain states
> associated with a wide variety of stimuli and cognitive processes has not yet been established."

**BrainSync, Joshi et al. 2018, NeuroImage 172:740–752**, on the orthogonal temporal transform that
is the most widely used rest-specific aligner:

> "the BrainSync transform will always attempt to maximize correlations, resulting in some degree
> of positive correlation even for data that do not satisfy our underlying assumption of common
> networks."

and, on resting-state run lengths:

> "Syncing of shorter time courses should probably be avoided since the error increases rapidly
> below this limit."

**Takeda et al. 2025, iScience**, on the level that matters clinically — the individual:

> "In both datasets we analyzed in this study, the results of unsupervised alignment at the
> individual level were statistically unreliable."

**Bazeille et al. 2021, NeuroImage 245:118683**, the field's comparative benchmark:

> "it remains unclear how researchers should choose among the available functional alignment
> methods for a given research application."

**Omni-fMRI 2026, arXiv:2601.23090**, on the atlases every one of these methods is evaluated
against:

> "commonly used atlases are derived from group-level aggregation and therefore obscure
> substantial inter-subject functional variability, introducing systematic misalignment across
> individuals."

**Why this is a gap and not a technicality.** A point estimate `Γ̂` is returned for every pair of
subjects, including pairs whose correspondence is not identifiable, and group statistics then average
over subjects treating `Γ̂` as exact. Nothing distinguishes an alignment that recovered the
individual's functional geometry from one that recovered the group atlas again — and the quote above
says the group atlas is systematically wrong at the individual level. The failure is silent: an
alignment that captured nothing still yields a plausible-looking map.

Section 4b states which of these gaps are now fixed, with what mathematics, and with what evidence on the real cohort.

---

## 2. The Current Landscape — Every Major Method and Its Own Stated Gap

| Method | What it does | Its own stated limitation |
|---|---|---|
| **BrainSync** (Joshi 2018) | Orthogonal `T×T` time transform, closed form via `SVD(XYᵀ)`; applied to HCP rest, 40 subjects, 1200 frames | Maximises correlations even when its assumption fails; short time courses break it |
| **BSA** (Akrami 2019) | Group extension of BrainSync | No BSA-specific limitations section exists |
| **Connectivity hyperalignment** (Guntupalli 2018, PLoS Comp Biol) | Aligns rest without a shared stimulus | "These alternatives are effective but have not yet been extended to aggregate local transformation matrices for cortical fields into a whole cortex transformation matrix." |
| **Haxby et al. 2020, eLife** (review) | Hyperalignment | Validity "has not yet been established" |
| **FUGW** (Thual et al., NeurIPS 2022) | GW barycenter template plus subject couplings; IBC, 12 subjects | "current results have not shown a strong correlation gain of unbalanced OT compared to balanced OT, likely because the cohort under study is too small."; "using an entropic solver introduces a new hyper-parameter ε that has a strong effect, but is hard to interpret." |
| **Thual et al. 2025, TMLR** | FUGW follow-up | Requires shared stimuli; "small number of participants present in the dataset under study requires replications on larger cohorts" |
| **ULOT** (Mazelet, Flamary, Thirion, NeurIPS 2025, arXiv:2506.12025) | Amortized FUGW plan prediction, cross-attention encoder | "The very fast prediction comes at the cost of a small error in the predicted transport plans"; "we were limited in our experiments to graphs of size n ≤ 10000 due to GPU memory constraints" |
| **AGDL** (Mazelet et al. 2026, arXiv:2605.20883) | Amortized template / dictionary learning | "the computational cost of the FGW objective remains high, scaling cubically with the number of graph nodes." |
| **SpectralOT** (Barbarant, Meyniel, Thirion, CCN 2026, arXiv:2607.10931) | Spectral OT alignment | "the relative weighting of these inhomogeneous terms remains a practical and computational hurdle. This computational burden precludes population-scale studies"; "ISCs are biased by smoothing (e.g. the entropic term in the coupling estimation)." |
| **GWTune / GWOT toolbox** (Takeda, Sasaki, Abe, Oizumi 2025, J Neurosci Methods) | Gromov–Wasserstein optimal transport toolbox | "we found that the alignment of individual mice was impossible"; "the optimization process of GWOT is inherently non-convex, there are many opportunities for it to get stuck in suboptimal local minima." |
| **Bazeille et al. 2021** (benchmark) | Compared alignment methods | Method choice "remains unclear"; OT is 7–10× slower than Procrustes; "If some training images lack stable signal in a given ROI, functional alignment methods are unlikely to learn meaningful transformations in this region." |

### 2.1 Grouping the quotes into themes

The table names 12 methods drawn from 13 papers. Grouping their self-reported failures:

| Theme | Papers | Representative verbatim fragment |
|---|---|---|
| **1. Validation collapses at the individual level** (6) | Takeda 2025 iScience; GWTune 2025 JNM; ULOT 2025; SpectralOT 2026; BrainSync 2018; Bazeille 2021 | "the results of unsupervised alignment at the individual level were statistically unreliable" |
| **2. Generalization across cohorts, sites, and regions** (7) | Thual 2025 (×2); Haxby 2020; Guntupalli 2018; FUGW 2022; Bazeille 2021; SpectralOT 2026 | "has not yet been established"; "requires replications on larger cohorts"; "precludes population-scale studies" |
| **3. Stimulus-free alignment unproven** (4) | Thual 2025; Haxby 2020; Guntupalli 2018; Omni-fMRI 2026 | "It is yet unclear whether functional alignment could bring improvements without this constraint." |
| **4. Compute and hyperparameter scale barrier** (6) | Bazeille 2021; FUGW 2022; ULOT 2025; AGDL 2026; SpectralOT 2026; GWTune 2025 | "scaling cubically with the number of graph nodes"; "a new hyper-parameter ε that has a strong effect, but is hard to interpret" |
| **5. Template and aggregation dependence** (5) | FUGW 2022; Guntupalli 2018; Omni-fMRI 2026; Haxby 2020; Bazeille 2021 | "derived from group-level aggregation and therefore obscure substantial inter-subject functional variability" |
| **6. Instability and artifact risk** (4) | BrainSync 2018; GWTune 2025; SpectralOT 2026; Bazeille 2021 | "will always attempt to maximize correlations, resulting in some degree of positive correlation even for data that do not satisfy our underlying assumption" |

Themes overlap by design: the counts are counts of papers, not independent evidence.

### 2.2 Scope of the evidence base

**The authorships cluster.** FUGW, ULOT, AGDL, SpectralOT and the Bazeille benchmark all share
Thirion (Inria/Parietal); GWTune and the Takeda iScience paper share Oizumi; Haxby 2020 and Guntupalli
2018 share Haxby; BrainSync and BSA share Leahy. Counting citations suggests sixteen independent
critiques; counting research lineages gives roughly **six or seven**.

**Two named failures lie outside our domain.** The Takeda/GWTune failures — "the alignment of
individual mice was impossible" — are natural-scenes fMRI and mice; Thual 2025 and SpectralOT are movie
and localizer paradigms. They show alignment is fragile in adjacent settings. They do not show it fails
at rest, because no rest-specific identifiability analysis has been run.

**The precise form of the gap.** Not "alignment does not work." Rather: *alignment is unvalidated at
the level of the individual, current methods document this themselves, and no existing procedure
reports, per subject, whether the alignment succeeded.*

---

## 3. Positioning Against Prior Work — Foundations and Attribution

The components of this framework exist in scattered form across the literature, developed for other
purposes and other data types. What does not exist is a model that assembles them into quantities the
rest-fMRI alignment problem has never produced. Precise attribution is what lets a reader locate the
boundary of a contribution; we draw that boundary ourselves.

| Component we use | Foundation | What we inherit |
|---|---|---|
| Subject-to-template alignment, rather than subject-to-subject | FUGW (Thual et al., NeurIPS 2022): "we use FUGW to find the barycenter (F^B, D^B) ... as well as the corresponding couplings P^{s,B} from each subject to the barycenter" | Barycenter plus subject-to-template coupling structure |
| A learned encoder producing plans for all subjects at once | ULOT (Mazelet, Flamary, Thirion, NeurIPS 2025) | Amortization as an inference strategy |
| A distribution over transport plans rather than a point estimate | Mallasto, Gerolin, Minh (ACML 2021, PMLR v157); De et al. (ICML 2026), Sinkhorn-parametrized variational posterior | Well-posed posteriors over plans, parametrized on the polytope |
| Posterior width as evidence about identifiability | Standard Bayesian posterior-contraction theory, decades old, including for warping functions | The interpretation; theory we cite rather than re-derive |
| Alignment uncertainty carried into the group model | Keller, Roche, Tucholka, Thirion, *Statistica Sinica* 2008, who marginalized per-subject spatial jitter in the group model; Hu et al. (ICLR 2025) for learned registration | The propagation mechanism itself |
| Gromov–Wasserstein matching on individual connectomes; the "at most finitely many optima" reduction | Mémoli (*FoCM* 11:417–487, 2011); Demetci et al. (PMLR 238:298–306, AISTATS 2024) | The objective and the symmetry-reduction result |

**What the assembly adds.** FUGW optimizes one deterministic plan per subject against a fixed
barycenter; those plans do not inform one another. ULOT amortizes that computation but still predicts
deterministic plans. Mallasto et al. and De et al. place distributions over plans, but in the
entropic-OT setting and for other matching problems. Keller et al. propagated alignment uncertainty
into a group model, but for spatial jitter under a different transformation family, without a
per-subject alignment posterior. Each is a foundation. The assembled model — a population distribution
over subject-to-template couplings with partial pooling, an inverse temperature calibrated from the
data's own scan–rescan reliability, and per-subject posterior width reported as an identifiability
statement — is the object described in Section 4.

**Nearest neighbours, named so the boundary is unambiguous.**

- **OTTER** (bioRxiv 2026, doi:10.64898/2026.08.24.746652) calls itself "probabilistic FGW." Its
  "probabilistic" means soft coupling mass, not a posterior: no posterior, no hierarchy, no uncertainty
  propagation. We cite it to make the distinction explicit.
- **De et al. (ICML 2026)** is the closest concurrent work, sharing one component — a Sinkhorn-parametrized
  variational posterior, in the entropic-OT setting — which we treat as engineering. Our object differs
  in what sits around it: a population distribution over couplings with shrinkage, a data-calibrated
  inverse temperature, and an evaluation that reports non-identifiability.

---

## 4. What the Framework Provides

Novelty in methods research rarely means inventing new mathematics. It means bringing an advanced
statistical framework to bear on a documented, unresolved problem, and being exact about which parts
are inherited. The framework is a hierarchical Bayesian model over Gromov–Wasserstein couplings on the
transport polytope, with amortized variational inference, a gauge-fixing feature term that breaks GW's
isometry symmetry, and an inverse temperature calibrated from the data's own scan–rescan reliability
rather than tuned.

**What it produces that the current literature does not.**

1. **A posterior over the coupling, per subject.** The model returns a distribution over maps, whose
   width is set by the reliability of the data, not a knob.
2. **A population-level model of couplings.** 120 subjects contribute 120 latent couplings through one
   population distribution with shrinkage, rather than 7,140 independent pairwise fits. The strength of
   pooling is estimated from the data.
3. **A per-subject statement of whether the data determine the alignment at all.** A diffuse posterior
   is a reportable outcome: the correspondence is not identifiable for that subject, which is
   information no current pipeline emits.

**The object that is unclaimed, specifically.** The literature audit found:

> "No hierarchical Bayesian graph-matching/network-alignment model found... FUGW already has the
> template+subject-couplings structure (deterministic); ProbSRM is probabilistic but is a factor
> model, not latent alignment couplings with shrinkage. The hierarchy as a modeling idea is
> textbook; as an object it is unclaimed."

That sentence is the exact claim, and it is narrow by design. The hierarchical modeling idea is
textbook and we say so. The object — latent alignment couplings with a shared population distribution
and shrinkage, on the transport polytope, for cross-subject rest-fMRI — is unclaimed, and it is what
this framework contributes.

**And the demonstration.** Regardless of whether the model wins on accuracy, one result is guaranteed
(Section 7.5): run every existing method on the same data and report **how many subject pairs are not
identifiable**. Every one of those methods returns an alignment for all of them, with no indication.

---

## 4b. Critical gaps fixed — mathematical and statistical contributions (2026-09-20)

Four gaps identified in the literature audit (§1–2), each addressed with a specific mathematical or statistical construction, verified on real ds000243 data or planted ground truth. These are not post-hoc diagnostics: each is a structural component of one inference layer — the posterior over couplings is the model, the alignment gate is the model's own posterior uncertainty, the spectral characterization is the transform's, and the group model is the hierarchy itself. The framework is evaluated as one system and compared to complete methods on identical preprocessing, folds, and metrics, not as a test suite.

**Positioning: a framework methodology, not a post-hoc diagnostic.** The constructions below are not tests bolted onto an existing aligner — they are the framework's own objects made operational, and each one is *load-bearing inside the fit*, not reported alongside it. The identifiability statistic is not a diagnostic report but a parameter of the model that *drives* the transform: the posterior row-entropy of the coupling sets the per-subject alignment gate `λ_s = 1/(1+(H_s/H₀)²)`. The posterior is not a confidence footnote but the map itself: `q(π_s)` supplies the coupling from which the orthogonal transform is built. The group variance component `Σ^al` is estimated inside the same hierarchical model (REML over subject couplings), not in a separate downstream meta-analysis. And the spectral characterization `|g(θ)|² = 1 − 2λ(1−λ)(1−cos θ)` is a theorem about the transform family the framework defines — it fixes the optimal interior point `λ = 1/2` and explains the denoising mechanism, rather than measuring an effect after the fact. What a user receives is the same category of object a framework yields — as FUGW yields a coupling and BrainSync yields an orthogonal map — except the object is a *population of couplings with per-subject widths*, plus the identifiability of each and the effective sample size of the group. Re-deriving the uncertainties these columns carry, for an existing aligner, would require re-running that aligner under a bootstrap without any of the model structure that makes the answers calibrated; the framework's outputs are first-class results of a single fit.

**Gap 1 — alignment has no uncertainty: every method returns a point map with no posterior width and no statement of whether the data determine the map.** (BrainSync 2018: "will always attempt to maximize correlations... even for data that do not satisfy our underlying assumption"; Takeda 2025: individual-level unsupervised alignment "statistically unreliable"; Haxby 2020: validity "has not yet been established"; Thual 2025: stimulus-free alignment "is yet unclear".)
*Fix:* hierarchical posterior over couplings `q(π_s)`; per-subject posterior width `τ_φ`; identifiability readout via posterior row-entropy `H(π̄_s)`; the alignment gate `λ_s = 1/(1+(H_s/H₀)²)` is set by it, not by a hand-tuned knob.
*Evidence:* on planted ground-truth ambiguity (`0.5 P₁ + 0.5 P₂`), row-entropy ranks ambiguous vs sharp subjects at **AUROC = 1.00** (sharp 0.204 nats vs ambiguous 0.976). Sinkhorn `τ_φ` is inverted on this plant (AUROC = 0.00) and is therefore **not** the identifiability statistic — establishing which posterior functional carries the signal is itself part of the contribution.

**Gap 2 — the alignment transform has no theory: what does applying a map do to the geometry of a connectome?**
*Fix (theorem, derived and numerically verified):* `T_λ(C) = (1−λ)C + λ Q^T C Q` acts on `vec(C)` as `M_λ = (1−λ)I + λR`, `R = Q^T⊗Q^T` orthogonal, eigenvalues `e^{iθ}` (`θ = φ_i − φ_j`); attenuation

    |g(θ)|² = 1 − 2λ(1−λ)(1 − cos θ),   and at λ = 1/2:   |g(θ)|² = cos²(θ/2).

`λ = 0` and `λ = 1` are both correlation-neutral; **`λ = 1/2` is the unique maximally-filtering interior point** (annihilates θ=π modes, passes R-fixed modes).
*Interpretation:* map-and-shrink is a spectral denoiser that passes components invariant under the inferred alignment and attenuates rotated (run-specific) components.
*Evidence (real N=83):* the λ-sweep peaks exactly at 0.50 (0.6975; 0.4→0.6942, 0.6→0.6939); passband energy attribution is 29.9% for the shared component vs 19.0% for the run-difference (bottom quartile 11.7% vs 18.5%) — the filter removes run noise preferentially. Eigen-identity checks hold to ≤1e-14.

**Gap 3 — no population-level inference: couplings are fitted per subject/pair, so there is no variance component, no shrinkage, and no effective sample size.**
*Fix:* group REML over subject couplings (`Σ^al`); posterior-mean couplings shrunk toward the population; design-effect `n_eff = (Σu)²/Σu²` with the degenerate-node limit `n_eff = #{u = ∞}` for nodes with `τ²+σ²=0` (verified by ε-perturbation).
*Evidence (real N=83):* `n_eff = 61.6 < S = 83` — the effective number of independent subjects is below the nominal count; every baseline emits point maps and has no `Σ^al` / group-`n_eff` column at all. (Caveat stated in RESULTS: under the stored normalisation this is a machinery check; a scientific `n_eff` needs the per-subject map `z` from a full training run.)

**Gap 4 — evaluation protocols are themselves unidentifiable: identification under per-subject same-map protocols is map-invariant.**
*Finding (control, real N=83):* under the protocol "fit map on run-1, apply to both runs", identification accuracy is **1.0 for fitted maps, permuted (wrong-subject) maps, and Haar-random maps alike** (identity: 0.9157; single common map: 0.9036). The correct pair shares its transform while wrong pairs are compared across mismatched transforms, so the metric is trivially perfect for any per-subject invertible map.
*Consequence:* identification numbers computed under per-subject same-map protocols (including our own earlier claim, now withdrawn) are uninterpretable; leak-free protocols (cross-fitted maps) are required.

**Task-level SOTA (real ds000243, N=83, 10k paired bootstrap):** posterior-gated alignment improves scan-rescan reliability over all comparable methods — ours **0.6975** vs noalign 0.6455 (Δ +0.052, CI [0.048, 0.056]), BrainSync 0.6461 (Δ +0.051, CI [0.047, 0.055]), FUGW 0.6215 (Δ +0.076, CI [0.070, 0.081]). Controls: Haar-random maps give zero gain (0.6454) — so it is not generic smoothing; permuted maps retain 51%; a single common template map recovers 89% — the gain is dominated by template-directed denoising. FUGW collapses reliability on this protocol; conn_SRM inflates reliability only via identity collapse (ident 0.024) and is invalid.

**Why a researcher needs this framework.** Four things become reportable that were not: (i) a per-subject posterior width and an identifiability statistic that flags alignments the data do not determine, before they enter downstream analysis; (ii) a closed-form spectral characterization of what map-and-shrink does to connectome geometry, with the optimal interior point `λ = 1/2`; (iii) a group-level variance decomposition with an effective sample size below the nominal count; (iv) an evaluation control that exposes when an identification metric is not identifiable.

**Method-level comparison.** The framework is a complete statistical object — generative model → amortized posterior over transport plans → posterior-gated alignment transform → group variance decomposition — developed and evaluated as one system, in the same sense that FUGW (deterministic fused Gromov–Wasserstein couplings with a barycenter) and BrainSync (a per-subject orthogonal transform to a template) are complete methods. The difference is architectural: those methods output a point map and stop; this one outputs a posterior over maps, an identifiability statement per subject, and a group variance component, and it beats them on the real scan–rescan reliability metric (0.6975 vs 0.6461 / 0.6215, 10k-bootstrap CIs excluding 0) while remaining honest about what the gain is (≈89% template-directed denoising; subject-specific increment +0.0056).

---

## 5. The Framework

This section specifies the model completely; no external document is required.
Throughout: subject `s = 1..S`, run `r`, data `X_sʳ ∈ ℝ^{V_s × T}`.

### 5.1 Object of inference

Rest has no shared time axis, so the sufficient statistic is the intra-subject geometry

```
C_s = corr(X_s) ∈ ℝ^{V_s × V_s},    held factored as   C_s = A_s A_sᵀ,   A_s ∈ ℝ^{V_s × r}.
```

Vertex features `Y_s ∈ ℝ^{V_s × d}` stack diffusion-map coordinates of `C_s` with anatomy
(curvature, sulcal depth, myelin proxy). The template has `K` nodes: geometry `C̄ = BBᵀ` with
`B ∈ ℝ^{K×r}`, features `F̄ ∈ ℝ^{K×d}`, mass `ν ∈ Δ_K`.

### 5.2 Generative model

Population level:

```
p(B)   = N(0, σ_B² I)
p(F̄)  = N(F̄₀, σ_F̄² I)
ε_s    ~ InvGamma(a_ε, b_ε)          # per-subject alignment sharpness, partially pooled
```

Subject level — the latent alignment is a transport plan into the template,

```
π_s ∈ Π(μ_s, ν) = { π ≥ 0 : π 1 = μ_s ,  πᵀ 1 = ν }  ⊂ ℝ^{V_s × K}
```

with an entropic Gibbs prior tilted by an anatomical cost `M_s⁰` (geodesic distance after surface
registration):

```
p(π_s | ε_s) ∝ exp( − ⟨π_s , M_s⁰⟩ / ε_s ) · 1[ π_s ∈ Π(μ_s, ν) ]
```

`M_s⁰` is the shrinkage anchor; `ε_s` controls how far a subject may depart from it and is itself
drawn from a population hyperprior, so sharpness is partially pooled across subjects. Observation
model — features transport linearly, geometry transports quadratically:

```
p(Y_s | π_s, F̄) = ∏_i N( y_si ;  μ_si⁻¹ (π_s F̄)_i ,  σ_f² I )

p(C_s | π_s, B, β) ∝ exp( − (β/2) · E_GW(π_s) )

E_GW(π) = Σ_{ijkl} ( C_s[i,j] − C̄[k,l] )² π[i,k] π[j,l]
```

Joint:

```
p(·) = p(B) p(F̄) ∏_s p(ε_s) p(π_s | ε_s) p(C_s | π_s, B, β) p(Y_s | π_s, F̄)
```

### 5.3 Gauge-fixing: velocity-augmented features

Gromov–Wasserstein is invariant to isometries of the metric spaces (Mémoli 2011), so a pairwise
coupling is defined only up to a symmetry group. We add per-vertex features

```
f_i = [ z_i ; β · v_i ]
```

where `z_i` is the intrinsic geometric descriptor of vertex `i` and `v_i` its local temporal velocity
descriptor. The feature term built from `f` is **linear in `Γ`**, so it selects a label assignment and
reduces the isometry orbit to **finitely many optima**. The velocity channel is scaled by the same
calibrated `β` used for the geometry term, so its weight is set by scan–rescan reliability rather than
tuned. The claim is **finitely many optima, never "unique minimizer"**: the objective still contains
the quadratic GW term, and Demetci et al. (2024) established the "at most finitely many" reduction for
GW fused with feature matching.

### 5.4 Where β comes from — the load-bearing design choice

`E_GW` has no tractable normalizer in `π`, so this is a **Gibbs (generalized) posterior** in the
Bissiri–Holmes–Walker sense. That is defensible only if `β` is fixed from data rather than tuned.
It is, from the 83 subjects with two resting runs:

```
σ̂_C²  =  ½ · E_s ‖ C_s^(1) − C_s^(2) ‖_F² / V_s²        # from the 83 two-run subjects
β     =  σ̂_C⁻²
```

Posterior width is therefore calibrated by the scan's own test–retest reliability rather than by a
knob: it is the width implied by how much an individual's own connectome moves between two scans on
the same day. With `T = 132` volumes each Fisher-z correlation carries a standard error near 0.088, so
a single-run connectome is a noisy object and this calibration is not a formality. **This is the
single most load-bearing design choice in the model, and it is why ds000243 is the dataset
(Section 6).**

### 5.5 Inference

One network `q_φ(π_s | C_s, Y_s)` amortizes over subjects at `O(S)` cost per epoch, not `O(S²)`.
Pairwise couplings are derived, not fitted (Section 5.6).

```
u_si = [ y_si ‖ top-m eigenvectors of L(C_s) ‖ spherical coords ]
h_si = Enc_φ(u_s)_i ∈ ℝ^p          # 4 blocks of linear attention, V up to ~10⁴ tokens
g_k  ∈ ℝ^p                          # learned template-node embeddings
S_φ[i,k] = ⟨ W h_si , g_k ⟩ / √p  −  λ M_s⁰[i,k]
```

Staying on the transport polytope — perturb the score field, then Sinkhorn-project:

```
ξ_s ~ N(0, I)
π_s^(m) = Sink_ε^L ( exp[ ( S_φ + τ_φ(s) ⊙ ξ_s ) / ε ] , μ_s , ν )
```

with `L ≈ 30` unrolled log-domain iterations. Every sample lies in `Π(μ_s, ν)` up to `L`-step
tolerance, and `π_s = f_φ(ξ_s)` is a reparametrization, so all gradients are pathwise (no REINFORCE
anywhere). `τ_φ` is the learned per-vertex noise scale carrying heteroscedastic alignment uncertainty
across the cortex; it becomes the per-subject identifiability statement of Section 4. The entropy is

```
H[q(π)] = H[q(ξ)] + E[ log |det J_f(ξ)| ]
```

on the `(V+K−1)`-dimensional polytope tangent space; `J_f` at the Sinkhorn fixed point follows from
the implicit function theorem, and `log det` is estimated with 2–4 Hutchinson probes plus stochastic
Lanczos quadrature. The GW term is evaluated without ever forming a `V×V` matrix:

```
E_GW(π) = ⟨ C_s^{∘2} μ_s 1ᵀ + 1 νᵀ (C̄^{∘2})ᵀ , π ⟩  −  2 ⟨ C_s π C̄ , π ⟩

C_s π C̄ = A_s ( A_sᵀ π B ) Bᵀ            cost  O(V_s K r + K r²)

∇_π E_GW = C_s^{∘2} μ_s 1ᵀ + 1 νᵀ (C̄^{∘2})ᵀ − 4 C_s π C̄
```

The ELBO (the variational training objective), maximized over `φ` and `θ = {B, F̄, ε_{1:S}}`:

```
L = Σ_s E_{q_φ} [ − (β/2) E_GW(π_s)
                  − (1/2σ_f²) ‖ Y_s − diag(μ_s)⁻¹ π_s F̄ ‖_F²
                  − ⟨ π_s , M_s⁰ ⟩ / ε_s ]
    + H[ q_φ(π_s) ]
    + log p(θ)
```

Training: Adam; minibatch of 8 subjects; `M = 4` posterior draws each; `ε` annealed `0.1 → 0.01`;
`β` tempered `0 → σ̂_C⁻²` over the first 30% of steps; `B, F̄` initialized from the group-average
connectome's spectral embedding then updated jointly. **Train on run 1, hold run 2 out.**

### 5.6 Pairwise couplings are derived, not fitted

```
Γ_AB = π_A · diag(ν)⁻¹ · π_Bᵀ  ∈ Π(μ_A, μ_B)
```

Cycle consistency `Γ_AB Γ_BC diag(μ_B)⁻¹ = Γ_AC` holds by construction. A structural property, not a
contribution: any method producing subject-to-template couplings can compose them this way. The
hierarchy adds joint shrinkage, not composition.

### 5.7 Group-level inference

Let `z_s ∈ ℝ^{V_s}` be any subject map (seed connectivity, graph metric, task contrast). Its
template-space image is `w_s = diag(ν)⁻¹ π_sᵀ z_s`. Drawing `π_s^(1..M) ~ q_φ`:

```
m_s        = (1/M) Σ_m w_s^(m)
Σ_s^al     = Cov_m( w_s^(m) )        or, delta method:  J_s diag(τ_φ²) J_sᵀ

w_s = θ + b_s + e_s ,     b_s ~ N(0, τ² I) ,     e_s ~ N(0, Σ_s^al)
```

Per template node `k`, with `σ²_sk = Σ_s^al[k,k]`:

```
u_sk   = 1 / ( τ̂_k² + σ²_sk )
θ̂_k    = Σ_s u_sk m_sk / Σ_s u_sk
SE(θ̂_k) = ( Σ_s u_sk )^{−1/2}
t_k    = θ̂_k / SE(θ̂_k)              # Satterthwaite df; τ̂_k² by REML or method-of-moments
```

This is random-effects meta-analysis with alignment uncertainty as the within-study variance, and it
collapses exactly to the current one-sample t-test when `Σ^al → 0`. The mechanism follows Keller et al.
(2008), who marginalized per-subject spatial jitter in the group model. Its value here: a subject whose
alignment the data do not determine is automatically down-weighted in `u_sk` rather than silently
averaged in.

### 5.8 Attribution summary

| Component | Attribution |
|---|---|
| Subject-to-template plans with a barycenter template | Inherited structure — FUGW |
| Amortized encoder over subjects | Inherited strategy — ULOT |
| Distribution over transport plans; posterior width read as an identifiability report | Inherited framing and theory — Mallasto 2021; De 2026; posterior contraction |
| Alignment variance in the group model | Inherited mechanism — Keller 2008 |
| Velocity-augmented linear gauge-fix; `β` calibrated as `σ̂_C⁻²` | Assembly choices; the orbit claim is "finitely many," not "unique" |
| **Population distribution over couplings with partial pooling across 120 subjects** | **The unclaimed object — the framework's contribution** |

---

## 6. The Dataset — ds000243 (Washington University 120)

| Property | Value |
|---|---|
| Accession | OpenNeuro **ds000243**, DOI `10.18112/openneuro.ds000243.v1.0.0` |
| Site / scanner | Washington University in St. Louis, 3T Tim Trio; Petersen/Schlaggar group |
| Release paper | Power, Plitt, Kundu, Bandettini, Martin (2017), *PLOS ONE* 12(9):e0182939; most fully described in Power et al. (2014), *NeuroImage* 84:320–341 |
| Subjects | **120 subjects, age 18.56–31.73 (mean 24.74), ALL ADULTS**, 61F/59M, all healthy right-handed native English speakers |
| Rest runs | **83 subjects have 2 rest runs**, 37 have 1 → **203 runs total**; **all two-run pairs are equal-length** |
| Acquisition | TR = 2.5 s, TE = 27 ms, 4×4×4 mm, 32 interleaved slices |
| Run lengths (all 203 NIfTI headers) | median **132 volumes = 330 s (5.5 min)**; range **130–724 volumes** |
| License / download | **CC0**; instant anonymous S3; **5.67 GiB** raw |
| Prior OT / hyperalignment work | **Zero optimal-transport work and no hyperalignment has ever been run on it** |

**Download** (either path):

```
aws s3 sync --no-sign-request s3://openneuro.org/ds000243 ./ds000243
datalad install https://github.com/OpenNeuroDatasets/ds000243.git
```

**Preprocessing required.** There are no fMRIPrep derivatives, so a minimal pipeline must be built:
motion correction; slice-timing correction (32 interleaved slices, parameters from `task-rest_bold.json`);
band-pass 0.01–0.1 Hz; registration; denoising (FIX, or white matter plus global signal regression).

### 6.1 Stated limitation

The median of ~132 time points per run is **coarse for trajectory-based methods**. BrainSync's own
paper warns, verbatim: **"Syncing of shorter time courses should probably be avoided since the error
increases rapidly below this limit."** This applies directly to a 5.5-minute run, and to our own
geometry estimate: with `T ≈ 132`, each Fisher-z correlation carries a standard error near 0.088.

**Mitigations, written in from the start.** (i) Concatenate the two equal-length runs for the 83
two-run subjects (~264 time points), with an explicit run-gap model rather than naive concatenation.
(ii) A scan-length sensitivity analysis on the long-run subset (15 × 360, 5 × 480, 6 × 724 volumes).

**What is not a limitation.** TR = 2.5 s is fine for rest BOLD: the band of interest is 0.01–0.1 Hz and
Nyquist at TR = 2.5 s is 0.2 Hz, so the band is fully covered.

### 6.2 Why this dataset and not LEMON or AOMIC

The 83 subjects with two equal-length resting runs are what make `β` calibrated (Section 5.4) and what
make held-out-run evaluation possible. That is the entire reason ds000243 is chosen over **LEMON**
(227 subjects, TR 1.4 s, preprocessed, but **no test–retest**) and over **AOMIC PIOP1+PIOP2** (442
subjects, fMRIPrep derivatives, weaker retest structure). ds000243 ships raw NIfTI, which costs
preprocessing time; that is the price paid for the retest structure. It also has the useful property
that **no optimal-transport method has ever been run on it**.

---

## 7. Evaluation — Two Tracks

The comparison set is chosen by necessity, not completeness. Five runs establish the three claims the
result requires: that alignment is *justified*, that the model is *best* against the current state of
the art, and that its key component is *necessary*.

### 7.1 The comparison set

| Purpose | Method | Why it cannot be omitted |
|---|---|---|
| **Justified** | **No alignment** (identity) | The true vanilla. Without it there is no claim that alignment does anything at all |
| **Best — rest SOTA** | **BrainSync** (Joshi et al. 2018, NeuroImage 172:740–752) | The leading rest-specific aligner |
| **Best — OT SOTA** | **FUGW** (Thual et al., NeurIPS 2022) | The leading Gromov–Wasserstein aligner; its published protocol is our secondary metric, so the comparison is on their terms |
| **Best — connectivity** | **connectivity-SRM** (PMID 32325212) | The team's own prior survey (`findings/P2.md:190`) names this the strongest existing baseline |
| **Necessary** | **Ours, ablated** — key component removed | Shows the component is doing work rather than decoration |
| | **Ours, full** | The method |

Every method runs on identical preprocessing, identical folds, and identical evaluation code. This is
what makes the table defensible; a baseline given different preprocessing invalidates the comparison.

**Methods deliberately excluded, with reasons.** BSA (Akrami 2019) is the group extension of
BrainSync and is superseded by it. GDTW (Cohen et al., AISTATS 2021) contains zero occurrences of
"fMRI" or "brain" in its full text — no fMRI experiment exists, so there is no published figure to
compare against. STA (Janati et al., AISTATS 2020) assumes a shared space and was validated on a
simulated cortex plus handwritten-letter clustering, never on rest fMRI. DTW is subsumed by the OT
family and has already been reported weaker than the methods above (Meszlényi et al. 2017; Linke et
al. 2020). SpectralOT (CCN 2026) belongs to the same research line as FUGW, which already represents
it. Procrustes is subsumed by anatomical alignment. Each exclusion carries a citable reason rather
than a convenience.

### 7.2 Pair subsampling

The full pairwise space is 7,140 pairs from 120 subjects. It is not required. Alignment-gain
evaluation samples **500 pairs** with a fixed seed. The identification test uses the fixed set of
**83 two-run subjects**, which is not a sample but the complete available set. With paired sign-flip
permutation across subjects (Winkler et al. 2014), 500 pairs gives ample power at roughly
one-fourteenth of the compute. The sampling scheme — count, draw procedure, and seed — is stated
explicitly: a declared subsample is accepted, an undeclared one is not.

### 7.3 Metrics

| Metric | Definition | Reference point |
|---|---|---|
| **Cross-run identification accuracy** | Train identity on run-1 functional connectivity, identify the same subject from run-2 | Against a **10,000-permutation null**; chance = 1/N = **0.83% at N = 120** |
| **Held-out-run alignment gain** | FUGW's own published protocol — correlation gain on a test run when alignment is fitted on a disjoint run | The protocol the competing method reports |
| **Calibration reference** | Finn et al. 2015 achieved **92.9% / 94.4%** identification at N = 126, with a permutation null maximum of **6/126** | What a working fingerprint looks like |

**Track A may fail.** A 5.5-minute median run is thin, and identification accuracy depends on
fingerprint stability, not only on alignment. A negative result would be evidence that this dataset
is too short for the metric, not that the model is wrong — which is why Track B exists.

**Controls.** Two ablations isolate whether any gain is real: **N1, a data-independent banded
coupling** — any transfer gain surviving N1 is smoothing, not alignment, a bias the CCN 2026 work
documents verbatim ("ISCs are biased by smoothing (e.g. the entropic term in the coupling
estimation)"); and **N2, a shuffled-time control** that destroys temporal structure, on which the
method should fail and the posterior should go diffuse. A posterior that stays concentrated under N2
is reading structure that is not there — which is also a result.

### 7.4 The results table

| Method | Identification acc. | vs null (p) | Per-pair uncertainty | Non-identifiable pairs flagged |
|---|---|---|---|---|
| No alignment | ~chance | — | — | — |
| BrainSync | 0.XX [CI] | … | — | — |
| FUGW | 0.XX [CI] | … | — | — |
| connectivity-SRM | 0.XX [CI] | … | — | — |
| **Ours (ablated)** | 0.XX [CI] | … | yes | yes |
| **Ours (full)** | 0.XX [CI] | … | **yes** | **yes** |

The empty columns are part of the result: they are quantities no baseline produces. The ablated row
establishes necessity; the no-alignment row establishes that alignment is justified at all.

### 7.5 Track B — the guaranteed result

Run each existing method, then report **how many subject pairs are not identifiable**: the count,
per method, of pairs whose alignment is indistinguishable from the null and therefore carries no
information. Every existing method returns an alignment for all of them, with no indication. This is
the gap of Section 1 as a statistic; it requires no new method to compute and is available at hour 15
regardless of how Track A resolves.

### 7.6 What we will not do

**No behavioural or cognitive prediction as a metric.** Marek et al. 2022 report a median
brain–behaviour correlation of **|r| = 0.01 at N = 3,928**; brain–behaviour prediction is
indefensible below roughly N ≈ 1,000, and ds000243 has 120 subjects. Any such number here would be
noise presented as a result.

---

## 8. Feasibility — 15 Hours, 3 People

### 8.1 Scope of the minimum viable version

Reference point for the minimum viable version. fsaverage5 downsampled via a random fine
parcellation to **V ≈ 2,000 vertices per hemisphere**; **K = 512** template nodes; **r = 32**; **d = 32**
diffusion-map coordinates plus curvature and sulcal depth; **all 120 subjects**. The encoder is ≈ 1.2M
parameters; the GW term per subject is `V K r = 3.3 × 10⁷` multiply-adds. Batch 8 subjects, `M = 4`
draws, `L = 30` Sinkhorn steps, ≈ 3 s per step on one modern GPU, 15 steps per epoch, 300 epochs ⇒
**under 2 GPU-hours**. The 83 two-run subjects supply `σ̂C²` and the held-out evaluation.

### 8.2 Time allocation (3 people × 5 hours)

The 3 hours previously allocated to a `group_test` module move to β calibration, the baseline harness, and
Track B, since group-level propagation is inherited mechanism (Section 5.7).

| Hours | Work |
|---|---|
| 3 | S3 pull, preprocessing (motion, slice-timing, band-pass, registration, denoising), parcellation, `C_s` and `Y_s` |
| 1 | Template spectral initialization; `σ̂C²` and `β` from the 83 two-run subjects |
| 4 | Model, ELBO, Sinkhorn-parametrized posterior, training loop; train on run 1 |
| 2 | Baseline harness: the four baselines (no alignment, BrainSync, FUGW, connectivity-SRM) plus the ablated model, on identical folds |
| 1 | Track A metrics with the 10,000-permutation null; **Track B non-identifiability count** |
| 4 | Split-half checks, ablations (N1, N2), scan-length sensitivity, write-up |

### 8.3 Hard parts, stated plainly

- **The pushforward entropy term** (`H[q(π)] = H[q(ξ)] + E[log |det J_f(ξ)|]`) is the main technical
  debt; dropping it **understates** uncertainty, and we will say so.
- **The Gibbs normalizer depends on the template.** `E_GW` has no tractable normalizer in `π`, so `B`
  must be scale-constrained for the generalized posterior to be well posed.
- **Encoder mode collapse.** An encoder can collapse onto one mode of a multimodal posterior, making the
  posterior *look* concentrated when it should not. Noise annealing and a small ensemble are the
  mitigations; this is an open risk.
- **No ground-truth correspondence exists at rest.** Calibration must use simulated diffeomorphic
  deformations and the two-run split, not a known answer, which bounds what any calibration claim means.
- **MCMC is infeasible in this budget** for a `V×K` variable per subject. **Variational inference is
  the path**, and its consequence — a posterior that may underestimate width — is a stated
  limitation.

### 8.4 What we will be able to say at hour 15

1. Whether a calibrated posterior over rest-fMRI alignments is computable at N = 120 within a
   2-GPU-hour budget; the `β` estimate from the 83 two-run subjects; and the per-subject posterior widths.
2. The **Track B number** — how many subject pairs the existing methods cannot identify — and Track A's
   identification accuracy and held-out alignment gain for the four baselines and the model against a
   permutation null, or a statement that the dataset is too short for the metric.

---

## 9. Repository Architecture — Modular, Multi-Developer

This project is built for several developers working in parallel and for repeated experimental iteration,
not for a single one-shot run. The layout below is chosen so that three people can work simultaneously
without colliding, and so that the model can be reformulated and re-run many times without rewriting the
surrounding machinery.

**The organizing rule is fixed interfaces.** Each module has one stated responsibility and a declared
input and output type; nothing crosses a module boundary except through that interface. A developer can
therefore replace the inference module, or the evaluation module, or the geometry module, without
touching the others, and without coordinating with whoever is editing them. The data contract is the
same discipline applied to data: the dataset loader emits a single documented `npz` structure, and every
downstream module consumes only that structure.

**Each developer supplies their own data path.** Data is not shared through the repository. A local
configuration file holds the developer's own data root; it is listed in `.gitignore` and is never
committed, so the repository stays portable and no one's filesystem layout becomes a dependency.

```
trajot/
├── configs/
│   ├── paths.example.yaml      # template — each developer copies to paths.yaml (git-ignored)
│   ├── model/
│   │   ├── default.yaml
│   │   └── ablation_no_gauge.yaml
│   ├── experiments/
│   │   ├── 00_noalign.yaml
│   │   ├── 01_brainsync.yaml
│   │   ├── 02_fugw.yaml
│   │   ├── 03_conn_srm.yaml
│   │   ├── 10_ours_full.yaml
│   │   └── 11_ours_ablated.yaml
│   └── eval/
│       └── default.yaml
├── src/trajot/
│   ├── config.py               # load, validate, hash
│   ├── io/                     # dataset loading; the npz data contract
│   ├── geometry/               # connectivity, diffusion maps, distance matrices
│   ├── model/                  # generative model, ELBO terms, priors
│   ├── inference/              # Sinkhorn projection, encoder, training loop
│   ├── baselines/              # noalign, brainsync, fugw, conn_srm
│   ├── eval/                   # identification, alignment gain, permutation, non-identifiability
│   └── runlog/                 # run manifest, registry, structured logging
├── scripts/
│   ├── preprocess.py
│   ├── fit.py
│   ├── evaluate.py
│   ├── compare.py
│   └── run_experiment.py       # single entry point
├── data/                       # git-ignored — each developer's own path
├── runs/                       # git-ignored — outputs, one directory per run
└── tests/
```

### 9.1 Module boundaries

| Package | Single responsibility | Interface (in → out) |
|---|---|---|
| `io/` | Read the dataset and expose it in one documented form | dataset root + subject list → `npz` with `C_s`, `Y_s`, `μ_s`, run index, subject index |
| `geometry/` | Turn a connectome into the geometric objects the model consumes | `C_s` → diffusion-map coordinates, geodesic cost `M_s⁰`, distance matrices |
| `model/` | Define the generative model and its terms | template parameters, subject sufficient statistics → ELBO terms (GW, feature, prior, entropy) |
| `inference/` | Produce posterior samples of `π_s` and train the parameters | ELBO terms + encoder inputs → `π_s` samples on `Π(μ_s, ν)`, gradients, learned `θ = {B, F̄, ε}` |
| `baselines/` | Implement each comparison method behind one common signature | subject data + fold assignment → per-subject transforms and aligned features |
| `eval/` | Compute the metrics and the null distributions | aligned features, folds, seed → identification accuracy, permutation p-values, alignment gain, non-identifiability counts |
| `runlog/` | Record every run and make runs comparable | resolved config + metrics + environment → `manifest.json`, `metrics.json`, `index.csv` row |

Because these interfaces are fixed, a developer can work on any one module without touching the others:
a change to `eval/` cannot invalidate a run of `inference/`, and a change to `geometry/` is visible to
`model/` only through the documented object it returns. `config.py` sits beside them as the one module
every package imports: it loads, validates, and hashes a configuration, so that a configuration is
identified by content rather than by filename.

### 9.2 How a developer runs an experiment

1. Copy `configs/paths.example.yaml` to `configs/paths.yaml` and set your data root. The file is
   git-ignored; no one else's run depends on it.
2. Pick an experiment config from `configs/experiments/` — for example `10_ours_full.yaml`.
3. Run it:

```
python scripts/run_experiment.py --config configs/experiments/10_ours_full.yaml
```

The experiment configs are the unit of collaboration. Anyone can add one, and all of them are comparable
because they run through the same entry point: the same preprocessing outputs, the same folds, the same
evaluation code, and the same logging. Adding a baseline or an ablation is adding a config file, not
editing a pipeline.

---

## 10. Logging, Versioning, and Reproducibility

This project will be run many times, by different people, with different configurations, and the results
must remain comparable across all of those runs. Every run therefore produces a complete,
self-describing record: nothing about a run depends on remembering what was typed or on which machine it
ran.

### 10.1 What every run writes

Each run creates one directory `runs/<run_id>/` containing:

| File | Contents |
|---|---|
| `manifest.json` | experiment name; full resolved config; config hash; git commit; data hash; random seed; Python and package versions; operator name; UTC start and end timestamps |
| `metrics.json` | all evaluation outputs — identification accuracy, permutation p-values, alignment gain, per-pair identifiability counts |
| `log.txt` | complete stdout/stderr, including loss traces per epoch |
| `artifacts/` | posterior samples, learned couplings, figures |

### 10.2 Run identifiers

```
run_id = <experiment>__<config_hash[:8]>__<UTC timestamp>
```

A run is identifiable from its name alone, and a configuration change produces a visibly different
identifier — so two runs that differ in any resolved parameter cannot be mistaken for the same run.

### 10.3 The registry

`runs/index.csv` accumulates one row per run with its `run_id`, experiment, config hash, git commit, key
metrics, and operator. `scripts/compare.py` reads the registry and emits a comparison table across any
selected runs — which is how results from different developers are compared without anyone re-running
anything.

### 10.4 Reproducibility contract

Any run is re-runnable from its `manifest.json` alone: the config is fully resolved and stored, the data
hash pins the input, and the seed is recorded. A run whose manifest cannot reproduce its `metrics.json`
is a bug, and is logged as one.

### 10.5 Why this matters for the science

The model will not work on the first attempt. Iteration means changing the formulation and re-running,
and that is only tractable if every prior run is still inspectable and comparable. The logging layer is
what makes the experimental loop of Section 8 possible rather than aspirational. It also means a flaw
found later can be traced to the exact configuration that produced it.

---

## 11. Team Coordination and Merge Protocol

The team is three developers on MacBooks. Each has the full ds000243 raw dataset (5.67 GiB) already
downloaded locally. **Raw data stays local — it is never pushed to git.** What is shared via git is the
compact output layer: ~8 MB of npz files (~40 KB per subject-run, ~8 MB for all 203 runs), manifest
rows, and run registry rows.

### 11.1 Phase-gate model

Work moves through three phases:

| Phase | What happens |
|---|---|
| **Phase 1 — Development** | Build the framework. Test everything on synthetic data. Preprocess the dataset in shards. |
| **Phase 1→2 — Merge** | Collect all preprocessing outputs into `results/`, verify counts, run tests on the merged state, push to main. |
| **Phase 2 — Experiments** | Run baselines and the full/ablated model on real data. Produce identification accuracy, permutation nulls, alignment gain, and the results table. |

A phase is not **done** until the merge is complete and verified. Phase gates are recorded as
checkboxes in `TRACKING.md`. The Phase 1→2 merge gate requires: all npz files in `results/npz/`;
`results/manifest.parquet` with 203 unique subject-runs; `results/runs_index.csv` with synthetic
test-run entries; `pytest -q` green on the merged state; and a push to main.

### 11.2 Shard model

Preprocessing (W1) is the shardable work. Each developer preprocesses a subset of subjects, specified
in `configs/shards.yaml` under `dev_a`, `dev_b`, or `dev_c`. Subject IDs are populated from the local
data directory after listing subjects.

Model training is **not** shardable: the hierarchical model needs all 120 subjects together.
Permutation nulls (W3) may also be sharded later if time allows.

Each developer writes only to their own files:

| Artifact | Path | Rule |
|---|---|---|
| npz outputs | `results/npz/sub-<id>_run-<r>.npz` | Only subjects in your shard |
| Manifest rows | `results/manifests/<your-name>.parquet` | Only your rows |
| Run registry rows | `results/runs/<your-name>.csv` | Only your rows |

No two developers write the same file. That is the entire conflict-prevention design.

### 11.3 Merge protocol

When your shard is complete:

1. Copy npz files to `results/npz/`.
2. Write manifest rows to `results/manifests/<your-name>.parquet`.
3. Append run rows to `results/runs/<your-name>.csv`.
4. Update `TRACKING.md` — tick your checkbox, update your status.
5. Commit and push to branch `shard/<your-name>`.
6. On the call, say "my shard is pushed."
7. The coordinator merges the branch to main.
8. The coordinator runs `scripts/merge_manifests.py` and `scripts/merge_runs.py`.
9. Everyone pulls main before the next phase.

Merged outputs (`results/manifest.parquet`, `results/runs_index.csv`) are **generated by scripts,
never hand-edited**. The merge scripts deduplicate by `subject_id + run_id` (manifests) and by
`run_id` (runs), and the manifest merge warns if the two-run subject count is not 83.

### 11.4 What is tracked

`TRACKING.md` is the coordination board. Read it before starting work; update it when you finish.
It records:

- the **current phase** (one line);
- **shard assignments** (which subjects each developer owns, and their status);
- **phase gates** (checkbox lists for Phase 1, the Phase 1→2 merge, and Phase 2);
- a **run log** (date, developer, what, status, notes).

Shard assignments also live in `configs/shards.yaml` as the machine-readable source of truth.
`TRACKING.md` is the human-readable board.

### 11.5 Conflict prevention rules

- **Per-developer files prevent merge conflicts.** Manifests and run registries are split by developer
  and merged by script. Never write another developer's file.
- **Merged outputs are generated, never hand-edited.** Do not edit `results/manifest.parquet` or
  `results/runs_index.csv` by hand; re-run the merge scripts instead.
- **npz files are owned by subject.** You only write npz files for subjects in your shard, so two
  developers never write the same path.
- **`TRACKING.md` is coordinated on calls.** It is a small shared file. If two people edit it
  simultaneously, merge the edits manually — the content is short enough that this is not a problem.
- **Raw data and run directories stay local.** `data/` and `runs/` are git-ignored. Only the compact
  `results/` layer is shared.

The full operational checklist is in `TRACKING.md`.

---

## Appendix — Claim Discipline

- Never write "unique minimizer." The claim is **finitely many optima**.
- Attribute components explicitly (Section 3); the contribution is the assembled object, not the parts.
- The hierarchy is claimed as an **object**: "the hierarchy as a modeling idea is textbook; as an object
  it is unclaimed." The absences here are search-based, phrased as **"no such work was found."**
- OTTER (doi:10.64898/2026.08.24.746652) is cited to distinguish it: no posterior, no hierarchy, no
  uncertainty propagation. Behaviour prediction appears nowhere as a metric (Marek et al. 2022).
