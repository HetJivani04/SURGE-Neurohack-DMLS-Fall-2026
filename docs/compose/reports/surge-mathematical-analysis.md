# SURGE Mathematical Root-Cause Analysis: Why TrajOT Loses Identification Accuracy

**Protocol**: N=49 frozen pilot, data_hash `ccce8212`, B=200, Schaefer-100, ds000243  
**Code base**: `/tmp/surge-coord/trajot` at analysis time  
**Status**: analysis only — no code changes  
**Directive**: if baselines win, analyze mathematically what to fix in our method (do not cheat baselines)

---

## 0. Definitive Results (frozen pilot)

| Method | Ident | Gain | NonIdent | feat_corr | tau_phi / per_pair_unc |
|--------|-------|------|----------|-----------|------------------------|
| noalign | 0.959 | 0.000 | 500 | 1.000 | — |
| brainsync | 0.959 | −0.00005 | 469 | ~0.998 | — |
| fugw | 0.918 | −0.005 | 466 | ~0.966 | — |
| conn_srm | 0.082 | +0.395 | 467 | ~0.586 | — |
| **ours_full** | **0.857** | **+0.018** | **472** | **0.658** | **0.075** |

ours_full: K=100, β=29.189 (scan-rescan calibrated), τ_φ mean=0.075, transforms_applied=true, max|aligned−raw|=2.58  
Source run: `runs/10_ours_full__dd67ab1e__20260920T074747Z`

---

## 1. What the code actually does (not what the task prompt assumed)

The task prompt described the transform as pure barycentric projection. **The current `ours.py` has already moved past that.** Reading `src/trajot/baselines/ours.py` carefully:

### 1.1 Actual transform pipeline (definitive path)

```
fit():
  load/train hierarchical artifacts (B, F_bar, nu, eps, tau_phi, beta, posterior π means)
  C_bar = B B^T                         # from learned loadings (NOT used as coupling target)
  C_pop = mean_s(C_s)                   # region-level population connectome (R×R)
  for each subject s:
    π_s = EMD(C_s, C_pop)               # ot.emd soft coupling to C_pop, not to C_bar
    Q_s = Procrustes(C_s, P C_pop P^T)  # orthogonal map from barycentric reference
    store (π_s, Q_s)

transform(C):
  Q = Q_s (reuse train-run Q by subject index for held-out run-2)
  return Q^T C Q                        # orthogonal conjugation
```

Key functions:

- `_soft_coupling(C, C_pop)`: `ot.emd(1/R, 1/R, _cdist_rows(C, C_pop))`
- `_orthogonal_from_coupling(C, C_pop, π)`:  
  `P = π/row_sums`, `C_ref = P C_pop P^T`, `Q = UV^T` from `SVD(C C_ref^T)`
- `transform`: `symmetrize(Q^T C Q)`

So the applied map is **orthogonal conjugation toward a population-mean template**, not mass-weighted barycentric mixing. The docstring of `_orthogonal_from_coupling` already states the motivation: preserve eigenvalues rather than collapse the between-subject spectrum.

### 1.2 Identification metric (`eval/identification.py`)

```
x_s^{(r)} = vec_u(C_s^{(r)})           # upper triangle, k=1
scores[i,j] = Pearson(x_i^{(1)}, x_j^{(2)})
ident = mean_i [ argmax_j scores[i,j] == i ]
```

Identification depends on **vectorized edge-pattern differences**, not on eigenvalues.

### 1.3 Alignment gain (`eval/alignment_gain.py`)

```
gain_ab = corr(aligned_a^{r1}, aligned_b^{r2}) − corr(raw_a^{r1}, raw_b^{r2})
```

Mean over 500 declared ordered pairs (seed 2026). Cross-subject pairs dominate the pair pool (N=49 → 49·48 = 2352 ordered off-diagonal pairs available; 500 drawn).

### 1.4 Track B non-identifiability (`eval/identifiability.py` + `scripts/evaluate.py`)

```
gain-null: permute run-2 subject labels, recompute per-pair gains, B times
threshold = quantile(null, 0.95)
nonidentifiable = count(gain_k ≤ threshold)
```

This count is filled for **all methods**, including baselines. Model-only extras:

- `per_pair_uncertainty = mean(τ_φ)` (ours only)
- `per_pair_flags`: gain-null flags **OR** subject-level mean τ_φ > q75 (ours only)

---

## 2. Root cause of the identification deficit (0.857 vs 0.959)

### 2.1 The ceiling is real

Raw Schaefer-100 connectomes already achieve **ident = 0.959 at N=49** (47/49 correct). Chance is 1/49 ≈ 0.020. There is almost no headroom for any alignment method to *improve* identification on this protocol. Any nontrivial transform that perturbs edge patterns risks falling below the raw ceiling.

Finn et al. 2015 reference: 92.9–94.4% at N=126 on higher-resolution features. We are at 95.9% at N=49 on 100 parcels — raw connectivity is already an excellent fingerprint at this cohort size.

### 2.2 Why orthogonal Procrustes still hurts identification

Orthogonal conjugation `C̃ = Q^T C Q` **preserves the spectrum** of C (eigenvalues identical). Identification does **not** use the spectrum. It uses Pearson correlation of upper-triangle edge vectors:

```
ρ_self  = corr(vec_u(Q_s^T C_s^{(1)} Q_s), vec_u(Q_s^T C_s^{(2)} Q_s))
ρ_cross = corr(vec_u(Q_s^T C_s^{(1)} Q_s), vec_u(Q_t^T C_t^{(2)} Q_t))
ident ≈ P( ρ_self > max_{t≠s} ρ_cross )
```

Write each connectome as `C_s = C̄ + Δ_s`, where `Δ_s` carries the individual fingerprint. The Procrustes map is estimated **per subject** from that subject's own C_s against a **shared** target `C_ref ≈ f(C_pop)`:

```
Q_s = argmin_Q ‖ Q C_s − C_ref ‖_F
```

This is an *empirical Bayes / shrinkage-style* map in disguise: Q_s rotates subject s's edge pattern so that it best matches a template-aligned version of itself. Consequences:

1. **Between-subject variance in the transformed edge space shrinks.** Subjects that start far from C_pop receive larger rotations toward it. After transform, the set `{Q_s^T C_s Q_s}` is more mutually similar than `{C_s}`.
2. **feat_corr = 0.658** quantifies the distortion: aligned features retain only ~66% linear correlation with raw features. Compare brainsync (~0.998) and fugw (~0.966), which preserve fingerprint structure.
3. **Cross-subject correlation rises slightly** (gain = +0.018 > 0), which is exactly the between-subject compression showing up as increased group similarity.

### 2.3 Quantifying the ident drop

| Method | Ident A | Approx. correct / 49 | Φ⁻¹(A) (discriminability proxy) |
|--------|---------|----------------------|----------------------------------|
| noalign | 0.959 | 47 | +1.74 |
| brainsync | 0.959 | 47 | +1.74 |
| fugw | 0.918 | 45 | +1.39 |
| ours_full | 0.857 | 42 | +1.07 |
| conn_srm | 0.082 | 4 | −1.39 (≈ chance) |

ours_full loses **5 subjects' fingerprints** relative to noalign. Discriminability (Φ⁻¹ of accuracy, a rough d′ proxy against chance) falls from ~1.74 to ~1.07 — a **~38% reduction in self-vs-cross separation**.

Under a two-Gaussian model of identification:

```
scores_self  ~ N(μ_self, σ²)
scores_cross ~ N(μ_cross, σ²)
margin M = μ_self − μ_cross
A = Φ( M / (σ√2) )   (approximately, for large N gallery competition is harder)
```

Alignment that raises μ_cross (via template pull) without raising μ_self by at least as much **strictly reduces M**. Our transform does exactly this: +cross-subject similarity at the cost of self-match distinctiveness.

### 2.4 Held-out protocol subtlety

`heldout_couplings=True`: run-2 reuses run-1's Q_s by subject index. This is protocol-correct (fit on run-1, test on run-2) and avoids leakage. But:

- If Q_s captures true subject geometry, applying it to run-2 should help ρ_self.
- If Q_s overfits run-1-specific noise / template proximity, run-2 does not benefit the same way, while cross-subject compression still applies.
- Empirically ident drops, so the second effect dominates: Q_s is more "how close is this subject to C_pop" than "what is this subject's stable fingerprint basis."

### 2.5 Root cause (one sentence)

**ours_full loses identification because a subject-specific orthogonal map estimated to pull each connectome toward the population mean compresses between-subject edge-pattern variance, and identification is a function of that variance — while raw connectomes are already at a 0.959 ceiling, so any nontrivial template-directed transform can only lose Track A.**

This is not a bug in Procrustes. Orthogonal conjugation preserves eigenvalues, not fingerprints. The deficit is structural.

---

## 3. Why conn_srm gets +0.395 gain but ident collapses to 0.08

### 3.1 SRM mathematics

Connectivity-SRM (`baselines/conn_srm.py`) assumes:

```
C_s ≈ A_s S
```

with shared response `S` (k×R) and subject loadings `A_s` (R×k). Alternating least squares estimates S and {A_s}. Transform finds region-space orthogonal Q aligning A_s to A̅_bar via Procrustes on `A_s A̅^T`, then applies `C̃_s = Q^T C_s Q`.

### 3.2 Why gain rises

SRM maximizes shared variance across subjects. After alignment, subjects' connectomes live in a common low-rank shared subspace. Cross-subject Pearson correlation of vectorized connectomes rises dramatically → **gain = +0.395**.

feat_corr ≈ 0.586 confirms severe distortion away from raw individual patterns.

### 3.3 Why ident collapses

Identification needs **residual individual differences** after removing group mean. SRM's generative story is:

```
C_s = C̄_shared + ε_s
```

and the transform projects onto the shared component. The residual ε_s — the fingerprint — is treated as noise to be explained away by A_s, not as signal to preserve in the evaluation space.

When subjects are forced toward a common response:

```
C̃_s → C̄  for all s
μ_cross → μ_self
ident → 1/N ≈ 0.02
```

Observed 0.082 is slightly above chance (4/49), consistent with residual loading differences leaking through imperfect Procrustes.

### 3.4 Bias–variance tradeoff

| Quantity | What alignment optimizes | Effect of strong alignment |
|----------|--------------------------|----------------------------|
| Cross-subject corr (bias toward group) | ↑ (this is the gain metric) | rises |
| Between-subject variance (fingerprint) | ↓ (destroyed as "noise") | collapses |
| Identification accuracy | needs the variance component | falls to chance |

**conn_srm is the extreme point on this tradeoff. ours_full is a milder point on the same curve**: gain +0.018, ident 0.857, feat_corr 0.658. fugw/brainsync/noalign sit near the other extreme (near-zero gain, near-ceiling ident).

The curve is not a free lunch. Under this protocol (ceiling raw ident, no ground-truth correspondence), **methods that buy cross-subject correlation pay for it in fingerprint variance**.

---

## 4. Track A vs Track B: mathematical tension

### 4.1 The two objectives

**Track A (identification)** requires maximizing the margin:

```
M = ρ_self − ρ_cross
    = corr(aligned_s^1, aligned_s^2) − corr(aligned_s^1, aligned_t^2)
```

Identification is a **discriminative** metric: it needs *between-subject differences*.

**Track B (alignment gain)** measures:

```
G = E_{(a,b)}[ corr(aligned_a^1, aligned_b^2) − corr(raw_a^1, raw_b^2) ]
```

Gain is a **commonality** metric: it needs *within-template / cross-subject similarity* to rise.

### 4.2 Is the opposition a mathematical identity?

**Not in full generality.** Counterexample: if the only source of low cross-subject correlation is a shared misregistration `R_s` (same biological fingerprint, rotated coordinates), a *true* alignment `R_s^{-1}` would raise both ρ_cross **and** ρ_self (by stabilizing run-1/run-2 of the same subject in a common frame), and identification could stay flat or improve.

The opposition **is** an empirical regularity under the conditions that actually hold here:

1. **No ground-truth correspondence at rest.** There is no known R_s to recover. Methods estimate maps from connectivity geometry alone.
2. **Raw ident is at ceiling (0.959).** There is no misregistration noise large enough to suppress fingerprints; any additional template-directed distortion can only hurt.
3. **Alignment methods increase cross-subject corr by projecting toward a template / shared subspace**, not by undoing a known bijective misregistration. Template projection shrinks Δ_s.
4. **Evaluation uses the same vectorized connectome features** for both metrics. A transform that makes subjects look alike in that space necessarily raises cross-corr and lowers self-vs-cross margin.

Under (1)–(4), the following holds approximately:

```
If ∂G/∂λ > 0 (alignment strength λ increases gain),
and alignment acts as shrinkage C̃_s(λ) = C̄ + α(λ) Δ_s + … with α'(λ) < 0,
then Var_s[C̃_s] ∝ α(λ)² decreases,
so M(λ) decreases and ident(λ) falls.
```

conn_srm is α ≈ 0. ours_full is intermediate α. noalign is α = 1.

**Honest phrasing for the paper**: *Under rest-fMRI evaluation with ceiling raw identification and no ground-truth correspondence, alignment methods that raise cross-subject correlation by shrinking subjects toward a template necessarily reduce fingerprint discriminability. This is not a universal theorem about all possible alignments; it is a property of template-directed shrinkage on this protocol.*

### 4.3 The quantification table (internal)

| Method | α-proxy (feat_corr) | Gain | Ident | Interpretation |
|--------|---------------------|------|-------|----------------|
| noalign | 1.00 | 0.000 | 0.959 | no shrinkage, ceiling ident |
| brainsync | ~1.00 | ~0.000 | 0.959 | orthogonal time-sync; connectomes nearly unchanged |
| fugw | ~0.97 | −0.005 | 0.918 | mild OT; slight ident loss, no gain |
| ours_full | 0.66 | +0.018 | 0.857 | moderate template pull; small gain, real ident cost |
| conn_srm | 0.59 | +0.395 | 0.082 | heavy shared-subspace projection; ident destroyed |

The frontier is visible: as feat_corr falls, gain can rise, and ident falls. ours_full is not "wrong"; it is sitting on a tradeoff curve that **cannot** win Track A against noalign at ceiling.

---

## 5. What PLAN.md actually claims ours should win on

### 5.1 Section 7.5 — Track B guaranteed result (verbatim essence)

> Run each existing method, then report **how many subject pairs are not identifiable**: the count, per method, of pairs whose alignment is indistinguishable from the null and therefore carries no information. Every existing method returns an alignment for all of them, with no indication. This is the gap of Section 1 as a statistic; it requires no new method to compute and is available at hour 15 regardless of how Track A resolves.

### 5.2 Section 7.4 — results table structure

| Method | Ident acc. | vs null (p) | Per-pair uncertainty | Non-identifiable pairs flagged |
|--------|------------|-------------|----------------------|--------------------------------|
| baselines | … | … | **—** | **—** |
| Ours (full) | … | … | **yes** | **yes** |

PLAN.md states explicitly:

> The empty columns are part of the result: they are quantities no baseline produces.

### 5.3 Section 7.3 — Track A may fail

> A 5.5-minute median run is thin, and identification accuracy depends on fingerprint stability, not only on alignment. A negative result would be evidence that this dataset is too short for the metric, not that the model is wrong — which is why Track B exists.

Empirically Track A did **not** fail for raw connectivity (0.959). It failed for *ours* because our transform hurts fingerprints. PLAN.md never claimed ours wins identification.

### 5.4 Section 4 — guaranteed demonstration

> Regardless of whether the model wins on accuracy, one result is guaranteed (Section 7.5): run every existing method on the same data and report how many subject pairs are not identifiable.

### 5.5 What the claim is NOT

The claim is **NOT**:
- "ours beats baselines on identification accuracy"
- "ours beats noalign on Track A"

The claim **IS**:
1. Ours produces **per-pair uncertainty** (τ_φ posterior width) that baselines cannot
2. Ours reports **which pairs/subjects are non-identifiable** via posterior width
3. Track B: the **count** of non-identifiable pairs is the gap statistic
4. The hierarchical object (population of couplings with shrinkage + calibrated β + posterior width as identifiability) is the contribution

### 5.6 Current evaluation framing problem

The frozen table reports NonIdent for **all** methods via the gain-null test:

| Method | NonIdent (gain-null, /500) |
|--------|----------------------------|
| noalign | 500 |
| brainsync | 469 |
| fugw | 466 |
| conn_srm | 467 |
| ours_full | 472 |

Problems with presenting this as the Track B win:

1. **noalign scores 500/500** — under gain-null, identity alignment has zero gain by definition, so *every* pair is "non-identifiable." The statistic is nearly tautological for noalign.
2. **Counts do not differentiate methods** (466–500). ours is not clearly "better at flagging."
3. **PLAN.md intended the flagged column to be model-only** (posterior-width based). Current code fills `nonidentifiable_pairs` for baselines too via gain-null, then adds τ_φ flags only to `per_pair_flags` for ours. The published table collapses these.
4. **`per_pair_uncertainty = 0.075` is a single mean τ_φ**, not a per-pair vector in the metrics schema. It does not yet show *which* pairs.
5. **RESULTS.md §5** (first-pass) warned that dropping the entropy term makes τ_φ uncalibrated. Current config has `model.entropy.weight = 1.0` and `scripts/fit.py` notes "wired entropy term," so this may be partially fixed — but calibration of posterior width as an identifiability statement is **not yet demonstrated** (no recovery experiment on known non-identifiable subjects, no correlation of τ_φ with gain-null flags, no N2 shuffled-time diffuse-posterior check reported in the frozen table).

**Conclusion on framing**: PLAN.md's defensible claim is real, but the current frozen table does not yet deliver it cleanly. The unique column (posterior uncertainty / posterior-width flags) is present but weakly specified and not validated as calibrated.

---

## 6. What would need to change for ours to beat noalign on ident

### Option 1: Shared (not subject-specific) orthogonal gauge

Use one Q̄ for all subjects (or Q_s ≈ Q̄ + small residual), estimated from the template only.

- Preserves between-subject differences up to a global isometry of the edge space.
- Ident would be **preserved** (≈ noalign), not improved.
- **Cannot beat noalign** when raw is already optimal for fingerprinting.
- Gain would also be near zero (nothing is aligned differentially).
- Verdict: necessary for "do no harm" on Track A; insufficient to win Track A.

### Option 2: Optimize within-subject reliability, not cross-subject gain

Change the transform objective from template fit to scan-rescan stability:

```
Q_s* = argmax_Q  corr( vec_u(Q^T C_s^{(1)} Q), vec_u(Q^T C_s^{(2)} Q) )
```

or a population version with hierarchical shrinkage on Q_s that maximizes expected run-1/run-2 self-match.

- This is a **reliability** objective, not an alignment-gain objective.
- Could raise ρ_self without raising ρ_cross as much → ident ↑.
- Would likely **not** produce positive cross-subject gain (may even be slightly negative).
- Aligns with β's philosophy (calibrate from scan-rescan) but currently β only sets inverse temperature; it does not enter the transform target.
- Verdict: the only mathematically coherent path to improving Track A *within the method*. Unlikely to beat a 0.959 ceiling by much; might close the gap to noalign (0.857 → ~0.95) if Q_s captures stable individual geometry rather than template proximity.

### Option 3: Harder protocol where raw ident is NOT at ceiling

Create headroom:

| Harder protocol | Why raw ident falls | Risk |
|-----------------|---------------------|------|
| More subjects (N=83 two-run set as planned) | larger gallery, harder top-1 | still may stay high |
| Fewer parcels (e.g. R=20) | fewer edges, less fingerprint signal | noisy |
| Shorter runs / split-half within run | more measurement noise | PLAN already flags short runs |
| Cross-session (if available) | session effects | ds000243 is same-day |
| Additive noise / motion-matched subsets | degrades features | artificial |
| Non-identical parcellations / resolution mismatch | raw features misaligned | closer to the real problem |

Only if raw ident drops into the 0.4–0.7 range is there room for alignment to help. At 0.959 there is none.

- Verdict: scientifically honest for "when does alignment help fingerprinting?" but does not rescue the current frozen comparison. Do **not** claim a Track A win on a protocol chosen after seeing the ceiling.

### Option 4: Reframe around Track B / uncertainty reporting (PLAN-faithful)

Make the unique column real and validated:

1. Report **per-subject posterior width** τ_φ(s), not only the global mean.
2. Define non-identifiability **from the posterior** (e.g. τ_φ(s) > calibrated threshold, or posterior mass of couplings that disagree on a region), not only from gain-null.
3. Show baselines produce **no such column** (PLAN §7.4 empty cells).
4. Validate calibration:
   - N2 shuffled-time control: posterior should go **diffuse** (RESULTS.md already states this requirement).
   - Synthetic planted-permutation: posterior width should be high when correspondence is ambiguous.
   - Correlation between τ_φ flags and gain-null flags (they need not match, but should be related for identifiable subjects).
5. Stop presenting NonIdent gain-null counts as if they differentiate methods when all methods score 466–500.
6. Optionally compute a **baseline-forced** uncertainty proxy (e.g. bootstrap instability of FUGW plans) to show ours is better calibrated, not merely that ours has a number and baselines have none.

- Verdict: **this is the PLAN.md claim.** It requires evaluation-framing changes and calibration evidence, not a transform hack.

### Option 5 (implementation honesty): actually apply the hierarchical posterior

Current transform ignores π_s from the hierarchical model and re-estimates EMD to C_pop. If PLAN §5 is the contribution, transform should use:

```
Γ_s = π_s  (posterior mean or samples)
aligned features via template:  w_s = diag(ν)^{-1} π_s^T z_s
or connectome in template space via derived pairwise Γ
```

and group metrics via §5.7 random-effects with Σ_s^al. Whether this improves ident is unknown — posterior-mean couplings may be even more shrinkage-like (worse for ident) — but then the **method under test would be the method in the plan**, and Track B uncertainty would come from the same object that produces the transform.

---

## 7. Is current ours_full faithful to PLAN.md §5?

### 7.1 Component-by-component audit

| PLAN §5 component | Present in current ours.py? | Used in transform? | Used in evaluation? |
|-------------------|----------------------------|--------------------|---------------------|
| Hierarchical population of couplings with shrinkage | Partially (train loads B, ε_s, π means) | **No** — transform uses EMD(C_s, C_pop) | No |
| Template C̄ = B B^T from learned loadings | Computed (`_template_connectome`) | **No** — C_pop = mean(C) is the coupling target | No |
| Posterior over π_s (Sinkhorn variational) | Loaded from artifacts when present | **No** | No |
| Gauge-fixing velocity features | Trained when `gauge_features=True` | **No** — ablation shows identical transform metrics | Only via τ_φ difference |
| β = σ̂_C⁻² from scan-rescan | **Yes** (29.189) | Indirect (shaped training) | Reported in metrics |
| Pairwise couplings Γ_AB = π_A diag(ν)⁻¹ π_Bᵀ | Not implemented in baselines/ours.py | **No** | No |
| τ_φ as identifiability statement | Loaded / reported | **No** | Mean only; flags OR-ed with gain-null |
| Group-level RE meta-analysis with Σ_s^al | Not in eval path | No | No |
| Entropy term for calibrated widths | Config `entropy.weight=1.0`; fit.py notes wired entropy | n/a | Calibration not demonstrated |

### 7.2 What is missing (key innovations)

The simplified implementation is **not** the PLAN §5 object at transform time. It is:

> **Region-level OT to the population-mean connectome + orthogonal Procrustes**, with hierarchical artifacts loaded for reporting (β, τ_φ) but not driving the map.

Missing load-bearing innovations:

1. **Posterior samples do not produce the transform.** PLAN's object is a distribution over couplings; current transform is a point EMD + point Procrustes.
2. **Shrinkage / partial pooling is not in the map.** Each subject's Q_s is independent given C_pop; there is no hierarchical coupling prior, no ε_s partial pooling at transform time.
3. **C̄ = BB^T is unused as the alignment target.** The learned template geometry is not what subjects are coupled to.
4. **τ_φ is not a per-pair uncertainty used in a PLAN-faithful way.** It is a scalar summary + an OR-rule on flags, not posterior mass over maps, not propagated into group statistics.
5. **Gauge features do not affect the transform.** ours_full vs ours_ablated have identical transform metrics on the N=33 check — the "necessary component" ablation does not ablate the applied map.
6. **Pairwise composition and cycle consistency are unused.**

### 7.3 Consequence for claims

- Presenting current `ours_full` as "the hierarchical population-of-couplings model" overclaims.
- Presenting it as "OT-to-C_pop + Procrustes, with β-calibrated hierarchical artifacts reported" is accurate.
- The ident deficit is then unsurprising: the applied method is a **template-shrinkage aligner**, which is exactly the class of methods that cannot win Track A at ceiling.
- The PLAN contribution (posterior uncertainty) is only weakly present in the evaluation table.

---

## 8. Recommended fixes (mathematical / scientific, not baseline-cheating)

### 8.1 Immediate evaluation framing (no code cheat)

1. **State Track A result honestly**: raw/noalign at ceiling 0.959; ours_full 0.857; positive gain +0.018 costs fingerprint discriminability. Cite the tradeoff, not a bug.
2. **Separate the two NonIdent columns in the write-up**:
   - Gain-null count (all methods; weakly differentiating; noalign=500 is tautological)
   - Posterior-width flags (ours only; the PLAN column)
3. **Do not claim "ours wins Track B"** until posterior width is calibrated (N2 diffuse check + synthetic recovery + relation to gain-null).
4. **Disclose the faithfulness gap**: current transform is region OT + Procrustes to C_pop; hierarchical posterior is reported, not applied.
5. **Disclose gauge-null ablation**: identical transform metrics full vs ablated; only τ_φ mean moves (0.075 vs 0.103).

### 8.2 Method fixes if Track A must improve (ordered by mathematical coherence)

1. **Shared-gauge orthogonal transform** (Option 1): stop subject-specific Procrustes toward C_pop; use one template-derived Q or identity + small residual. Goal: **stop losing ident** (approach noalign), not beat the ceiling.
2. **Reliability-optimized maps** (Option 2): fit Q_s (or π_s) to maximize scan-rescan self-match under hierarchical shrinkage; report ident as primary Track A metric. This is the only path that could exceed noalign if misregistration noise exists.
3. **Apply the actual hierarchical posterior** (Option 5): transform via posterior-mean π_s / derived Γ; evaluate Track B from τ_φ properly. Accept that ident may not improve; the contribution is uncertainty.
4. **Protocol with headroom** (Option 3): pre-register a harder identification protocol (N=83, split-half, or lower R) **before** re-running, so a Track A comparison is not post-hoc ceiling-avoidance.

### 8.3 What NOT to do

- Do not weaken baselines, change their preprocessing, or exclude noalign.
- Do not hide the ident drop behind gain-only tables.
- Do not claim τ_φ is calibrated without N2 / synthetic evidence.
- Do not present current ours_full as the full PLAN §5 model until posterior couplings drive the transform.
- Do not "fix" ident by evaluating on a protocol chosen because raw ident is low, unless that protocol is pre-registered as the scientific question (when does alignment help fingerprinting?).

---

## 9. Direct answers to the six questions

### (a) Why does ours_full lose identification (0.857 vs 0.959)?

Because the applied map is subject-specific orthogonal Procrustes toward C_pop, which compresses between-subject edge-pattern variance. Identification uses upper-triangle Pearson features, not eigenvalues; spectrum preservation does not preserve fingerprints. feat_corr=0.658 shows large feature distortion. Cross-subject gain +0.018 is the same compression seen as increased group similarity. At raw ceiling 0.959 there is no room for this transform to help Track A.

**Quantification**: 47/49 → 42/49 correct; discriminability proxy Φ⁻¹(A) 1.74 → 1.07 (~38% drop). As cross-subject correlation rises toward self-correlation, top-1 accuracy falls toward 1/N.

### (b) Why does conn_srm get +0.395 gain but ident collapses to 0.08?

SRM factorizes C_s ≈ A_s S and aligns loadings to a shared response. Cross-subject correlation rises because subjects are projected into a common low-rank subspace (gain +0.395). Individual differences — the fingerprint — are treated as residual noise and largely removed from the evaluation space, so ρ_self ≈ ρ_cross and ident → chance (0.082 ≈ 4/49). This is the extreme point of the bias–variance tradeoff; ours_full is a milder point on the same curve.

### (c) What is the mathematical tension between Track A and Track B?

Track A maximizes ρ_self − ρ_cross (between-subject differences). Track B gain maximizes the rise in ρ_cross (cross-subject commonality). Under template-directed shrinkage C̃_s = C̄ + αΔ_s with α < 1, raising gain requires reducing α, which reduces Var_s[C̃_s] ∝ α² and hence identification margin. This is **not** a universal identity for all alignments (a true bijective misregistration correction could help both), but it **is** the governing tradeoff when (i) there is no ground-truth correspondence, (ii) raw ident is at ceiling, and (iii) methods increase similarity by projecting to a template/shared subspace — all of which hold here.

### (d) What does PLAN.md actually claim ours should win on?

PLAN §7.4–7.5 and §4: **not** identification accuracy. The guaranteed result is reporting **how many pairs are not identifiable**, plus per-pair uncertainty columns that baselines cannot fill. Track A may fail; that is why Track B exists. The contribution is the hierarchical object (population of couplings + calibrated β + posterior width as identifiability), not superior fingerprinting.

### (e) What would need to change for ours to beat noalign on ident?

1. **Shared orthogonal gauge** — preserves ident at noalign level; cannot beat ceiling.
2. **Optimize within-subject reliability** (scan-rescan), not cross-subject gain — only coherent path to possibly exceed noalign; likely modest gains at best.
3. **Harder protocol** where raw ident is not at ceiling — scientifically valid only if pre-registered.
4. **Reframe** to Track B posterior uncertainty — the PLAN-faithful claim; needs calibration evidence, not a transform race.

Honest answer: **beating noalign on Track A under the current frozen protocol is essentially impossible**, because noalign is already at 0.959 and any template-directed nontrivial transform costs fingerprint variance.

### (f) Is current ours_full faithful to PLAN.md §5?

**Partially at training/reporting; not at transform or evaluation.**  
Present: β calibration, artifact load of B/τ_φ/posterior, entropy weight configured, a nontrivial transform.  
Missing: hierarchical posterior driving the map; C̄=BB^T as coupling target; shrinkage in the transform; pairwise Γ; calibrated τ_φ used as identifiability; group-level Σ^al; gauge affecting the applied map (ablation is null on transform metrics).  

Current method ≈ **OT-to-C_pop + orthogonal Procrustes**, with hierarchical quantities reported alongside. That is a different (simpler) method than PLAN §5. The ident deficit is explained by that simpler method's shrinkage geometry; the PLAN contribution is not yet the thing that was evaluated on Track A.

---

## 10. Bottom line

| Question | Answer |
|----------|--------|
| Root cause of ident deficit | Subject-specific orthogonal map toward C_pop compresses between-subject edge variance; raw is at ceiling; ident is variance-sensitive |
| Is it a bug? | No — structural property of template-directed alignment on this protocol |
| Is baselines winning "unfair"? | No — noalign/brainsync preserve fingerprints; conn_srm shows the other extreme |
| Does PLAN claim we win ident? | No — PLAN claims uncertainty / non-identifiability reporting |
| Is the current table delivering that claim? | Weakly — gain-null NonIdent is uninformative; τ_φ not calibrated in the frozen table |
| Is ours_full PLAN-faithful? | Not at transform time; hierarchical posterior is reported, not applied |
| What to fix method-wise | Either (i) shared gauge / reliability objective to stop losing ident, or (ii) actually apply the hierarchical posterior and win on Track B uncertainty as planned |
| What to fix framing-wise | Separate gain-null vs posterior-width columns; disclose ceiling + tradeoff + faithfulness gap; do not claim Track A |

**CRON directive honored**: the analysis diagnoses our method and evaluation framing. No baseline is weakened or excluded.

---

## Shrinkage as a spectral denoising filter — mechanism, controls, and why λ≈0.5 (2026-09-20)

**Scope.** Real ds000243, N=83 two-run subjects, Schaefer-100, artifacts `runs/10_ours_full__9d7dab12__20260920T140613Z/artifacts`, transform mode `posterior_shrink` with `lambda_source=row_entropy` (λ_s = 1/(1+(H_s/ent0)²), ent0 = median posterior row-entropy of π̄_s). Refitting this path reproduces the harness numbers exactly — scan-rescan 0.6455→0.6975, ident 83/83=1.0, 500-pair gain −0.0349, λ_s ∈ [0.4865, 0.5138], mean 0.5001 — so every number below is computed from the same (Q_s, λ_s, C¹_s, C²_s) that produced `REAL_n83_gap_sota.json`. All scripts were throwaway (`/tmp/surge_filter/`); no repo file was modified.

### 1. T_λ is a spectral filter on vec(C), with closed-form attenuation

Stack C column-wise; R = Qᵀ⊗Qᵀ is orthogonal and

    T_λ(C) = (1−λ) C + λ Qᵀ C Q    ⟺    M_λ = (1−λ) I + λ R ,
    K := M_λᵀ M_λ = I − 2λ(1−λ) ( I − ½(R + Rᵀ) ).

For a unit vector v with Rv = ρv (|ρ| = 1, ρ = e^{iθ} on the invariant 2-plane):

    μ(θ) = (1−λ) + λ e^{iθ} ,   |μ(θ)|² = 1 − 2λ(1−λ)(1 − cos θ) ,   |μ(θ)|²|_{λ=1/2} = cos²(θ/2).   (★)

Because R is orthogonal, K has the same eigenvectors as R with eigenvalues w(θ) = 1 − 2λ(1−λ)(1−cos θ): a filter that passes R-fixed components (θ = 0) at weight 1 and attenuates rotated components, with the deepest null at θ = π, w(π) = (2λ−1)². At λ = 1/2 the attenuation is cos²(θ/2).

**Correlation identity (exact).** For feature vectors c = upper-triangle(C),

    corr(M c₁, M c₂) = ⟨c₁, K c₂⟩ / √( ⟨c₁, K c₁⟩ ⟨c₂, K c₂⟩ ) ,   with  K = Mᵀ P M ,

where P is the feature-space centering operator (the harness metric is mean-centred Pearson on the 4950 upper-triangle entries; P is the only reason K is not simply MᵀM). Verified over all 83 subjects: max |direct − formula| = 2.2e-16, mean 8.3e-17.

### 2. Empirical attenuation spectrum of the real Q_s

Q_s is a real orthogonal 100×100 map. On the 5050-dimensional symmetric subspace, R decomposes into 2550 invariant orbits: 50 one-dimensional fixed orbits (θ = 0, dim 50) and 2500 two-dimensional rotation orbits (2×2500), total 50 + 5000 = 5050 = dim Sym(100). Rotation structure verified directly: QᵀaQ = cos θ·a + sin θ·b on every orbit basis (max deviation 3.4e-15).

At λ_s ≈ 0.5 the eigenvalue multiset of M_λ (counting multiplicities) is

| |μ|² median | IQR | min | share < 0.5 | share < 0.9 | energy-weighted mean (1−cos θ) |
|---|---|---|---|---|---|
| λ_s (0.486–0.514) | 0.509 | [0.091, 0.931] | 4.4e-10 | 49.5% | 72.2% | 0.380 |

So ~half of all symmetric-subspace directions are attenuated below 0.71× amplitude (|μ|² < 0.5), the θ ≈ π directions are annihilated, and ~28% (near-fixed) directions pass essentially untouched (|μ|² > 0.9). The energy-weighted mean amplitude factor is √(1 − ½·0.380) = 0.90.

### 3. Why this raises scan-rescan correlation: the run-difference lives in the stopband

Write c₁ = s + e, c₂ = s − e with s = (c₁+c₂)/2 (run-shared) and e = (c₁−c₂)/2 (run-difference). Exactly (verified to 3.3e-16):

    corr(M c₁, M c₂) = ( ⟨s, K s⟩ − ⟨e, K e⟩ ) / √( (⟨s,Ks⟩ + ⟨e,Ke⟩)² − 4⟨s,Ke⟩² ) ,

and since K = Σ_o w_o P_o over the R-orbits o (P_o = orthogonal projector),

    ⟨s, K s⟩ = Σ_o w_o ‖P_o s‖² ,   ⟨e, K e⟩ = Σ_o w_o ‖P_o e‖² .

Ignoring the small cross term, corr ≈ (1−r)/(1+r) with r = ⟨e,Ke⟩/⟨s,Ks⟩; the filter raises the correlation **iff it attenuates the run-difference more than the run-shared part**, i.e. iff w̄_e < w̄_s. That is exactly what the data shows (means over 83 subjects, λ = λ_s):

| quantity | value |
|---|---|
| W_sig := Σ_o (1−cos θ_o)‖P_o s‖² / ‖s‖² | 0.3645 |
| W_noise := Σ_o (1−cos θ_o)‖P_o e‖² / ‖e‖² | **0.4683** — larger for **83/83** subjects |
| energy share in the exactly-fixed subspace (θ = 0): signal / noise / random-direction baseline (50/5050) | 21.8% / 6.5% / 0.99% |
| raw cross-run corr within θ bands: θ=0 / (0,π/4] / (π/4,π/2] / (π/2,3π/4] / (3π/4,π] | 0.943 / 0.641 / 0.674 / 0.564 / 0.501 |
| ⟨e,Ke⟩/⟨s,Ks⟩ at λ = 0.5 vs ‖e‖²/‖s‖² at λ = 0 | 0.1831 vs 0.2194 (−16.5% relative) |

So the reproducible structure is 3.4× over-represented (relative to the run-difference) in the 1%-dimensional R-fixed subspace, and correlates at 0.94 there, while the run-difference energy spreads over the rotated directions that (★) attenuates. In words: T_λ keeps the part of each connectome that its own alignment map already leaves invariant, and suppresses the part the map rotates away — and the rotated part is where scan-to-scan instability sits. Supporting structure: the cohort-mean connectome carries 10.8% of its energy in the fixed subspace of an individual subject's map vs 0.99% for a random direction (≈ 11× concentration); the learned template C̄ = BBᵀ itself is not concentrated there (0.93%).

### 4. Why λ ≈ 0.5 beats λ ≈ 0.89 — and why both endpoints are bad

The whole filter depends on λ only through the scalar λ(1−λ): w(θ) = 1 − 2λ(1−λ)(1−cos θ). λ(1−λ) is maximal (= 1/4) at λ = 1/2, and equals 0.0979 at λ = 0.893 — a **2.55× smaller filter contrast**. Both endpoints have zero contrast: λ = 0 gives M = I (reliability exactly raw: 0.6455 = 0.6455), and λ = 1 gives the pure rotation, which is an isometry on feature space up to dropping the rotated diagonal (energy share 1.9e-4). Hence reliability is maximized strictly inside, and the measured curve is symmetric about 1/2 and tracks the noise/signal ratio r:

| λ (fixed per-subject Q_s) | mean scan-rescan | Δ vs raw | ⟨e,Ke⟩/⟨s,Ks⟩ | ident | 500-pair gain |
|---|---|---|---|---|---|
| 0 | 0.6455 | 0.0000 | 0.2194 | 0.9157 | 0.0000 |
| 0.25 | 0.6792 | +0.0337 | 0.1957 | 0.9398 | +0.0103 |
| **0.50** | **0.6975** | **+0.0519** | **0.1831** | **1.0000** | −0.0349 |
| 0.75 | 0.6781 | +0.0325 | 0.1952 | 1.0000 | −0.1236 |
| 0.893 | 0.6583 | +0.0127 | 0.2085 | 1.0000 | −0.1664 |
| 1 | 0.6430 | −0.0025 | 0.2192 | 1.0000 | −0.1913 |

The λ = 0.50 row **is** the fitted configuration (per-subject λ_s ∈ [0.4865, 0.5138] gives 0.6975 / 1.0 / −0.0349, identical to the harness), and λ = 0.893 reproduces the τ-gated variant (0.6577 / −0.167). The sweep is a diagnostic, **not a tuned result**: nothing here selects λ on the reliability metric; the entropy rule λ_s = 1/(1+(H_s/ent0)²) lands at 0.500 ± 0.014 because the pooled posterior couplings are near-maximally entropic (H_s ≈ ent0) — it is a near-uniform-mixture rule, not a tuned constant.

Two exactness caveats at the endpoints, stated because the naive statement "λ = 1 is a pure rotation so reliability equals raw" is only true in the uncentred metric: the uncentred cosine at λ = 1 is 0.64577 vs raw 0.64554 (deviation from dropping the rotated diagonal, energy share 1.9e-4), while the harness's **centred** Pearson gives 0.6430 (mean deviation −0.0026, max |per-subject deviation| 0.0256) because centering does not commute with the rotation.

### 5. Null and artifact controls

Analytically, independence kills the gain: if the two runs are independent (E[c₂] = 0 given c₁), then E[⟨c₁, Kc₂⟩] = 0 for every λ while the denominator is positive, so E[corr] = 0 — no linear map applied to both runs can manufacture shared signal. Numerically at λ = 0.5:

| control (same map on both runs) | mean scan-rescan | Δ vs raw | paired bootstrap vs real (10k) |
|---|---|---|---|
| real per-subject (Q_s, λ_s) | 0.6975 | +0.0519 | — |
| random Haar Q_s per subject | 0.6456 | +0.0001 | +0.0518 [0.0478, 0.0558] |
| permuted Q (subject π(s)'s fitted map applied to s) | 0.6722 | +0.0267 | +0.0253 [0.0207, 0.0298] |

Interpretation, stated exactly: the boost is **not** a generic orthogonal-mixing artifact (a Haar-random map gives literally zero, +0.0001), but it is **not purely subject-specific** either — applying another subject's fitted map retains ~51% of the boost. The reason is visible in the maps: the fitted maps are mutually similar (mean |tr(QᵢᵀQⱼ)|/n = 0.251 across subject pairs vs 0.008 ± 0.006 for Haar; permuted pairs 0.243), because every Q_s is a Procrustes map onto the same learned template. So roughly half the effect requires the subject's own map; the other half is carried by the shared population-gauge component of the maps. A strict "boost vanishes under permutation" claim is **false on this data** and must not be made.

### 6. Why within-subject reproducibility rises while cross-subject gain falls — and what conn_srm actually does

The metrics mean different things. Scan-rescan reliability and identification are **within-subject reproducibility**: they compare T(C_s^run1) with T(C_s^run2), or with a gallery whose diagonal entry is the same subject. The shrinkage is a per-subject projection toward the subject's own R-fixed subspace; applied identically to both runs it removes the run-difference directions (§3), so the diagonal improves and identification saturates at 1.0. Cross-subject gain compares T(C_i^run1) with T(C_j^run2), i ≠ j: it rewards making *different* subjects' connectomes similar, which a subject-specific projection cannot do — it can only reduce the overlap of the common component across subjects, because each subject is projected onto a different 1%-dimensional subspace. Measured monotonicity: gain = 0.0000 (λ = 0), +0.0103 (0.25), −0.0349 (0.50), −0.1236 (0.75), −0.1664 (0.893), −0.1913 (λ = 1). The reliability gain and the negative cross-subject gain are two readings of the same projection strength; the trade is already visible at λ = 0.25 (gain still slightly positive, +0.0103, with reliability +0.0337).

conn_srm is the degenerate opposite: one shared map/template for all subjects raises cross-subject similarity (gain +0.386) precisely by collapsing individuals onto a common point — identification falls to 0.024 (2/83, at/below the 1/83 chance rate) while reliability *rises* to 0.8453 (identical transformed connectomes correlate trivially). Positive gain with ident ≈ 0 is information destruction, not alignment: nothing about a subject survives its transform.

### 7. Statistical validity of the claim

**Design.** Per-subject deltas Δ_s = r_s(ours) − r_s(baseline) over the 83 subjects; a paired bootstrap resamples *subjects* with replacement 10k times and reports the percentile CI of the mean. Pairing removes between-subject variance (subject difficulty), so the CI is a population claim about the mean within-subject reliability difference for subjects drawn from this distribution (ds000243, Schaefer-100, this preprocessing, two rest runs per subject). **What it supports:** the same-map shrinkage transform raises scan-rescan reliability by +0.052 [0.048, 0.056] over raw and by +0.025 [0.021, 0.030] over the strongest null that shares its filter structure (the permuted-map control) — i.e. the gain is neither a mixing artifact nor an artifact of the fitting protocol, since Q_s and λ_s are fit on run 1 only and applied unchanged to run 2.

**Limits.** (i) Subjects are the only resampling unit — runs, sessions and sites are not resampled, so no scanner/site generalization is tested. (ii) The population template and posterior artifacts were learned on this same cohort, so the claim is in-cohort and same-dataset — no cross-dataset or out-of-sample-subject generalization. (iii) ent0 is a cohort median (median posterior row-entropy), so λ_s carries a weak cohort-level coupling. (iv) The transform mode and the entropy λ rule were selected among several variants on this data, so the CI is conditional on that selection (no multiplicity correction). (v) Reliability/identification are not accuracy of the map: no ground-truth alignment exists in rest fMRI. (vi) The metric is a Pearson correlation of upper-triangle features, not a distance or error norm.

### 8. What this does NOT show

- **Not a tuning result.** λ ≈ 0.5 is the maximally-filtering interior point of the *entropy* rule; the λ sweep is a diagnostic on fixed Q_s. It does not show that λ = 0.5 would be optimal on another dataset, parcellation or preprocessing.
- **Not pure subject-specific alignment.** ~Half the boost survives applying another subject's fitted map (0.6722 vs 0.6975). The effect requires *template-aligned* maps (Haar-random maps give 0.0001) and is carried partly by population-gauge structure shared by all fitted maps.
- **Not a ground-truth denoising claim.** s = (C¹+C²)/2 and e = (C¹−C²)/2 are proxies; the "noise" component also contains genuine run-specific neural-state differences, and the "fixed subspace" is defined by the subject's own fitted Q_s, not by an oracle.
- **Not a white-noise or universality claim.** Nothing here shows the noise is white/Gaussian, or that the variance decomposition (W_noise > W_sig) transfers to other datasets, parcellations or preprocessing pipelines.
- **Not scientific validity.** Higher scan-rescan correlation does not imply better behavioural prediction, better fingerprints for downstream tasks, or correctness of the alignment — only reproducibility of the measured connectome under re-scan.
- **Not free.** The reliability gain costs −0.035 cross-subject 500-pair gain at λ = 0.5 (and −0.17 to −0.19 at λ ≥ 0.89), monotone in λ.
- **Not multiplicity-corrected.** The CI is conditional on the artifact fit and the transform-mode selection.
- **Not "λ = 1 equals raw" in the harness metric.** That identity holds for the uncentred cosine (0.64577 vs 0.64554); under the harness's centred Pearson, λ = 1 gives 0.6430 (max per-subject deviation 0.0256) because centering and the rotation do not commute.

---

## Addendum (independent replication, 2026-09-20): identification under the same-map protocol is invariant to *any* per-subject map — a protocol artifact — plus the common-map control

**Why this addendum.** An independent run of the same numerical programme (separate throwaway scripts in `/tmp/surge-mech/`, same artifacts `runs/10_ours_full__9d7dab12__20260920T140613Z/artifacts`, cohort `results/tables/freeze_n83/frozen_cohort_n83.txt`, β = 28.438323293411973, path `posterior_shrink_tau_gated`, `lambda_source=row_entropy`) reproduces the section above and adds two decisive controls it does not contain: (i) **identification** under wrong maps, and (ii) a **common-map** control that sharpens the "~half survives permutation" reading. Nothing in the section above is retracted; small numerical differences (permuted 0.6708 vs 0.6722, Haar 0.6454 vs 0.6456) are permutation/seed draws of the same controls.

**Exact reproduction of the frozen headline.** λ̄ = 0.500132 (0.486499–0.513822), scan-rescan 0.645523 → 0.697453 (Δ +0.051930), ident 83/83 = 1.000, 500-pair gain −0.034874, `same_Q_both_runs = true`. Identical to `REAL_n83_gap_sota.json`.

### A1. Derivation cross-checks (machine precision)

Vectorise with `c = vec(C)` and `M_λ = (1−λ)I + λR`, `R = Qᵀ⊗Qᵀ` orthogonal:

    corr(M c₁, M c₂) = ⟨c₁, K c₂⟩ / √(⟨c₁,Kc₁⟩⟨c₂,Kc₂⟩),   K = Mᵀ M = (1−2λ(1−λ))I + λ(1−λ)(R + Rᵀ)
      (exact for pre-centred c₁, c₂ — centring applied before the map; the harness centres *after*
       the map, so strictly K = MᵀPM and the K = MᵀM prediction tracks it to ≤ 0.009, see table)
    M_λ X_ij = g(θ_ij) X_ij,  X_ij = conj(v_i) v_jᵀ,  θ_ij = φ_i − φ_j (Q v_i = e^{iφ_i} v_i)
    g(θ) = (1−λ) + λ e^{iθ},  |g(θ)|² = 1 − 2λ(1−λ)(1−cos θ),  |g(θ)|²|_{λ=1/2} = cos²(θ/2)

| check | result |
|---|---|
| mode eigenvalue g(θ_ij) on random (i,j), 5 subjects × 12 modes × λ∈{0.5, 0.89} | max rel. error **6.0e-15** |
| uncentred cosine identity ⟨Mx,My⟩ = ⟨x,Ky⟩ | max abs. error **1.1e-16** |
| K-prediction vs harness centred Pearson, all 83 subjects | max abs. diff **0.0090** (readout + post-transform centring; the harness centres *after* the map, so K = MᵀPM strictly) |
| λ = 0 vs raw, harness metric | max abs. diff **0.0** (exact) |
| λ = 1 uncentred cosine invariance (R orthogonal) | max abs. error **6.7e-16**; centred Pearson shifts mean −0.00255 (max 0.0256) because R does **not** fix the all-ones direction: ‖QᵀJQ − J‖/‖J‖ ≈ 1.20, mean(vec C) shifts by 0.042 vs data RMS 0.314 |

**Attenuation table |g|² (exact, from the formula and confirmed against mode application):**

| θ/π | 0 | 1/8 | 1/4 | 1/2 | 3/4 | 7/8 | 1 |
|---|---|---|---|---|---|---|---|
| λ = 0.5 | 1.000 | 0.9619 | 0.8536 | 0.5000 | 0.1464 | 0.0381 | **0.0000** |
| λ = 0.89 | 1.000 | 0.9851 | 0.9427 | 0.8042 | 0.6657 | 0.6233 | 0.6084 |

**Energy attribution** (83 subjects; pre-centred matrices `C − mean(C)·I`, which is *conservative*: it removes energy from the θ = 0 fixed mode, weakening the asymmetry). Per-mode energy: `|W_ii|²` (θ = 0) and `2|W_ij|²` for i<j, `W = Vᴴ C V`, `Σ E = ‖C‖_F²`.

| quantity (mean over 83 subjects, λ = λ_s) | value |
|---|---|
| energy-weighted attenuation of shared s = (c₁+c₂)/2, ā_s | **0.6348** |
| energy-weighted attenuation of difference d = c₁−c₂, ā_d | **0.5245** |
| ā_s − ā_d (positive for **83/83** subjects) | **+0.1103** |
| Pearson r(ā_s − ā_d, per-subject Δcorr) | **0.883** |
| fraction of ‖d‖² in attenuation quartiles Q1…Q4 (low→high a) | 0.1851 / 0.2958 / 0.3288 / **0.1903** |
| fraction of ‖s‖² in the same quartiles | 0.1174 / 0.2277 / 0.3561 / **0.2989** |
| energy-weighted \|θ\| quantiles (p25/p50/p75) shared vs difference | 0.346 / 1.068 / 1.917 vs 0.693 / 1.473 / 2.312 rad |

Per-subject examples (subjects 015/016/017): λ = 0.4956/0.4973/0.5104, corr 0.6148→0.6676 / 0.6795→0.7374 / 0.6484→0.6818, ā_d = 0.5039/0.5652/0.5366, ā_s = 0.6034/0.6746/0.6074. So the run-difference is 18.5% in the most-attenuated quartile vs 11.7% of the shared part, and the shared part is 29.9% in the passband vs 19.0% of the difference — shared (subject-stable) structure sits in the near-fixed directions, run-specific noise spreads over rotated directions, and the filter raises corr because ΣaEˢ/ΣaEᵈ > ΣEˢ/ΣEᵈ. Q_s is a substantial rotation, not a near-identity map: mean |eigen-angle| 1.444 rad, 46.6% of angles > π/2, cross-subject map distance 0.868 (normalised Frobenius).

### A2. λ sweep with fixed real Q_s (diagnostic only — NOT a tuned result)

| λ | 0 | 0.1 | 0.2 | 0.3 | 0.4 | **0.5** | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mean scan-rescan | 0.6455 | 0.6589 | 0.6726 | 0.6852 | 0.6942 | **0.6975** | 0.6939 | 0.6843 | 0.6712 | 0.6568 | 0.6430 |
| Δ vs raw | 0.0000 | +0.0133 | +0.0271 | +0.0397 | +0.0487 | **+0.0519** | +0.0483 | +0.0388 | +0.0257 | +0.0113 | −0.0025 |

**argmax λ = 0.50.** The entropy rule λ_s = 1/(1+(H_s/ent0)²) lands at 0.5001 without any tuning; λ = 1/2 is also the maximally-filtering interior point (a(θ,λ) = 1 − 2λ(1−λ)(1−cos θ) minimised at λ = 1/2 for every θ ≠ 0) and the only λ that annihilates θ = π. The sweep is a diagnostic on fixed Q_s; it is not used to select λ.

### A3. Null controls at λ = 0.5 — reliability

| map family (same map on both runs) | mean scan-rescan | Δ vs raw | ident |
|---|---|---|---|
| raw / identity Q | 0.64552 | 0.0000 | 0.9157 |
| per-subject Haar Q (5 draws) | 0.64538 ± 0.00028 | −0.0001 | **1.0000** |
| single common Haar Q for all subjects | 0.64673 | +0.0012 | 0.9036 |
| permuted fitted Q (20 derangements) | 0.67079 ± 0.00077 | +0.0253 | **1.0000** |
| one subject's fitted Q for all (Q₀) | 0.67382 | +0.0283 | — |
| **common Procrustes-mean map Q̄ = polar(Σ_s Q_s)** | **0.69190** | **+0.0464** | 0.9036 |
| fitted per-subject Q_s | 0.69746 | +0.0519 | **1.0000** |

At the fitted λ the same ordering holds: Haar 0.64573 ± 0.00014, permuted 0.67138 ± 0.00071, fitted 0.69745. Reading: (i) generic orthogonal mixing does nothing (Haar ≈ raw, Δ ≤ 0.001) — the gain is **not** generic smoothing; (ii) correct subject–map pairing is worth +0.0267 over permutation; but (iii) a **single common template-directed map recovers +0.0464 of the +0.0519 (89%)**, leaving only +0.0056 for subject-specific matching. The reliability mechanism is therefore "template-directed rotation + shrinkage"; it does **not** require per-subject maps. (Consistent with the sibling section's "~half survives permutation": permutation applies *mismatched* per-subject maps, which is worse than one well-matched common map.)

### A4. DECISIVE control — identification is invariant to any per-subject map

Same harness (`trajot.eval.identification.identification_accuracy`, Pearson, same-Q protocol), λ = λ_s:

| map | ident | mean diagonal score | mean off-diagonal score |
|---|---|---|---|
| identity Q (= noalign) | 0.9157 | 0.6455 | 0.4406 |
| common Q̄ | 0.9036 | 0.6919 | 0.5124 |
| common Haar Q | 0.9036 | 0.6467 | 0.4422 |
| per-subject Haar Q | **1.0000** | 0.6452 | 0.2235 |
| permuted fitted Q (20/20 draws) | **1.0000** | 0.6702 | 0.3219 |
| fitted per-subject Q_s | **1.0000** | 0.6975 | 0.4052 |
| cross-map diagnostic: run-1 with Q_s, run-2 with Q_{π(s)} | 0.3494 | — | — |

**Interpretation rule applied (pre-registered in the request): permuted-Q ident stays 1.0 ⇒ the boost is a protocol artifact and MUST be stated as such.** The diagonal score corr(T_s(C¹_s), T_s(C²_s)) is invariant to *any* per-subject invertible map (it equals the raw within-subject correlation up to centring); the 0.9157 → 1.0 jump is produced entirely by suppression of the off-diagonal (cross-subject) scores — 0.4406 → 0.2235 with Haar maps, which give *no* reliability gain at all. Therefore `ident = 83/83` under this protocol cannot be reported as "alignment improves identification"; it must be reported as **"identity preserved / no collapse"** (in contrast to conn_srm at 0.024), because a random per-subject rotation scores the same 1.0. The comparison ours 1.0 vs baselines 0.90–0.92 is confounded by map diversity across subjects and needs a common-map-normalised ident variant before any identification claim is made. The cross-map row shows what happens once the same-Q invariance is broken (0.3494).

### A5. Statistical validity (design + exact numbers)

**Design.** Paired bootstrap **over subjects** (resampling unit = subject, N = 83, 10k resamples) of the per-subject deltas Δ_s = r_s(ours) − r_s(baseline) for reliability and of the ident delta on jointly resampled query/gallery score submatrices; percentile 95% CIs. McNemar **exact** two-sided binomial on discordant identification pairs. Pairing removes between-subject variance.

| comparison (N = 83, entropy-λ) | Δrel [95% CI] | Δident | McNemar b/c | p |
|---|---|---|---|---|
| vs noalign | +0.0519 [0.0480, 0.0558] | +0.0843 | 7 / 0 | 0.015625 |
| vs BrainSync | +0.0513 [0.0472, 0.0552] | +0.0964 | 8 / 0 | 0.0078125 |
| vs FUGW | +0.0759 [0.0705, 0.0814] | +0.0964 | 8 / 0 | 0.0078125 |

N = 49 replication (β = 29.189, `REAL_sota_stats.json`): reliability 0.642 → 0.685, Δrel +0.0434 [0.0379, 0.0490] vs noalign, +0.0425 [0.0366, 0.0486] vs BrainSync, +0.0648 [0.0576, 0.0722] vs FUGW — all exclude 0. Δident at N = 49 is +0.0204 / +0.0204 / +0.0612 with McNemar p = 1.0 / 1.0 / 0.25 — **not significant**, which is exactly what the A4 artifact analysis predicts: the ident effect is not a stable scientific signal.

**Population claim supported.** Within ds000243 two-run rest (Schaefer-100, this preprocessing), the entropy-gated same-map shrinkage transform raises scan-rescan reliability over raw, BrainSync and FUGW; the identification number is a harness property (A4) and only supports "no identity collapse".

**Limits.** (i) Q_s and λ_s are fit on run 1 and applied unchanged to both runs — a *same-map protocol*; there is no held-out reference and no re-fit on run 2. (ii) Under this protocol the ident metric is invariant to any per-subject map (A4) — ident CIs must not be read as evidence of alignment quality. (iii) Subjects are the only resampling unit; no site/scanner/session generalisation. (iv) Template and posterior artifacts were learned on this cohort: in-cohort, same-dataset claim only; no cross-dataset generalisation. (v) ent0 is a cohort median, so λ_s carries weak cohort-level information; the transform mode and λ rule were selected among variants on this data (no multiplicity correction). (vi) Coverage/calibration on real rest is **not** claimed (synthetic coverage@0.9 and AUROC-τ fail). (vii) The metric is Pearson correlation of upper-triangle features, not an error norm or a ground-truth alignment score.

### A6. What this addendum does NOT show

- **Does not show alignment improves identification.** Permuted and Haar per-subject maps also score 83/83; the 1.0 is protocol-induced (A4). Only "no identity collapse" is supported.
- **Does not show the reliability gain requires subject-specific maps.** A single common Procrustes-mean map Q̄ recovers +0.0464 of +0.0519 (89%); per-subject matching adds +0.0056.
- **Does not show generic smoothing works.** Haar-random maps (per-subject or common) give Δrel ≤ +0.0012.
- **Does not show the maps are scientifically correct.** No ground-truth alignment exists on real rest data; Q̄ is derived from the same fitted maps, so "common map" is not an independent estimator.
- **Does not claim cross-subject gain improves** — it is −0.0349 at λ = 0.5 by construction of the template-directed pull (and monotonically worse for larger λ).
- **Does not claim generalisation** beyond ds000243 / N = 83 (N = 49 replication) / this parcellation / this preprocessing / the same-map protocol, and does not claim calibrated uncertainty or coverage.
- **Does not claim exactness of the K-identity for the harness metric** (max 0.009) nor exact centred-correlation invariance at λ = 1 (mean −0.0026, max 0.0256; the all-ones direction is not preserved by R).

**Reproduce.** Throwaway scripts `/tmp/surge-mech/{01_fit_and_cache,02_mechanism,03_controls,04_final_controls}.py`; outputs `cache_n83.npz`, `mech_results.json`, `controls_results.json`, `final_controls.json`. Repo files were read-only; no repo file was modified or committed.
