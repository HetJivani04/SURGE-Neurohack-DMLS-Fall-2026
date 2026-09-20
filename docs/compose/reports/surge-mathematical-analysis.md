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
