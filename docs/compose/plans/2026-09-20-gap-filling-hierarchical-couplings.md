# Gap-Filling Hierarchical Couplings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use compose:subagent or compose:execute. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Wire the hierarchical posterior into alignment + evaluation so TrajOT fills the literature gap (point estimates → calibrated uncertainty, identifiability, uncertainty-aware group inference) that baselines cannot fill.

**Architecture:** Posterior-gated transform (`λ(τ)` shrinkage + coupling to learned `C_bar`); new uncertainty metrics; synthetic planted-GT harness; group REML driver on real/synthetic posteriors.

**Tech Stack:** numpy float64, POT `ot.emd`, existing `trajot.inference.train` / `report.group` / `eval`, pytest.

## Global Constraints

- Repo: `/tmp/surge-coord`, code under `trajot/`; push to `origin/main` after green tests; **never force-push**; **never commit files >100MB** (gitignore `frozen_ds000243/`, `private_freeze_*/`, `trajot/runs_frozen/`)
- Python: `/Users/anandlo/.central_venv/bin/python3`
- β real scan-rescan **29.189**; synthetic noise `σ_C² = 1/β ≈ 0.03426`
- Method keys: `ours_full`, `ours_ablated`; baselines `per_pair_flags=null`, `per_pair_uncertainty=null`
- **Do not claim:** ID superiority, unique minimizer, dynamics, real-rest calibrated coverage without GT
- **Literature gap (the point):** no existing rest-fMRI aligner quantifies alignment uncertainty or identifiability; hierarchical population-of-couplings + group Σ^al is the contribution
- TDD mandatory; long trainings OK — wait; kill nothing that is mid-run on large frozen jobs
- `model.K` must equal `n_regions` when transform is region-space (100 for Schaefer)
- Debug runs may cap epochs at 3 — synthetic gap harness **must** use `model.train.epochs >= 10`
- Claim discipline: finitely many optima; band prior not dynamics; "no such work was found"; Marek 2022

## Pre-registered success (synthetic)

| Metric | Pass |
|--------|------|
| Coverage @ nominal 0.90 vs planted π* | ours ∈ [0.85, 0.95]; baselines N/A (0-width) |
| AUROC(τ, planted-ambiguous) | > 0.8 |
| Group FPR / n_eff | weighted REML FPR ≤ naive+0.02 when noisy subjects planted; n_eff < S when τ high |
| Held-out score | ours better than baseline residual under common score |
| Ablation | full ≠ ablated on coverage or group FPR (not only τ scalar) |

---

### Task 1: Uncertainty metrics module

**Covers:** [S2 Approach A], [S4]

**Files:**
- Create: `trajot/src/trajot/eval/uncertainty.py`
- Modify: `trajot/src/trajot/eval/__init__.py`, `trajot/src/trajot/eval/metrics.py`
- Test: `trajot/tests/eval/test_uncertainty.py`

**Interfaces (produces):**
```python
UNCERTAINTY_KEYS = ("posterior_coverage", "nonident_auroc", "heldout_score", "group_neff")

def shrinkage_transform(C: np.ndarray, Q: np.ndarray, lam: float) -> np.ndarray:
    """C̃ = (1-λ) C + λ Q^T C Q, λ∈[0,1]; λ=0 identity, λ=1 full Q^T C Q."""

def posterior_coverage(pi_samples: np.ndarray, pi_star: np.ndarray, alpha: float = 0.1) -> float:
    """Fraction of (i,k) where pi_star[i,k] in (1-alpha) empirical interval of draws.
    pi_samples (M,R,R) or (M,V,K); pi_star same trailing shape. Interval = quantiles."""

def nonident_auroc(tau_phi: Sequence[np.ndarray] | np.ndarray, ambiguous: np.ndarray) -> float:
    """AUROC of subject-level mean(tau) at detecting ambiguous[i]==True.
    Use rank-based AUROC (no sklearn): handle ties, return 0.5 if one class empty."""

def heldout_predictive_score(C_run2: np.ndarray, C_aligned_from_run1: np.ndarray) -> float:
    """Common score for all methods: -||C_run2 - aligned||_F^2 (higher better).
    Ours may later use ELBO; this keeps baselines comparable."""

def subject_mean_tau(tau_phi: Sequence[np.ndarray]) -> np.ndarray:
    """(S,) mean tau per subject."""
```

- [ ] Write failing tests:
```python
def test_shrinkage_limits():
    C = np.random.randn(20, 20); C = 0.5*(C+C.T); np.fill_diagonal(C, 0)
    U, _, Vt = np.linalg.svd(np.random.randn(20, 20)); Q = U @ Vt
    assert np.allclose(shrinkage_transform(C, Q, 0.0), C)
    assert np.allclose(shrinkage_transform(C, Q, 1.0), Q.T @ C @ Q)

def test_coverage_perfect_posterior():
    pi_star = np.eye(8)[None].repeat(50, 0)
    assert posterior_coverage(pi_star + 0.0, pi_star[0], 0.1) == 1.0

def test_auroc_separation():
    tau = np.array([0.01]*10 + [0.5]*10)
    amb = np.array([False]*10 + [True]*10)
    assert nonident_auroc(tau, amb) > 0.95

def test_metrics_keys_exported():
    from trajot.eval import UNCERTAINTY_KEYS, shrinkage_transform, posterior_coverage
```
- [ ] Implement `uncertainty.py`; export from `eval/__init__.py`
- [ ] `metrics.py`: add `OPTIONAL_UNCERTAINTY_KEYS = set(UNCERTAINTY_KEYS)` — do **not** put them in METHOD_KEYS (baselines stay valid without them)
- [ ] `pytest tests/eval/test_uncertainty.py -q` green
- [ ] Commit: `feat(eval): uncertainty gap-filling metrics (coverage, AUROC, heldout, shrinkage)`

---

### Task 2: Posterior-gated transform in OursFull

**Covers:** [S2 architecture fix], [S5]

**Files:**
- Modify: `trajot/src/trajot/baselines/ours.py`
- Test: `trajot/tests/baselines/test_ours_posterior.py`

**Interfaces:**
```python
def pool_pi_to_regions(pi: np.ndarray, region_index: np.ndarray, R: int) -> np.ndarray:
    """(V,K) or (R,K) -> (R,K) mass-preserving block sum; if already (R,K) return copy."""

def mix_identity_orthogonal(Q: np.ndarray, lam: float) -> np.ndarray:
    """Frobenius-ish mix: interpolate on O(R) via SVD re-orthogonalization of (1-λ)I + λ Q."""

class OursFull:
    transform_mode: str = "posterior_shrink"  # or "point_procrustes" for ablation
    tau0: float = 0.1
    def pairwise_gamma(self, i: int, j: int) -> np.ndarray:
        """Γ_ij = rowstoch(π̄_i) @ diag(ν)^{-1} @ rowstoch(π̄_j).T  (R,R) after pooling."""
```

**Transform math (posterior_shrink):**
```
π̄_s = pool_pi_to_regions(_pi_means[s])     # (R,K)
C_bar_region = _template_connectome(B) if K==R else spectral embed to R
P_s = π̄_s / row_sums
C_ref = P_s @ C_bar_region @ P_s.T
Q_s = UV^T of SVD(C_s @ C_ref.T)
λ_s = 1 / (1 + (mean(tau_phi[s]) / tau0)**2) if tau_phi else 1.0
C̃ = shrinkage_transform(C_s, Q_s, λ_s)      # Task 1
```

- [ ] Failing tests: posterior path uses `_pi_means` when present (monkeypatch emd to explode if called); τ→∞ ⇒ identity; τ→0 ⇒ Q^T C Q; `pairwise_gamma` shape (R,R); docstring does not claim BB^T drive when mode is point_procrustes
- [ ] Implement; when artifacts missing fall back to region EMD+Procrustes but **meta["transform"]** must say which path ran
- [ ] `OursAblated`: `gauge_features=False` **and** `transform_mode="point_procrustes"` (hierarchy ablates out of the map)
- [ ] `pytest tests/baselines -q` green
- [ ] Commit: `feat(ours): posterior-gated tau-shrink transform; ablation drops hierarchy from map`

---

### Task 3: Group analysis driver

**Covers:** [S2 Approach B], [S5]

**Files:**
- Create: `trajot/scripts/run_group_analysis.py`
- Test: `trajot/tests/scripts/test_group_driver.py`

**Interfaces:**
```python
def run_group(posteriors_dir: Path, z: np.ndarray | None, nu: np.ndarray | None,
              method: str = "REML") -> dict:
    """Load posterior_samples.npz + tau_phi.npz + template.npz nu.
    If z is None, use subject mean |connectome| strength as z_s (R,) after region pooling of π.
    Call meta_analysis_map / meta_analysis vs one_sample_ttest.
    Returns {n_subjects, n_eff, mean_weight, mean_sigma2, reml_theta_se, ttest_theta_se, ci_ratio}."""
```

- [ ] Test with synthetic draws: two subjects high τ, others low ⇒ n_eff < S; REML CI wider than t-test when sigma2 large
- [ ] Implement CLI: `python scripts/run_group_analysis.py --artifacts runs/.../artifacts [--z path.npy]`
- [ ] Smoke on any existing `runs/10_ours_full__*/artifacts` if present
- [ ] Commit: `feat(report): run_group_analysis driver on real/synthetic posteriors`

---

### Task 4: Synthetic gap-filling harness

**Covers:** [S3], [S4], pre-registered metrics

**Files:**
- Create: `trajot/scripts/synthetic_coverage.py`
- Test: `trajot/tests/scripts/test_synthetic_coverage.py` (small N)

**Generative model:**
```
sigma2 = 1.0 / 29.189
C_pop: random PSD R x R, zero diag, symmetrized
70%: P* near-permutation (row-stochastic, low entropy)
30%: ambiguous: 0.5 P1 + 0.5 P2 (two permutations)
C_s^(r) = P* C_pop P*^T + E, E ~ N(0, sigma2) symmetrized zero diag
```

**API:**
```python
def plant_and_fit(N=60, R=50, beta=29.189, frac_ambiguous=0.3, M=20, epochs=12, seed=0,
                  K=50) -> dict:
    """Returns keys: coverage_90, coverage_80, auroc_tau, heldout_ours,
    heldout_noalign, group_neff, group_fpr_reml, group_fpr_ttest, notes."""
```

- [ ] Small test N=8 R=10 epochs=2 (smoke shapes only)
- [ ] Full harness uses `pick_device(prefer_mps=False)`, `model.K=R`, gauge on for full / transform_mode point for ablated
- [ ] CLI writes `trajot/results/tables/synthetic_gap_results.json` + markdown
- [ ] Commit: `feat(experiments): synthetic planted-GT gap-filling harness`

---

### Task 5: Evaluate uncertainty keys (optional)

**Covers:** [S5]

**Files:**
- Modify: `trajot/scripts/evaluate.py` (only if ours artifacts present)
- Test: extend `tests/scripts/test_evaluate_script.py`

- [ ] When method is ours_* and artifacts exist, fill optional keys: `posterior_coverage=none` (no GT on real), `group_neff` if group driver run, keep `per_pair_uncertainty`
- [ ] Baselines unchanged (null model-only columns)
- [ ] Commit: `feat(evaluate): optional uncertainty keys for ours rows`

---

### Task 6: Run synthetic gap experiment + real group smoke

**Covers:** [S3], [S4]

- [ ] `cd trajot && /Users/anandlo/.central_venv/bin/python3 scripts/synthetic_coverage.py --full` (or plant_and_fit N=60)
- [ ] `python scripts/run_group_analysis.py --artifacts <latest ours artifacts>` if any exist
- [ ] Record JSON/MD under `trajot/results/tables/`
- [ ] Do **not** kill long frozen real-data trainings
- [ ] Commit results tables only (no large npz)

---

### Task 7: Honest RESULTS update + push

**Covers:** [S1], [S4], [S5]

**Files:**
- Modify: `trajot/reports/RESULTS.md`, `TRACKING.md`, root `RESULTS.md` pointer
- Create: `docs/compose/reports/2026-09-20-gap-filling-results.md`

- [ ] Pre-registered table filled from Task 6 artifacts (coverage, AUROC, group n_eff, heldout)
- [ ] Keep N=49 ID table as **secondary** honesty: ceiling, tradeoff, no ID win claim
- [ ] Literature gap section: verbatim that existing aligners emit point estimates; this framework produces calibrated uncertainty + identifiability + group Σ^al
- [ ] If synthetic metrics FAIL pre-registered gates: report failure honestly; diagnose τ calibration / entropy / epochs — do not invent wins
- [ ] `git add` compact files only; commit; `git push origin main` (no force, no >100MB)
- [ ] `gh issue comment 1` with gap-filling table
- [ ] Commit: `docs: gap-filling results — uncertainty metrics vs baselines N/A`

---

## Self-review

- [x] S1–S5 covered by tasks 1–7
- [x] Interfaces consistent: `shrinkage_transform` used by ours.py; UNCERTAINTY_KEYS separate from METHOD_KEYS
- [x] Literature gap is the success criterion — not ID acc
- [x] No placeholders in math; concrete β, N, R, epochs
