# Fix-All Audit Findings — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use compose:subagent (recommended) or compose:execute to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make W0–W3 consistent with issues #2–#5 and PLAN.md, unblock real-data fit on Apple Silicon, implement the missing W3 algorithms/eval formulas, wire W2→W3 orchestration, and revive the coordination mechanism — before Phase 2.

**Architecture:** Four parallel workstreams with strict file ownership. Runtime data plane stays `data_root/derivatives/trajot/` + `runs/`; coordination docs are updated to match it. NPZ contract (`connectivity/timeseries/embedding/features/coords/tr/n_volumes/subject_id/run_id`) is frozen — do not rename keys.

**Tech Stack:** Python 3.11, numpy/scipy float64 for OT, torch (MPS float32 encoder only), POT for OT baselines, pandas/pyarrow manifests.

## Global Constraints

- Repo clone: `/tmp/surge-coord` (branch `main`). Project root for code: `/tmp/surge-coord/trajot`
- Deadline: Sunday Sept 20, 2026 1:00 PM Atlantic (16:00 UTC)
- TDD mandatory: failing test → implement → pass. Untested code is unfinished.
- float64 OT/GW **on CPU**. MPS has no float64. Encoder may be MPS float32.
- Combined `.to(device=cpu, dtype=float64)` **fails** on Apple Silicon MPS. Always `.cpu().to(torch.float64)` first.
- NPZ contract keys frozen (9 keys above). Gauge features / masses / A live in `*_geometry.npz` sidecars.
- Band prior is a **band prior**, never "dynamics/trajectory".
- β is **calibrated**, never tuned. No behavioral prediction metrics (Marek 2022).
- No scope reduction. No `_v2`/`_backup` files. Edit in place.
- Code comments: no edit-changelog comments. Only non-obvious WHY.
- Do not touch files outside your ownership list.
- Commit frequently on `main` in `/tmp/surge-coord` after each green test batch.
- Tests: `cd /tmp/surge-coord/trajot && python -m pytest -q`
- Install: `cd /tmp/surge-coord/trajot && pip install -e '.[dev]'` if needed
- `configs/paths.yaml` is gitignored; tests synthesize it in tmp_path. For CLI smoke, copy `paths.example.yaml`.

## Scientific decisions (locked — implement these, do not re-litigate)

### D1. MPS float64 (CRITICAL)
`encoder.SinkhornPosterior.sample_torch` must promote via `.cpu().to(dtype=torch.float64)`, never `.to(device=cpu, dtype=float64)` in one call when the source may be MPS. Keep autograd graph: `.cpu()` is differentiable.

### D2. β normalizer (CRITICAL clarification)
`σ̂_C² = ½ E_s ‖C_s^(1)−C_s^(2)‖_F² / R²` where **R = connectome side (parcels)**, the matrix dimension of `connectivity (R,R)`. This is the per-entry scan-rescan variance. `train.py`'s scale convention (unit row mass, E_GW sums over connectome entries) presupposes this. Issue #4's "V = n_vertices" is **wrong relative to the actual observation model** — C_s is region-level, not vertex-level. Keep dividing by `C.shape[-1]**2`. Document in `beta.py` that V in the formula := R (connectome nodes), not surface vertices. Write `n_regions` into `beta.json`.

### D3. Gauge "velocity"
Code computes a **spatial** finite difference on the diffusion embedding along lex-sorted surface coords, divided by TR. PLAN prose saying "temporal/kinematic" is inaccurate; the implemented quantity matches the formula `f_i=[z_i; β v_i]` with spatial v. Keep the implementation. Fix docstrings/notes to say "spatial gradient of the diffusion embedding used as a gauge-breaking channel", never "temporal velocity" / "captures dynamics". Finitely many optima wording stays.

### D4. Gibbs anchor M0
Prefer geodesic `geometry.anatomical_cost` when surface faces are available; fallback Euclidean `cdist(coords, node_coords)/scale` and record `anchor: "euclidean"` vs `"geodesic"` in run metrics. Faces may come from nilearn fsaverage or geometry sidecar if present.

### D5. Entropy term
Wire `trajot.inference.entropy.entropy_estimator` (Hutchinson + SLQ) into `train.py` so ELBO entropy is no longer hardcoded zero. `tau_phi` must become an uncertainty quantity (W4 handoff). If a full Jacobian log-det is too unstable in remaining time, wire `H[q(ξ)]` plus the Sinkhorn log-det estimator already tested, and document residual caveats in metrics notes — but **do not ship zeros silently**.

### D6. Band prior
Default `band.weight: 0.0` remains OK if documented; do not claim dynamics. Optional small weight only if tests stay green.

### D7. W3 evaluation statistics (from issue #5 / PLAN §7)
- **Identification:** Pearson correlation on flattened upper-triangle of **aligned** features/connectomes. Not cosine as the only score (cosine may remain as a secondary, but spec metric is Pearson).
- **Alignment gain:** `mean_s corr(aligned_s^(1), aligned_s^(2)) - mean_s corr(raw_s^(1), raw_s^(2))` over the **declared pair subsample** (default n=500, seed from config), held-out-run protocol where applicable.
- **Non-identifiable pairs (Track B):** per-pair alignment gain compared to the **method's own permutation null** at α=0.05. Pair flagged non-identifiable if gain is inside the null (not significantly above chance). Return count + boolean vector.
- **Controls:** N1 `banded_coupling` (restrict coupling to a spatial band — residual gain = smoothing); N2 `shuffle_time` (shuffle timepoints / destroy temporal structure — ID → chance). Keep existing `random_permutation_control`.
- **Null max:** report `null_max` alongside `perm_p`.
- **CI:** bootstrap percentile CI (`accuracy_ci`, n_boot from config, default 10000) not Wald-only.
- **Folds:** `make_folds` samples ordered distinct pairs **without replacement** via `default_rng(seed)`, writes seed/count/procedure into metrics.
- **Baselines unfilled columns:** write JSON `null`, never 0.

### D8. W3 baselines (real algorithms)
- **noalign:** identity (already OK).
- **brainsync:** time-series BrainSync. Given X,Y as (T,V) or (V,T): SVD(X Yᵀ)=U S Vᵀ (thin), Q=U Vᵀ, X_aligned = X Q (or Qᵀ X depending on orientation — pin one convention and test). `to_connectome_transform`: C' = Qᵀ C Q when C is in the aligned space. Fit needs timeseries when available; if only connectomes, raise a clear error rather than silently Procrustes.
- **fugw:** Fused Gromov-Wasserstein via POT (`ot.gromov.fused_gromov_wasserstein` or `gromov_wasserstein`) on connectivity-derived costs. Barycenter optional if time-boxed; subject-to-template coupling + transform derived from coupling is enough for MVP. float64 CPU. Record device in meta.
- **conn_srm:** connectivity-SRM — shared response on connectivity features (subject-specific transforms to a shared space), not eigenvalue truncation.
- **ours / ablated:** consume W2 artifacts. `ablation_no_gauge.yaml` (`gauge_features: false`) vs `default.yaml`. AblatedModel must call `trajot.inference.train.train` (or load its artifacts), not return identity. Register `ours_full` and `ours_ablated` (and aliases `10_ours_full`, `11_ours_ablated`) in the baseline registry.
- **harness.py:** `run_baseline(name, data, folds, cfg) -> BaselineResult`, `align_features(result, embeddings)`.

### D9. Orchestration
`run_experiment.py` EXPERIMENT_RUNNERS must stop being all stubs. Each experiment loads data, fits the method, evaluates with D7 metrics, writes schema-valid `metrics.json`. `scripts/compare.py` builds the comparison table from registry/metrics. `evaluate.py --synthetic` should exercise the real NPZ path via `make_synthetic_npz` when practical (or call contract writers).

### D10. Coordination
- TRACKING.md path convention must match **runtime** paths the code actually reads: `data_root/derivatives/trajot/` for npz+manifest, `trajot/runs/index.csv` for the run registry. Coordination can still use per-developer files under `results/` as a **staging** area, with an explicit copy/merge step documented that feeds `derivatives/trajot/` and `runs/`.
- Tick phase gates that are done, fill run log from PRs #7–#10, note W4 in progress.
- Comment/close GitHub issues #2/#4/#5 when their code is verified merged and consistent (or leave open with "audit fixes landing" if still incomplete).
- Add a short W4 coordination note (inputs come from merged W2/W3 metrics after runners exist).
- shards.yaml: leave structure; populate only if subjects are listed. Document the command.

## File ownership (do not cross)

| Agent | Owns (write) | May read |
|---|---|---|
| A — W2 critical | `trajot/src/trajot/inference/**`, `trajot/scripts/fit.py`, `trajot/tests/inference/**`, `trajot/tests/scripts/test_fit_script.py`, `trajot/tests/model/**`, `trajot/configs/model/**` | everything |
| B — W3 baselines | `trajot/src/trajot/baselines/**`, `trajot/tests/baselines/**` | inference, geometry, io, eval (read-only) |
| C — W3 eval | `trajot/src/trajot/eval/**`, `trajot/tests/eval/**` | baselines interface, io, issue #5 |
| D — Orchestration + coord | `trajot/scripts/{run_experiment,evaluate,compare}.py`, `trajot/tests/scripts/**` (except test_fit_script.py), `TRACKING.md`, `trajot/results/**`, `trajot/configs/shards.yaml`, `trajot/configs/experiments/**`, GitHub issues via `gh` | everything |

D runs after B and C have landed interfaces (or codes against this plan's interface block). A is independent.

## Interfaces (B produces, C+D consume)

```python
# baselines/base.py — keep Baseline Protocol: fit(connectomes, cfg), transform(connectome)
# Extend BaselineResult: name, transforms, aligned_features (S,R,d) or (S, R*(R-1)//2),
#   meta (device, algorithm, n_regions, notes), device: str

# baselines/harness.py
def run_baseline(name: str, data: dict, folds, cfg) -> BaselineResult: ...
def align_features(result: BaselineResult, embeddings: np.ndarray | None) -> np.ndarray: ...

# Registry names (lowercase): noalign, brainsync, fugw, conn_srm,
#   ours_full, ours_ablated  (accept experiment filenames as aliases)

# eval/folds.py
@dataclass
class FoldAssignment:
    pairs: list[tuple[str, str]]
    seed: int
    n_pairs: int
    procedure: str  # "default_rng.choice without replacement"

def make_folds(manifest_or_subjects, scheme: str, n_pairs: int, seed: int) -> FoldAssignment: ...
def identification_subjects(manifest) -> list[str]: ...  # delegate two_run_subjects

# eval/identification.py
def flatten_features(x: np.ndarray) -> np.ndarray: ...  # upper triangle
def pearson_scores(query, gallery) -> np.ndarray: ...
def identification_accuracy(run1, run2, *, metric="pearson") -> IdentificationResult: ...
def accuracy_ci(correct: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]: ...

# eval/permutation.py
def permutation_null(stat_fn, items, *, B: int, seed: int) -> PermutationResult: ...
def sign_flip_permutation(values, *, B: int, seed: int) -> np.ndarray: ...
# keep permutation_p; add null_max to result

# eval/alignment_gain.py
def pair_gain(aligned1, aligned2, raw1, raw2, pairs_idx) -> np.ndarray: ...  # per-pair corr gain
def alignment_gain(aligned1, aligned2, raw1, raw2, pairs_idx) -> float: ...  # mean pair_gain

# eval/identifiability.py
def nonidentifiable_count(per_pair_gains, null_dist, alpha: float = 0.05) -> tuple[int, np.ndarray]: ...
def posterior_width_flag(tau_phi, threshold) -> np.ndarray: ...  # W4 column

# eval/controls.py
def banded_coupling(run1, run2, *, band: int, seed: int) -> ControlResult: ...  # N1
def shuffle_time_control(timeseries_or_connectomes, *, seed: int) -> ControlResult: ...  # N2
# keep random_permutation_control

# eval/metrics.py METHOD_KEYS
METHOD_KEYS = {
    "ident_accuracy", "ident_ci", "perm_p", "null_max",
    "alignment_gain", "nonidentifiable_pairs",
    "per_pair_uncertainty", "per_pair_flags",
}
# null_max: float | None  (null if not computed)
```

---

## Task A1: MPS float64 + fit device policy

**Covers:** D1, audit CRITICAL #1

**Files:**
- Modify: `trajot/src/trajot/inference/encoder.py:126-134`
- Modify: `trajot/scripts/fit.py` (real-data device policy)
- Test: `trajot/tests/inference/test_encoder.py`, `trajot/tests/scripts/test_fit_script.py`

- [ ] Failing test: `sample_torch` with `S_phi` on MPS (or CPU float32 simulating the promotion path) returns float64 CPU `pi` and does not raise; gradients flow to `S_phi` when it requires grad.
- [ ] Implement: promote with `S_phi.cpu().to(dtype=torch.float64)` and same for `tau`; ensure `mu_s`/`nu` are CPU float64 before Sinkhorn.
- [ ] Real fit path: `fit.py` must not crash on MPS Macs — either force encoder+OT through the safe promotion path, or `pick_device(prefer_mps=False)` for real runs until entropy+MPS path is proven.
- [ ] Full `pytest tests/inference tests/scripts/test_fit_script.py -q` green including the previously failing real-path test (may use synthetic manifest fixtures).
- [ ] Commit.

## Task A2: β documentation + beta.json n_regions

**Covers:** D2

**Files:**
- Modify: `trajot/src/trajot/inference/beta.py`
- Test: `trajot/tests/inference/test_beta.py`

- [ ] Test: `calibrate_beta` output dict contains `n_regions` equal to `C.shape[-1]`; docstring states V:=R connectome nodes not surface vertices.
- [ ] Keep formula `0.5 * mean ||dC||_F^2 / R^2`. Do **not** switch to n_vertices.
- [ ] Commit.

## Task A3: Entropy wired into train

**Covers:** D5

**Files:**
- Modify: `trajot/src/trajot/inference/train.py`
- Modify: `trajot/src/trajot/inference/entropy.py` only if API gaps
- Test: `trajot/tests/inference/test_train.py`

- [ ] Test: ELBO `entropy` term is not identically zero on a short synthetic train; metrics note records entropy estimator name.
- [ ] Replace `torch.zeros(M, dtype=torch.float64)` with an entropy estimate from `entropy_estimator` / Hutchinson+SLQ on the reparametrized draws. Keep numerically stable (clamp, float64 CPU).
- [ ] If divergence risk: add a small `model.entropy.weight` config (default 1.0) so it can be scaled; document.
- [ ] Short synthetic train still finishes; loss trace has finite entropy.
- [ ] Commit.

## Task A4: Gibbs anchor + gauge docs + band prior honesty

**Covers:** D3, D4, D6

**Files:**
- Modify: `trajot/src/trajot/inference/train.py` (M0 construction + print/notes)
- Modify: `trajot/src/trajot/geometry/diffusion.py` (docstrings only if they claim temporal)
- Test: `trajot/tests/inference/test_train.py`, `trajot/tests/geometry/test_diffusion.py`

- [ ] Docstrings never say temporal velocity / dynamics for gauge features.
- [ ] M0 uses geodesic path when faces available; else Euclidean + `anchor` field in artifacts/metrics.
- [ ] Commit.

## Task B1: Baseline interface + noalign + harness skeleton

**Covers:** D8

**Files:**
- Modify: `trajot/src/trajot/baselines/base.py`
- Create: `trajot/src/trajot/baselines/harness.py`
- Modify: `trajot/src/trajot/baselines/__init__.py`
- Test: `trajot/tests/baselines/test_w3_baselines.py`

- [ ] `BaselineResult` + `run_baseline` + `align_features` exist; noalign registered and tested.
- [ ] Commit.

## Task B2: Real BrainSync

**Covers:** D8

**Files:**
- Modify: `trajot/src/trajot/baselines/brainsync.py`
- Test: `trajot/tests/baselines/test_w3_baselines.py`

- [ ] Test: on synthetic paired timeseries with known orthogonal misalignment, BrainSync recovers higher cross-run correlation than noalign; Q is orthogonal (`Q.T@Q≈I`).
- [ ] Implement thin SVD `Q = U Vᵀ` from `X Yᵀ` (pin orientation); `to_connectome_transform(Q, C)`.
- [ ] `fit` may accept timeseries via optional `data` dict in harness; connectome-only Procrustes is forbidden as silent substitute.
- [ ] Commit.

## Task B3: Real FUGW (POT) + conn-SRM

**Covers:** D8

**Files:**
- Modify: `trajot/src/trajot/baselines/fugw.py`, `conn_srm.py`
- Test: `trajot/tests/baselines/test_w3_baselines.py`

- [ ] FUGW: POT GW/FGW float64 CPU on (S,R,R) connectomes; `transform` produces an aligned connectome; test runs on small synthetic (S=4,R=16) and returns finite symmetric zero-diag matrices.
- [ ] conn-SRM: shared-response on connectivity features (not eigenvalue truncation); test shape + improved within-subject similarity on synthetic shared-signal data.
- [ ] Commit.

## Task B4: Ours full + ablated consume W2

**Covers:** D8, D9 handoff

**Files:**
- Modify: `trajot/src/trajot/baselines/ablated.py`
- Create or Modify: `trajot/src/trajot/baselines/ours.py` (if cleaner)
- Test: `trajot/tests/baselines/test_w3_baselines.py`

- [ ] `ours_full` loads artifacts from `runs/*/artifacts/{template,posterior_samples,tau_phi}.npz` if present, else can fit via `train` with `default.yaml` on synthetic.
- [ ] `ours_ablated` fits/loads with `gauge_features: false`.
- [ ] Test: ablated differs from full on synthetic when gauge channel is informative (or at least both produce valid BaselineResult with different meta `gauge_features` flag).
- [ ] Registry exports both names.
- [ ] Commit.

## Task C1: Folds + identification (Pearson + bootstrap CI)

**Covers:** D7

**Files:**
- Modify: `trajot/src/trajot/eval/folds.py`, `identification.py`
- Test: `trajot/tests/eval/test_w3_eval_core.py`

- [ ] `make_folds` without replacement; `identification_subjects` uses `two_run_subjects`.
- [ ] Pearson scores on upper-triangle features; `accuracy_ci` bootstrap.
- [ ] Tests: perfect identity → acc=1; independent noise → acc≈1/S; CI contains point estimate; fold pairs unique.
- [ ] Commit.

## Task C2: Alignment gain + non-identifiability + null_max

**Covers:** D7 (Track B statistic)

**Files:**
- Modify: `trajot/src/trajot/eval/alignment_gain.py`, `identifiability.py`, `permutation.py`, `metrics.py`
- Test: `trajot/tests/eval/test_w3_eval_core.py`

- [ ] Formulas per D7. `null_max` in METHOD_KEYS (allow None).
- [ ] Test: when aligned==raw, gain≈0; when alignment improves correlation, gain>0; nonidentifiable flags fire when gain inside null.
- [ ] Commit.

## Task C3: Controls N1/N2

**Covers:** D7

**Files:**
- Modify: `trajot/src/trajot/eval/controls.py`, `eval/__init__.py`
- Test: `trajot/tests/eval/test_w3_eval_core.py`

- [ ] `banded_coupling` + `shuffle_time_control` implemented and exported.
- [ ] Tests: N2 drives identification toward chance on synthetic time-shuffled data; N1 returns ControlResult schema.
- [ ] Commit.

## Task D1: Wire run_experiment runners

**Covers:** D9

**Files:**
- Modify: `trajot/scripts/run_experiment.py`
- Modify: `trajot/scripts/evaluate.py`
- Test: `trajot/tests/scripts/test_evaluate_script.py` (+ new run_experiment tests)

- [ ] Each of the 6 experiment names dispatches to real fit+eval (or load artifacts+eval), writes schema-valid metrics with all METHOD_KEYS including `null_max`.
- [ ] Baselines unfilled uncertainty → JSON `null`.
- [ ] `evaluate.py --synthetic` uses contract/synthetic npz path when possible.
- [ ] Commit.

## Task D2: compare.py

**Covers:** D9

**Files:**
- Modify: `trajot/scripts/compare.py` (currently empty)
- Test: `trajot/tests/scripts/test_compare.py` (create)

- [ ] Reads `runs/index.csv` + `runs/*/metrics.json`, writes `results/tables/comparison.csv` (or stdout markdown) with method × metric columns; nulls shown as empty.
- [ ] Commit.

## Task D3: Coordination mechanism revival

**Covers:** D10

**Files:**
- Modify: `TRACKING.md`
- Modify: `trajot/results/README.md`
- Modify: `trajot/configs/shards.yaml` (comments/command only unless data listed)
- GitHub: issues #2 #4 #5 #6 via `gh`

- [ ] TRACKING path section rewritten: runtime paths authoritative; results/ = staging with explicit merge commands that copy into `derivatives/trajot/` and document `runs/index.csv`.
- [ ] Phase gates: tick W0–W3 code-complete with notes on remaining gaps if any; W4 in progress; run log entries for PRs #7–#10 + audit-fix PR.
- [ ] `gh issue comment` on #2/#4/#5 summarizing audit status; close when fixes verified.
- [ ] Add W4 note: inputs = metrics.json from runners; tau_phi uncertainty if entropy wired.
- [ ] Commit + push `main`.

## Task E1: Full-suite verification + push

**Covers:** all

- [ ] `cd /tmp/surge-coord/trajot && python -m pytest -q` — report counts. Prefer 0 failures; document any remaining skip (real ds000243 absent).
- [ ] Grep: no `.to(device=.*, dtype=torch.float64)` on MPS-eligible tensors; no "temporal velocity" claims; band prior not described as dynamics.
- [ ] Push origin main.
- [ ] Update TRACKING run log with verification result.

---

## Self-review checklist

- [x] Spec coverage: audit findings A–D and coordination all have tasks
- [x] Locked scientific decisions D1–D10 referenced by tasks
- [x] Interfaces block matches what B produces and C/D consume
- [x] File ownership prevents write collisions
- [x] No TBD placeholders in critical formulas
- [x] β decision explained (R² kept, issue text was wrong)
