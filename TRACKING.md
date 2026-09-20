# TRACKING.md — Team Coordination Board

**Team coordination for SURGE Neurohack 2026. Deadline: Sunday Sept 20, 1:00 PM Atlantic. Update this file when you finish a step. Read it before you start.**

**Path convention:** coordination paths under `results/`, `configs/`, and `scripts/` are relative to the **`trajot/`** project root. Runtime data paths are absolute via `configs/paths.yaml` (`data_root`). `cd trajot` before running scripts. The repository root holds this file, `PLAN.md`, and `docs/`.

---

## Current phase

**Phase 1 — Development (framework code landing; W0–W3 code merged; audit-fix wave in flight)**

---

## AUTHORITATIVE runtime paths (what the code actually uses)

The scripts do **not** read coordination staging files as their primary data plane. These are the paths every runner, evaluator, and trainer uses:

| What | Runtime path | Read by |
|---|---|---|
| Per-run NPZ + manifest | `<data_root>/derivatives/trajot/` (`sub-<id>_run-<r>.npz`, `manifest.parquet`) | `evaluate.py`, `train.load_train_data`, `run_experiment.py`, `fit.py` |
| Run registry | `trajot/runs/index.csv` | `run_experiment.py`, `compare.py`, W4 |
| Run artifacts | `trajot/runs/<run_id>/artifacts/` | `ours_full` / `ours_ablated`, W4 |
| Per-run metrics | `trajot/runs/<run_id>/metrics.json` | `compare.py`, W4 |
| Local path config | `trajot/configs/paths.yaml` (**gitignored**; copy from `paths.example.yaml`) | `trajot.config.load_config` |

`data_root` is set in `configs/paths.yaml`. On a developer Mac that is typically the local ds000243 root. It is **not** `trajot/results/`.

**Critical:** `evaluate.py` and `trajot.inference.train.load_train_data` read the subject manifest from `cfg.data_root/derivatives/trajot/manifest.parquet`. Coordinating only under `results/` does **nothing** to the runtime until either:

1. `data_root` is populated (npz + manifest copied/synced into `<data_root>/derivatives/trajot/`), or
2. `configs/paths.yaml` is pointed at a shared derivatives folder that already has that layout.

---

## STAGING coordination area: `trajot/results/`

`results/` is a **git-tracked staging area** for multi-developer merges. It is not the runtime data plane.

| Staging path | Contents | Git? |
|---|---|---|
| `results/npz/` | Optional compact copies of subject npz for sharing (~40 KB each) | yes |
| `results/manifests/<dev>.parquet` | Per-developer manifest rows **BEFORE** merge | yes |
| `results/registry/<dev>.csv` | Per-developer run-registry rows **BEFORE** merge | yes |
| `results/tables/` | Comparison tables from W4 / `compare.py` | yes |
| `results/manifest.parquet` | Merged manifest produced by `scripts/merge_manifests.py` | optional artifact |
| `results/runs_index.csv` | Merged run index produced by `scripts/merge_runs.py` | optional artifact |

**Not in git:**

- `data/` — raw ds000243 (~5.67 GiB), each developer has it locally
- `trajot/runs/` — per-run directories written by `run_experiment.py` / `fit.py` (gitignored)
- `trajot/configs/paths.yaml` — per-developer local paths (gitignored)

Registry CSVs in `results/registry/<dev>.csv` **are** git-tracked (staging). The live runtime registry `runs/index.csv` is **gitignored**.

---

## MERGE COMMANDS (staging → runtime)

After shard branches merge to `main`, the coordinator runs from `trajot/`:

```bash
# 1) Merge staged manifests and staged run rows
python scripts/merge_manifests.py
#   reads  results/manifests/*.parquet
#   writes results/manifest.parquet
python scripts/merge_runs.py
#   reads  results/registry/*.csv
#   writes results/runs_index.csv

# 2) COPY/SYNC into the data_root that configs/paths.yaml points at,
#    OR point paths.yaml at a shared derivatives folder that already has this layout:
#      <data_root>/derivatives/trajot/manifest.parquet
#      <data_root>/derivatives/trajot/sub-<id>_run-<r>.npz
mkdir -p "$DATA_ROOT/derivatives/trajot"
cp results/manifest.parquet "$DATA_ROOT/derivatives/trajot/manifest.parquet"
# cp results/npz/*.npz "$DATA_ROOT/derivatives/trajot/"   # if npz were staged

# 3) Sync the merged run registry into the runtime path the code reads
mkdir -p runs
cp results/runs_index.csv runs/index.csv
```

Alternative: set `data_root:` in each developer's `configs/paths.yaml` to a shared directory that already contains `derivatives/trajot/` populated from staging.

Until step 2 (or the paths.yaml retarget) is done, `evaluate.py` / `train.py` will not see merged subjects.

---

## Shard assignments

| Developer | Subjects | Status |
|---|---|---|
| dev_a |  | Not started |
| dev_b |  | Not started |
| dev_c |  | Not started |

Populate subject IDs only after listing them on the machine that has ds000243:

```bash
ls data/ds000243/ | grep '^sub-' | sort
```

Then distribute evenly across developers and record the assignment in `configs/shards.yaml`. Do **not** invent subject IDs in git.

---

## Phase gates (updated from git reality)

### Phase 1 — Development

- [x] W0 scaffolding complete (PR #8)
- [x] W1 code complete (PR #7) — real 120-subject npz still local / Phase 2
- [x] W2 model + inference code complete (PR #10) — audit fixes in flight
- [x] W3 baselines + eval code landing (PR #9 + audit-fix wave)
- [ ] W4 analysis + reporting (PR #11 on origin/main; local merge + contract alignment in progress)
- [x] All tests pass (audit-fix integration: `pytest -q` **298 passed, 2 skipped** on 2026-09-20; skips are missing real ds000243 cases)

### Verification — audit-fix wave (2026-09-20)

| Suite | Result |
|---|---|
| Full `python -m pytest -q` (pre-W4 merge, local audit-fix) | **298 passed, 2 skipped** (10m39s) |
| `tests/inference` excluding long recovery | 13 passed (entropy/MPS/β wired) |
| `tests/baselines + tests/eval + tests/scripts + tests/model` | 107 passed |
| Long `tests/inference/test_train.py` (incl. 80-epoch permutation recovery) | 16 passed (~10m; entropy pathwise Shannon + detached Hutchinson/SLQ) |

**Locked scientific decisions (do not re-litigate):**

- **D1 MPS:** `encoder.sample_torch` promotes via `.cpu().to(dtype=torch.float64)`, never combined `.to(device=cpu, dtype=float64)`.
- **D2 β:** `σ̂_C² = ½ E_s ‖C_s^(1)−C_s^(2)‖_F² / R²` with **R = connectome side (parcels)**; `beta.json` writes `n_regions`.
- **D3 gauge:** spatial finite difference / gauge-breaking channel — never "temporal velocity" / "captures dynamics".
- **D5 entropy:** WIRED (not zero). Pathwise Shannon `H[pi]` (differentiable) + detached `projected_sinkhorn_logdet` (Hutchinson/SLQ, `subspace_dim=4`, `n_iter=max(10, min(L,20))` in train). Estimator id: `shannon_pi+hutchinson_slq`. Scale via `model.entropy.weight` (default 1.0). Residual caveat: Jacobian logdet is detached.
- **D7 folds:** `default_rng.choice` **without replacement** over ordered distinct pairs (`eval.folds.FOLD_PROCEDURE`).
- **METHOD_KEYS:** include `null_max`; unfilled columns are JSON `null`, never `0`.
- **Track B:** non-identifiable if pair gain is inside the method's own permutation null at α=0.05 (not ID-miss definition).

**W4 merge contract (frozen):**

- Method keys canonical: `ours_full`, `ours_ablated` (aliases accepted in table collection).
- `evaluate.py`: baselines write `per_pair_flags=null` and `per_pair_uncertainty=null`.
- `scripts/compare.py` = W4 paper 6×5 CLI (PLAN §7.4). Flat all-method CSV → `scripts/compare_flat.py`.
- Nine macOS junk `* 2.py` files are deleted on merge (RULE 3).

### Phase 1→2 merge gates

These remain **unchecked** until real data + green full suite:

- [ ] All npz files available under `<data_root>/derivatives/trajot/` (staging copies optional in `results/npz/`)
- [ ] `manifest.parquet` merged and verified (203 rows / 83 two-run subjects) at the **runtime** path
- [ ] `runs/index.csv` has entries for synthetic and real test runs
- [ ] Someone runs `pytest -q` on merged state — all green
- [ ] Push to main

### Phase 2 — Experiments

- [ ] Run baselines on real data
- [ ] Run ours_full on real data
- [ ] Run ours_ablated on real data
- [ ] Compute identification accuracy + permutation null
- [ ] Compute alignment gain
- [ ] Count non-identifiable pairs
- [ ] Generate results table
- [ ] Write up

---

## Merge protocol

### What goes in git vs what does not

See the staging/runtime tables above. Summary:

**In git:** `results/npz/*.npz` (optional), `results/manifests/<dev>.parquet`, `results/registry/<dev>.csv`, `results/tables/`, `configs/shards.yaml`, `TRACKING.md`

**Not in git:** `data/`, `runs/`, `configs/paths.yaml`

### How to merge your shard

1. Finish preprocessing your subjects.
2. Copy your npz files to `results/npz/` (optional sharing copy; runtime still needs `<data_root>/derivatives/trajot/`).
3. Write your manifest rows to `results/manifests/<your-name>.parquet`.
4. Append your run rows to `results/registry/<your-name>.csv`.
5. Update this `TRACKING.md` — tick your checkbox, update your status.
6. Commit and push to branch `shard/<your-name>`.
7. On the call say "my shard is pushed".
8. Coordinator merges the branch to main.
9. Coordinator runs `scripts/merge_manifests.py` and `scripts/merge_runs.py`, then syncs into `data_root/derivatives/trajot/` and `runs/index.csv` (see MERGE COMMANDS).
10. Everyone pulls main and points `configs/paths.yaml` at the populated `data_root`.

---

## Conflict prevention

- Each developer only writes npz files for their own subjects — no file conflicts.
- Manifest rows go in per-developer files (`results/manifests/<name>.parquet`), merged by script — never hand-edit `results/manifest.parquet`.
- Run rows go in per-developer files (`results/registry/<name>.csv`), merged by script.
- Runtime `runs/` is local and gitignored; do not commit it.
- `TRACKING.md`: coordinate on call; if two people edit simultaneously, merge manually — it's a small file.

---

## W4 coordination note

W4 analysis and reporting (issue #6) starts after the orchestration runners produce real metrics.

**Inputs W4 consumes:**

- `runs/*/metrics.json` — schema published by W0; `METHOD_KEYS` now include `null_max` (may be JSON `null` when a method cannot fill it)
- `runs/index.csv` — run registry
- Comparison table from `scripts/compare.py` → `trajot/results/tables/comparison.csv` (also prints markdown)

**Dependencies:**

- `METHOD_KEYS` including `null_max` (Agent C eval)
- `tau_phi` uncertainty column once entropy is wired (Agent A) — model-only; baselines report `null`
- Track B statistic: non-identifiable pairs via permutation null — `nonidentifiable_count(gains, null, alpha=0.05)` (Agent C)
- `compare.py` produces the method × metric comparison table

Coordinate status changes in this file; do not invent analysis numbers before runners land on real data.

---

## Run log

| Date | Developer | What | Status | Notes |
|---|---|---|---|---|
| 2026-09-20 | team | PR #7 W1 IO/geometry foundation | merged | Preprocess/contract/geometry code; real 120-subject npz still local |
| 2026-09-20 | team | PR #8 W0 scaffolding | merged | Config, runlog, paths.example, experiment configs |
| 2026-09-20 | team | PR #9 W3 baselines + eval | merged | Skeleton baselines + eval core; several baselines placeholder-only |
| 2026-09-20 | team | PR #10 W2 model + inference | merged | train/encoder/entropy/beta/synthetic; audit fixes in flight |
| 2026-09-20 | coord | TRACKING.md coordination revival | in progress | Runtime paths authoritative; results/ = staging; phase gates ticked |
| 2026-09-20 | A/B/C/D | Audit-fix wave | in progress | D1–D10: MPS float64, real baselines, corrected eval formulas, runners, compare.py |
| 2026-09-20 | Agent D | Orchestration + coordination | done on disk | TRACKING paths/gates, results/README, shards comments, gh #2/#4/#5/#6; runners wired to B/C APIs; compare.py live; `tests/scripts` + `test_entrypoint` 42 passed. Remaining suite failures live in baselines/eval/config owned by A/B/C (entropy config key, brainsync/conn_srm unit tests). Full `pytest -q` still long-running on inference/baseline suites — re-check after A/B/C land fixes. |
| 2026-09-20 | Integration | Audit-fix wave verification | done on disk | **pytest (pre-W4-merge): 298 passed, 2 skipped** (10:39 wall; long train/entropy OK). Entropy: wired pathwise Shannon(pi)+detached projected Hutchinson/SLQ logdet (`model.entropy.weight`, estimator `shannon_pi+hutchinson_slq`); training calls `projected_sinkhorn_logdet` with `subspace_dim=4`, `n_iter=max(10,min(L,20))` — nonzero finite entropy on short synthetic fits. MPS: encoder `sample_torch` promotes via `.cpu().to(dtype=float64)`; no combined `.to(device=..., dtype=float64)`. β: V:=R connectome nodes; `n_regions` in beta.json; formula unchanged. Folds: `default_rng.choice` **without replacement** (`eval.folds.FOLD_PROCEDURE`). METHOD_KEYS includes `null_max`. Gauge/band docs: spatial gradient / band prior — no temporal-velocity claims. evaluate.py `pair_gain` import intact (no NameError). Greps clean. |
| 2026-09-20 | Integration | W4 merge contract | in progress | Method keys frozen `ours_full`/`ours_ablated` (aliases ok). Baselines write `per_pair_flags=null` + `per_pair_uncertainty=null`. W4 `scripts/compare.py` primary (6×5 PLAN table); Agent D flat CSV → `scripts/compare_flat.py`. Delete 9 macOS `* 2.py` junk after merge. Import METHOD_KEYS from `trajot.eval.metrics` in report/table.py. Fold meta text must say without replacement. |
