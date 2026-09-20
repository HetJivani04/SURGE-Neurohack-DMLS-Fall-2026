# TRACKING.md — Team Coordination Board

**Team coordination for SURGE Neurohack 2026. Deadline: Sunday Sept 20, 1:00 PM Atlantic. Update this file when you finish a step. Read it before you start.**

**Path convention:** coordination paths under `results/`, `configs/`, and `scripts/` are relative to the **`trajot/`** project root. Runtime data paths are absolute via `configs/paths.yaml` (`data_root`). `cd trajot` before running scripts. The repository root holds this file, `PLAN.md`, and `docs/`.

---

| 2026-09-20 | Phase2 lead | Frozen pilot n=24 | in progress | Cohort 015–038, frozen beta 29.189, B=200 debug. Ours transform = region-level OT coupling to C_pop then orthogonal Procrustes (not identity; tests assert non-identity). BrainSync wired with region-level timeseries lists + common-T crop. Ablated gauge-off train produces tau_phi. |

## Current phase

**Phase 2 — Frozen N=49 primary comparison complete (pilot B=200). Official write-up: `trajot/reports/RESULTS.md`.**

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

**Status: complete for the Phase 2 pilot freeze (N=49).** Canonical write-up: `trajot/reports/RESULTS.md`. CLI table: `trajot/reports/results_table.md` (also `trajot/results/tables/results_table.md`).

**W4 frozen N=49 verdict (do not invent wins):**

| Method | ID acc | alignment_gain | per_pair_uncertainty | run_id |
|---|---|---|---|---|
| noalign | 0.959 | 0.0 | — | `00_noalign__eebd2f8e__20260920T074351Z` |
| brainsync | 0.959 | ≈0 | — | `01_brainsync__b4f31914__20260920T074433Z` |
| fugw | 0.918 | −0.005 | — | `02_fugw__cf2f95f4__20260920T074534Z` |
| conn_srm | 0.082 | +0.395 | — | `03_conn_srm__0777ce05__20260920T074652Z` |
| ours_ablated | 0.85 (N=33) | ~0 | **0.103** | `11_ours_ablated__56a17400__20260920T071555Z` |
| ours_full | 0.857 | **+0.018** | **0.075** | `10_ours_full__dd67ab1e__20260920T074747Z` |

**Track A/B narrative:**

1. **Track A ceiling diagnosis.** Raw Schaefer-100 connectomes identify subjects at 0.94–0.96 (self-corr ≈0.67 vs cross ≈0.46). Track A cannot discriminate alignment methods at this parcellation — no headroom. TrajOT does **not** win identification (0.857 vs 0.959 for noalign/brainsync).
2. **Track B guaranteed result.** Only the hierarchical population-of-couplings framework produces `per_pair_uncertainty` from `tau_phi` (0.075 full / 0.103 ablated). Every baseline reports an alignment for every subject pair with **no** uncertainty. That is the PLAN §7.5 contribution — calibrated uncertainty / identifiability statements, not a magic accuracy boost.
3. **alignment_gain tradeoff.** Methods that force cross-subject correlation (conn_srm +0.395) destroy individual identity (acc 0.082). Ours is the only method with positive gain without identity collapse (+0.018). FUGW slightly negative.
4. **Gauge ablation.** Point transforms nearly identical; gauge moves posterior width (`tau_phi` 0.075 vs 0.103) — uncertainty channel, not point accuracy.
5. **BrainSync.** Time-domain orthogonal Q leaves spatial connectome invariant (`XQQ^T X^T = XX^T`); feat_corr≈0.998. Not a failed code path — structural limitation when scoring post-hoc connectomes.
6. **nonident caveat.** When gain-null is degenerate, all pairs flag — not scientific uncertainty.
7. **Pilot limits.** CPU-only, short runs T≈130, B=200 not 10000, N=49 of 83 two-run, ablated row is N=33, entropy Jacobian log-det detached.

Coordinate status changes in this file; do not invent analysis numbers.

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
| 2026-09-20 | Phase2 lead | Env + preprocess smoke | done | `paths.yaml` data_root=`/Users/anandlo/Surge2026F/ds000243-master`; pot/nibabel/nilearn ok; contract npz validated (100×100, V=5124, T=130). Smoke sub-015/016/017 both runs green (qc_pass). SHA 45d79f8 |
| 2026-09-20 | Phase2 lead | Synthetic pollution fix | done | `evaluate.py` `_synthetic_via_contract` wrote into real data_root and overwrote manifest; sandboxed to `derivatives/trajot_synthetic` / `TRAJOT_SYNTHETIC_ROOT`. Removed sub-001–008 synthetic npz. SHA 6523ad5 |
| 2026-09-20 | Phase2 lead | Full two-run preprocess | in progress | Background PID on all 83 two-run IDs (015–068, 092–120), n_jobs=1 (RAM ~2.8 GiB free). Pilot gate: ≥20 complete pairs → beta + baselines + ours; then scale to 83 for final table. Chance=1/83. Method keys: `ours_full`/`ours_ablated`. |
| 2026-09-20 | Phase2 lead | Frozen pilot table | done | Cohort n=24 (015–038), beta=29.189, B=200. Ours transform = region OT coupling to C_pop + orthogonal Procrustes (not identity; tests green). BrainSync uses region timeseries with ragged-T crop. Model rows report tau_phi unc. compare.py 6x5 printed. Preprocess continues toward 83. |
| 2026-09-20 | Phase2 | N=49 frozen primary | done | Cohort 015–063, data_hash `ccce8212b978`, beta=29.189, B=200. Verdict: Track A ceiling (noalign 0.959); ours_full 0.857 with +0.018 gain + tau_phi 0.075 only. Official RESULTS.md + tables committed. 210MB `template_geometry.npz` gitignored — no force-push. Longer ours_* trainings still running on CPU (do not kill). |
