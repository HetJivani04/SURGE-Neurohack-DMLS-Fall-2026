# TRACKING.md — Team Coordination Board

**Team coordination for SURGE Neurohack 2026. Deadline: Sunday Sept 20, 1:00 PM Atlantic. Update this file when you finish a step. Read it before you start.**

**Path convention:** all `results/`, `configs/`, and `scripts/` paths below are relative to the **`trajot/`** project root. `cd trajot` before running anything. The only file at the repository root is this one and `PLAN.md`.

---

## Current phase

**Phase 1 — Development (building the framework, testing on synthetic data)**

---

## Shard assignments

| Developer | Subjects | Status |
|---|---|---|
| dev_a |  | Not started |
| dev_b |  | Not started |
| dev_c |  | Not started |

> Populate subject IDs from participants.tsv after listing subjects — run: `ls data/ds000243/ | grep sub-`
> Then distribute evenly across developers and record the assignment in `configs/shards.yaml`.

---

## Phase gates

### Phase 1 — Development

- [ ] W0 scaffolding complete
- [ ] W1 preprocessing all 120 subjects npz'd
- [ ] W2 model + inference on synthetic data
- [ ] W3 baselines + evaluation on synthetic data
- [ ] W4 analysis + reporting tools
- [ ] All tests pass

### Phase 1→2 merge gates

- [ ] All npz files in `results/npz/`
- [ ] `results/manifest.parquet` merged and verified (203 rows)
- [ ] `results/runs_index.csv` has entries for synthetic test runs
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

**In git:**

- `results/npz/*.npz`
- `results/manifests/<developer>.parquet`
- `results/registry/<developer>.csv`
- `results/tables/`
- `configs/shards.yaml`
- `TRACKING.md`

**Not in git:**

- `data/` — raw, 5.67 GiB, each developer has it locally
- `runs/` — per-developer run directories
- `configs/paths.yaml` — per-developer local paths

### How to merge your shard

1. Finish preprocessing your subjects.
2. Copy your npz files to `results/npz/`.
3. Write your manifest rows to `results/manifests/<your-name>.parquet`.
4. Append your run rows to `results/registry/<your-name>.csv`.
5. Update this `TRACKING.md` — tick your checkbox, update your status.
6. Commit and push to branch `shard/<your-name>`.
7. On the call say "my shard is pushed".
8. Coordinator merges the branch to main.
9. Coordinator runs `scripts/merge_manifests.py` and `scripts/merge_runs.py`.
10. Everyone pulls main before the next phase.

---

## Conflict prevention

- Each developer only writes npz files for their own subjects — no file conflicts.
- Manifest rows go in per-developer files (`results/manifests/<name>.parquet`), merged by script — never hand-edit `results/manifest.parquet`.
- Run rows go in per-developer files (`results/registry/<name>.csv`), merged by script.
- `TRACKING.md`: coordinate on call; if two people edit simultaneously, merge manually — it's a small file.

---

## Run log

| Date | Developer | What | Status | Notes |
|---|---|---|---|---|
|  |  |  |  |  |
