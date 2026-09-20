# results/ — staging area shared via git

**All paths in this file are relative to the `trajot/` project root.** Run commands from inside `trajot/`.

This directory is a **coordination staging area**, not the runtime data plane.

**Raw data stays in `data/`** (git-ignored). Each developer already has the 5.67 GiB ds000243 dataset on their own Mac; it is never pushed to GitHub.

**Runtime run directories stay in `runs/`** (git-ignored). `run_experiment.py` / `fit.py` write `runs/<run_id>/` and append to `runs/index.csv`. That live registry is **not** git-tracked.

Only small, mergeable staging outputs are committed here:

| Path | Contents | Written by | Git-tracked? |
|---|---|---|---|
| `npz/` | Compact subject outputs (`sub-<id>_run-<r>.npz`, ~40 KB each) | Each developer, for their shard subjects only | yes (optional) |
| `manifests/` | Per-developer manifest files (`<dev>.parquet` or `manifest_<name>.parquet`) | Each developer, their own rows only | yes |
| `registry/` | Per-developer run registry rows (`<dev>.csv` or `runs_<name>.csv`) | Each developer, their own rows only | yes |
| `tables/` | Comparison tables (`comparison.csv`) and figures | W4 / `scripts/compare.py` | yes |

Each developer writes only their own files to avoid conflicts.

## Registry: staging vs runtime

These are **two different things** — do not conflate them:

1. **Staging (git-tracked):** `results/registry/<dev>.csv` — per-developer run rows **before** merge. Safe to commit.
2. **Runtime (gitignored):** `trajot/runs/index.csv` — the live registry every script reads/writes. Never commit this file.

After merge, staged registry rows are combined and copied into the runtime path (see `TRACKING.md` MERGE COMMANDS).

## Merged outputs (generated, never hand-edited)

- `results/manifest.parquet` ← `scripts/merge_manifests.py` (reads `results/manifests/*.parquet`)
- `results/runs_index.csv` ← `scripts/merge_runs.py` (reads `results/registry/*.csv`)

Those merged staging files must then be synced into the **runtime** locations the code actually reads:

- `results/manifest.parquet` → `<data_root>/derivatives/trajot/manifest.parquet`
- staged npz → `<data_root>/derivatives/trajot/`
- `results/runs_index.csv` → `trajot/runs/index.csv`

`evaluate.py` and `train.load_train_data` read the manifest from `cfg.data_root/derivatives/trajot/manifest.parquet`. Staging alone does not feed them.

See `TRACKING.md` for the full merge protocol and `PLAN.md` §11 for the team coordination model.
