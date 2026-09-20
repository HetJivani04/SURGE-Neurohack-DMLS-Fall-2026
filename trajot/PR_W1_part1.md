# W1: Freeze Data Contract + IO/Geometry Foundation (Part of #1)

## Summary
This PR implements the W1 foundation under the `trajot/` workspace:
- Frozen NPZ data contract with strict validation and manifest helpers.
- Dataset discovery + streaming primitives.
- Preprocessing pipeline scaffolding with slice timing, motion correction, denoising, band-pass, and memory discipline (`del` + `gc.collect()`).
- Geometry modules for connectivity, diffusion, gauge features, and cost matrices.
- Preprocess entry script and fixture artifacts for downstream workstreams.
- Test suite expansion for IO, preprocessing, geometry, and script wiring.

Linked issue: #1 (Part of #1, W1 scope)

## Scope and Structure
All additions are inside `trajot/`:
- `trajot/src/trajot/io/`
- `trajot/src/trajot/geometry/`
- `trajot/scripts/preprocess.py`
- `trajot/tests/`

## What was added
### IO contract and manifest
- `NPZ_SCHEMA`, `SCHEMA_VERSION`, `ContractError`.
- `subject_run_path`, `write_subject_run`, `read_subject_run`, `validate_subject_run`.
- `write_manifest`, `read_manifest`, `two_run_subjects`.
- `load_connectomes`, `load_embeddings`.

### Dataset discovery and streaming
- `RunSpec` dataclass.
- `list_bold_runs` (BIDS parsing + NIfTI/JSON metadata read).
- `read_bold_json` (required keys validation).
- `iter_bold_chunks` streaming blocks `(V_chunk, T)` as float32.

### Preprocessing module
- `slice_time_correct` Fourier-domain shift.
- `motion_correct` with volumetric translation alignment path and `(T, 6)` motion params.
- `bandpass` zero-phase Butterworth.
- `denoise` confound regression in float64 arithmetic.
- `preprocess_run` orchestration with explicit intermediate cleanup.

### Geometry
- Connectivity: `parcellate`, `correlation_matrix`, `fisher_z`, `connectivity`, `rank_factorize`.
- Diffusion: `diffusion_map`, `laplacian_eigenvectors`, `gauge_velocity`, `gauge_features`.
- Cost: `geodesic_cost`, `anatomical_cost`, `vertex_mass`, `feature_cost`.

### Script
- `scripts/preprocess.py` with `--subjects`, `--runs`, `--n-jobs`, `--force`, `--dry-run`.
- Per-run writes + manifest rebuild.
- Stage-1 behavior forces `n_jobs=1` and includes a compatibility hook for `runlog.parallel.parallel_map` if present.

### Fixtures and docs
- Fixture NPZ: `trajot/src/trajot/io/fixtures/sub-fixture_run-1.npz`.
- Fixture manifest: `trajot/src/trajot/io/fixtures/derivatives/trajot/manifest.parquet`.
- Dataset notes: `trajot/src/trajot/io/DATASET.md`.

## Tests
Added/updated tests:
- `trajot/tests/io/test_contract.py`
- `trajot/tests/io/test_dataset.py`
- `trajot/tests/io/test_preprocess.py`
- `trajot/tests/geometry/test_connectivity.py`
- `trajot/tests/geometry/test_diffusion.py`
- `trajot/tests/geometry/test_cost.py`
- `trajot/tests/scripts/test_preprocess_script.py`

Current result:
- `pytest -q` => 24 passed

## Checklist vs Part of #1
### Completed
- Frozen NPZ contract + validating reader + manifest writer/reader.
- `two_run_subjects`, `load_connectomes`, `load_embeddings` interfaces.
- Dataset discovery/streaming interfaces.
- Geometry modules and required core tests (contract, diffusion map, gauge features, connectivity factorization).
- Fixture NPZ and fixture manifest published.
- Script CLI shape and resumable behavior (`--force` skip logic).

### Partially completed / Known deviations
- Preprocessing includes practical registration/confound proxies, not full production-grade fsaverage registration with WM/CSF tissue masks.
- Connectivity factorization is produced and written as sidecar `*_factor.npz` (not in the frozen NPZ schema keys).
- Full experiment-run acceptance checks from issue text (for example 203-row real-data manifest build, RSS budget run, band-pass plot artifact in runs, byte-for-byte 3-subject rerun proof) are not executed in this PR.

## Notes for reviewers
- This PR is intentionally scoped to W1 foundations and strict interfaces to unblock W2/W3 development.
- No changes were made in model/inference/baselines/eval/runlog ownership areas.
