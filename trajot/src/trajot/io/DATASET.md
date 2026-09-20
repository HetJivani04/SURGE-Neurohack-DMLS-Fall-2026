# ds000243: dataset facts and the W1 preprocessing record

Every number below was measured on the real files in `data/ds000243`, not copied from the plan.

## Source and download

```bash
# from trajot/; the aws CLI is `pip install awscli`, and the bucket allows anonymous access
aws s3 sync --no-sign-request s3://openneuro.org/ds000243 ./data/ds000243
```

The download is 8.7 GB on disk: **5.67 GiB of raw NIfTI** (4.37 GiB functional, 1.30 GiB T1w) plus the BIDS
metadata and `derivatives/mriqc`. The dataset is CC0. Set `data_root` in `configs/paths.yaml` to this folder.
A `datalad install` clone is not enough on its own: its NIfTI files are git-annex placeholders (broken symlinks)
until `datalad get` fetches them.

## Verified counts

| Fact | Value |
|---|---|
| Subjects | 120 |
| Rest runs | 203 (`sub-*/func/*_task-rest_run-*_bold.nii.gz`) |
| Two-run subjects | 83, and **all 83 have equal `n_volumes` in their two runs** |
| One-run subjects | 37 |
| Grid, voxel size | 64 x 64 x 32, 4 x 4 x 4 mm, in all 203 runs |
| TR, TE | 2.5 s, 27 ms |
| Volumes per run | median 132, range 130 to 724: 130 x 28, 132 x 102, 133 x 36, 184 x 1, 240 x 10, 360 x 15, 480 x 5, 724 x 6 |
| Raw NIfTI | 5.67 GiB |

The command that produced them (from `trajot/`; it reads only the NIfTI headers):

```python
from pathlib import Path
from collections import Counter
from trajot.io.dataset import list_bold_runs

specs = list_bold_runs(Path("data/ds000243"))
per_subject = Counter(s.subject_id for s in specs)
print(len(per_subject), len(specs))                                   # 120 203
print(sum(n == 2 for n in per_subject.values()), sum(n == 1 for n in per_subject.values()))  # 83 37
print(sorted(Counter(s.n_volumes for s in specs).items()))            # the run-length distribution
```

`tests/io/test_real_dataset.py` asserts all of this whenever the data is present.

**Do not take run lengths from `derivatives/mriqc`.** Its `size_t` says 131 volumes for sub-050 and sub-067 run 2,
while the NIfTI headers say 132 (as for the first run of each), so counting from MRIQC wrongly suggests that only
81 of the 83 two-run subjects have equal lengths.

## Slice timing is not in the sidecar

`task-rest_bold.json` holds `RepetitionTime` and `EchoTime` but **no `SliceTiming`**, there are no per-run
sidecars, and the NIfTI headers carry no slice-order code. The README says only "interleaved ascending".
`read_bold_json` therefore treats `SliceTiming` as optional when asked (`require_slice_timing=False`; a present key
is still validated), and `preprocess.slice_order` in the config supplies the order. The value used is
`interleaved_odd_first`: slices z = 1, 3, ..., 31 are acquired first, then z = 0, 2, ..., 30, each slot lasting
TR / 32 (AFNI `alt+z2`, and the Siemens convention for an even slice count).

The order was checked against the data rather than assumed, because the two candidate interleavings differ by half
a TR and a wrong guess is worse than no correction. The global signal is shared across slices, so each slice's
phase lag against the whole-brain mean reveals when it was acquired:

```python
import glob, numpy as np, nibabel as nib
TR, delays = 2.5, []
for p in sorted(glob.glob("data/ds000243/sub-*/func/*_task-rest_*bold.nii.gz")):
    d = np.asarray(nib.load(p).dataobj, dtype=np.float32); T = d.shape[3]; m = d.mean(3)
    mask = m > 0.4 * np.percentile(m, 98)
    ok = [mask[:, :, z].sum() > 50 for z in range(32)]
    sl = np.stack([d[:, :, z][mask[:, :, z]].mean(0) if ok[z] else np.zeros(T) for z in range(32)])
    sl -= sl.mean(1, keepdims=True); t = np.linspace(-1, 1, T); X = np.stack([np.ones(T), t, t**2], 1)
    sl -= (X @ np.linalg.lstsq(X, sl.T, rcond=None)[0]).T                      # remove slow drift
    F = np.fft.rfft(sl * np.hanning(T), axis=1); f = np.fft.rfftfreq(T, TR); band = (f >= .02) & (f <= .12)
    cross = F[:, band] * np.conj(F.mean(0)[band])                              # phase = 2 pi f (lag of slice vs mean)
    lag = (np.angle(cross) / (2 * np.pi * f[band])).clip(-2, 2); w = np.abs(cross)
    lag = (lag * w).sum(1) / np.maximum(w.sum(1), 1e-9); lag[~np.array(ok)] = np.nan; delays.append(lag)
mu = np.nanmean(delays, 0)                                                      # seconds, per slice index
print(np.nanmean(mu[1::2]) - np.nanmean(mu[0::2]))                              # odd - even
```

Over all 203 runs, even-indexed slices lag odd-indexed ones: the odd-minus-even lag is **-0.80 s**, and the lag rises
with slice index inside each parity group. That is interleaved ascending with odd slices first (predicted -1.25 s;
the measurement is smaller because the shared signal is only partly coherent). An even-first order would give
+1.25 s and a sequential order about 0 s. If a sidecar does carry `SliceTiming`, it is used instead.

## What `preprocess_run` does

One run at a time; each large intermediate is freed with `del` and `gc.collect()` before the next is allocated.

1. **Read** the run once, sequentially, in its on-disk dtype (int16), converted to float32 block by block
   (`iter_bold_chunks`; the only place a BOLD file is opened).
2. **Slice timing**, per z slice in Fourier space, reference time 0, offsets from the order above.
3. **Rigid motion correction**, 6 degrees of freedom, every volume to volume 0 (inverse-compositional Gauss-Newton on
   one-voxel-smoothed images, trilinear resampling). The `(T, 6)` parameters (mm, radians) go to the QC record;
   framewise displacement (50 mm sphere) is derived from them.
4. **Registration**: the mean corrected EPI is registered directly to the MNI152 template with a 12-parameter affine
   that maximizes normalized mutual information (coarse rigid search from five head-pitch starts at 8 mm, then affine
   refinement at 4 mm). The subjects' T1w images are not used.
5. **Surface**: the fsaverage4 pial and white surfaces (2,562 vertices per hemisphere, **V = 5,124**, left first) are
   mapped into native space through the registration, and the run is sampled trilinearly at five depths between white
   and pial and averaged. A vertex is valid if all samples fall inside the field of view and its mid-thickness point
   lies inside the EPI brain mask; invalid vertices are all zeros.
6. **Denoising** (not FIX): the six motion parameters, their first differences, the mean signal of eroded white
   matter (MNI152 probability > 0.9, eroded once) and of the eroded Harvard-Oxford lateral ventricles (CSF), and the
   global signal (`preprocess.gsr`), regressed out in float64.
7. **Band-pass** 0.01 to 0.1 Hz, zero-phase second-order Butterworth.

Then `scripts/preprocess.py` labels each vertex with a Schaefer-2018 region (100 parcels, 7 networks; nearest
labelled voxel in the vertex's own hemisphere), averages vertices per region, and computes:

* `connectivity`: Fisher-z of the region correlation, `(100, 100)` float64, symmetric, zero diagonal
* `embedding`: `diffusion_map` of the vertex-level correlation matrix, `(V, 32)` float32
* `features`: the fsaverage curvature and sulcal depth, `(V, 2)` float32
* `coords`: mid-thickness vertex coordinates in template space, `(V, 3)` float32

## Files under `<data_root>/derivatives/trajot/`

| File | Contents |
|---|---|
| `sub-<id>_run-<r>.npz` | the frozen contract (`trajot.io.contract`, schema 1.0.0); written last, so a killed run leaves no contract file |
| `sub-<id>_run-<r>_geometry.npz` | `A` `(100, 32)` float64 with `A A^T ~ C_s`, `retained_variance`, `mass` `(V,)` float64 (`mu_s`: uniform over valid vertices, sums to 1) |
| `sub-<id>_run-<r>_qc.json` | the QC record: slice-timing source, motion parameters and FD, registration metrics and matrix, denoising method, band-pass spectra, `qc_pass` |
| `template_geometry.npz` | `anatomical_cost` `(V, V)` float64 (`M^0`), `faces`, `coords`, `n_left`, `region_labels` |
| `manifest.parquet` | the contract index, one row per run |

`M^0` is `(V, V)` (surface geodesic distance on the fsaverage mesh, scaled to [0, 1], the two hemispheres
mutually at distance 1); the `(V, K)` block for the `K` template nodes is a column selection made by W2. Every
subject shares this one matrix, so it is written once instead of once per subject.

`qc_pass` is true when the mean framewise displacement is at most `preprocess.qc.max_mean_fd_mm` (0.5 mm), the
registration brain-mask Dice is at least `preprocess.qc.min_mask_dice` (0.75), and at least 20 white-matter and
20 CSF voxels were found. It is informational: every run stays in the manifest, so the 83 two-run subjects are
identifiable from the table alone whatever their QC.

## Limitations, stated plainly

* There are no per-subject surface reconstructions (that would need FreeSurfer's `recon-all`, hours per subject).
  All subjects are sampled on the same fsaverage mesh, so `features` (curvature, sulcal depth), `coords` and
  `M^0` are properties of the template and **identical across subjects**; only the signal differs.
* Registration is one affine EPI-to-MNI step with no nonlinear warp and no T1w. Expect residual misalignment of
  about a voxel (4 mm). fsaverage is aligned to MNI305 rather than exactly to MNI152.
* White matter and CSF come from template masks, not from each subject's segmentation.
* The slice order is inferred from the data and the Siemens convention, not read from a sidecar.
* With T = 130 to 133 volumes for most runs, each Fisher-z correlation has a standard error near 0.088.

## Running it

```bash
python scripts/preprocess.py --dry-run                      # list the runs
python scripts/preprocess.py --subjects 001,002 --runs 1    # some runs
python scripts/preprocess.py                                # all 203 runs, resumable; --force redoes finished runs
```

Each invocation is recorded under `runs/<run_id>/` (manifest, log, and `artifacts/bandpass_check.png`, the
band-pass check for one spot-checked run) and in `runs/index.csv`. Peak memory is about 1.3 GiB per process, and the
script refuses to start if less than 2 GiB per worker is available.
