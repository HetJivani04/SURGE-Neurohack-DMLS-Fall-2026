"""Minimal preprocessing of one subject-run at a time (ds000243 ships raw NIfTI, no derivatives).

Pipeline (``preprocess_run``): stream the run, slice-time correct, rigid motion correct,
register the mean image to the MNI152 template (12-parameter affine, normalized mutual
information), sample the fsaverage cortical surface, regress motion / white matter / CSF /
global signal, band-pass. Everything is float32 except the small regressions and the
optimizer, which run in float64. Each big intermediate is freed with ``del`` and
``gc.collect()`` before the next is allocated.

Volumes are flattened to ``(V, T)`` in C order over ``(X, Y, Z)``.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize
from scipy.signal import butter, filtfilt, welch

from trajot.io.contract import subject_run_path
from trajot.io.dataset import RunSpec, iter_bold_chunks, read_bold_geometry, read_bold_json

FD_RADIUS_MM = 50.0  # Power et al. 2012 framewise displacement: rotations on a 50 mm sphere
_RADIUS = 50.0  # mm-equivalent scaling of rotation / scale / shear parameters in the optimizer
_DEPTHS = np.linspace(0.0, 1.0, 5)  # samples from the white to the pial surface


# --------------------------------------------------------------------------- #
# Signal-processing steps
# --------------------------------------------------------------------------- #
def slice_time_correct(
    bold: np.ndarray,
    slice_timing: np.ndarray,
    tr: float,
    t0: float = 0.0,
) -> np.ndarray:
    """Fourier-domain slice timing correction along the time axis.

    ``slice_timing`` holds per-slice acquisition offsets in seconds. Row ``i`` of ``bold``
    is assigned offset ``slice_timing[i % len(slice_timing)]``; ``preprocess_run`` calls this
    once per slice with a single offset, so no voxel layout is assumed. Each row is delayed
    by ``offset - t0`` seconds, which is exact for band-limited periodic signals.
    Input and output ``(V, T)`` float32.
    """

    x = np.asarray(bold, dtype=np.float32)
    offsets = np.asarray(slice_timing, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"bold must be (V, T), found {x.shape}")
    if offsets.ndim != 1:
        raise ValueError(f"slice_timing must be 1D, found {offsets.shape}")
    if tr <= 0:
        raise ValueError("tr must be positive")

    V, T = x.shape
    freqs = np.fft.rfftfreq(T, d=tr)
    corrected = np.empty_like(x, dtype=np.float32)

    offset_per_vertex = offsets[np.arange(V) % offsets.size]
    unique_offsets = np.unique(offset_per_vertex)

    for offset in unique_offsets:
        idx = np.where(offset_per_vertex == offset)[0]
        phase = np.exp(-2j * np.pi * freqs * (offset - t0))
        xf = np.fft.rfft(x[idx].astype(np.float64), axis=1)
        yf = xf * phase[None, :]
        corrected[idx] = np.fft.irfft(yf, n=T, axis=1).astype(np.float32)

    return corrected


def slice_timing_from_order(order: str, n_slices: int, tr: float) -> np.ndarray:
    """Per-slice acquisition offsets (seconds, by array z index) for a documented slice order.

    ``interleaved_odd_first`` acquires z = 1, 3, ..., then 0, 2, ... (AFNI ``alt+z2``, the
    Siemens convention for an even slice count); ``interleaved_even_first`` is AFNI
    ``alt+z``; ``sequential_ascending`` is 0, 1, 2, .... Slices are equally spaced
    over the TR.
    """

    if order == "interleaved_odd_first":
        acquired = list(range(1, n_slices, 2)) + list(range(0, n_slices, 2))
    elif order == "interleaved_even_first":
        acquired = list(range(0, n_slices, 2)) + list(range(1, n_slices, 2))
    elif order == "sequential_ascending":
        acquired = list(range(n_slices))
    else:
        raise ValueError(f"unknown slice_order {order!r}")
    times = np.empty(n_slices, dtype=np.float64)
    times[acquired] = np.arange(n_slices) * (tr / n_slices)
    return times


def _rigid_matrix(q: np.ndarray) -> np.ndarray:
    """4x4 rigid transform from ``(tx, ty, tz, rx, ry, rz)``; ``R = Rz @ Ry @ Rx``."""

    tx, ty, tz, rx, ry, rz = (float(v) for v in q)
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    matrix = np.eye(4)
    matrix[:3, :3] = rot_z @ rot_y @ rot_x
    matrix[:3, 3] = (tx, ty, tz)
    return matrix


def _rigid_params(matrix: np.ndarray) -> np.ndarray:
    rot = matrix[:3, :3]
    return np.array([
        *matrix[:3, 3],
        np.arctan2(rot[2, 1], rot[2, 2]),
        -np.arcsin(np.clip(rot[2, 0], -1.0, 1.0)),
        np.arctan2(rot[1, 0], rot[0, 0]),
    ])


def motion_correct(
    bold: np.ndarray,
    ref_volume: int = 0,
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Rigid-body (6 degrees of freedom) registration of every volume to ``ref_volume``.

    ``bold`` is ``(X, Y, Z, T)``. Each volume is aligned by inverse-compositional
    Gauss-Newton on the sum of squared differences of images smoothed by one voxel (the Jacobian is
    built once from the reference gradient), warm-started from the previous volume, and
    resampled by trilinear interpolation.

    Returns the corrected data as ``(V, T)`` float32 (C order over ``X, Y, Z``) and the
    ``(T, 6)`` float64 parameters of the transform that maps reference to moving space,
    about the volume centre: three translations in mm (``voxel_size`` scales voxels to mm)
    then three rotations in radians. Framewise displacement is derived from these
    parameters, not from the data.
    """

    x = np.asarray(bold)
    if x.ndim != 4:
        raise ValueError(f"bold must be (X, Y, Z, T); a flattened (V, T) array has no geometry, found {x.shape}")
    X, Y, Z, T = x.shape
    if not (0 <= ref_volume < T):
        raise ValueError(f"ref_volume {ref_volume} out of range for T={T}")

    vs = np.asarray(voxel_size, dtype=np.float64)
    sigma = 1.0  # voxels (4 mm on ds000243)
    center = (np.array([X, Y, Z], dtype=np.float64) - 1.0) / 2.0

    ref = ndi.gaussian_filter(x[..., ref_volume].astype(np.float64), sigma)
    inside = ref > 0.25 * np.percentile(ref, 98)
    p = (np.argwhere(inside) - center) * vs  # (N, 3) mm, relative to the centre
    grad = np.stack(np.gradient(ref, *vs), axis=-1)[inside]
    gx, gy, gz = grad.T
    px, py, pz = p.T
    jac = np.column_stack([gx, gy, gz, -gy * pz + gz * py, gx * pz - gz * px, -gx * py + gy * px])
    pinv = np.linalg.solve(jac.T @ jac, jac.T)  # (6, N), computed once
    ref_vals = ref[inside]

    all_p = (np.indices((X, Y, Z), dtype=np.float64).reshape(3, -1).T - center) * vs
    corrected = np.empty((X * Y * Z, T), dtype=np.float32)
    params = np.zeros((T, 6), dtype=np.float64)
    matrix = np.eye(4)

    for t in range(T):
        vol = x[..., t].astype(np.float64)
        if t == ref_volume:
            corrected[:, t] = vol.reshape(-1)
            continue
        smoothed = ndi.gaussian_filter(vol, sigma)
        for _ in range(15):
            moving = (p @ matrix[:3, :3].T + matrix[:3, 3]) / vs + center
            values = ndi.map_coordinates(smoothed, moving.T, order=1, mode="nearest")
            delta = pinv @ (values - ref_vals)
            matrix = matrix @ np.linalg.inv(_rigid_matrix(delta))
            if np.abs(delta[:3]).max() < 5e-3 and np.abs(delta[3:]).max() < 5e-5:
                break
        params[t] = _rigid_params(matrix)
        moving = (all_p @ matrix[:3, :3].T + matrix[:3, 3]) / vs + center
        corrected[:, t] = ndi.map_coordinates(vol, moving.T, order=1, mode="nearest").astype(np.float32)

    return corrected, params


def framewise_displacement(motion_params: np.ndarray) -> np.ndarray:
    """Framewise displacement (mm) from ``(T, 6)`` motion parameters; the first frame is 0."""

    diff = np.abs(np.diff(np.asarray(motion_params, dtype=np.float64), axis=0))
    fd = diff[:, :3].sum(axis=1) + FD_RADIUS_MM * diff[:, 3:].sum(axis=1)
    return np.concatenate([[0.0], fd])


def bandpass(
    bold: np.ndarray,
    tr: float,
    low: float = 0.01,
    high: float = 0.1,
    order: int = 2,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter along the time axis."""

    x = np.asarray(bold, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"bold must be (V, T), found {x.shape}")
    if tr <= 0:
        raise ValueError("tr must be positive")

    low_norm = low * 2.0 * tr
    high_norm = high * 2.0 * tr
    if not (0.0 < low_norm < high_norm < 1.0):
        raise ValueError(
            f"Invalid band edges after normalization: low={low_norm}, high={high_norm}"
        )

    b, a = butter(order, [low_norm, high_norm], btype="band")

    # filtfilt needs enough samples for padding. For very short signals we return
    # the input unchanged instead of producing edge-dominated artifacts.
    if x.shape[1] <= 3 * max(len(a), len(b)):
        return x.copy()

    y = filtfilt(b, a, x.astype(np.float64), axis=1)
    return y.astype(np.float32)


def denoise(timeseries: np.ndarray, confounds: np.ndarray, gsr: bool = True) -> np.ndarray:
    """Regress confounds out of the time series, in float64, and return float32.

    ``confounds`` is ``(T, C)``: the motion parameters, their temporal derivatives, and the
    white-matter and CSF means. With ``gsr=True`` the global signal (the mean over the rows
    of ``timeseries`` that are not identically zero) is regressed out as well.
    """

    x = np.asarray(timeseries, dtype=np.float64)
    c = np.asarray(confounds, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError(f"timeseries must be (V, T), found {x.shape}")
    if c.ndim != 2:
        raise ValueError(f"confounds must be (T, C), found {c.shape}")
    if c.shape[0] != x.shape[1]:
        raise ValueError(
            f"confounds first axis must match T={x.shape[1]}, found {c.shape[0]}"
        )

    columns = [c]
    if gsr:
        live = x.std(axis=1) > 0
        columns.append(x[live].mean(axis=0)[:, None] if live.any() else np.zeros((x.shape[1], 1)))
    design = np.column_stack([*columns, np.ones(x.shape[1], dtype=np.float64)])
    beta, *_ = np.linalg.lstsq(design, x.T, rcond=None)
    return (x.T - design @ beta).T.astype(np.float32)


# --------------------------------------------------------------------------- #
# Templates: fsaverage surface, atlas, MNI152 tissue masks
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class SurfaceTemplate:
    """An fsaverage cortical surface (left hemisphere first) with its per-vertex anatomy."""

    name: str
    n_left: int
    pial: np.ndarray  # (V, 3) float64, MNI mm
    white: np.ndarray  # (V, 3) float64, MNI mm
    faces: np.ndarray  # (F, 3) int64; right-hemisphere indices are offset by n_left
    features: np.ndarray  # (V, 2) float32: curvature, sulcal depth
    labels: np.ndarray  # (V,) int64: atlas region 0..R-1 of each vertex
    n_regions: int

    @property
    def coords(self) -> np.ndarray:
        """Mid-thickness vertex coordinates ``(V, 3)`` float32."""
        return (0.5 * (self.pial + self.white)).astype(np.float32)


@lru_cache(maxsize=2)
def load_surface_template(name: str = "fsaverage4", n_regions: int = 100) -> SurfaceTemplate:
    """Load an fsaverage mesh with its real curvature and sulcal depth (nilearn) and label each
    vertex with a Schaefer-2018 region (``n_regions`` parcels, 7 networks) by nearest-labelled
    voxel in its own hemisphere."""

    import nibabel as nib
    from nilearn import datasets, surface

    fs = datasets.fetch_surf_fsaverage(name)

    def per_hemi(kind: str) -> list[Any]:
        return [surface.load_surf_mesh(fs[f"{kind}_{hemi}"]) for hemi in ("left", "right")]

    pial, white = per_hemi("pial"), per_hemi("white")
    n_left = pial[0].coordinates.shape[0]
    faces = np.concatenate([pial[0].faces, pial[1].faces + n_left]).astype(np.int64)
    pial_xyz = np.concatenate([m.coordinates for m in pial]).astype(np.float64)
    white_xyz = np.concatenate([m.coordinates for m in white]).astype(np.float64)

    def anatomy(kind: str) -> np.ndarray:
        return np.concatenate([surface.load_surf_data(fs[f"{kind}_{h}"]) for h in ("left", "right")])

    features = np.column_stack([anatomy("curv"), anatomy("sulc")]).astype(np.float32)

    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=n_regions, yeo_networks=7, resolution_mm=2)
    image = nib.load(atlas["maps"])
    volume = np.asarray(image.dataobj).astype(np.int64)
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in atlas["labels"]]
    hemisphere_of_label = {i: ("L" if "_LH_" in name else "R") for i, name in enumerate(names) if i > 0}

    mid = 0.5 * (pial_xyz + white_xyz)
    ijk = np.rint(nib.affines.apply_affine(np.linalg.inv(image.affine), mid)).astype(int)
    ijk = np.clip(ijk, 0, np.array(volume.shape) - 1)
    labels = np.zeros(mid.shape[0], dtype=np.int64)
    for hemi, sl in (("L", slice(0, n_left)), ("R", slice(n_left, None))):
        keep = [i for i, h in hemisphere_of_label.items() if h == hemi]
        own = np.where(np.isin(volume, keep), volume, 0)
        nearest = ndi.distance_transform_edt(own == 0, return_indices=True)[1]
        filled = own[tuple(nearest)]
        labels[sl] = filled[tuple(ijk[sl].T)] - 1

    if set(labels.tolist()) != set(range(n_regions)):
        raise RuntimeError(f"{name}: atlas labels cover {len(set(labels.tolist()))} of {n_regions} regions")
    return SurfaceTemplate(name, n_left, pial_xyz, white_xyz, faces, features, labels, n_regions)


@dataclass(frozen=True, eq=False)
class _MNI:
    affine: np.ndarray  # voxel -> MNI mm of the 2 mm template grid
    template: np.ndarray  # smoothed T1 template
    brain: np.ndarray  # bool
    wm_core: np.ndarray  # bool: MNI152 white-matter probability > 0.9, eroded once (template grid)
    csf_core: np.ndarray  # bool: Harvard-Oxford lateral ventricles, eroded once (its own grid)
    csf_affine: np.ndarray  # voxel -> MNI mm of the Harvard-Oxford grid


@lru_cache(maxsize=1)
def _mni() -> _MNI:
    import nibabel as nib
    from nilearn import datasets

    template = datasets.load_mni152_template(resolution=2)
    brain = np.asarray(datasets.load_mni152_brain_mask(resolution=2).dataobj) > 0
    wm_probability = np.asarray(datasets.load_mni152_wm_template(resolution=2).dataobj, dtype=np.float64)

    atlas = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    atlas_img = atlas.maps if hasattr(atlas.maps, "dataobj") else nib.load(atlas.maps)
    ventricles = [i for i, name in enumerate(atlas.labels) if "Lateral Ventricle" in name]
    return _MNI(
        affine=np.asarray(template.affine, dtype=np.float64),
        template=ndi.gaussian_filter(np.asarray(template.dataobj, dtype=np.float64), 1.0),
        brain=brain,
        wm_core=ndi.binary_erosion(wm_probability > 0.9, iterations=1),
        csf_core=ndi.binary_erosion(np.isin(np.asarray(atlas_img.dataobj), ventricles), iterations=1),
        csf_affine=np.asarray(atlas_img.affine, dtype=np.float64),
    )


# --------------------------------------------------------------------------- #
# Registration of the mean EPI to the MNI152 template
# --------------------------------------------------------------------------- #
def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _nmi(a: np.ndarray, b: np.ndarray, ranges: tuple[tuple[float, float], tuple[float, float]]) -> float:
    """Studholme normalized mutual information (H(a) + H(b)) / H(a, b) on a 32 x 32 joint histogram."""

    hist, _, _ = np.histogram2d(np.clip(a, *ranges[0]), np.clip(b, *ranges[1]), bins=32, range=ranges)
    joint = hist / max(hist.sum(), 1.0)
    pa, pb = joint.sum(1), joint.sum(0)

    def entropy(p: np.ndarray) -> float:
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    h_joint = entropy(joint.ravel())
    return (entropy(pa) + entropy(pb)) / h_joint if h_joint > 0 else 1.0


def _affine_from_params(p: np.ndarray, center: np.ndarray) -> np.ndarray:
    """12 parameters (mm, and mm-equivalents on a 50 mm sphere for angles / scales / shears)
    -> 4x4 affine mapping template mm to EPI world mm, acting about ``center``."""

    rot = _rigid_matrix(np.array([0, 0, 0, *(p[3:6] / _RADIUS)]))[:3, :3]
    s, k = 1.0 + p[6:9] / _RADIUS, p[9:12] / _RADIUS
    linear = rot @ np.array([[s[0], k[0], k[1]], [0, s[1], k[2]], [0, 0, s[2]]])
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    matrix[:3, 3] = center + p[:3] - linear @ center
    return matrix


@lru_cache(maxsize=4)
def _template_grid(spacing: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regular MNI grid over the brain's bounding box: ``(points mm, template values, in-brain flags)``.

    Cached, so callers must not modify the returned arrays."""

    mni = _mni()
    world = _apply(mni.affine, np.argwhere(mni.brain).astype(np.float64))
    axes = [np.arange(lo, hi + 1e-6, spacing) for lo, hi in zip(world.min(0), world.max(0))]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    voxels = _apply(np.linalg.inv(mni.affine), points)
    values = ndi.map_coordinates(ndi.gaussian_filter(mni.template, spacing / 4.0), voxels.T, order=1)
    near_brain = ndi.map_coordinates(ndi.binary_dilation(mni.brain, iterations=2).astype(np.uint8),
                                     voxels.T, order=0) > 0
    return points, values, near_brain


def _register_to_template(mean_epi: np.ndarray, epi_affine: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Affine (12 degrees of freedom) registration of the mean EPI to the MNI152 template.

    Maximizes normalized mutual information (which tolerates the T2* / T1 contrast reversal):
    a coarse rigid search from five head-pitch starts at 8 mm, then a full affine refinement
    at 4 mm. Returns the 4x4 matrix mapping MNI mm to EPI world mm, the EPI brain mask, and
    QC metrics.
    """

    import nibabel as nib
    from nilearn.masking import compute_epi_mask

    epi_mask = np.asarray(compute_epi_mask(nib.Nifti1Image(mean_epi.astype(np.float32), epi_affine)).dataobj) > 0
    epi_s = ndi.gaussian_filter(mean_epi.astype(np.float64), 1.0)
    inv_epi = np.linalg.inv(epi_affine)
    dims = np.array(epi_s.shape) - 1.0
    epi_range = tuple(float(v) for v in np.percentile(epi_s[epi_mask], [2, 98]))

    tpl_range: tuple[float, float] = (0.0, 1.0)  # set below, once the template values are known

    def cost(p: np.ndarray, level: tuple[np.ndarray, np.ndarray], center: np.ndarray, rigid: bool) -> float:
        points, values = level
        full = np.concatenate([p, np.zeros(6)]) if rigid else p
        voxels = _apply(inv_epi, _apply(_affine_from_params(full, center), points))
        inside = ((voxels >= 0) & (voxels <= dims)).all(axis=1)
        if inside.mean() < 0.6:
            return -1.0
        sampled = ndi.map_coordinates(epi_s, voxels[inside].T, order=1)
        return -_nmi(values[inside], sampled, (tpl_range, epi_range))

    levels = {}
    for spacing in (8.0, 4.0):
        points, values, near = _template_grid(spacing)
        levels[spacing] = (points[near], values[near])
    all_points, _, brain_flag = _template_grid(4.0)
    brain_flag = brain_flag & (ndi.map_coordinates(
        _mni().brain.astype(np.uint8), _apply(np.linalg.inv(_mni().affine), all_points).T, order=0) > 0)
    tpl_range = (float(np.percentile(levels[4.0][1], 2)), float(np.percentile(levels[4.0][1], 98)))

    center = levels[4.0][0].mean(axis=0)
    epi_center = _apply(epi_affine, np.argwhere(epi_mask).mean(axis=0))
    shift = epi_center - center

    n_evals = 0
    best_p, best_cost = None, np.inf
    for pitch in (-30.0, -15.0, 0.0, 15.0, 30.0):
        start = np.array([*shift, np.deg2rad(pitch) * _RADIUS, 0.0, 0.0])
        result = minimize(cost, start, args=(levels[8.0], center, True), method="Powell",
                          bounds=[(s - 40, s + 40) for s in shift] + [(-40, 40)] * 3,
                          options={"xtol": 1e-2, "ftol": 1e-4, "maxfev": 800})
        n_evals += result.nfev
        if result.fun < best_cost:
            best_p, best_cost = result.x, result.fun

    start12 = np.concatenate([best_p, np.zeros(6)])
    bounds = [(s - 40, s + 40) for s in shift] + [(-45, 45)] * 3 + [(-12, 12)] * 3 + [(-8, 8)] * 3
    initial = -cost(np.concatenate([shift, np.zeros(9)]), levels[4.0], center, False)
    result = minimize(cost, start12, args=(levels[4.0], center, False), method="Powell", bounds=bounds,
                      options={"xtol": 1e-2, "ftol": 1e-5, "maxfev": 2500})
    n_evals += result.nfev
    matrix = _affine_from_params(result.x, center)

    voxels = _apply(inv_epi, _apply(matrix, all_points))
    inside = ((voxels >= 0) & (voxels <= dims)).all(axis=1)
    covered = np.zeros(all_points.shape[0], dtype=bool)
    covered[inside] = ndi.map_coordinates(epi_mask.astype(np.uint8), voxels[inside].T, order=0) > 0
    dice = 2.0 * (covered & brain_flag).sum() / max(int(covered.sum()) + int(brain_flag.sum()), 1)
    info = {"method": "affine 12-DOF, normalized mutual information, EPI -> MNI152 (nilearn), no T1w",
            "nmi_initial": float(initial), "nmi_final": float(-result.fun), "mask_dice": float(dice),
            "n_evaluations": int(n_evals), "mni_to_epi_world": matrix.tolist()}
    return matrix, epi_mask, info


def _native_tissue_masks(shape3: tuple[int, int, int], epi_affine: np.ndarray, to_epi: np.ndarray,
                         epi_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """White-matter and CSF (lateral ventricle) cores resampled into native EPI voxels (nearest
    neighbour) and restricted to the EPI brain mask."""

    mni = _mni()
    ijk = np.indices(shape3, dtype=np.float64).reshape(3, -1).T
    mni_mm = _apply(np.linalg.inv(to_epi), _apply(epi_affine, ijk))

    def sample(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
        voxels = _apply(np.linalg.inv(affine), mni_mm)
        return (ndi.map_coordinates(mask.astype(np.uint8), voxels.T, order=0) > 0).reshape(shape3)

    return sample(mni.wm_core, mni.affine) & epi_mask, sample(mni.csf_core, mni.csf_affine) & epi_mask


# --------------------------------------------------------------------------- #
# Sampling the cortical surface
# --------------------------------------------------------------------------- #
def _sample_surface(data_vt: np.ndarray, shape3: tuple[int, int, int], epi_affine: np.ndarray, to_epi: np.ndarray,
                    surface: SurfaceTemplate, epi_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trilinear sampling of ``(V, T)`` native data at ``len(_DEPTHS)`` depths between the white
    and pial surfaces, averaged over depth. Returns ``(n_vertices, T)`` float32 and a validity
    mask (all samples inside the field of view, mid-thickness point inside the EPI brain mask)."""

    X, Y, Z = shape3
    dims = np.array(shape3)
    inv_epi = np.linalg.inv(epi_affine)
    out = np.zeros((surface.pial.shape[0], data_vt.shape[1]), dtype=np.float32)
    valid = np.ones(surface.pial.shape[0], dtype=bool)

    for depth in _DEPTHS:
        points = surface.white + depth * (surface.pial - surface.white)
        vox = _apply(inv_epi, _apply(to_epi, points))
        valid &= ((vox >= 0) & (vox <= dims - 1)).all(axis=1)
        base = np.clip(np.floor(vox).astype(np.int64), 0, dims - 2)
        frac = np.clip(vox - base, 0.0, 1.0)
        for corner in np.ndindex(2, 2, 2):
            weight = np.prod(np.where(np.array(corner) == 1, frac, 1.0 - frac), axis=1).astype(np.float32)
            i, j, k = (base + np.array(corner)).T
            out += (weight / len(_DEPTHS))[:, None] * data_vt[(i * Y + j) * Z + k]

    mid = _apply(inv_epi, _apply(to_epi, 0.5 * (surface.pial + surface.white)))
    mid_idx = np.clip(np.rint(mid).astype(int), 0, dims - 1).T
    valid &= epi_mask[tuple(mid_idx)]
    out[~valid] = 0.0
    return out, valid


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def qc_path(root: Path, subject_id: str, run_id: str) -> Path:
    """``<root>/derivatives/trajot/sub-<id>_run-<r>_qc.json``: the run's QC record."""

    npz = subject_run_path(root, subject_id, run_id)
    return npz.with_name(npz.stem + "_qc.json")


def _band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    keep = (freqs >= low) & (freqs <= high)
    return float(psd[keep].sum())


def preprocess_run(spec: RunSpec, cfg: Any) -> tuple[np.ndarray, np.ndarray, float]:
    """Preprocess one subject-run; return ``(timeseries (V, T) float32, coords (V, 3) float32, tr)``.

    ``V`` is the number of fsaverage vertices (both hemispheres); vertices outside the field
    of view are all zeros. The run's QC record (slice-timing source, motion parameters and
    framewise displacement, registration metrics, denoising method, band-pass spectra,
    ``qc_pass``) is written next to the output npz (see ``qc_path``). Reads
    ``preprocess.*`` and ``data.root`` from ``cfg``.
    """

    started = time.time()
    stage: dict[str, float] = {}

    def lap(name: str) -> None:
        stage[name] = round(time.time() - started - sum(stage.values()), 2)

    surface = load_surface_template(cfg.get("preprocess.surface"), int(cfg.get("preprocess.n_regions")))
    shape, voxel_size, epi_affine = read_bold_geometry(spec)
    X, Y, Z, T = shape
    tr = float(spec.tr)

    sidecar = read_bold_json(spec.json_path, require_slice_timing=False)
    if "SliceTiming" in sidecar:
        slice_timing, timing_source = np.asarray(sidecar["SliceTiming"], dtype=np.float64), "sidecar"
    else:
        order = cfg.get("preprocess.slice_order")
        slice_timing, timing_source = slice_timing_from_order(order, Z, tr), f"derived:{order}"

    volume = np.empty((X * Y * Z, T), dtype=np.float32)
    start = 0
    for block in iter_bold_chunks(spec, chunk=4000):
        volume[start : start + block.shape[0]] = block
        start += block.shape[0]
    del block
    vol4 = volume.reshape(X, Y, Z, T)
    for z in range(Z):
        plane = np.ascontiguousarray(vol4[:, :, z, :]).reshape(-1, T)
        vol4[:, :, z, :] = slice_time_correct(plane, slice_timing[z : z + 1], tr).reshape(X, Y, T)
    del plane
    gc.collect()
    lap("read+slice_timing")

    corrected, motion = motion_correct(vol4, ref_volume=0, voxel_size=voxel_size)
    del volume, vol4
    gc.collect()
    lap("motion")

    mean_epi = corrected.mean(axis=1, dtype=np.float64).reshape(X, Y, Z)
    to_epi, epi_mask, registration = _register_to_template(mean_epi, epi_affine)
    wm, csf = _native_tissue_masks((X, Y, Z), epi_affine, to_epi, epi_mask)
    lap("registration")

    wm_signal = corrected[wm.reshape(-1)].mean(axis=0, dtype=np.float64) if wm.any() else np.zeros(T)
    csf_signal = corrected[csf.reshape(-1)].mean(axis=0, dtype=np.float64) if csf.any() else np.zeros(T)
    timeseries, valid = _sample_surface(corrected, (X, Y, Z), epi_affine, to_epi, surface, epi_mask)
    del corrected
    gc.collect()
    lap("surface")

    derivative = np.vstack([np.zeros((1, 6)), np.diff(motion, axis=0)])
    confounds = np.column_stack([motion, derivative, wm_signal, csf_signal])
    gsr = bool(cfg.get("preprocess.gsr"))
    denoised = denoise(timeseries, confounds, gsr=gsr)
    del timeseries, confounds
    gc.collect()

    low, high, order_ = (cfg.get(f"preprocess.bandpass.{k}") for k in ("low", "high", "order"))
    filtered = bandpass(denoised, tr=tr, low=float(low), high=float(high), order=int(order_))
    nperseg = min(T, 64)
    freqs, before = welch(denoised[valid], fs=1.0 / tr, nperseg=nperseg, axis=1)
    _, after = welch(filtered[valid], fs=1.0 / tr, nperseg=nperseg, axis=1)
    before, after = before.mean(axis=0), after.mean(axis=0)
    del denoised
    gc.collect()
    lap("denoise+bandpass")

    fd = framewise_displacement(motion)
    limits = {"max_mean_fd_mm": float(cfg.get("preprocess.qc.max_mean_fd_mm")),
              "min_mask_dice": float(cfg.get("preprocess.qc.min_mask_dice"))}
    failed = []
    if fd.mean() > limits["max_mean_fd_mm"]:
        failed.append(f"mean framewise displacement {fd.mean():.3f} mm > {limits['max_mean_fd_mm']}")
    if registration["mask_dice"] < limits["min_mask_dice"]:
        failed.append(f"registration mask Dice {registration['mask_dice']:.3f} < {limits['min_mask_dice']}")
    if wm.sum() < 20 or csf.sum() < 20:
        failed.append(f"too few tissue voxels (white matter {int(wm.sum())}, CSF {int(csf.sum())})")
    empty_regions = sorted(set(range(surface.n_regions)) - set(np.unique(surface.labels[valid]).tolist()))
    if empty_regions:
        failed.append(f"{len(empty_regions)} atlas regions outside the field of view: {empty_regions}")

    record = {
        "subject_id": spec.subject_id, "run_id": spec.run_id, "n_volumes": T, "tr": tr,
        "slice_timing": {"source": timing_source, "seconds_by_z": slice_timing.tolist()},
        "motion": {"parameters_mm_rad": motion.tolist(), "fd_mean_mm": float(fd.mean()),
                   "fd_max_mm": float(fd.max()), "n_frames_fd_over_0p5mm": int((fd > 0.5).sum())},
        "registration": registration,
        "denoise": {"method": "motion(6)+derivatives(6)+white matter(MNI152 p>0.9, eroded)+CSF(lateral ventricles, eroded)" + ("+global signal" if gsr else ""),
                    "n_white_matter_voxels": int(wm.sum()), "n_csf_voxels": int(csf.sum()), "gsr": gsr},
        "surface": {"name": surface.name, "n_vertices": int(surface.pial.shape[0]), "n_valid_vertices": int(valid.sum()),
                    "empty_regions": empty_regions},
        "bandpass": {"low": float(low), "high": float(high), "order": int(order_), "freqs_hz": freqs.tolist(),
                     "psd_before": before.tolist(), "psd_after": after.tolist(),
                     "in_band_power_ratio": _band_power(freqs, after, low, high) / max(_band_power(freqs, before, low, high), 1e-30),
                     "out_of_band_power_ratio": _band_power(freqs, after, 1.5 * high, 1.0 / (2 * tr)) / max(_band_power(freqs, before, 1.5 * high, 1.0 / (2 * tr)), 1e-30)},
        "qc_limits": limits, "qc_pass": not failed, "qc_failed": failed,
    }
    path = qc_path(Path(cfg.data_root), spec.subject_id, spec.run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    print(f"sub-{spec.subject_id} run-{spec.run_id}: {T} volumes, {time.time() - started:.1f}s "
          f"({', '.join(f'{k} {v}s' for k, v in stage.items())}), qc_pass={not failed}")

    return filtered, surface.coords, tr
