from __future__ import annotations

import gc
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.ndimage import center_of_mass, shift as ndi_shift
from scipy.signal import butter, filtfilt

from .dataset import RunSpec, iter_bold_chunks, read_bold_json


def slice_time_correct(
    bold: np.ndarray,
    slice_timing: np.ndarray,
    tr: float,
    t0: float = 0.0,
) -> np.ndarray:
    """Fourier-domain slice timing correction along the time axis.

    The caller provides per-slice timing offsets from BIDS `SliceTiming`.
    Vertices are mapped to slice offsets by modulo indexing, which is a practical
    layout-agnostic default for flattened volumes.
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


def motion_correct(
    bold: np.ndarray,
    ref_volume: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Rigid-align each volume to a reference and return motion parameters.

    Accepted inputs:
    - (V, T): flattened time series. Spatial registration cannot be recovered
      from flattened data, so the signal is returned unchanged with zero motion.
    - (X, Y, Z, T): volumetric data. A translation-only rigid approximation is
      estimated by center-of-mass registration to the reference volume.

    Returns
    -------
    corrected:
        Flattened data with shape (V, T), float32.
    params:
        Motion parameters with shape (T, 6), float64. First three columns are
        translations, rotations are set to zero when not estimated.
    """

    x = np.asarray(bold, dtype=np.float32)

    if x.ndim == 2:
        if not (0 <= ref_volume < x.shape[1]):
            raise ValueError(f"ref_volume {ref_volume} out of range for T={x.shape[1]}")
        params = np.zeros((x.shape[1], 6), dtype=np.float64)
        return x.copy(), params

    if x.ndim != 4:
        raise ValueError(f"bold must be (V, T) or (X, Y, Z, T), found {x.shape}")
    if not (0 <= ref_volume < x.shape[3]):
        raise ValueError(f"ref_volume {ref_volume} out of range for T={x.shape[3]}")

    n_t = x.shape[3]
    corrected = np.empty_like(x, dtype=np.float32)
    params = np.zeros((n_t, 6), dtype=np.float64)

    reference = x[..., ref_volume]
    ref_com = np.asarray(center_of_mass(np.clip(reference, a_min=0.0, a_max=None) + 1e-6))

    for t in range(n_t):
        vol = x[..., t]
        mov_com = np.asarray(center_of_mass(np.clip(vol, a_min=0.0, a_max=None) + 1e-6))
        translation = ref_com - mov_com

        corrected[..., t] = ndi_shift(
            vol,
            shift=translation,
            order=1,
            mode="nearest",
            prefilter=False,
        ).astype(np.float32)
        params[t, :3] = translation

    return corrected.reshape(-1, n_t), params


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
    """Regress confounds out of vertex time series in float64 arithmetic."""

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

    if not gsr and c.shape[1] > 0:
        c = c[:, :-1]

    design = np.column_stack([c, np.ones(c.shape[0], dtype=np.float64)])
    beta, *_ = np.linalg.lstsq(design, x.T, rcond=None)
    fitted = design @ beta
    residual = x.T - fitted

    return residual.T.astype(np.float32)


def _motion_derivatives(motion_params: np.ndarray) -> np.ndarray:
    deriv = np.zeros_like(motion_params, dtype=np.float64)
    if motion_params.shape[0] > 1:
        deriv[1:] = np.diff(motion_params, axis=0)
    return deriv


def _estimate_wm_csf_means(timeseries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean_signal = timeseries.mean(axis=1)
    hi = mean_signal >= np.quantile(mean_signal, 0.8)
    lo = mean_signal <= np.quantile(mean_signal, 0.2)

    if not np.any(hi):
        hi = np.ones(timeseries.shape[0], dtype=bool)
    if not np.any(lo):
        lo = np.ones(timeseries.shape[0], dtype=bool)

    wm = timeseries[hi].mean(axis=0, dtype=np.float64)
    csf = timeseries[lo].mean(axis=0, dtype=np.float64)
    return wm, csf


def _build_confounds(
    timeseries: np.ndarray,
    motion_params: np.ndarray,
    gsr: bool,
) -> np.ndarray:
    ts = np.asarray(timeseries, dtype=np.float64)
    motion = np.asarray(motion_params, dtype=np.float64)

    if motion.shape != (ts.shape[1], 6):
        raise ValueError(
            f"motion_params must have shape ({ts.shape[1]}, 6), found {motion.shape}"
        )

    motion_dt = _motion_derivatives(motion)
    wm, csf = _estimate_wm_csf_means(ts)

    cols = [motion, motion_dt, wm[:, None], csf[:, None]]
    if gsr:
        global_signal = ts.mean(axis=0, dtype=np.float64)
        cols.append(global_signal[:, None])

    return np.column_stack(cols).astype(np.float64)


def _coords_from_nifti(spec: RunSpec) -> tuple[np.ndarray, tuple[int, int, int]]:
    import nibabel as nib

    image = nib.load(str(spec.bold_path))
    nx, ny, nz, _ = image.shape

    grid = np.stack(
        np.meshgrid(
            np.arange(nx, dtype=np.float32),
            np.arange(ny, dtype=np.float32),
            np.arange(nz, dtype=np.float32),
            indexing="ij",
        ),
        axis=-1,
    )
    return grid.reshape(-1, 3).astype(np.float32), (nx, ny, nz)


def _register_to_surface(coords: np.ndarray) -> np.ndarray:
    """Apply a deterministic affine normalization as a lightweight registration proxy."""

    centered = coords - coords.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, keepdims=True)
    scale[scale == 0.0] = 1.0
    return (centered / scale).astype(np.float32)


def _resample_vertices(
    timeseries: np.ndarray,
    coords: np.ndarray,
    target_per_hemi: int,
) -> tuple[np.ndarray, np.ndarray]:
    if target_per_hemi <= 0:
        return timeseries, coords

    x = coords[:, 0]
    split = np.median(x)
    left = np.where(x <= split)[0]
    right = np.where(x > split)[0]

    def _take(idx: np.ndarray) -> np.ndarray:
        if idx.size <= target_per_hemi:
            return idx
        sample_idx = np.linspace(0, idx.size - 1, num=target_per_hemi, dtype=int)
        return idx[sample_idx]

    keep = np.concatenate([_take(left), _take(right)])
    keep = np.unique(np.sort(keep))
    return timeseries[keep], coords[keep]


def preprocess_run(spec: RunSpec, cfg: Any) -> tuple[np.ndarray, np.ndarray, float]:
    """Preprocess one subject-run and return (timeseries, coords, tr).

    Intermediates are explicitly released between stages to keep peak RSS low.
    """

    if cfg is None:
        cfg = SimpleNamespace()

    chunk_size = int(getattr(cfg, "chunk_size", 4000))
    low = float(getattr(cfg, "low", 0.01))
    high = float(getattr(cfg, "high", 0.1))
    order = int(getattr(cfg, "order", 2))
    gsr = bool(getattr(cfg, "gsr", True))
    target_vertices_per_hemi = int(getattr(cfg, "target_vertices_per_hemi", 2000))

    sidecar = read_bold_json(spec.json_path)
    slice_timing = np.asarray(sidecar["SliceTiming"], dtype=np.float64)

    coords, volume_shape = _coords_from_nifti(spec)

    chunks = [block for block in iter_bold_chunks(spec, chunk=chunk_size)]
    timeseries = np.vstack(chunks).astype(np.float32)
    del chunks
    gc.collect()

    stc = slice_time_correct(timeseries, slice_timing=slice_timing, tr=spec.tr)
    del timeseries
    gc.collect()

    stc_4d = stc.reshape((*volume_shape, stc.shape[1]), order="C")
    motion_corrected, motion_params = motion_correct(stc_4d)
    del stc
    del stc_4d
    gc.collect()

    confounds = _build_confounds(motion_corrected, motion_params=motion_params, gsr=gsr)
    denoised = denoise(motion_corrected, confounds=confounds, gsr=gsr)
    del motion_corrected
    del confounds
    del motion_params
    gc.collect()

    filtered = bandpass(denoised, tr=spec.tr, low=low, high=high, order=order)
    del denoised
    gc.collect()

    coords = _register_to_surface(coords)
    filtered, coords = _resample_vertices(
        filtered,
        coords,
        target_per_hemi=target_vertices_per_hemi,
    )

    return filtered.astype(np.float32), coords.astype(np.float32), float(spec.tr)
