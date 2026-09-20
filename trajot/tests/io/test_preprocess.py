from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
from scipy.ndimage import center_of_mass, shift as ndi_shift

from trajot.io.dataset import RunSpec
from trajot.io.preprocess import (
    bandpass,
    denoise,
    motion_correct,
    preprocess_run,
    slice_time_correct,
)


def test_slice_time_correct_preserves_shape_and_dtype() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 120)).astype(np.float32)
    slice_timing = np.linspace(0.0, 2.5 - 2.5 / 32.0, 32)

    y = slice_time_correct(x, slice_timing=slice_timing, tr=2.5)
    assert y.shape == x.shape
    assert y.dtype == np.float32


def test_motion_correct_4d_outputs_params_and_improves_alignment() -> None:
    grid = np.indices((12, 12, 8)).astype(np.float64)
    center = np.array([[6.0], [6.0], [4.0]])
    sq = ((grid.reshape(3, -1) - center) ** 2).sum(axis=0).reshape(12, 12, 8)
    ref = np.exp(-sq / 6.0).astype(np.float32)

    shifts = [(0.0, 0.0, 0.0), (0.8, -0.4, 0.2), (-0.5, 0.3, -0.2)]
    vols = np.stack([ndi_shift(ref, s, order=1, mode="nearest") for s in shifts], axis=-1)

    corrected_flat, params = motion_correct(vols, ref_volume=0)
    corrected = corrected_flat.reshape(12, 12, 8, 3)

    before = np.linalg.norm(np.asarray(center_of_mass(vols[..., 1])) - np.asarray(center_of_mass(ref)))
    after = np.linalg.norm(
        np.asarray(center_of_mass(corrected[..., 1])) - np.asarray(center_of_mass(ref))
    )

    assert params.shape == (3, 6)
    assert params.dtype == np.float64
    assert after < before


def test_bandpass_retains_low_frequency_and_attenuates_high_frequency() -> None:
    tr = 2.5
    t = np.arange(240, dtype=np.float64) * tr
    low = np.sin(2 * np.pi * 0.03 * t)
    high = 0.8 * np.sin(2 * np.pi * 0.16 * t)

    x = np.vstack([low + high, low + high]).astype(np.float32)
    y = bandpass(x, tr=tr, low=0.01, high=0.1, order=2)

    fft = np.fft.rfft(y[0])
    freqs = np.fft.rfftfreq(y.shape[1], d=tr)
    amp_low = np.abs(fft[np.argmin(np.abs(freqs - 0.03))])
    amp_high = np.abs(fft[np.argmin(np.abs(freqs - 0.16))])
    assert amp_low > amp_high


def test_denoise_reduces_confound_correlation() -> None:
    rng = np.random.default_rng(3)
    T = 150
    conf = rng.normal(size=(T, 1))
    weights = rng.normal(size=(40, 1))
    signal = weights @ conf.T + 0.3 * rng.normal(size=(40, T))

    before = np.mean([np.corrcoef(signal[i], conf[:, 0])[0, 1] ** 2 for i in range(40)])
    cleaned = denoise(signal.astype(np.float32), confounds=conf.astype(np.float64), gsr=False)
    after = np.mean([np.corrcoef(cleaned[i], conf[:, 0])[0, 1] ** 2 for i in range(40)])

    assert after < before


def test_preprocess_run_on_synthetic_nifti(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    func_dir = tmp_path / "sub-001" / "func"
    func_dir.mkdir(parents=True, exist_ok=True)

    bold_path = func_dir / "sub-001_task-rest_run-1_bold.nii.gz"
    data = rng.normal(size=(6, 6, 32, 40)).astype(np.float32)
    nib.save(nib.Nifti1Image(data, affine=np.eye(4)), str(bold_path))

    slice_timing = [float(i) * (2.5 / 32.0) for i in range(32)]
    json_path = func_dir / "sub-001_task-rest_run-1_bold.json"
    json_path.write_text(
        "{\n"
        '  "RepetitionTime": 2.5,\n'
        '  "EchoTime": 0.027,\n'
        f'  "SliceTiming": {slice_timing}\n'
        "}\n",
        encoding="utf-8",
    )

    spec = RunSpec(
        subject_id="001",
        run_id="1",
        bold_path=bold_path,
        json_path=json_path,
        n_volumes=40,
        tr=2.5,
    )

    cfg = SimpleNamespace(
        chunk_size=100,
        low=0.01,
        high=0.1,
        order=2,
        gsr=True,
        target_vertices_per_hemi=10,
    )

    ts, coords, tr = preprocess_run(spec, cfg)
    assert tr == 2.5
    assert ts.dtype == np.float32
    assert coords.dtype == np.float32
    assert ts.shape[0] == coords.shape[0]
    assert ts.shape[1] == 40
