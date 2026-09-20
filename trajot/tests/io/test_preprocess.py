from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from scipy import ndimage as ndi

from trajot.config import Config
from trajot.io.dataset import RunSpec
from trajot.io.preprocess import (
    _rigid_matrix,
    bandpass,
    denoise,
    framewise_displacement,
    motion_correct,
    preprocess_run,
    qc_path,
    slice_time_correct,
    slice_timing_from_order,
)

TR = 2.5


# --------------------------------------------------------------------------- slice timing
def test_slice_time_correct_preserves_shape_and_dtype() -> None:
    x = np.random.default_rng(0).normal(size=(64, 120)).astype(np.float32)
    y = slice_time_correct(x, slice_timing=np.linspace(0.0, TR - TR / 32.0, 32), tr=TR)
    assert y.shape == x.shape and y.dtype == np.float32


def test_slice_time_correct_delays_by_the_slice_offset() -> None:
    """A slice sampled ``offset`` seconds late is moved back to the reference time."""
    n, offset = 120, 1.25
    t = np.arange(n) * TR
    f1, f2 = 12 / (n * TR), 24 / (n * TR)  # exact Fourier bins, so the shift is exact
    signal = lambda tt: np.sin(2 * np.pi * f1 * tt) + 0.5 * np.sin(2 * np.pi * f2 * tt + 1.0)  # noqa: E731
    late = signal(t + offset)[None, :].astype(np.float32)  # what a slice acquired `offset` late records
    corrected = slice_time_correct(late, np.array([offset]), tr=TR)[0]
    assert np.abs(corrected - signal(t)).max() < 1e-4


def test_slice_timing_from_order_interleaved_odd_first() -> None:
    times = slice_timing_from_order("interleaved_odd_first", 8, TR)
    dt = TR / 8
    assert np.allclose(times, np.array([4, 0, 5, 1, 6, 2, 7, 3]) * dt)  # z = 1,3,5,7 first, then 0,2,4,6
    assert np.allclose(slice_timing_from_order("interleaved_even_first", 8, TR), np.array([0, 4, 1, 5, 2, 6, 3, 7]) * dt)
    assert np.allclose(slice_timing_from_order("sequential_ascending", 8, TR), np.arange(8) * dt)
    with pytest.raises(ValueError, match="slice_order"):
        slice_timing_from_order("bogus", 8, TR)
    assert sorted(slice_timing_from_order("interleaved_odd_first", 32, TR)) == pytest.approx(list(np.arange(32) * TR / 32))


# --------------------------------------------------------------------------- motion
def _blob_volume(shape=(32, 32, 20)) -> np.ndarray:
    g = np.indices(shape).astype(float)
    blobs = [(100, 16, 16, 10, 7), (60, 10, 20, 8, 3), (70, 22, 12, 12, 3), (50, 16, 8, 6, 2.5)]
    return sum(a * np.exp(-(((g[0] - cx) ** 2 + (g[1] - cy) ** 2 + (g[2] - cz) ** 2) / (2 * s**2)))
               for a, cx, cy, cz, s in blobs) + 5


def test_motion_correct_recovers_known_rigid_transforms() -> None:
    shape, vs = (32, 32, 20), np.array([4.0, 4.0, 4.0])
    center = (np.array(shape) - 1) / 2
    vol = _blob_volume(shape)
    rng = np.random.default_rng(0)
    # parameters are the transform M with f(M x) = reference(x); build each frame as reference(M^-1 y)
    truth = [np.zeros(6), np.array([1.2, -0.8, 0.5, 0.02, -0.03, 0.04]), np.array([-2.0, 1.5, -1.0, -0.05, 0.02, -0.03])]
    frames = []
    p = (np.indices(shape, dtype=float).reshape(3, -1).T - center) * vs
    for q in truth:
        inv = np.linalg.inv(_rigid_matrix(q))
        coords = ((p @ inv[:3, :3].T + inv[:3, 3]) / vs + center).T
        frames.append(ndi.map_coordinates(vol, coords, order=1, mode="nearest").reshape(shape) + rng.normal(0, 0.5, shape))

    corrected, params = motion_correct(np.stack(frames, -1), ref_volume=0, voxel_size=tuple(vs))
    assert corrected.shape == (32 * 32 * 20, 3) and corrected.dtype == np.float32
    assert params.shape == (3, 6) and params.dtype == np.float64
    for est, true in zip(params, truth):
        assert np.abs(est[:3] - true[:3]).max() < 0.15  # mm
        assert np.abs(est[3:] - true[3:]).max() < 0.005  # rad
    residual_before = np.abs(frames[2] - frames[0]).mean()
    residual_after = np.abs(corrected[:, 2].reshape(shape) - frames[0]).mean()
    assert residual_after < 0.6 * residual_before


def test_motion_correct_needs_a_4d_volume_and_a_valid_reference() -> None:
    with pytest.raises(ValueError, match="X, Y, Z, T"):
        motion_correct(np.zeros((100, 5)))
    with pytest.raises(ValueError, match="ref_volume"):
        motion_correct(np.zeros((4, 4, 4, 3)) + 1, ref_volume=3)


def test_framewise_displacement_from_parameters() -> None:
    params = np.zeros((4, 6))
    params[1, 0] = 1.0  # 1 mm translation
    params[2, 0] = 1.0  # no change
    params[3, 0] = 1.0
    params[3, 3] = 0.02  # 0.02 rad on a 50 mm sphere = 1 mm
    fd = framewise_displacement(params)
    assert np.allclose(fd, [0.0, 1.0, 0.0, 1.0])


# --------------------------------------------------------------------------- band-pass, denoise
def test_bandpass_retains_low_frequency_and_attenuates_high_frequency() -> None:
    t = np.arange(240, dtype=np.float64) * TR
    x = np.vstack([np.sin(2 * np.pi * 0.03 * t) + 0.8 * np.sin(2 * np.pi * 0.16 * t)] * 2).astype(np.float32)
    y = bandpass(x, tr=TR, low=0.01, high=0.1, order=2)
    fft = np.fft.rfft(y[0])
    freqs = np.fft.rfftfreq(y.shape[1], d=TR)
    assert np.abs(fft[np.argmin(np.abs(freqs - 0.03))]) > 20 * np.abs(fft[np.argmin(np.abs(freqs - 0.16))])


def test_denoise_removes_confounds_and_optionally_the_global_signal() -> None:
    rng = np.random.default_rng(3)
    T = 150
    conf = rng.normal(size=(T, 1))
    shared = rng.normal(size=T)  # a global fluctuation carried by every vertex
    x = (rng.normal(size=(40, 1)) @ conf.T + shared[None, :] + 0.3 * rng.normal(size=(40, T))).astype(np.float32)

    def r2(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean([np.corrcoef(row, b)[0, 1] ** 2 for row in a]))

    no_gsr = denoise(x, conf, gsr=False)
    with_gsr = denoise(x, conf, gsr=True)
    assert no_gsr.dtype == np.float32
    assert r2(no_gsr, conf[:, 0]) < 1e-3 < r2(x, conf[:, 0])
    assert r2(no_gsr, shared) > 0.5 > r2(with_gsr, shared)  # only gsr=True removes the shared signal
    assert abs(float(with_gsr.mean(axis=0).std())) < 1e-4  # the mean signal is gone


def test_denoise_leaves_all_zero_rows_zero() -> None:
    x = np.random.default_rng(1).normal(size=(6, 50)).astype(np.float32)
    x[2] = 0.0
    y = denoise(x, np.random.default_rng(2).normal(size=(50, 3)), gsr=True)
    assert np.all(y[2] == 0.0)


# --------------------------------------------------------------------------- end to end
def _template_resources():
    try:
        from trajot.io.preprocess import _mni, load_surface_template

        return _mni(), load_surface_template("fsaverage4", 100)
    except Exception as exc:  # no network and no cached nilearn data
        pytest.skip(f"nilearn template data unavailable: {exc}")


def _build_synthetic_run(root: Path, z_origin: float):
    """An EPI-like run built from the MNI template through a known misalignment.

    ``z_origin`` is the world z of the bottom slice: raising it cuts the inferior brain out of view.
    """
    mni, _ = _template_resources()
    n_t, shape = 40, (64, 64, 32)
    epi_affine = np.diag([4.0, 4.0, 4.0, 1.0])
    epi_affine[:3, 3] = [-126.0, -126.0, z_origin]

    truth = _rigid_matrix(np.array([7.0, -6.0, 4.0, np.deg2rad(6), np.deg2rad(-4), np.deg2rad(5)]))  # MNI mm -> EPI world
    ijk = np.indices(shape, dtype=float).reshape(3, -1).T
    world = ijk @ epi_affine[:3, :3].T + epi_affine[:3, 3]
    mni_mm = (world - truth[:3, 3]) @ np.linalg.inv(truth[:3, :3]).T
    template_vox = (mni_mm - mni.affine[:3, 3]) @ np.linalg.inv(mni.affine[:3, :3]).T
    tpl = ndi.map_coordinates(mni.template, template_vox.T, order=1, cval=0).reshape(shape)
    brain = ndi.map_coordinates(mni.brain.astype(np.uint8), template_vox.T, order=0).reshape(shape) > 0
    mean = np.where(brain, 1400.0 - 800.0 * tpl / tpl.max(), 20.0)  # T2*-like contrast (reversed relative to T1)

    rng = np.random.default_rng(5)
    t = np.arange(n_t) * TR
    slow = 1.0 + 0.02 * np.sin(2 * np.pi * 0.03 * t)
    data = (mean[..., None] * slow + rng.normal(0, 8.0, shape + (n_t,))).astype(np.float32)
    func = root / "sub-001" / "func"
    func.mkdir(parents=True)
    bold = func / "sub-001_task-rest_run-1_bold.nii.gz"
    nib.save(nib.Nifti1Image(data, epi_affine), str(bold))
    sidecar = func / "sub-001_task-rest_run-1_bold.json"
    sidecar.write_text('{"RepetitionTime": 2.5, "EchoTime": 0.027}')  # like ds000243: no SliceTiming
    spec = RunSpec("001", "1", bold, sidecar, n_t, TR)

    cfg = Config({
        "experiment": "test", "data": {"root": str(root), "contract_version": "1.0.0"},
        "model": {"d": 32, "r": 32},
        "preprocess": {"slice_order": "interleaved_odd_first", "bandpass": {"low": 0.01, "high": 0.1, "order": 2},
                       "gsr": True, "surface": "fsaverage4", "n_regions": 100,
                       "qc": {"max_mean_fd_mm": 0.5, "min_mask_dice": 0.75}},
    })
    return spec, cfg, truth, mni


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    return _build_synthetic_run(tmp_path_factory.mktemp("synthetic_ds"), z_origin=-50.0)


def test_preprocess_run_end_to_end_recovers_the_known_registration(synthetic_run) -> None:
    spec, cfg, truth, mni = synthetic_run
    timeseries, coords, tr = preprocess_run(spec, cfg)

    assert tr == TR
    assert timeseries.dtype == np.float32 and timeseries.shape == (5124, 40)  # fsaverage4, both hemispheres
    assert coords.dtype == np.float32 and coords.shape == (5124, 3)
    assert np.isfinite(timeseries).all()

    record = json.loads(qc_path(Path(cfg.data_root), "001", "1").read_text())
    assert record["slice_timing"]["source"] == "derived:interleaved_odd_first"
    assert record["registration"]["mask_dice"] > 0.85
    assert record["registration"]["nmi_final"] > record["registration"]["nmi_initial"]
    assert record["surface"]["n_valid_vertices"] > 0.8 * 5124 and record["surface"]["empty_regions"] == []
    assert record["denoise"]["n_white_matter_voxels"] >= 20 and record["denoise"]["n_csf_voxels"] >= 20
    assert record["bandpass"]["in_band_power_ratio"] > 0.5
    assert record["bandpass"]["out_of_band_power_ratio"] < 0.01
    assert record["qc_pass"] is True and record["qc_failed"] == []
    assert len(record["motion"]["parameters_mm_rad"]) == 40

    # the recovered MNI -> EPI transform agrees with the known one to about a voxel over the brain
    estimated = np.array(record["registration"]["mni_to_epi_world"])
    points = np.argwhere(mni.brain)[::400].astype(float) @ mni.affine[:3, :3].T + mni.affine[:3, 3]
    error = np.linalg.norm((points @ estimated[:3, :3].T + estimated[:3, 3]) - (points @ truth[:3, :3].T + truth[:3, 3]), axis=1)
    assert error.mean() < 4.0 and np.percentile(error, 95) < 8.0

    # invalid (out of view) vertices are all zeros; valid ones carry signal
    valid = timeseries.std(axis=1) > 0
    assert valid.sum() == record["surface"]["n_valid_vertices"]


def test_preprocess_run_is_deterministic(synthetic_run) -> None:
    spec, cfg, _, _ = synthetic_run
    first = preprocess_run(spec, cfg)[0]
    second = preprocess_run(spec, cfg)[0]
    assert np.array_equal(first, second)


def test_a_field_of_view_that_misses_part_of_the_cortex_is_reported_not_hidden(tmp_path) -> None:
    """A scan starting 40 mm higher loses the temporal poles and other inferior cortex."""
    spec, cfg, _, _ = _build_synthetic_run(tmp_path, z_origin=-10.0)
    timeseries, _, _ = preprocess_run(spec, cfg)

    record = json.loads(qc_path(Path(cfg.data_root), "001", "1").read_text())
    empty = record["surface"]["empty_regions"]
    assert empty and all(0 <= region < 100 for region in empty)
    assert record["surface"]["n_valid_vertices"] < 5124 and (timeseries.std(axis=1) > 0).sum() == record["surface"]["n_valid_vertices"]
    assert record["qc_pass"] is False
    assert any("outside the field of view" in reason and str(len(empty)) in reason for reason in record["qc_failed"])
