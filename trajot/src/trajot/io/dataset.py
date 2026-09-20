from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

_BOLD_RE = re.compile(
    r"sub-(?P<subject>[A-Za-z0-9]+)_task-rest(?:_run-(?P<run>[A-Za-z0-9]+))?_bold\.nii\.gz$"
)


def _resolve_sidecar(root: Path, bold_path: Path) -> Path:
    run_level = bold_path.with_suffix("").with_suffix(".json")
    task_level = bold_path.parent / f"sub-{bold_path.parts[-3].replace('sub-', '')}_task-rest_bold.json"
    dataset_level = root / "task-rest_bold.json"

    for candidate in (run_level, task_level, dataset_level):
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Missing sidecar JSON for "
        f"{bold_path}. Tried: {run_level}, {task_level}, {dataset_level}"
    )


@dataclass(frozen=True)
class RunSpec:
    subject_id: str
    run_id: str
    bold_path: Path
    json_path: Path
    n_volumes: int
    tr: float


def read_bold_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    missing = [k for k in ("RepetitionTime", "SliceTiming", "EchoTime") if k not in payload]
    if missing:
        raise ValueError(f"Missing required sidecar key(s) in {path}: {', '.join(missing)}")

    slice_timing = payload["SliceTiming"]
    if not isinstance(slice_timing, list) or len(slice_timing) != 32:
        raise ValueError(
            f"SliceTiming must be a list of length 32 in {path}, found {type(slice_timing).__name__} with length {len(slice_timing) if isinstance(slice_timing, list) else 'n/a'}"
        )

    return payload


def list_bold_runs(root: Path) -> list[RunSpec]:
    try:
        import nibabel as nib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "nibabel is required for list_bold_runs(); install it before discovery"
        ) from exc

    specs: list[RunSpec] = []
    for bold_path in sorted(root.glob("sub-*/func/*_task-rest*_bold.nii.gz")):
        match = _BOLD_RE.search(bold_path.name)
        if match is None:
            continue

        subject_id = match.group("subject")
        run_id = match.group("run") or "1"

        json_path = _resolve_sidecar(root, bold_path)

        sidecar = read_bold_json(json_path)
        tr = float(sidecar["RepetitionTime"])

        nii = nib.load(str(bold_path))
        if len(nii.shape) != 4:
            raise ValueError(f"Expected 4D BOLD NIfTI at {bold_path}, found shape {nii.shape}")
        n_volumes = int(nii.shape[3])

        specs.append(
            RunSpec(
                subject_id=subject_id,
                run_id=str(run_id),
                bold_path=bold_path,
                json_path=json_path,
                n_volumes=n_volumes,
                tr=tr,
            )
        )

    return specs


def iter_bold_chunks(spec: RunSpec, chunk: int = 20) -> Iterator[np.ndarray]:
    """Yield (V_chunk, T) float32 slices from a 4D BOLD volume.

    This function streams across x-slices so the full run is never materialized as
    float64. Chunk is interpreted as an approximate vertex count.
    """

    try:
        import nibabel as nib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "nibabel is required for iter_bold_chunks(); install it before streaming"
        ) from exc

    image = nib.load(str(spec.bold_path))
    if len(image.shape) != 4:
        raise ValueError(
            f"Expected 4D BOLD NIfTI at {spec.bold_path}, found shape {image.shape}"
        )

    nx, ny, nz, n_t = image.shape
    dataobj = image.dataobj

    voxels_per_x = ny * nz
    x_step = max(1, int(np.ceil(chunk / max(voxels_per_x, 1))))

    for x0 in range(0, nx, x_step):
        x1 = min(nx, x0 + x_step)
        block = np.asarray(dataobj[x0:x1, :, :, :], dtype=np.float32)
        yield block.reshape(-1, n_t)
