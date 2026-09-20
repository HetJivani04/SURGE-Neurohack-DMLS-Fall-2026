"""The single thread-environment owner.

This is the only module that assigns the BLAS/OpenMP thread variables and the only one that
calls ``torch.device(...)``. ``setup_threads`` must run before the first BLAS call in the
process, so torch is imported lazily inside the functions.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def setup_threads(mode: Literal["serial", "outer", "inner"], n_jobs: int) -> dict[str, str]:
    """Set the four thread variables and ``torch.set_num_threads``; return the snapshot.

    * ``serial``: all four variables ``"1"``, torch threads 1.
    * ``outer``: all four variables ``"1"``; used when joblib fans out.
    * ``inner``: all four variables ``str(n_jobs)``, no joblib.

    Must be called **before the first BLAS call in the process**, because several BLAS
    implementations read these variables once at import time.
    """
    if mode in ("serial", "outer"):
        n_threads = 1
    elif mode == "inner":
        n_threads = n_jobs
    else:
        raise ValueError(f"unknown mode {mode!r}")
    for var in THREAD_VARS:
        os.environ[var] = str(n_threads)

    import torch

    torch.set_num_threads(n_threads)
    return thread_env_snapshot()


def thread_env_snapshot() -> dict[str, str]:
    """Current values of the four variables plus ``torch.get_num_threads()``, for ``manifest.json``."""
    import torch

    snapshot = {var: os.environ.get(var, "") for var in THREAD_VARS}
    snapshot["torch_threads"] = str(torch.get_num_threads())
    return snapshot


def pick_device(prefer_mps: bool = True) -> "torch.device":
    """``torch.device("mps")`` if ``prefer_mps`` and MPS is available, else ``cpu``.

    **Any float64 tensor must go to CPU**, since MPS has no float64 at all, so OT/GW code
    calls ``pick_device(prefer_mps=False)``.
    """
    import torch

    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parallel_map(fn: Callable, items: Sequence, n_jobs: int, mode: str = "outer") -> list:
    """``joblib.Parallel(n_jobs=n_jobs, backend="loky")`` wrapper; results in input order.

    Calls :func:`setup_threads` first. At ``n_jobs == 1`` it runs in-process, so Stage 1
    code paths are identical to Stage 3 code paths.
    """
    setup_threads(mode, n_jobs)
    if n_jobs == 1:
        return [fn(item) for item in items]

    from joblib import Parallel, delayed

    return Parallel(n_jobs=n_jobs, backend="loky")(delayed(fn)(item) for item in items)
