#!/usr/bin/env python
"""Group-level random-effects driver on hierarchical posterior artifacts.

    python scripts/run_group_analysis.py --artifacts runs/10_ours_full__.../artifacts
    python scripts/run_group_analysis.py --artifacts runs/.../artifacts --z maps.npy --method REML

Loads ``posterior_samples.npz`` + ``tau_phi.npz`` + ``template.npz`` (nu) and
calls :func:`trajot.report.group.meta_analysis_map` /
:func:`trajot.report.group.one_sample_ttest`. Alignment uncertainty enters as
``Sigma_s^al`` from the coupling draws: subjects the data do not identify are
down-weighted instead of silently averaged.

Returns ``n_eff``, ``mean_weight``, ``mean_sigma2``, REML vs t-test SE ratio.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from trajot.report.group import meta_analysis, meta_analysis_map, one_sample_ttest


def _resolve_artifacts_dir(posteriors_dir: Path) -> Path:
    art = Path(posteriors_dir)
    if art.name != "artifacts" and (art / "artifacts").is_dir():
        art = art / "artifacts"
    return art


def _load_blocks(
    art: Path,
    subjects: Sequence[str] | None = None,
) -> tuple[list[str], list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Return subject ids, pi draws list, tau_phi list, nu from an artifacts directory."""
    post_path = art / "posterior_samples.npz"
    tau_path = art / "tau_phi.npz"
    template_path = art / "template.npz"
    for path in (post_path, tau_path, template_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing required artifact: {path}")

    with np.load(template_path) as z:
        nu = np.asarray(z["nu"], dtype=np.float64)
        ids = [str(s) for s in z["subject_ids"]] if "subject_ids" in z.files else None

    with np.load(post_path) as post, np.load(tau_path) as tau_z:
        keys = [f"sub-{s}" for s in (ids or []) if f"sub-{s}" in post.files]
        if not keys:
            keys = [k for k in post.files if k in tau_z.files]
        if subjects is not None:
            wanted = {str(s) for s in subjects}
            keys = [k for k in keys if k.removeprefix("sub-") in wanted or k in wanted]
        if not keys:
            raise ValueError(f"no overlapping subject keys in {post_path} and {tau_path}")
        pi_list = [np.asarray(post[k], dtype=np.float64) for k in keys]
        tau_list = [np.asarray(tau_z[k], dtype=np.float64) for k in keys]
    sids = [k.removeprefix("sub-") for k in keys]
    return sids, pi_list, tau_list, nu


def _default_z(pi_list: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Unit vertex maps: ``w_sk`` then measures coupling-mass uncertainty alone."""
    return [np.ones(pi.shape[1], dtype=np.float64) for pi in pi_list]


def _as_z_list(z: np.ndarray | None, pi_list: Sequence[np.ndarray]) -> list[np.ndarray]:
    if z is None:
        return _default_z(pi_list)
    arr = np.asarray(z, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != pi_list[0].shape[1]:
            raise ValueError(
                f"z has length {arr.shape[0]} but couplings have V={pi_list[0].shape[1]}"
            )
        return [arr.copy() for _ in pi_list]
    if arr.ndim == 2:
        if arr.shape[0] != len(pi_list):
            raise ValueError(f"z has S={arr.shape[0]} rows but {len(pi_list)} subjects")
        if arr.shape[1] != pi_list[0].shape[1]:
            raise ValueError(
                f"z has V={arr.shape[1]} but couplings have V={pi_list[0].shape[1]}"
            )
        return [np.asarray(arr[i], dtype=np.float64) for i in range(arr.shape[0])]
    raise ValueError(f"z must be 1-D (V,) or 2-D (S,V), got shape {arr.shape}")


def run_group_from_arrays(
    pi_samples: Sequence[np.ndarray],
    tau_phi: Sequence[np.ndarray],
    z_s: Sequence[np.ndarray],
    nu: np.ndarray,
    method: str = "REML",
) -> dict[str, Any]:
    """Group REML / t-test on in-memory posterior draws."""
    if len(pi_samples) != len(tau_phi) or len(pi_samples) != len(z_s):
        raise ValueError("pi_samples, tau_phi and z_s must have the same length")
    S = len(pi_samples)
    if S < 2:
        raise ValueError("need at least two subjects")
    nu = np.asarray(nu, dtype=np.float64)
    reml = meta_analysis_map(pi_samples, tau_phi, z_s, nu, method=method)
    ttest = one_sample_ttest(reml["m"])

    reml_se = np.asarray(reml["se_theta"], dtype=np.float64)
    ttest_se = np.asarray(ttest["se_theta"], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ttest_se > 0, reml_se / ttest_se, np.nan)
    finite = ratio[np.isfinite(ratio)]
    ci_ratio = float(np.mean(finite)) if finite.size else float("nan")

    n_eff = np.asarray(reml["n_eff"], dtype=np.float64)
    mean_weight = np.asarray(reml["mean_weight"], dtype=np.float64)
    mean_sigma2 = np.asarray(reml["mean_sigma2"], dtype=np.float64)

    reml_meta = meta_analysis(reml["m"], reml["sigma2"], method=method)
    n_degenerate_nodes = int(reml.get("n_degenerate_nodes", 0))
    notes = (
        f"method={method}; n_eff mean over {n_eff.size} template nodes; "
        f"ci_ratio = mean(se_reml/se_ttest); "
        f"high-tau subjects down-weighted when mean_sigma2 is large; "
        f"n_degenerate_nodes = {n_degenerate_nodes} "
        "(nodes with tau^2 + sigma^2 = 0 exactly; n_eff takes the equal-weight limit #{u = inf})"
    )
    return {
        "n_subjects": int(S),
        "n_eff": float(np.mean(n_eff)),
        "n_eff_min": float(np.min(n_eff)),
        "n_eff_per_node": n_eff.tolist(),
        "mean_weight": float(np.mean(mean_weight)),
        "mean_weight_per_subject": mean_weight.tolist(),
        "mean_sigma2": float(np.mean(mean_sigma2)),
        "mean_sigma2_per_subject": mean_sigma2.tolist(),
        "n_degenerate_nodes": n_degenerate_nodes,
        "mean_tau_phi": np.asarray(reml["mean_tau_phi"], dtype=np.float64).tolist(),
        "reml_theta_se": float(np.mean(reml_se)),
        "ttest_theta_se": float(np.mean(ttest_se)),
        "ci_ratio": ci_ratio,
        "reml_mean_p": float(np.mean(reml["p_k"])),
        "ttest_mean_p": float(np.mean(ttest["p_k"])),
        "reml_tau_hat_mean": float(np.mean(reml_meta["tau_hat_k2"])),
        "method": method,
        "K": int(n_eff.size),
        "notes": notes,
    }


def run_group(
    posteriors_dir: Path,
    z: np.ndarray | None = None,
    nu: np.ndarray | None = None,
    method: str = "REML",
    subjects: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Load posterior artifacts and run group REML vs one-sample t-test.

    Parameters
    ----------
    posteriors_dir:
        Directory containing ``posterior_samples.npz``, ``tau_phi.npz``,
        ``template.npz`` (or a run dir with an ``artifacts/`` child).
    z:
        Optional subject maps. ``None`` uses unit vertex maps so weights
        reflect alignment covariance alone. ``(V,)`` broadcasts; ``(S,V)``
        is per-subject.
    nu:
        Optional template mass ``(K,)``; defaults to ``template.npz['nu']``.
    method:
        ``"REML"`` or ``"MoM"`` for the random-effects tau estimator.
    """
    art = _resolve_artifacts_dir(Path(posteriors_dir))
    sids, pi_list, tau_list, nu_loaded = _load_blocks(art, subjects=subjects)
    nu_use = np.asarray(nu_loaded if nu is None else nu, dtype=np.float64)
    z_list = _as_z_list(z, pi_list)
    out = run_group_from_arrays(pi_list, tau_list, z_list, nu_use, method=method)
    out["subjects"] = sids
    out["artifacts"] = str(art)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        required=True,
        help="run dir or artifacts/ with posterior_samples.npz, tau_phi.npz, template.npz",
    )
    parser.add_argument("--z", type=Path, default=None, help="optional (S,V) or (V,) maps .npy")
    parser.add_argument("--method", type=str, default="REML", choices=["REML", "MoM", "reml", "mom"])
    parser.add_argument("--out", type=Path, default=None, help="optional JSON output path")
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="comma-separated subject ids to include (default: all in artifacts)",
    )
    args = parser.parse_args(argv)

    z = np.load(args.z) if args.z is not None else None
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] if args.subjects else None
    result = run_group(args.artifacts, z=z, method=args.method, subjects=subjects)

    text = json.dumps(result, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
