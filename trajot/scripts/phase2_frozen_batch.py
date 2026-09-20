#!/usr/bin/env python
"""Freeze the manifest to a declared subject list, run all experiments, restore manifest."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from trajot.io.contract import read_manifest, two_run_subjects, write_manifest

ROOT = Path("/Users/anandlo/Surge2026F/ds000243-master")
DERIV = ROOT / "derivatives" / "trajot"
TRAJOT = Path(__file__).resolve().parents[1]
RUNS = TRAJOT / "runs"
ART_FULL = RUNS / "10_ours_full__0963e0b9__20260920T061730Z"  # real-beta + tau_phi artifacts
SUBJECTS = ["015", "016", "017", "018", "019", "020", "021", "022", "023", "024", "025", "026"]


def rebuild_manifest() -> None:
    spec = importlib.util.spec_from_file_location("pre", TRAJOT / "scripts" / "preprocess.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._rebuild_manifest(ROOT)


def freeze_manifest(ids: list[str]) -> pd.DataFrame:
    rebuild_manifest()
    table = read_manifest(ROOT)
    frozen = table[table["subject_id"].astype(str).isin(ids)].copy()
    backup = DERIV / "manifest.full.parquet"
    shutil.copy2(DERIV / "manifest.parquet", backup)
    rows = frozen.to_dict(orient="records")
    # write_manifest expects the standard columns
    write_manifest(rows, ROOT)
    got = two_run_subjects(read_manifest(ROOT), strict=False)
    print(f"frozen n={len(got)} ids={got}")
    if got != sorted(ids):
        raise SystemExit(f"frozen manifest mismatch: {got} != {sorted(ids)}")
    return table


def run_experiment(config: str, extra: list[str]) -> dict:
    cmd = [
        sys.executable,
        str(TRAJOT / "scripts" / "run_experiment.py"),
        "--config",
        str(TRAJOT / "configs" / "experiments" / config),
        "--override",
        "eval.permutations.B=200",
        "--debug",
        *extra,
    ]
    print("RUN", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=TRAJOT, capture_output=True, text=True)
    print(proc.stdout[-2000:] if proc.stdout else "")
    if proc.returncode != 0:
        print(proc.stderr[-2000:] if proc.stderr else "")
        raise SystemExit(f"failed: {config}")
    # newest metrics for this experiment prefix
    exp = config.split("_", 1)[0] if False else Path(config).stem
    cands = sorted(RUNS.glob(f"{exp}__*/metrics.json"), key=lambda p: p.stat().st_mtime)
    if not cands:
        # experiment name is the yaml stem
        stem = Path(config).stem
        cands = sorted(RUNS.glob(f"{stem}__*/metrics.json"), key=lambda p: p.stat().st_mtime)
    path = cands[-1]
    payload = json.loads(path.read_text())
    print("METRICS", path, json.dumps(payload.get("methods"), indent=2)[:1500])
    return payload


def main() -> int:
    ids = SUBJECTS
    full = freeze_manifest(ids)
    try:
        results = {}
        results["noalign"] = run_experiment("00_noalign.yaml", [])
        results["brainsync"] = run_experiment("01_brainsync.yaml", [])
        results["fugw"] = run_experiment("02_fugw.yaml", [])
        results["conn_srm"] = run_experiment("03_conn_srm.yaml", [])
        results["ours_full"] = run_experiment(
            "10_ours_full.yaml",
            ["--override", f"run.ours_artifacts={ART_FULL}"],
        )
        # ablated: train inline on frozen cohort (gauge off)
        results["ours_ablated"] = run_experiment(
            "11_ours_ablated.yaml",
            ["--override", f"run.ours_artifacts={ART_FULL.parent}", ],
        )
        # Prefer a dedicated ablated train if present; if artifacts are full-model, retrain without them.
        abl = results["ours_ablated"]
        method = next(iter(abl.get("methods", {}).values()), {})
        if method.get("per_pair_uncertainty") is None:
            print("ablated unc still null; retraining ablated without shared artifacts", flush=True)
            results["ours_ablated"] = run_experiment("11_ours_ablated.yaml", [])
        summary = {k: next(iter(v.get("methods", {}).values()), {}) for k, v in results.items()}
        print("\n=== FROZEN SUMMARY ===")
        for k, v in summary.items():
            print(k, {x: v.get(x) for x in [
                "ident_accuracy", "ident_ci", "perm_p", "null_max", "alignment_gain",
                "nonidentifiable_pairs", "per_pair_uncertainty"]},
                "flags", None if v.get("per_pair_flags") is None else len(v["per_pair_flags"]))
            beta = results[k].get("beta")
            print("  beta", beta, "S", results[k].get("n_subjects"))
    finally:
        # restore full manifest so preprocess continue is visible
        backup = DERIV / "manifest.full.parquet"
        if backup.exists():
            shutil.copy2(backup, DERIV / "manifest.parquet")
            print("restored full manifest")
        rebuild_manifest()
        print("rebuilt live manifest", two_run_subjects(read_manifest(ROOT), strict=False))

    print("\n=== COMPARE ===")
    proc = subprocess.run(
        [sys.executable, str(TRAJOT / "scripts" / "compare.py"), "--experiments", "all"],
        cwd=TRAJOT,
        capture_output=True,
        text=True,
    )
    print(proc.stdout)
    print(proc.stderr)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
