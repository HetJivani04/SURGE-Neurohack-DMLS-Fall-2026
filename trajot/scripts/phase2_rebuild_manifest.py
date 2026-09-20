#!/usr/bin/env python
"""Rebuild manifest from existing contract npz and report two-run cohort size."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from trajot.io.contract import read_manifest, two_run_subjects

ROOT = Path("/Users/anandlo/Surge2026F/ds000243-master")
PRE = Path(__file__).resolve().parent / "preprocess.py"


def main() -> int:
    spec = importlib.util.spec_from_file_location("pre", PRE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    path = mod._rebuild_manifest(ROOT)
    table = read_manifest(ROOT)
    ids = two_run_subjects(table, strict=False)
    print(f"manifest={path}")
    print(f"rows={len(table)} two_run={len(ids)}")
    print("ids=" + ",".join(ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
