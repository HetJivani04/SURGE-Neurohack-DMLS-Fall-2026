#!/usr/bin/env python3
"""Merge per-developer manifest files into results/manifest.parquet."""
import sys
from pathlib import Path
import pandas as pd

def main():
    manifests_dir = Path("results/manifests")
    files = sorted(manifests_dir.glob("*.parquet"))
    if not files:
        print("No manifest files found in results/manifests/")
        return 1
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        dfs.append(df)
        print(f"  {f.name}: {len(df)} rows")
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=["subject_id", "run_id"], keep="first")
    out = Path("results/manifest.parquet")
    merged.to_parquet(out, index=False)
    two_run = merged.groupby("subject_id").size()
    n_two = (two_run == 2).sum()
    print(f"Merged: {len(files)} files -> {len(merged)} unique subject-runs")
    print(f"Two-run subjects: {n_two}")
    if n_two != 83:
        print(f"WARNING: expected 83 two-run subjects, got {n_two}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
