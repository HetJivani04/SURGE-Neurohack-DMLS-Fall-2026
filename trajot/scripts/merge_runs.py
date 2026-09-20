#!/usr/bin/env python3
"""Merge per-developer run registry files into results/runs_index.csv."""
import sys
from pathlib import Path
import pandas as pd

def main():
    runs_dir = Path("results/registry")
    files = sorted(runs_dir.glob("*.csv"))
    if not files:
        print("No run registry files found in results/registry/")
        return 1
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"  {f.name}: {len(df)} rows")
    merged = pd.concat(dfs, ignore_index=True)
    if "run_id" in merged.columns:
        merged = merged.drop_duplicates(subset=["run_id"], keep="first")
    out = Path("results/runs_index.csv")
    merged.to_csv(out, index=False)
    print(f"Merged: {len(files)} files -> {len(merged)} unique runs")
    if "experiment" in merged.columns:
        print(f"Experiments: {sorted(merged['experiment'].unique())}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
