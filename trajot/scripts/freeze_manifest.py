#!/usr/bin/env python3
"""Freeze manifest to current state for fair comparison."""
from pathlib import Path
import shutil
from trajot.io.contract import read_manifest, two_run_subjects, write_manifest

ROOT = Path('/Users/anandlo/Surge2026F/ds000243-master')
DERIV = ROOT / 'derivatives' / 'trajot'

# Backup current
if not (DERIV / 'manifest.full.parquet').exists():
    shutil.copy2(DERIV / 'manifest.parquet', DERIV / 'manifest.full.parquet')

# Read current and freeze to two-run subjects only
table = read_manifest(ROOT)
ids = two_run_subjects(table, strict=False)
print(f'Current two-run subjects: {len(ids)} -> {ids}')

frozen = table[table['subject_id'].astype(str).isin(ids)]
write_manifest(frozen.to_dict('records'), ROOT)
print(f'Froze manifest to {len(frozen)} rows, {len(ids)} subjects')
