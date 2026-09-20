# ds000243 Dataset Notes

This file freezes the verified dataset facts used by W1 and downstream modules.

## Source and download

```bash
aws s3 sync --no-sign-request s3://openneuro.org/ds000243 ./data/ds000243
```

## Verified counts

- Subjects: 120
- Rest runs: 203 total
- Two-run subjects with equal run length: 83
- One-run subjects: 37
- TR: 2.5 s
- TE: 27 ms
- Voxel size: 4 x 4 x 4 mm
- Slice count: 32 (interleaved, from `task-rest_bold.json`)
- Volumes per run: median 132, range 130-724
- Raw dataset size: 5.67 GiB
- License/access: CC0, anonymous S3 access (`--no-sign-request`)
- Derivatives: no fMRIPrep derivatives present in this dataset

## Verification commands

The following commands were used to compute counts and run-level summaries from raw files:

```bash
# count subjects
find ./ds000243 -maxdepth 1 -type d -name 'sub-*' | wc -l

# count runs
find ./ds000243/sub-*/func -type f -name '*_task-rest*_bold.nii.gz' | wc -l
```

```bash
# summarize per-subject run counts and per-run volume counts
python - <<'PY'
from pathlib import Path
import nibabel as nib
from collections import Counter

root = Path('./ds000243')
runs = sorted(root.glob('sub-*/func/*_task-rest*_bold.nii.gz'))
by_subject = Counter(p.parts[-3] for p in runs)
print('subjects', len(by_subject))
print('runs', len(runs))
print('two_run_subjects', sum(1 for _, n in by_subject.items() if n == 2))

volumes = []
for run in runs:
    img = nib.load(str(run))
    volumes.append(int(img.shape[3]))
volumes.sort()
print('median', volumes[len(volumes)//2])
print('min', volumes[0], 'max', volumes[-1])
PY
```
