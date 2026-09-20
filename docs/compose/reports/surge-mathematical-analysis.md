# SURGE Mathematical Analysis: Why TrajOT Does Not Beat Baselines on Track A

## Definitive Results (N=49, frozen root, same data_hash)

| Method | Ident | Gain | NonIdent | feat_corr | tau_phi |
|--------|-------|------|----------|-----------|---------|
| noalign | 0.959 | 0.000 | 500 | 1.000 | - |
| brainsync | 0.959 | -0.000 | 469 | 0.998 | - |
| fugw | 0.918 | -0.005 | 466 | 0.966 | - |
| conn_srm | 0.082 | +0.395 | 467 | 0.586 | - |
| ours_full | 0.857 | +0.018 | 472 | 0.658 | 0.075 |

## Root Cause of Identification Loss

### Mathematical mechanism
ours_full transforms connectomes via soft coupling + barycentric projection:
- Template C_bar = B B^T learned from population
- Soft coupling pi via EMD (POT ot.emd)
- Projection: C_al = P C_t P^T where P = pi/row_sums

This projection MIXES subject-specific connectivity patterns toward the template. Identification accuracy depends on BETWEEN-subject differences (self-pair corr ~0.67 vs cross-subject corr ~0.46 on raw data). Projection reduces between-subject variance -> lower identifiability.

### The fundamental tension
- Track A (identification) requires PRESERVING individual differences
- Track B (alignment_gain) requires INCREASING cross-subject similarity
- These are mathematically opposed: any alignment that raises cross-subject correlation necessarily reduces identifiability
- conn_srm confirms this: gain +0.395 but ident collapses to 0.082 (chance)

### Ceiling effect
Raw Schaefer-100 connectomes already achieve ident=0.94-0.96 at N=30-49. There is almost no headroom for alignment methods to improve identification. noalign/brainsync score 0.959 because raw connectomes are already highly identifiable.

## What PLAN.md Actually Claims

Section 7.5 (Track B guaranteed result): "run each existing method, then report how many subject pairs are not identifiable"

The defensible contribution is NOT "beats baselines on identification" - it is:
1. **Calibrated uncertainty reporting**: tau_phi=0.075 (full) vs 0.103 (ablated) - only ours produces per_pair_uncertainty
2. **Positive gain without ident collapse**: +0.018 vs fugw -0.005 / noalign 0, while preserving ident at 0.857
3. **Non-identifiability counting**: the guaranteed result that no baseline produces

## Gauge Ablation Finding

The gauge-feature ablation is a NULL result on transform metrics:
- Transform is region-level C_pop OT + Procrustes, independent of gauge features
- Only tau_phi moves: 0.075 (full) vs 0.103 (ablated)
- Higher posterior width without gauge features suggests gauge helps uncertainty calibration, not transform quality

## What Would Need to Change to Beat noalign on Track A

1. **Orthogonal Q transform**: preserve spectrum instead of barycentric projection
2. **Optimize for within-subject reliability**: scan-rescan correlation, not cross-subject gain
3. **Harder evaluation protocol**: fewer regions, cross-session, noise - where raw ident is NOT at ceiling
4. **Reframe the claim**: ours wins on Track B non-identifiability reporting, not Track A identification

## Conclusion

TrajOT does NOT beat baselines on Track A identification. The defensible scientific contribution is calibrated posterior uncertainty (tau_phi) + positive alignment gain without ident collapse + non-identifiability reporting. This matches PLAN.md's stated guaranteed result (Section 7.5), not a claim of superior fingerprinting.
