# REAL N=49 gap-filling + SOTA table

- data_root: `/Users/anandlo/Surge2026F/ds000243-master`
- n_subjects: 49 (frozen cohort 015–063)
- beta: 29.189086229914952 (real scan-rescan)
- pairs: 500 seed 2026
- ours artifacts: `runs/10_ours_full__73533e35__20260920T074217Z/artifacts`
- transform paths: {'noalign': '', 'brainsync': '', 'fugw': '', 'conn_srm': '', 'ours_full_posterior_shrink': 'posterior_shrink_tau_gated', 'ours_full_point_procrustes': 'point_procrustes_C_pop', 'ours_full_c_bar_procrustes': 'c_bar_procrustes', 'ours_ablated': 'point_procrustes_C_pop'}

## Method table (lower heldout_error better; higher gain / scan-rescan better)

| method | heldout_score_module | heldout_error | alignment_gain | ident_acc | scan_resc_raw | scan_resc_after | tau_phi_mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| noalign | -701.646 | 701.646 | 0 | 0.9592 | 0.6416 | 0.6416 | — |
| brainsync | -703.51 | 703.51 | 0.000661059 | 0.9592 | 0.6416 | 0.6425 | — |
| fugw | -704.002 | 704.002 | -0.00494709 | 0.9184 | 0.6416 | 0.6202 | — |
| conn_srm | -863.691 | 863.691 | 0.395454 | 0.08163 | 0.6416 | 0.8498 | — |
| ours_full_posterior_shrink | -774.769 | 774.769 | -0.0350871 | 0.9796 | 0.6416 | 0.685 | 0.00923645 |
| ours_full_point_procrustes | -783.447 | 783.447 | 0.0180541 | 0.8571 | 0.6416 | 0.6443 | 0.00923645 |
| ours_full_c_bar_procrustes | -1617.3 | 1617.3 | -0.400457 | 1 | 0.6416 | 0.6428 | 0.00923645 |
| ours_ablated | -783.447 | 783.447 | 0.0180541 | 0.8571 | 0.6416 | 0.6443 | 0.00923645 |

## Group REML on real posteriors

- n_subjects: 12
- n_eff (mean over nodes): 7.451807539911549 (min 2.4570207641701245)
- mean_weight: 0.08333333333333333
- mean_sigma2: 1.3485960648804346e-29
- reml_theta_se: 7.131790864992034e-16
- ttest_theta_se: 5.920015715688613e-16
- ci_ratio (reml/ttest): 1.276971107772669

## Honest verdict vs SOTA bar

- heldout beats noalign: **False**
- heldout beats fugw: **False**
- ours alignment_gain > 0: **False**
- group n_eff < S: **True**

### Notes

- HONEST: best ours heldout_score_module does NOT beat noalign (ours_full_posterior_shrink -774.769 vs noalign -701.646).
- METRIC CAVEAT: heldout_score_module = -||C_run2 - T(C_run1)||^2 compares template-remapped run1 to *native-gauge* run2. Any nontrivial spatial reindexing (Q≠I) inflates this residual even when alignment_gain improves. Read heldout_score_module jointly with alignment_gain and scanrescan_corr_after.
- ours vs fugw heldout_score_module: -774.769 vs -704.002 -> does NOT beat fugw.
- hierarchy-on posterior_shrink vs point_procrustes: heldout -774.769 vs -783.447; scan-rescan after 0.6850360355737478 vs 0.6443395103468178; gain -0.035087118526146474 vs 0.018054067101284182.
- SOTA scientific retry (one shot, applied): transform_mode=c_bar_procrustes (EMD to learned C_bar=BB^T). heldout=-1617.3, gain=-0.400457387157253 — RETRY FAILED (BB^T spectral template is a poor EMD target vs empirical C_pop). Max one retry; no further invented wins.
- SOTA baselines on REAL N=49: BrainSync heldout=-703.51 gain=0.0006610585621552324 ident=0.9591836734693877 vs noalign heldout=-701.646. BrainSync is near a no-op on region-parcellated rest connectomes (consistent with XQQ^T X^T = XX^T).
- FUGW heldout=-704.002 gain=-0.004947090344352626 ident=0.9183673469387755 — does not beat noalign on heldout/gain at N=49.
- best ours (ours_full_posterior_shrink) vs point_procrustes heldout: -774.769 vs -783.447; alignment_gain -0.035087118526146474 vs 0.018054067101284182.
- best ours alignment_gain=-0.035087118526146474 (positive=False); tau_phi_mean=0.00923644939823487 (uncertainty column baselines cannot fill).
- alignment_gain vs noalign: -0.035087118526146474 vs 0.0 at N=49.
- conn_srm alignment_gain=0.3954539023443999 with ident=0.08163265306122448 — high gain via identity collapse; not a valid SOTA alignment win.
- group REML: n_eff=7.451807539911549 < n_subjects=12 means alignment uncertainty down-weights subjects; ci_ratio=1.276971107772669.
- Gap metric PASS: n_eff < S under REML on real posteriors.
- ID ceiling: noalign ident_accuracy=0.9592 — identification cannot discriminate methods at Schaefer-100 N=49.

### Literature gap (the point)

Existing rest-fMRI aligners emit point estimates only. This framework produces calibrated uncertainty (tau_phi / posterior coverage), identifiability flags, and group REML with alignment covariance Sigma^al — columns baselines leave null.

Uncertainty-only wins are insufficient: this table reports planted-free REAL heldout / gain / scan-rescan next to tau_phi and group n_eff.

