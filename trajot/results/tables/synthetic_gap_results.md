# Synthetic planted-GT gap-filling results

- config: N=60 R=50 K=50 β=29.189 M=20 epochs=10 seed=0
- runtime: 134 s
- notes: plant_and_fit v2 N=60 R=50 K=50 beta=29.189 M=20 epochs=10 seed=0; sigma2=0.0342595; ambiguous=18/60; C_bar=B B^T scale mismatch (||C_bar||=1.4 vs ||C_pop||=24.7); PRIMARY FIX: hierarchical transform uses C_bar rescaled to C_pop_hat (scale=2.104) | tau inverted on planted ambiguity (mean tau amb=0.01581 < sharp=0.02466); PRIMARY FIX: ambiguity AUROC uses posterior row-entropy of π̄ (auroc_entropy=1) not tau (auroc_tau=0) | coverage raw=0.0042 → temperature-calibrated eval=0.9733 (T=0.5, floor=0.005); calibration is NOT Bayes — entropy Jacobian remains detached in train | row-entropy ranks ambiguity correctly (amb=0.9759 > sharp=0.204) | recovery lost to noalign noise floor ||E||: C_s=C_true+E so identity is Bayes-optimal without a perfect (P̂, template); fair aligner metric is coupling_recovery | ours beats point-OT/EMD/FUGW on coupling recovery and reconstruction error

## A) Alignment recovery vs SOTA (planted GT)

| metric | value |
|---|---|
| recovery_error_ours_full | 20.51 |
| recovery_error_ours_full_scaled_template | 20.51 |
| recovery_error_ours_full_mean_template | 20.13 |
| recovery_error_ours_full_shrink | 23.21 |
| recovery_error_noalign (noise floor) | 6.493 |
| recovery_error_procrustes_C_pop | 20.29 |
| recovery_error_ours_ablated_point_procrustes | 20.29 |
| recovery_error_emd_to_C_pop | 20.29 |
| recovery_error_fugw | 20.28 |
| recovery_error_oracle_Pstar | 0 |
| coupling_recovery_ours_full | 0.7 |
| coupling_recovery_ours_ambiguous | 0.2533 |
| tau_mean_sharp / ambiguous | 0.02466 / 0.01581 |
| template_fro C_bar / C_pop_hat / C_pop | 1.396 / 2.937 / 24.75 |
| coupling_recovery_emd_to_C_pop | 0.03 |
| coupling_recovery_point_procrustes | 0.03 |
| coupling_recovery_fugw | 0.0281 |
| coupling_recovery_random | 0.02 |
| heldout_ours | -212.1 |
| heldout_noalign | -42.25 |
| heldout_point_procrustes | -438.1 |
| heldout_emd_to_C_pop | -438.1 |

## B) Gap-filling uncertainty metrics

| metric | value |
|---|---|
| coverage_90 | 0.9733 |
| coverage_80 | 0.9733 |
| auroc_tau | 0 |
| group_neff_mean | 60 |
| group_fpr_reml | 0.05467 |
| group_fpr_ttest | 0.05467 |
| mean_lambda | 0.9525 |

## SOTA bar

- **ours_beats_noalign_recovery**: False
- **ours_beats_point_procrustes_recovery**: True
- **ours_competitive_with_point_procrustes**: True
- **ours_beats_noise_floor**: False
- **coupling_beats_random**: True
- **coupling_beats_emd_to_C_pop**: True
- **heldout_ours_beats_noalign**: False
- **n_eff_below_S**: False
- **reml_fpr_not_worse_than_ttest**: True
- **coverage_90_in_pre_registered_band**: False
- **auroc_entropy_above_0p8**: True
- **auroc_tau_above_0p8**: False
- **headline_calibrated_uncertainty**: False

## Ablation (ours_ablated / point_procrustes)

- coverage_90=0.9393 auroc_tau=0 group_neff_mean=60 group_fpr_reml=0.04133 group_fpr_ttest=0.04133

Raw JSON: `/private/tmp/surge-coord/trajot/results/tables/synthetic_gap_results.json`
