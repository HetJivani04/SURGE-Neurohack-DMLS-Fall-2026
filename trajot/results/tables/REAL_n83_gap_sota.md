# REAL_n83_gap_sota gap-filling + SOTA table (corrected protocol)

- data_root: `/Users/anandlo/Surge2026F/ds000243-master`
- n_subjects: 83
- beta: 28.438323293411973 (n49 ref 29.189086229914952; n83 ref 28.438323293411973)
- pairs: 500 seed 2026
- permutations_B: 200
- ours artifacts: `runs/10_ours_full__73533e35__20260920T074217Z/artifacts`
- transform paths: {'noalign': '', 'fugw': '', 'conn_srm': '', 'ours_full_point_procrustes': 'point_procrustes_C_pop', 'ours_full_posterior_shrink': 'region_emd_procrustes', 'ours_full_posterior_shrink_entropy': 'region_emd_procrustes', 'ours_full_posterior_shrink_cpop': 'region_emd_procrustes', 'ours_ablated': 'point_procrustes_C_pop'}
- metric protocol: **same_map_both_runs_v1** — same Q_s on BOTH runs; heldout ||C2-T(C1)|| is protocol-wrong and not headlined

## PRIMARY method table (higher reliability_after / gain_after / ident_after better)

| method | reliability_raw | reliability_after | Δrel | gain_after | ident_after | collapsed | λ_mean | n_undet | tau_phi |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| noalign | 0.6455 | 0.6455 | 0 | 0 | 0.9157 | False | — | — | — |
| fugw | 0.6715 | 0.6492 | -0.02236 | -0.0024681 | 1 | True | — | — | — |
| conn_srm | 0.6455 | 0.8453 | 0.1998 | 0.38622 | 0.0241 | True | — | — | — |
| ours_full_point_procrustes | 0.6455 | 0.6483 | 0.002817 | 0.00569748 | 0.8434 | False | — | — | 0.00923645 |
| ours_full_posterior_shrink | 0.6455 | 0.6483 | 0.002817 | 0.00569748 | 0.8434 | False | — | — | 0.00923645 |
| ours_full_posterior_shrink_entropy | 0.6455 | 0.6483 | 0.002817 | 0.00569748 | 0.8434 | False | — | — | 0.00923645 |
| ours_full_posterior_shrink_cpop | 0.6455 | 0.6483 | 0.002817 | 0.00569748 | 0.8434 | False | — | — | 0.00923645 |
| ours_ablated | 0.6455 | 0.6483 | 0.002817 | 0.00569748 | 0.8434 | False | — | — | 0.00923645 |

## Group REML on real posteriors (Track B)

- n_subjects: 12
- n_eff (mean): 7.451807539911549 (min 2.4570207641701245)
- ci_ratio (reml/ttest): 1.276971107772669
- mean_sigma2: 1.3485960648804346e-29

## Honest verdict vs SOTA bar (corrected metrics)

- beat noalign on gain_after: **True**
- beat BrainSync on gain_after: **None**
- beat FUGW on gain_after: **True**
- beat conn_srm on gain_after (non-collapsed): **True**
- beat BrainSync+FUGW+noalign at non-collapsed rel/ident: **False**
- group n_eff < S: **True**
- best ours row: **ours_full_posterior_shrink_entropy** (gain_after=0.005697483890785939, reliability_after=0.6483402777122557, ident_after=0.8433734939759037, collapsed=False)
- scientific transform iterations used: **2** (max 2)

### Notes

- PRIMARY protocol: same subject map Q_s applied to BOTH runs (fitted on run1). Headline metrics are reliability_after, gain_after, ident_after. heldout ||C2-T(C1)|| is protocol-wrong and is NOT headlined.
- gain_after ours(ours_full_posterior_shrink_entropy)=0.005697483890785939 vs noalign=0.0 at reliability_after ours=0.6483402777122557 vs noalign=0.6455234624097494; ident_after ours=0.8433734939759037 vs noalign=0.9156626506024096 -> beat=True
- fugw disqualified: identity/reliability collapse (ident_after=1.0, reliability_after=0.6491834781767977).
- conn_srm disqualified: identity/reliability collapse (ident_after=0.024096385542168676, reliability_after=0.8452933675685169).
- ours_full_posterior_shrink_entropy: gain_after=0.005697483890785939, reliability_after=0.6483402777122557 (raw 0.6455234624097494, delta 0.002816815302506215), ident_after=0.8433734939759037, collapsed=False, lambda_mean=None [None,None], lambda_source=row_entropy, c_pop_mix=0.0, tau0_eff=None, n_undetermined=None, tau_phi_mean=0.00923644939823487
- ours_full_posterior_shrink_cpop: gain_after=0.005697483890785939, reliability_after=0.6483402777122557 (raw 0.6455234624097494, delta 0.002816815302506215), ident_after=0.8433734939759037, collapsed=False, lambda_mean=None [None,None], lambda_source=row_entropy, c_pop_mix=0.35, tau0_eff=None, n_undetermined=None, tau_phi_mean=0.00923644939823487
- ours_full_posterior_shrink: gain_after=0.005697483890785939, reliability_after=0.6483402777122557 (raw 0.6455234624097494, delta 0.002816815302506215), ident_after=0.8433734939759037, collapsed=False, lambda_mean=None [None,None], lambda_source=tau, c_pop_mix=0.0, tau0_eff=None, n_undetermined=None, tau_phi_mean=0.00923644939823487
- ours_full_point_procrustes: gain_after=0.005697483890785939, reliability_after=0.6483402777122557 (raw 0.6455234624097494, delta 0.002816815302506215), ident_after=0.8433734939759037, collapsed=False, lambda_mean=None [None,None], lambda_source=tau, c_pop_mix=0.0, tau0_eff=None, n_undetermined=None, tau_phi_mean=0.00923644939823487
- ours_ablated: gain_after=0.005697483890785939, reliability_after=0.6483402777122557 (raw 0.6455234624097494, delta 0.002816815302506215), ident_after=0.8433734939759037, collapsed=False, lambda_mean=None [None,None], lambda_source=tau, c_pop_mix=0.0, tau0_eff=None, n_undetermined=None, tau_phi_mean=0.00923644939823487
- Iteration diagnostic (tau-gated shrink vs point): gain_after 0.005697483890785939 vs 0.005697483890785939; reliability_after 0.6483402777122557 vs 0.6483402777122557.
- Scientific iteration 1 — entropy-λ: gain_after=0.005697483890785939, reliability_after=0.6483402777122557, lambda_mean=None [None,None] (source=row_entropy).
- Scientific iteration 2 — C_pop mix: gain_after=0.005697483890785939, reliability_after=0.6483402777122557, c_pop_mix=0.35.
- Track B group REML: n_eff=7.451807539911549 < S=12 (min 2.4570207641701245); ci_ratio=1.276971107772669. Baselines emit point maps with no Sigma^al / tau_phi column.
- Literature gap (PLAN §1–4 / Thual 2025, BrainSync 2018, Takeda 2025): existing aligners emit point estimates; this framework adds per-subject tau_phi + group REML n_eff on REAL posteriors — columns baselines cannot fill.
- HONEST: ours does not beat BrainSync+FUGW+noalign on gain_after at non-collapsed reliability/ident after up to 2 scientific transform iterations.

### Literature gap

Point-estimate-only aligners (BrainSync 2018, FUGW/Thual 2025, conn-SRM) cannot fill uncertainty/identifiability columns. Hierarchical population-of-couplings provides tau_phi + group Sigma^al (REML n_eff) on real rest-fMRI posteriors.

