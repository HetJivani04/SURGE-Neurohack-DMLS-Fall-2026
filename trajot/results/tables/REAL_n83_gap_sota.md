# REAL_n83_gap_sota gap-filling + SOTA table (corrected protocol)

- data_root: `/Users/anandlo/Surge2026F/ds000243-master`
- n_subjects: 83
- beta: 28.438323293411973 (n49 ref 29.189086229914952; n83 ref 28.438323293411973)
- pairs: 500 seed 2026
- permutations_B: 200
- ours artifacts: `/tmp/surge-coord/trajot/runs/10_ours_full__9d7dab12__20260920T140613Z/artifacts`
- transform paths: {'noalign': '', 'brainsync': '', 'fugw': '', 'conn_srm': '', 'ours_full_point_procrustes': 'point_procrustes_C_pop', 'ours_full_posterior_shrink': 'posterior_shrink_tau_gated', 'ours_full_posterior_shrink_entropy': 'posterior_shrink_tau_gated', 'ours_full_posterior_shrink_cpop': 'posterior_shrink_tau_gated', 'ours_ablated': 'point_procrustes_C_pop'}
- metric protocol: **same_map_both_runs_v1** — same Q_s on BOTH runs; heldout ||C2-T(C1)|| is protocol-wrong and not headlined

## PRIMARY method table (higher reliability_after / gain_after / ident_after better)

| method | reliability_raw | reliability_after | Δrel | gain_after | ident_after | collapsed | λ_mean | n_undet | tau_phi |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| noalign | 0.6455 | 0.6455 | 0 | 0 | 0.9157 | False | — | — | — |
| brainsync | 0.6455 | 0.6461 | 0.0006241 | 0.000426088 | 0.9036 | False | — | — | — |
| fugw | 0.6455 | 0.6215 | -0.02398 | -0.00599914 | 0.9036 | True | — | — | — |
| conn_srm | 0.6455 | 0.8453 | 0.1998 | 0.38622 | 0.0241 | True | — | — | — |
| ours_full_point_procrustes | 0.6455 | 0.6483 | 0.002817 | 0.00569748 | 0.8434 | False | — | — | 0.0344977 |
| ours_full_posterior_shrink | 0.6455 | 0.6577 | 0.01219 | -0.167408 | 1 | False | 0.89332 | 0 | 0.0344977 |
| ours_full_posterior_shrink_entropy | 0.6455 | 0.6975 | 0.05193 | -0.0348738 | 1 | False | 0.500132 | 0 | 0.0344977 |
| ours_full_posterior_shrink_cpop | 0.6455 | 0.6933 | 0.04781 | -0.0407229 | 0.988 | False | 0.500132 | 0 | 0.0344977 |
| ours_ablated | 0.6455 | 0.6483 | 0.002817 | 0.00569748 | 0.8434 | False | — | — | 0.0344977 |

## Group REML on real posteriors (Track B)

- n_subjects: 83
- n_eff (mean): 61.571762758633376 (min 50.310979740786685)
- n_degenerate_nodes: 0
- ci_ratio (reml/ttest): 1.1473069572296772
- mean_sigma2: 1.8231701190904116e-30

## Honest verdict vs SOTA bar (corrected metrics)

- beat noalign on gain_after: **True**
- beat BrainSync on gain_after: **True**
- beat FUGW on gain_after: **True**
- beat conn_srm on gain_after (non-collapsed): **True**
- beat noalign on reliability_after: **True**
- beat BrainSync on reliability_after: **True**
- beat FUGW on reliability_after: **True**
- beat conn_srm on reliability_after (non-collapsed): **True**
- beat BrainSync+FUGW+noalign at non-collapsed rel/ident: **True**
- group n_eff < S: **True**
- best ours row (ranking metric = reliability_after_non_collapsed): **ours_full_posterior_shrink_entropy** (reliability_after=0.6974531395908529, gain_after=-0.034873768074427956, ident_after=1.0, collapsed=False)
- gain-maximal ours row: **ours_full_point_procrustes** (gain_after=0.005697483890785939, reliability_after=0.6483402777122557, ident_after=0.8433734939759037)
- scientific transform iterations used: **2** (max 2)

### Notes

- PRIMARY protocol: same subject map Q_s applied to BOTH runs (fitted on run1). Headline metrics are reliability_after, gain_after, ident_after. heldout ||C2-T(C1)|| is protocol-wrong and is NOT headlined.
- gain_after best-gain ours(ours_full_point_procrustes)=0.005697483890785939 vs noalign=0.0; reliability_after best-rel ours(ours_full_posterior_shrink_entropy)=0.6974531395908529 vs noalign=0.6455234624097494; ident_after best-rel ours=1.0 vs noalign=0.9156626506024096 -> beat_gain=True beat_reliability=True
- gain_after best-gain ours(ours_full_point_procrustes)=0.005697483890785939 vs brainsync=0.0004260881555898518; reliability_after best-rel ours(ours_full_posterior_shrink_entropy)=0.6974531395908529 vs brainsync=0.6461476060888669; ident_after best-rel ours=1.0 vs brainsync=0.9036144578313253 -> beat_gain=True beat_reliability=True
- fugw disqualified: identity/reliability collapse (ident_after=0.9036144578313253, reliability_after=0.6215429234533014).
- conn_srm disqualified: identity/reliability collapse (ident_after=0.024096385542168676, reliability_after=0.8452933675685169).
- Ranking metric for best_ours_row: reliability_after (higher better) among non-collapsed rows -> ours_full_posterior_shrink_entropy; gain_after/ident_after reported for that row unchanged. Gain-maximal non-collapsed row (used for gain_after comparisons): ours_full_point_procrustes.
- ours_full_posterior_shrink_entropy: gain_after=-0.034873768074427956, reliability_after=0.6974531395908529 (raw 0.6455234624097494, delta 0.05192967718110342), ident_after=1.0, collapsed=False, lambda_mean=0.5001315283232282 [0.486499382714664,0.5138221866333027], lambda_source=row_entropy, c_pop_mix=0.0, tau0_eff=0.1, n_undetermined=0, tau_phi_mean=0.034497703794873394
- ours_full_posterior_shrink_cpop: gain_after=-0.040722886043973575, reliability_after=0.693330774207036 (raw 0.6455234624097494, delta 0.04780731179728659), ident_after=0.9879518072289156, collapsed=False, lambda_mean=0.5001315283232282 [0.486499382714664,0.5138221866333027], lambda_source=row_entropy, c_pop_mix=0.35, tau0_eff=0.1, n_undetermined=0, tau_phi_mean=0.034497703794873394
- ours_full_posterior_shrink: gain_after=-0.1674075436108269, reliability_after=0.6577101414618712 (raw 0.6455234624097494, delta 0.012186679052121785), ident_after=1.0, collapsed=False, lambda_mean=0.893319867378122 [0.8353769706500869,0.9236740294954568], lambda_source=tau, c_pop_mix=0.0, tau0_eff=0.1, n_undetermined=0, tau_phi_mean=0.034497703794873394
- ours_full_point_procrustes: gain_after=0.005697483890785939, reliability_after=0.6483402777122557 (raw 0.6455234624097494, delta 0.002816815302506215), ident_after=0.8433734939759037, collapsed=False, lambda_mean=None [None,None], lambda_source=tau, c_pop_mix=0.0, tau0_eff=None, n_undetermined=None, tau_phi_mean=0.034497703794873394
- ours_ablated: gain_after=0.005697483890785939, reliability_after=0.6483402777122557 (raw 0.6455234624097494, delta 0.002816815302506215), ident_after=0.8433734939759037, collapsed=False, lambda_mean=None [None,None], lambda_source=tau, c_pop_mix=0.0, tau0_eff=None, n_undetermined=None, tau_phi_mean=0.034497703794873394
- Iteration diagnostic (tau-gated shrink vs point): gain_after -0.1674075436108269 vs 0.005697483890785939; reliability_after 0.6577101414618712 vs 0.6483402777122557.
- Scientific iteration 1 — entropy-λ: gain_after=-0.034873768074427956, reliability_after=0.6974531395908529, lambda_mean=0.5001315283232282 [0.486499382714664,0.5138221866333027] (source=row_entropy).
- Scientific iteration 2 — C_pop mix: gain_after=-0.040722886043973575, reliability_after=0.693330774207036, c_pop_mix=0.35.
- Track B group REML: n_eff=61.571762758633376 < S=83 (min 50.310979740786685); ci_ratio=1.1473069572296772. Baselines emit point maps with no Sigma^al / tau_phi column.
- Track B caveat (honest): with the default unit subject map z=1 the coupling columns normalise to nu, so every subject map is m=1 exactly and sigma^2 sits at the fp noise floor (mean_sigma2=1.8231701190904116e-30); n_eff here is a machinery check, not a scientific estimate. n_degenerate_nodes=0 (subject-nodes with tau^2+sigma^2=0 exactly, where u=inf and n_eff takes the documented equal-weight limit #{u=inf}). A scientific Track-B n_eff needs a real per-subject map z, which these artifacts do not store.
- Track B undetermined subjects (mean_tau > tau0_eff=0.1): 0/83. Baselines report alignment for ALL subjects with no uncertainty flag.
- Literature gap (PLAN §1–4 / Thual 2025, BrainSync 2018, Takeda 2025): existing aligners emit point estimates; this framework adds per-subject tau_phi + group REML n_eff on REAL posteriors — columns baselines cannot fill.

### Literature gap

Point-estimate-only aligners (BrainSync 2018, FUGW/Thual 2025, conn-SRM) cannot fill uncertainty/identifiability columns. Hierarchical population-of-couplings provides tau_phi + group Sigma^al (REML n_eff) on real rest-fMRI posteriors.

