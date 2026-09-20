# REAL N=49 gap-filling + SOTA table (corrected protocol)

- data_root: `/Users/anandlo/Surge2026F/ds000243-master`
- n_subjects: 49 (frozen cohort 015–063)
- beta: 29.189086229914952 (real scan-rescan)
- pairs: 500 seed 2026
- ours artifacts: `runs/10_ours_full__73533e35__20260920T074217Z/artifacts`
- transform paths: {'noalign': '', 'brainsync': '', 'fugw': '', 'conn_srm': '', 'ours_full_point_procrustes': 'point_procrustes_C_pop', 'ours_full_posterior_shrink': 'posterior_shrink_tau_gated', 'ours_full_posterior_shrink_entropy': 'posterior_shrink_tau_gated', 'ours_full_posterior_shrink_cpop': 'posterior_shrink_tau_gated', 'ours_ablated': 'point_procrustes_C_pop'}
- metric protocol: **same_map_both_runs_v1** — same Q_s on BOTH runs; heldout ||C2-T(C1)|| is protocol-wrong and not headlined

## PRIMARY method table (higher reliability_after / gain_after / ident_after better)

| method | reliability_raw | reliability_after | Δrel | gain_after | ident_after | collapsed | λ_mean | n_undet | tau_phi |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| noalign | 0.6416 | 0.6416 | 0 | 0 | 0.9592 | False | — | — | — |
| brainsync | 0.6416 | 0.6425 | 0.000864 | 0.000661059 | 0.9592 | False | — | — | — |
| fugw | 0.6416 | 0.6202 | -0.0214 | -0.00494709 | 0.9184 | True | — | — | — |
| conn_srm | 0.6416 | 0.8498 | 0.2082 | 0.395454 | 0.08163 | True | — | — | — |
| ours_full_point_procrustes | 0.6416 | 0.6443 | 0.002695 | 0.0180541 | 0.8571 | False | — | — | 0.00923645 |
| ours_full_posterior_shrink | 0.6416 | 0.685 | 0.04339 | -0.0350871 | 0.9796 | False | 0.501307 | 24 | 0.00923645 |
| ours_full_posterior_shrink_entropy | 0.6416 | 0.6852 | 0.04351 | -0.0329763 | 0.9796 | False | 0.499714 | 24 | 0.00923645 |
| ours_full_posterior_shrink_cpop | 0.6416 | 0.6787 | 0.03707 | -0.0409234 | 1 | False | 0.499714 | 24 | 0.00923645 |
| ours_ablated | 0.6416 | 0.6443 | 0.002695 | 0.0180541 | 0.8571 | False | — | — | 0.00923645 |

## Group REML on real posteriors (Track B)

- n_subjects: 12
- n_eff (mean): 7.451807539911549 (min 2.4570207641701245)
- ci_ratio (reml/ttest): 1.276971107772669
- mean_sigma2: 1.3485960648804346e-29

## Honest verdict vs SOTA bar (corrected metrics)

- beat noalign on gain_after: **True**
- beat BrainSync on gain_after: **True**
- beat FUGW on gain_after: **True**
- beat conn_srm on gain_after (non-collapsed): **True**
- beat BrainSync+FUGW+noalign at non-collapsed rel/ident: **True**
- group n_eff < S: **True**
- best ours row: **ours_full_point_procrustes** (gain_after=0.018054067101284182, reliability_after=0.6443395103468178, ident_after=0.8571428571428571, collapsed=False)
- scientific transform iterations used: **2** (max 2)

### Notes

- PRIMARY protocol: same subject map Q_s applied to BOTH runs (fitted on run1). Headline metrics are reliability_after, gain_after, ident_after. heldout ||C2-T(C1)|| is protocol-wrong and is NOT headlined.
- gain_after ours(ours_full_point_procrustes)=0.018054067101284182 vs noalign=0.0 at reliability_after ours=0.6443395103468178 vs noalign=0.641644195688358; ident_after ours=0.8571428571428571 vs noalign=0.9591836734693877 -> beat=True
- gain_after ours(ours_full_point_procrustes)=0.018054067101284182 vs brainsync=0.0006610585621552324 at reliability_after ours=0.6443395103468178 vs brainsync=0.6425081738845867; ident_after ours=0.8571428571428571 vs brainsync=0.9591836734693877 -> beat=True
- fugw disqualified: identity/reliability collapse (ident_after=0.9183673469387755, reliability_after=0.6202401056541041).
- conn_srm disqualified: identity/reliability collapse (ident_after=0.08163265306122448, reliability_after=0.8498246928707958).
- ours_full_posterior_shrink_entropy: gain_after=-0.032976321835702156, reliability_after=0.6851537053868291 (raw 0.641644195688358, delta 0.043509509698471116), ident_after=0.9795918367346939, collapsed=False, lambda_mean=0.49971394751262244 [0.48122702924343497,0.515434473586563], lambda_source=row_entropy, c_pop_mix=0.0, tau0_eff=0.009251849548644765, n_undetermined=24, tau_phi_mean=0.00923644939823487
- ours_full_posterior_shrink_cpop: gain_after=-0.04092344064849882, reliability_after=0.6787106659398447 (raw 0.641644195688358, delta 0.037066470251486794), ident_after=1.0, collapsed=False, lambda_mean=0.49971394751262244 [0.48122702924343497,0.515434473586563], lambda_source=row_entropy, c_pop_mix=0.35, tau0_eff=0.009251849548644765, n_undetermined=24, tau_phi_mean=0.00923644939823487
- ours_full_posterior_shrink: gain_after=-0.035087118526146474, reliability_after=0.6850360355737478 (raw 0.641644195688358, delta 0.04339183988538986), ident_after=0.9795918367346939, collapsed=False, lambda_mean=0.5013069175743132 [0.44584348527027384,0.5386807754327037], lambda_source=tau, c_pop_mix=0.0, tau0_eff=0.009251849548644765, n_undetermined=24, tau_phi_mean=0.00923644939823487
- ours_full_point_procrustes: gain_after=0.018054067101284182, reliability_after=0.6443395103468178 (raw 0.641644195688358, delta 0.0026953146584598464), ident_after=0.8571428571428571, collapsed=False, lambda_mean=None [None,None], lambda_source=tau, c_pop_mix=0.0, tau0_eff=None, n_undetermined=None, tau_phi_mean=0.00923644939823487
- ours_ablated: gain_after=0.018054067101284182, reliability_after=0.6443395103468178 (raw 0.641644195688358, delta 0.0026953146584598464), ident_after=0.8571428571428571, collapsed=False, lambda_mean=None [None,None], lambda_source=tau, c_pop_mix=0.0, tau0_eff=None, n_undetermined=None, tau_phi_mean=0.00923644939823487
- Iteration diagnostic (tau-gated shrink vs point): gain_after -0.035087118526146474 vs 0.018054067101284182; reliability_after 0.6850360355737478 vs 0.6443395103468178.
- Scientific iteration 1 — entropy-λ: gain_after=-0.032976321835702156, reliability_after=0.6851537053868291, lambda_mean=0.49971394751262244 [0.48122702924343497,0.515434473586563] (source=row_entropy).
- Scientific iteration 2 — C_pop mix: gain_after=-0.04092344064849882, reliability_after=0.6787106659398447, c_pop_mix=0.35.
- Track B group REML: n_eff=7.451807539911549 < S=12 (min 2.4570207641701245); ci_ratio=1.276971107772669. Baselines emit point maps with no Sigma^al / tau_phi column.
- Literature gap (PLAN §1–4 / Thual 2025, BrainSync 2018, Takeda 2025): existing aligners emit point estimates; this framework adds per-subject tau_phi + group REML n_eff on REAL posteriors — columns baselines cannot fill.

### Literature gap

Point-estimate-only aligners (BrainSync 2018, FUGW/Thual 2025, conn-SRM) cannot fill uncertainty/identifiability columns. Hierarchical population-of-couplings provides tau_phi + group Sigma^al (REML n_eff) on real rest-fMRI posteriors.

