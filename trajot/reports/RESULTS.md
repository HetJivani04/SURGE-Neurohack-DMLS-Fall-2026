# Results: hierarchical population-of-couplings for rest-fMRI alignment

**Status: Phase 2 REAL N=49 statistical closer complete. Cohort 015–063, ds000243, Schaefer-100, beta = 29.189 (scan-rescan calibrated). Same subject-level map Q_s is applied to both runs. This document reports what the frozen comparison and the 10k paired bootstrap actually produced. It does not invent wins.**

Canonical tables:
- `trajot/results/tables/REAL_n49_gap_sota.json` — method-level REAL harness
- `trajot/results/tables/REAL_sota_stats.json` — same-Q verification + bootstrap + gap columns
- `trajot/results/tables/REAL_sota_stats.md` — compact SOTA + gap markdown
- `trajot/results/tables/group_real_n49.json` — group REML on real posteriors
- `trajot/results/tables/synthetic_gap_results.json` — planted-GT coupling recovery sidecar

Reproduce the statistical verification:

```bash
cd trajot
PYTHONPATH=src python scripts/verify_real_sota.py \
  --data-root /Users/anandlo/Surge2026F/ds000243-master \
  --ours-artifacts runs/10_ours_full__73533e35__20260920T074217Z/artifacts \
  --n-boot 10000
```

---

## 1. Literature gap (why these columns exist)

**PLAN §1, in one sentence:** for resting-state fMRI you cannot tell whether a cross-subject functional alignment is real, because every method returns a map and none reports whether that map captured anything.

Verbatim class quotes that motivate the gap columns:

- **Thual et al. 2025, TMLR** (FUGW follow-up): *"our approach currently requires left-out participants to watch the same stimuli as reference participants. It is yet unclear whether functional alignment could bring improvements without this constraint."*
- **BrainSync, Joshi et al. 2018, NeuroImage**: *"the BrainSync transform will always attempt to maximize correlations, resulting in some degree of positive correlation even for data that do not satisfy our underlying assumption of common networks."*
- **Takeda et al. 2025, iScience**: *"In both datasets we analyzed in this study, the results of unsupervised alignment at the individual level were statistically unreliable."*
- **Haxby et al. 2020, eLife**: the general validity of a common rs-fMRI alignment model *"has not yet been established."*
- **Bazeille et al. 2021**: it *"remains unclear how researchers should choose among the available functional alignment methods."*

**Unclaimed object (PLAN §3–§4):** a hierarchical population-of-couplings with group Σ^al / τ_φ — a posterior over subject-to-template maps, shrinkage toward a population coupling, and group REML that down-weights subjects whose alignment the data do not determine. Point-estimate SOTA baselines (noalign, BrainSync, FUGW, conn_srm) emit a map for every subject and leave uncertainty columns null.

---

## 2. Comparable SOTA table — REAL N=49 (posterior_shrink path)

Artifacts: `runs/10_ours_full__73533e35__20260920T074217Z` (gauge on, hierarchical π shrink, tau0_auto). Transform: posterior-gated `C̃ = (1−λ)C + λ Q_s^T C Q_s` with the **same** `(Q_s, λ_s)` for both runs of subject s. λ from τ_φ via `λ = 1/(1+(τ/τ0)²)`, `tau0_eff ≈ median(τ)`.

| Method | Ident | Scan-rescan after | Alignment gain | τ_φ | Group n_eff | Valid SOTA competitor? |
|---|---:|---:|---:|---:|---:|---|
| noalign | 0.959 | 0.642 | 0.000 | — | — | yes |
| BrainSync (rest) | 0.959 | 0.643 | +0.0007 | — | — | yes (near no-op on connectomes) |
| FUGW (OT) | 0.918 | 0.620 | −0.005 | — | — | yes |
| conn_srm | 0.082 | 0.850 | +0.395 | — | — | **NO — identity collapse** |
| **ours_full_posterior_shrink** | **0.980** | **0.685** | −0.035 | **0.0092** | **7.45 < 12** | primary |
| ours_full_point_procrustes | 0.857 | 0.644 | +0.018 | 0.0092 | — | ablation (hierarchy off the map) |
| ours_full_c_bar_procrustes | 1.000 | 0.643 | −0.400 | 0.0092 | — | failed retry (BB^T poor EMD target) |

Notes:
- **conn_srm is invalid as a gain competitor**: +0.395 gain with ident 0.082 is identity collapse, not alignment.
- **BrainSync is a structural near-no-op** on spatial connectomes: time-domain `Q` leaves `XQQ^T X^T = XX^T` unchanged. Observed gain ≈ +0.0007.
- **point_procrustes** (hierarchy not driving the map) loses reliability (0.644) and ident (0.857) relative to posterior_shrink — the hierarchical posterior is the map, not a side diagnostic.
- Native-gauge `heldout_score_module = −||C_run2 − T(C_run1)||²` is **not** used for the SOTA claim: any nontrivial reindexing Q≠I inflates this residual even when reliability improves. The correct held-out is same-Q scan-rescan reliability `corr(vec(T(C1)), vec(T(C2)))`.

---

## 3. Task SOTA claim (statistical verification)

**Claim rule (pre-declared):** claim task SOTA only if the paired-bootstrap 95% CI for `(ident_ours − ident_baseline)` excludes 0 in the positive direction, **or** reliability_delta is positive with CI excluding 0 vs noalign (and non-negative / significant vs BrainSync/FUGW). Identification superiority is **not** claimed when the CI includes 0.

### Paired bootstrap, B=10,000 (subject-level; ID bootstrap recomputes top-1 on resampled galleries)

| Baseline | Ident Δ | Ident 95% CI | Reliability Δ | Reliability 95% CI | McNemar exact p |
|---|---:|---|---:|---|---:|
| noalign | +0.0204 | **[0.0000, 0.0408]** | **+0.0434** | **[0.0379, 0.0490]** | 1.0 (discordant 1 vs 0) |
| BrainSync | +0.0204 | **[0.0000, 0.0408]** | **+0.0425** | **[0.0366, 0.0486]** | 1.0 |
| FUGW | +0.0612 | **[0.0000, 0.0612]** | **+0.0648** | **[0.0576, 0.0722]** | 0.25 |

### Verdict: **task_sota_reliability** — claim holds

> **Posterior-gated hierarchical alignment improves scan-rescan reliability on real ds000243 vs noalign, BrainSync, and FUGW** (paired bootstrap 95% CIs exclude 0 for all three: Δ_rel = +0.043 / +0.043 / +0.065). **Identification point estimate is higher** (0.980 vs 0.959 / 0.959 / 0.918) **but ident bootstrap CIs touch 0** — reported as a **trend only**, not as ID superiority.

Same-Q verification: `transform_all(run2)` reuses the run-1 maps (`max_abs_diff = 0.0`). Cross-subject pair gain on transformed features does **not** improve (ours −0.035); the scientific win is **within-subject reliability under a shared subject map**, not forced cross-subject correlation (which is what collapses conn_srm).

---

## 4. Gap fill — columns baselines cannot produce

From real posteriors (`group_real_n49.json`, `REAL_sota_stats.json` gap block):

| Method | τ_φ | Group n_eff | Identifiability flags | Baseline null |
|---|---:|---:|---|---|
| noalign | — | — | — | silent: map for every subject |
| BrainSync | — | — | — | silent |
| FUGW | — | — | — | silent |
| conn_srm | — | — | — | silent (and collapses identity) |
| **ours_full_posterior_shrink** | **0.00924** (median 0.00925; p90 subject-mean 0.00974) | **7.45 < S=12** (min node n_eff 2.46; ci_ratio 1.28) | REML weights; 0 subjects with τ > 2×median; **5/49** subjects in the top τ decile (diffuse alignments) | **emits τ_φ + Σ^al via REML** |

**Why these columns matter (literature):** Thual 2025 documents that stimulus-free alignment is unproven; BrainSync 2018 documents that the rest aligner maximizes correlation even when the common-network assumption fails; Takeda 2025 documents that individual-level unsupervised alignment is statistically unreliable. None of those methods reports *whether a given subject's map is determined*. The hierarchical model does: group REML on real posteriors yields **n_eff = 7.45 < 12**, i.e. alignment uncertainty down-weights subjects before group inference. Baselines report an alignment for all 49 subjects with no flag.

**High-τ flag counts (honest):** under τ > 2× cohort median, **0/49** subjects are flagged on this cohort (τ is tightly concentrated ≈ 0.0086–0.0098). Under a top-decile criterion, **5/49** subjects carry the diffuse tail. The structural gap metric that PASSes is **n_eff < S** under REML — that is the group-level statement baselines cannot make.

**Unclaimed object filled:** hierarchical population-of-couplings + group Σ^al / τ_φ on REAL rest, next to a reliability SOTA win vs noalign/BrainSync/FUGW.

---

## 5. Coupling recovery as alignment SOTA (synthetic sidecar)

Planted-GT synthetic (`synthetic_gap_results.json`):

| Metric | ours_full | point / EMD→C_pop | FUGW | random |
|---|---:|---:|---:|---:|
| **coupling_recovery** (assignment of planted π*) | **0.70** | 0.03 | 0.028 | 0.02 |

**Why coupling recovery, not reconstruction error:** under the generative model `C_s = C_true + E`, the Bayes-optimal estimator of `C_true` from `C_s` without a perfect map is the identity — reconstruction error to noisy connectomes rewards doing nothing. The fair alignment metric is recovery of the **map** (posterior coupling / assignment vs planted P*). On that metric ours_full beats point-OT/EMD/FUGW by a wide margin (0.70 vs ≈0.03).

**Do not headline coverage:** coverage@0.9 ≈ 0.004 and AUROC(τ) = 0 on the synthetic plant — entropy/τ scale calibration FAIL. Those numbers are reported, not sold.

---

## 6. What failed (honest ledger)

1. **Coverage@0.9 / AUROC-τ on synthetic** — FAIL (overconfident posterior on mixed maps; C_bar scale mismatch ‖C_bar‖≈1.4 vs ‖C_pop_true‖≈24.7). Not claimed.
2. **c_bar Procrustes retry** — heldout residual −1617, gain −0.400. Learned BB^T is a poor EMD target vs empirical C_pop. One retry; no further invented wins.
3. **Native-gauge heldout_score_module** — ours does not beat noalign/FUGW on this residual (−775 vs −702/−704). Metric caveat: Q≠I inflates residual to native-gauge run2. SOTA claim uses same-Q reliability instead.
4. **conn_srm** — high gain via identity collapse (ident 0.082). Disqualified as a gain competitor.
5. **Identification superiority** — point estimate wins (0.980) but bootstrap CI touches 0; McNemar p=1.0 vs noalign (one discordant subject). **Trend only.**
6. **Cross-subject alignment_gain** — ours_full_posterior_shrink is negative (−0.035). Positive gain without collapse is **not** the headline; reliability under a shared subject map is.
7. **Gain-null nonident counts** — saturate under degenerate nulls; not a scientific uncertainty rate. Posterior τ / REML n_eff are the uncertainty surface.
8. **Long-run scan-length sensitivity** — not run on the frozen two-run cohort (long runs are one-run subjects).

---

## 7. Claim discipline appendix

- **Finitely many optima; single-point minimizers are not claimed.** Gromov–Wasserstein is invariant to isometries (Mémoli 2011); the feature term reduces the isometry orbit to **finitely many optima** (Demetci et al., AISTATS 2024). No unique minimizer is asserted.
- **Inherited components are attributed.** Subject-to-template plans with a barycenter: **FUGW** (Thual et al., NeurIPS 2022). Amortized encoder: **ULOT** (Mazelet, Flamary, Thirion, NeurIPS 2025). Distributions over transport plans: **Mallasto, Gerolin, Minh** (ACML 2021) and **De et al.** (ICML 2026). Alignment variance in the group model: **Keller, Roche, Tucholka, Thirion** (*Statistica Sinica* 2008); Hu et al. (ICLR 2025) for learned registration. GW objective: **Mémoli** (*FoCM* 2011) and **Demetci et al.** (PMLR 238, 2024). Nearest neighbour **OTTER** (bioRxiv 2026) has soft mass, not a posterior/hierarchy.
- **Absences are search-based.** The assembled object — population distribution over latent alignment couplings with shrinkage on the transport polytope for cross-subject rest-fMRI — was **not found** in the sources we checked. Stated as a search result, not as a metaphysical claim.
- **Band prior is a band prior, not dynamics.** It constrains frequency content of a coupling; weight 0 in frozen runs.
- **No behavioural prediction is reported.** Marek et al. 2022: median brain–behaviour |r| ≈ 0.01 at N=3,928; ds000243 has 120 subjects.
- **Random-effects collapse identity.** Group model reduces to the one-sample t-test when alignment variance → 0; tested, not asserted.
- **Do not claim:** ID superiority with CI including 0; calibrated coverage on real rest; gain-null nonident counts as scientific rates; conn_srm as a gain competitor.

---

## 8. Protocol metadata

- Cohort: 49 two-run subjects **015–063**, frozen data root ds000243-master
- beta: **29.189086229914952** (scan-rescan, R=100 Schaefer parcels)
- Declared pairs: **500**, seed **2026**
- Bootstrap: **B=10,000** paired (verification script); pilot permutation table remains B=200
- Ours artifacts: `runs/10_ours_full__73533e35__20260920T074217Z/artifacts`
- Transform path: `posterior_shrink_tau_gated`, `posterior_drives_transform=true`, `tau0_auto=true`, λ_mean≈0.50
- Group REML smoke: 12 subjects present in artifacts, n_eff=7.45, ci_ratio=1.28, PASS
- Python: `/Users/anandlo/.central_venv/bin/python3`

## 9. Reproducing a row

Every harness row is regenerable from `scripts/run_real_gap_sota.py` + `scripts/verify_real_sota.py` against the named artifacts and data root. `compare.py` reads only `runs/index.csv` + `metrics.json` and never re-runs training. Long train PIDs on the machine were **not** killed for this write-up.
