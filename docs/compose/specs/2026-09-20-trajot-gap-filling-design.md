# [SURGE-TrajOT] Gap-Filling Architecture + Non-Marginal Evaluation Design

> **Status:** Design draft for user approval before compose:plan.

**Goal:** Make the hierarchical population-of-couplings model the *actual* alignment + inference object, and demonstrate non-marginal advantages that baselines **cannot** produce — not ID wins on a ceiling metric.

## [S1] Why the current story is weak (three failures)

1. **Fidelity gap:** `ours.py` trains `B, F_bar, nu, eps_s, pi_s, tau_phi` then applies `ot.emd(C_s, C_pop)` + Procrustes (`ours.py:408-412,455-464`). Hierarchical posterior never drives the map. Docstring overclaims.
2. **Wrong primary metric:** Schaefer-100 raw ID is 0.959 (ceiling). Any template-shrinkage map loses ID by construction (variance theorem). Hierarchy **cannot** win Track A; ID **cannot** score uncertainty.
3. **Dead inference layer:** `report/group.py` REML / Σ^al / sign-flip is unit-tested but never run on real posteriors. Gain-null NonIdent saturates (466–500/500) — degenerate null artifact, not science.

## [S2] Approaches

### Approach A — Posterior-gated transform + synthetic gap metrics (RECOMMENDED)

**Architecture fix (Candidates 2+3):**
- Transform uses posterior mean coupling to **learned** `C_bar=BB^T` (not `C_pop=mean(C)`).
- **Uncertainty-gated shrinkage:** `λ_s = 1/(1+(τ_s/τ0)²)`, `C̃_s = (1-λ)C_s + λ Q_s^T C_s Q_s`.
  - Sharp posterior → aligned; diffuse → fingerprint preserved (`τ` load-bearing).
- Residual-preserving option: `C̃_s = C_pop + Q_s^T (C_s − C_ref,s) Q_s`.
- Bridge vertex π `(V,K)` → region `(R,K)` via mass-preserving region pooling (or train K=R=100 region-level).

**Evaluation (pre-registered, synthetic first):**
| Metric | Success | Why baselines lose |
|--------|---------|-------------------|
| Posterior coverage @ nominal 0.90 | ours ∈ [0.85,0.95] on planted π* | point maps: 0-width, coverage N/A |
| Non-ident AUROC(τ, planted-ambiguous) | >0.8 | always full confidence |
| Group FPR / n_eff (REML vs t-test) | weighted ≤ naive+0.02; n_eff < S when τ high | Σ^al=0 → silent averaging |
| Held-out predictive score | ours wins likelihood on run-2 | no generative model |
| Ablation | full ≠ ablated on coverage/FPR | gauge only moves τ scalar today |

**Trade-off:** ID may stay ~0.85 or improve if λ preserves fingerprints — either way ID is secondary.

### Approach B — Group-inference-first (real data today)

Wire `meta_analysis_map` on frozen real posteriors + age/node-strength contrast. Claim: uncertainty-aware group CIs + subject down-weighting vs baselines' error-free maps. **Risk:** π are vertex→template vs Schaefer R — need pooling or synthetic-only group.

### Approach C — Keep Procrustes, only relabel Track B

Document fidelity gap; ship uncertainty column only. **Rejected:** marginal; user explicitly wants gap-filling.

## [S3] Recommended pipeline (A + B light)

1. Wire posterior-driven + τ-gated transform in `ours.py` (honest docstring/meta).
2. `pool_pi_to_regions` + `pairwise_gamma` (PLAN object).
3. New eval: `posterior_coverage`, `nonident_auroc`, `heldout_predictive_score`, `shrinkage_transform`.
4. `scripts/synthetic_coverage.py`: N=60, R=50, β=29.19, 30% planted-ambiguous, M≥20, epochs≥10 (not debug cap 3).
5. `scripts/run_group_analysis.py`: real artifacts smoke + synthetic FPR comparison.
6. Re-run ours_full vs ours_ablated on synthetic gap metrics.
7. Update RESULTS.md with pre-registered table; keep N=49 ID table as secondary honesty.

## [S4] Testing / claim discipline

- TDD: shrinkage limits (τ→0 full align, τ→∞ identity); coverage helper; group n_eff∈[1,S].
- Do **not** claim: ID superiority, unique minimizer, dynamics, real-rest calibrated coverage without GT, PLAN §5 faithfulness until posterior drives transform.
- Do **claim if tests pass:** calibrated uncertainty on synthetic, non-ident detection, uncertainty-aware group inference, held-out likelihood — baselines structurally silent on these.

## [S5] Files

| Change | Path |
|--------|------|
| Posterior transform + Γ + pooling | `trajot/src/trajot/baselines/ours.py` |
| Uncertainty metrics | `trajot/src/trajot/eval/uncertainty.py` (new), `metrics.py` |
| Group driver | `trajot/scripts/run_group_analysis.py` (new) |
| Synthetic harness | `trajot/scripts/synthetic_coverage.py` (new) |
| Evaluate optional keys | `trajot/scripts/evaluate.py` |
| Write-up | `trajot/reports/RESULTS.md`, TRACKING.md |
