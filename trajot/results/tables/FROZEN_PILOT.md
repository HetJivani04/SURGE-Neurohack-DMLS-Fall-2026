# Frozen pilot cohort (Phase 2)

- **Subjects (n=24)**: 015,016,017,018,019,020,021,022,023,024,025,026,027,028,029,030,031,032,033,034,035,036,037,038
- **beta**: 29.189086229914952 (sigma_hat_C^2 = 0.034259380102660844, calibrated on the 12-subject subset 015–026; frozen for fair comparison)
- **n_regions**: 100
- **pairs**: 500, seed 2026
- **permutations B**: 200 (debug/pilot); final paper numbers re-run at B=10000 after N=83
- **ours_full artifacts**: runs/10_ours_full__0963e0b9__20260920T061730Z (template + tau_phi, real-beta train)
- **ours transform**: region-level OT coupling to C_pop (R,R) then orthogonal Procrustes; NOT identity when artifacts/load succeed
- **BrainSync**: region-level timeseries (R,T) from contract npz via template_geometry labels
