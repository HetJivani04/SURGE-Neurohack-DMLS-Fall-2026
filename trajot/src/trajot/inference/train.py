"""Training: template initialization, the amortized variational loop, and the artifacts it produces.

Adam over the encoder ``phi`` and the population parameters ``theta = {B, F_bar, eps_{1:S}}``, minibatches of
subjects, ``M`` reparametrized posterior draws each, ``eps`` annealed geometrically and ``beta`` tempered from 0 to
``sigma_hat_C^-2``. **Training uses run 1 only; run 2 is held out** (``TrainData.holdout``, never read here).

Scale convention. Inside the loop the coupling has **unit row mass** (total mass ``V_s``): ``mu = V_s * mu_s`` and
``nu = V_s * nu``. Then ``E_GW`` is the sum over the entries of the region-level connectome, which is what
``sigma_hat_C^2 = 0.5 ||dC||_F^2 / R^2`` (a per-entry variance with ``R`` the connectome side / parcel count, not
surface vertices) presupposes, and the feature and Gibbs terms are sums over vertices, commensurate with the entropy
of the ``V_s K``-dimensional posterior. With probability masses (total 1) those terms are weighted *means*, smaller
by ``V_s^2`` and ``V_s``, and an unscaled ambient entropy dominates: ``tau`` diverges and nothing is learned.
Returned draws are divided by ``V_s``, so they lie in ``Pi(mu_s, nu)`` with ``mu_s`` the probability vertex mass of
:func:`trajot.geometry.cost.vertex_mass`.

Entropy (wired). The ELBO entropy is no longer hardcoded zero. True ``H[q(pi)] = H[q(xi)] + E[log|det J_sinkhorn|]``
is estimated as a **pathwise Shannon entropy of the coupling draws** (differentiable through the unrolled Sinkhorn
map; bounded on the polytope) plus a **detached Hutchinson/SLQ log-determinant** of a projected Sinkhorn Jacobian
(:func:`trajot.inference.entropy.projected_sinkhorn_logdet`, clipped). Analytic ``H[q(xi)] = 0.5 V K (1 + log 2π)``
is logged for reference; it is not added at full ambient scale inside the unit-row-mass objective (that sum is what
made ``tau`` diverge when the Sinkhorn log-det was omitted). Scale the bonus with ``model.entropy.weight`` (default
1.0). Residual caveat: the Jacobian term is detached, so ``tau`` receives entropy gradients only through the
pathwise Shannon term — not through ``log|det J|``.

Gibbs anchor. ``M0`` prefers geodesic distances from ``geometry.anatomical_cost`` when surface faces are available
(``template_geometry.npz`` or a geometry sidecar under the derivatives tree). Otherwise Euclidean
``cdist(coords, node_coords) / scale`` is used. Which path ran is recorded as ``TrainData.anchor`` /
``TrainResult.anchor`` (``"geodesic"`` | ``"euclidean"``) and written into ``template.npz``.

Gauge features. ``gauge_velocity`` is a **spatial** gradient of the diffusion embedding used as a gauge-breaking
channel (breaks GW isometry to finitely many optima) — not a time derivative and not a model of dynamics.
The band prior (``model.band.weight``) is a band prior, never a model of dynamics; default weight is 0.0.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
from scipy import stats
from scipy.spatial.distance import cdist

from trajot.geometry.cost import anatomical_cost, vertex_mass
from trajot.geometry.diffusion import gauge_velocity, laplacian_eigenvectors
from trajot.inference.encoder import ScoreEncoder, SinkhornPosterior, build_inputs
from trajot.inference.entropy import projected_sinkhorn_logdet
from trajot.io.contract import load_connectomes, read_manifest, read_subject_run, subject_run_path
from trajot.model.elbo import beta_schedule, elbo_terms, eps_schedule
from trajot.model.prior import band_penalty_matrix, constrain_scale, log_prior_theta
from trajot.runlog.parallel import pick_device


# --------------------------------------------------------------------------- parameters
@dataclass
class TemplateParams:
    """The population parameters ``theta``."""

    B: np.ndarray  # (K, r) float32: template geometry loadings, C_bar = B B^T
    F_bar: np.ndarray  # (K, F) float32: template features
    nu: np.ndarray  # (K,) float64: template mass, uniform
    eps: np.ndarray  # (S,) float64: per-subject prior sharpness, from the InvGamma hyperprior


def init_template(
    connectomes: np.ndarray, K: int = 512, r: int = 32, seed: int = 0, *, n_features: int | None = None,
    eps_prior: tuple[float, float] = (3.0, 0.4),
) -> TemplateParams:
    """Initialize ``B`` and ``F_bar`` from the spectral embedding of the group-average connectome.

    ``connectomes`` is ``(S, R, R)``. The group average is decomposed and the ``r`` largest-magnitude components,
    scaled by ``sqrt|lambda|``, give an ``(R, r)`` embedding; each of the ``K`` template nodes takes a row
    (without replacement while ``K <= R``, then with replacement) plus a little noise. ``F_bar`` is ``(K,
    n_features)`` (default ``r``) filled the same way. ``nu`` is uniform and ``eps`` is drawn from
    ``InvGamma(*eps_prior)``. Deterministic in ``seed``.
    """
    rng = np.random.default_rng(seed)
    connectomes = np.asarray(connectomes, dtype=np.float64)
    S, R = connectomes.shape[0], connectomes.shape[1]
    average = connectomes.mean(axis=0)
    values, vectors = np.linalg.eigh(0.5 * (average + average.T))
    order = np.argsort(np.abs(values))[::-1][: min(r, R)]
    embedding = vectors[:, order] * np.sqrt(np.abs(values[order]))  # (R, <=r)
    scale = float(embedding.std()) or 1.0

    def fill(width: int) -> np.ndarray:
        rows = rng.permutation(R)[:K] if K <= R else rng.integers(0, R, size=K)
        out = np.zeros((K, width))
        out[:, : min(width, embedding.shape[1])] = embedding[rows][:, :width]
        return (out + 0.05 * scale * rng.normal(size=out.shape)).astype(np.float32)

    a_eps, b_eps = eps_prior
    return TemplateParams(
        B=fill(r), F_bar=fill(n_features if n_features is not None else r), nu=np.full(K, 1.0 / K),
        eps=stats.invgamma.rvs(a_eps, scale=b_eps, size=S, random_state=rng).astype(np.float64))


# --------------------------------------------------------------------------- data
@dataclass
class SubjectData:
    """One subject-run reduced to what the model needs, over its *valid* vertices only."""

    subject_id: str
    tr: float
    valid: np.ndarray  # (V_all,) bool: vertices inside the field of view
    A: np.ndarray  # (V, r) float64: rank-r factor of the vertex-level correlation matrix, C_s = A A^T
    mu: np.ndarray  # (V,) float64: vertex mass, uniform
    Y: np.ndarray  # (V, F0) float64: [diffusion embedding, anatomical features]
    v: np.ndarray | None  # (V, d) float64: spatial gradient of the embedding (gauge-breaking channel), or None
    M0: np.ndarray  # (V, K) float64: anatomical anchor to the template nodes
    u: np.ndarray  # (V, in_dim) float32: encoder input
    anchor: str = "euclidean"  # "geodesic" when M0 came from anatomical_cost faces; else Euclidean cdist


@dataclass
class TrainData:
    train: list[SubjectData]  # run 1
    holdout: list[SubjectData]  # run 2: never used for training
    connectomes: np.ndarray  # (S, R, R) float64, run 1, for the template initialization
    beta_target: float
    node_coords: np.ndarray  # (K, 3): where the template nodes sit anatomically
    anchor: str = "euclidean"  # how M0 was built: "geodesic" | "euclidean"


def farthest_point_sampling(points: np.ndarray, k: int) -> np.ndarray:
    """Indices of ``k`` points chosen greedily to be far apart (starts from point 0)."""
    if k > len(points):
        raise ValueError(f"cannot pick K={k} template nodes from {len(points)} vertices")
    chosen = [0]
    distance = np.linalg.norm(points - points[0], axis=1)
    for _ in range(k - 1):
        chosen.append(int(distance.argmax()))
        distance = np.minimum(distance, np.linalg.norm(points - points[chosen[-1]], axis=1))
    return np.array(chosen)


def _load_faces_for(root: Path, n_vertices: int) -> np.ndarray | None:
    """Surface faces from the derivatives tree if their indices are valid for ``n_vertices``."""
    deriv = Path(root) / "derivatives" / "trajot"
    candidates = [deriv / "template_geometry.npz", *sorted(deriv.glob("*_geometry.npz"))]
    for path in candidates:
        if not path.exists():
            continue
        try:
            with np.load(path) as z:
                if "faces" not in z.files:
                    continue
                faces = np.asarray(z["faces"])
        except OSError:
            continue
        if faces.size and faces.ndim == 2 and faces.shape[1] == 3 and int(faces.max()) < n_vertices:
            return faces.astype(np.int64)
    return None


def prepare_subject(
    npz: dict[str, Any], cfg: Any, node_coords: np.ndarray, distance_scale: float, *,
    node_dist: np.ndarray | None = None,
) -> SubjectData:
    """Turn a contract file into :class:`SubjectData`.

    ``A_s`` is the rank-``r`` factor of the vertex-level correlation matrix, taken from the standardized time
    series (a thin SVD), so no ``(V, V)`` matrix is needed to define the geometry.

    ``M0`` is geodesic when ``node_dist (V_all, K)`` is given (``anatomical_cost`` columns to the template nodes);
    otherwise Euclidean ``cdist(coords, node_coords) / distance_scale``. The choice is recorded on the result.
    """
    r, m = int(cfg.get("model.r")), int(cfg.get("model.encoder.m_eigvecs"))
    timeseries = np.asarray(npz["timeseries"], dtype=np.float64)
    valid = timeseries.std(axis=1) > 0
    T = timeseries.shape[1]

    X = timeseries[valid]
    X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True) / math.sqrt(T - 1)  # X X^T = correlation
    U, sv, _ = np.linalg.svd(X, full_matrices=False)
    A = np.zeros((X.shape[0], r))
    A[:, : min(r, sv.size)] = (U * sv)[:, :r]

    embedding = np.asarray(npz["embedding"], dtype=np.float64)[valid]
    features = np.asarray(npz["features"], dtype=np.float64)[valid]
    coords = np.asarray(npz["coords"], dtype=np.float64)[valid]
    tr = float(npz["tr"])
    gauge = bool(cfg.get("model.gauge_features", True))
    laplacian = laplacian_eigenvectors(X @ X.T, m)
    if node_dist is not None:
        M0 = np.asarray(node_dist, dtype=np.float64)[valid]
        anchor = "geodesic"
    else:
        M0 = cdist(coords, node_coords) / distance_scale
        anchor = "euclidean"
    return SubjectData(
        subject_id=str(npz["subject_id"]), tr=tr, valid=valid, A=A, mu=vertex_mass(X.shape[0]),
        Y=np.concatenate([embedding, features], axis=1), v=gauge_velocity(embedding, coords, tr) if gauge else None,
        M0=M0,
        u=build_inputs(np.asarray(npz["timeseries"])[valid], embedding, coords, laplacian, features=features),
        anchor=anchor)


def load_train_data(
    root: Path, cfg: Any, subjects: Sequence[str] | None = None, beta_target: float | None = None,
    train_run: str = "1", holdout_run: str = "2",
) -> TrainData:
    """Read the contract files under ``root``: run 1 of every subject for training, run 2 held out.

    The ``K`` template nodes are placed at the vertices of the first subject that are farthest apart. When surface
    faces are available under ``root/derivatives/trajot`` (``template_geometry.npz`` or a ``*_geometry.npz``
    sidecar), the anatomical anchor ``M0`` is the geodesic distance from each vertex to each node
    (:func:`trajot.geometry.cost.anatomical_cost`, already scaled to ``[0, 1]``) and ``anchor`` is
    ``"geodesic"``. Otherwise ``M0`` is Euclidean ``cdist(coords, node_coords) / scale`` and ``anchor`` is
    ``"euclidean"``. If ``beta_target`` is ``None`` it comes from ``cfg`` (``model.beta.synthetic``): calibrating
    ``beta`` from real two-run data is :func:`trajot.inference.beta.calibrate_beta`'s job alone.
    """
    root = Path(root)
    manifest = read_manifest(root)
    counts = manifest.groupby("subject_id")["run_id"].apply(set)
    ids = sorted(s for s, runs in counts.items() if {train_run, holdout_run} <= runs)
    if subjects is not None:
        ids = [s for s in ids if s in {str(x) for x in subjects}]
    if not ids:
        raise ValueError(f"no subjects with runs {train_run} and {holdout_run} under {root}")

    K = int(cfg.get("model.K"))
    reference = read_subject_run(subject_run_path(root, ids[0], train_run))["coords"].astype(np.float64)
    node_indices = farthest_point_sampling(reference, K)
    node_coords = reference[node_indices]
    faces = _load_faces_for(root, reference.shape[0])
    node_dist = None
    if faces is not None:
        full_cost = anatomical_cost(reference, faces)  # (V_all, V_all) in [0, 1]
        node_dist = full_cost[:, node_indices]
        scale = 1.0
        anchor = "geodesic"
    else:
        scale = float(cdist(node_coords, node_coords).max()) or 1.0
        anchor = "euclidean"

    def load(run: str) -> list[SubjectData]:
        return [
            prepare_subject(read_subject_run(subject_run_path(root, s, run)), cfg, node_coords, scale,
                            node_dist=node_dist)
            for s in ids
        ]

    connectomes, _ = load_connectomes(root, ids, run=train_run)
    beta = float(cfg.get("model.beta.synthetic")) if beta_target is None else float(beta_target)
    return TrainData(train=load(train_run), holdout=load(holdout_run), connectomes=connectomes,
                     beta_target=beta, node_coords=node_coords, anchor=anchor)


# --------------------------------------------------------------------------- training
@dataclass
class TrainResult:
    params: TemplateParams  # the learned theta
    encoder: ScoreEncoder
    tau_phi: list[np.ndarray]  # per subject, (V_s,) float32: the noise scale = the identifiability statement
    pi_samples: list[np.ndarray]  # per subject, (M, V_s, K) float64 posterior draws on Pi(mu_s, nu)
    loss_trace: list[dict[str, float]] = field(default_factory=list)  # one entry per epoch
    subject_ids: list[str] = field(default_factory=list)
    beta_target: float = 0.0
    anchor: str = "euclidean"  # how M0 was built: "geodesic" | "euclidean"
    entropy_weight: float = 1.0
    entropy_estimator: str = "shannon_pi+hutchinson_slq"


def posterior_samples(
    encoder: ScoreEncoder, subject: SubjectData, nu: np.ndarray, eps: float, M: int, L: int, enc_device: Any,
    generator: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(pi (M, V, K) float64, tau (V,) float32)`` for one subject with the current encoder; no gradients.

    ``pi`` lies in ``Pi(mu_s, nu)`` with probability masses (rows sum to ``subject.mu``, total 1); ``nu`` is the
    ``(K,)`` probability vector of the template."""
    V = subject.mu.shape[0]
    with torch.no_grad():
        u = torch.from_numpy(subject.u).to(enc_device)
        S_phi, tau = encoder(u, torch.from_numpy(subject.M0).to(device=enc_device, dtype=torch.float32))
        pi = SinkhornPosterior().sample(S_phi.cpu().numpy(), tau.cpu().numpy(), subject.mu * V, nu * V, eps, M, L, generator)
    return pi / V, tau.cpu().numpy()


def _entropy_log_q(
    pi: torch.Tensor, S_phi: torch.Tensor, tau: torch.Tensor, xi: torch.Tensor, mu: torch.Tensor,
    nu: torch.Tensor, eps: float, L: int, entropy_weight: float, generator: np.random.Generator,
) -> tuple[torch.Tensor, float]:
    """ELBO ``log_q`` such that ``-mean(log_q) = entropy_weight * H_est`` (float64, CPU graph).

    ``H_est`` is a pathwise Shannon entropy of the coupling draws (differentiable through Sinkhorn; bounded on
    the polytope) plus a detached Hutchinson/SLQ log-determinant of a projected Sinkhorn Jacobian, clipped.
    Analytic ``H[q(xi)]`` is returned separately for the run log and is not added at full ambient scale.
    """
    V, K = int(xi.shape[-2]), int(xi.shape[-1])
    H_xi = 0.5 * V * K * (1.0 + math.log(2.0 * math.pi))
    H_shannon = -(pi * torch.log(pi.clamp_min(1e-300))).sum(dim=(-2, -1)).mean()
    with torch.no_grad():
        score = (
            S_phi.cpu().to(dtype=torch.float64)
            + tau.cpu().to(dtype=torch.float64)[:, None] * xi[0].cpu().to(dtype=torch.float64)
        )
        mu_np = np.asarray(mu.cpu().numpy() if isinstance(mu, torch.Tensor) else mu, dtype=np.float64)
        nu_np = np.asarray(nu.cpu().numpy() if isinstance(nu, torch.Tensor) else nu, dtype=np.float64)
        logdet = projected_sinkhorn_logdet(
            score.numpy(), mu_np, nu_np, eps, n_iter=max(10, min(L, 20)), subspace_dim=4,
            generator=generator)
    logdet_t = torch.as_tensor(logdet, dtype=torch.float64, device=pi.device)
    entropy = entropy_weight * (H_shannon + logdet_t)
    return -entropy, float(H_xi)


def train(cfg: Any, data: TrainData, device: Any) -> TrainResult:
    """Fit the model on ``data.train`` (run 1). ``device`` is where the float32 encoder runs; OT / GW arithmetic
    is float64 on CPU (``pick_device(prefer_mps=False)``). Prints one line per epoch (the run logger captures it)."""
    seed = int(cfg.get("run.seed", 0))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    ot_device = pick_device(prefer_mps=False)

    K, r, M = int(cfg.get("model.K")), int(cfg.get("model.r")), int(cfg.get("model.m_draws"))
    batch_size, L = int(cfg.get("model.batch_subjects")), int(cfg.get("model.sinkhorn.L"))
    eps_start, eps_end = (float(e) for e in cfg.get("model.sinkhorn.eps"))
    warmup, sigma_f2 = float(cfg.get("model.beta.warmup_frac")), float(cfg.get("model.sigma_f2"))
    epochs, lr = int(cfg.get("model.train.epochs")), float(cfg.get("model.train.lr"))
    max_row_norm = float(cfg.get("model.prior.B_max_row_norm"))
    band_weight = float(cfg.get("model.band.weight", 0.0))
    entropy_weight = float(cfg.get("model.entropy.weight", 1.0))
    gauge = bool(cfg.get("model.gauge_features", True))
    subjects = data.train
    S = len(subjects)
    entropy_rng = np.random.default_rng(seed + 17)

    F0 = subjects[0].Y.shape[1]
    F = F0 + (subjects[0].v.shape[1] if gauge else 0)
    template = init_template(data.connectomes, K, r, seed, n_features=F,
                             eps_prior=(float(cfg.get("model.prior.a_eps")), float(cfg.get("model.prior.b_eps"))))
    B = torch.nn.Parameter(torch.from_numpy(template.B).to(ot_device))
    F_bar = torch.nn.Parameter(torch.from_numpy(template.F_bar).to(ot_device))
    log_eps = torch.nn.Parameter(torch.log(torch.from_numpy(template.eps)).to(ot_device))
    encoder = ScoreEncoder(subjects[0].u.shape[1], int(cfg.get("model.encoder.p")), K,
                           n_blocks=int(cfg.get("model.encoder.n_blocks")), device=device,
                           heads=int(cfg.get("model.encoder.heads")), lambda_init=float(cfg.get("model.encoder.lambda_init")))
    optimizer = torch.optim.Adam([*encoder.parameters(), B, F_bar, log_eps], lr=lr)
    nu = torch.from_numpy(template.nu).to(ot_device)
    posterior = SinkhornPosterior()

    steps_per_epoch = math.ceil(S / batch_size)
    total_steps = epochs * steps_per_epoch
    print(f"train: {S} subjects (run 1), K={K}, r={r}, F={F}, M={M}, L={L}, {epochs} epochs x {steps_per_epoch} steps, "
          f"beta target {data.beta_target:.4g}, gauge features {gauge}, band prior weight {band_weight}, "
          f"anchor {data.anchor}")
    print(f"train: entropy WIRED (model.entropy.weight={entropy_weight:g}): pathwise Shannon(pi) + detached "
          f"Hutchinson/SLQ logdet of a projected Sinkhorn Jacobian (estimator=shannon_pi+hutchinson_slq); "
          f"analytic H[q(xi)] is logged for reference, not summed at full ambient scale")

    trace: list[dict[str, float]] = []
    step = 0
    H_xi_ref = 0.0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(S)
        sums: dict[str, float] = {}
        for start in range(0, S, batch_size):
            batch = order[start : start + batch_size]
            beta_t = beta_schedule(step, total_steps, data.beta_target, warmup)
            eps_t = eps_schedule(step, total_steps, eps_start, eps_end)
            optimizer.zero_grad(set_to_none=True)
            for s in batch:
                sub = subjects[s]
                as_t = lambda x: torch.from_numpy(x).to(ot_device)  # noqa: E731
                S_phi, tau = encoder(torch.from_numpy(sub.u).to(device),
                                     torch.from_numpy(sub.M0).to(device=device, dtype=torch.float32))
                V_s = sub.mu.shape[0]
                mu, A, M0, nu_s = as_t(sub.mu) * V_s, as_t(sub.A), as_t(sub.M0), nu * V_s  # unit row mass (see module doc)
                pi, xi = posterior.sample_torch(S_phi, tau, mu, nu_s, eps_t, M, L, generator)
                Y = torch.cat([as_t(sub.Y), beta_t * as_t(sub.v)], dim=1) if gauge else as_t(sub.Y)
                B64 = B.double()
                penalty = None
                if band_weight > 0:
                    penalty = band_weight * band_penalty_matrix(
                        A, B64, sub.tr, tuple(cfg.get("model.band.seconds")), float(cfg.get("model.band.bandwidth")))
                log_prior = log_prior_theta(
                    SimpleNamespace(B=B64, F_bar=F_bar.double(), eps=torch.exp(log_eps)), cfg)
                log_q, H_xi_ref = _entropy_log_q(pi, S_phi, tau, xi, mu, nu_s, eps_t, L, entropy_weight, entropy_rng)
                terms = elbo_terms(pi, A, B64, nu_s, mu, Y, F_bar.double(), M0, torch.exp(log_eps[s]), sigma_f2, beta_t,
                                   log_q, band_penalty=penalty, log_prior=log_prior)
                (-terms["total"] / len(batch)).backward()
                with torch.no_grad():
                    for name, value in terms.items():
                        sums[name] = sums.get(name, 0.0) + value.item() / len(batch)
                    sums["row_marginal_error"] = max(sums.get("row_marginal_error", 0.0),
                                                     (pi.sum(-1) - mu).abs().max().item() / V_s)
                    sums["tau_mean"] = sums.get("tau_mean", 0.0) + tau.mean().item() / len(batch)
            optimizer.step()
            with torch.no_grad():
                B.copy_(constrain_scale(B, max_row_norm))
            step += 1
        record = {name: value / steps_per_epoch if name not in ("row_marginal_error",) else value
                  for name, value in sums.items()}
        record.update(epoch=epoch, beta=beta_t, eps=eps_t)
        trace.append(record)
        print(f"epoch {epoch}/{epochs}: total {record['total']:.4g} gw {record['gw']:.4g} feature {record['feature']:.4g} "
              f"prior {record['prior']:.4g} entropy {record['entropy']:.4g} tau {record['tau_mean']:.3g} "
              f"beta {beta_t:.4g} eps {eps_t:.4g} row-marginal-error {record['row_marginal_error']:.2e}")

    params = TemplateParams(B=B.data.cpu().numpy(), F_bar=F_bar.data.cpu().numpy(), nu=template.nu,
                            eps=torch.exp(log_eps).data.cpu().numpy())
    draws, taus = [], []
    for sub in subjects:
        pi, tau = posterior_samples(encoder, sub, template.nu, eps_end, M, L, device, rng)
        draws.append(pi)
        taus.append(tau)
    return TrainResult(params=params, encoder=encoder, tau_phi=taus, pi_samples=draws, loss_trace=trace,
                       subject_ids=[s.subject_id for s in subjects], beta_target=data.beta_target,
                       anchor=data.anchor, entropy_weight=entropy_weight)


def save_artifacts(run_dir: Path, result: TrainResult) -> None:
    """Write the posterior samples, ``tau_phi`` and the learned template into ``<run_dir>/artifacts/``."""
    out = Path(run_dir) / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "posterior_samples.npz", **{f"sub-{sid}": pi for sid, pi in zip(result.subject_ids, result.pi_samples)})
    np.savez(out / "tau_phi.npz", **{f"sub-{sid}": tau for sid, tau in zip(result.subject_ids, result.tau_phi)})
    np.savez(out / "template.npz", B=result.params.B, F_bar=result.params.F_bar, nu=result.params.nu,
             eps=result.params.eps, subject_ids=np.array(result.subject_ids),
             anchor=np.array(result.anchor), entropy_weight=np.float64(result.entropy_weight),
             entropy_estimator=np.array(result.entropy_estimator))
    (out / "loss_trace.json").write_text(json.dumps(result.loss_trace, indent=2) + "\n")
