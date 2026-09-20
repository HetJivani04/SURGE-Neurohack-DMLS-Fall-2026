"""The amortized encoder ``q_phi(pi_s | C_s, Y_s)``: inputs, score network, and the Sinkhorn posterior.

::

    u_si = [ y_si || top-m eigenvectors of L(C_s) || spherical coords ]
    h_si = Enc_phi(u_s)_i in R^p                     # 4 blocks of linear attention over the V vertex tokens
    S_phi[i,k] = < W h_si , g_k > / sqrt(p)  -  lambda * M_s0[i,k]

``tau_phi`` is a learned per-vertex noise scale: it carries heteroscedastic alignment uncertainty across the
cortex and is the per-subject identifiability statement (a large ``tau_phi[i]`` says the data do not determine
where vertex ``i`` goes). The encoder runs in float32 (on MPS if available); the posterior draws are float64 on CPU.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from trajot.inference.sinkhorn import perturb_then_project, sinkhorn_log_torch


def build_inputs(
    timeseries: np.ndarray, embedding: np.ndarray, coords: np.ndarray, lap_eigvecs: np.ndarray,
    features: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble ``u_si = [y_si || top-m Laplacian eigenvectors || spherical coords]`` as ``(V, F + m + 2)`` float32.

    ``y_si`` is the diffusion embedding ``embedding (V, d)`` (followed by ``features (V, F_a)`` when given).
    ``lap_eigvecs (V, m)`` comes from ``geometry.laplacian_eigenvectors``. The two spherical angles
    ``(theta, phi)`` are computed here from ``coords (V, 3)`` about their centroid. Vertices whose ``timeseries``
    row is all zero (outside the field of view) get an all-zero input row.
    """
    embedding = np.asarray(embedding, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    V = embedding.shape[0]
    if not (timeseries.shape[0] == coords.shape[0] == lap_eigvecs.shape[0] == V):
        raise ValueError("timeseries, embedding, coords and lap_eigvecs must have the same number of vertices")

    centred = coords - coords.mean(axis=0, keepdims=True)
    radius = np.maximum(np.linalg.norm(centred, axis=1), 1e-12)
    theta = np.arccos(np.clip(centred[:, 2] / radius, -1.0, 1.0))
    phi = np.arctan2(centred[:, 1], centred[:, 0])

    blocks = [embedding] + ([np.asarray(features, dtype=np.float64)] if features is not None else [])
    u = np.column_stack([*blocks, np.asarray(lap_eigvecs, dtype=np.float64), theta, phi])
    u[np.asarray(timeseries).std(axis=1) == 0] = 0.0
    return u.astype(np.float32)


class _LinearAttentionBlock(nn.Module):
    """Pre-norm block: kernelized linear attention (``elu + 1`` feature map) plus an MLP, both residual.
    Cost ``O(V p^2)``: no ``(V, V)`` attention matrix is ever formed."""

    def __init__(self, p: int, heads: int, mlp_ratio: int) -> None:
        super().__init__()
        if p % heads:
            raise ValueError(f"p={p} must be divisible by the number of heads ({heads})")
        self.heads, self.norm1, self.norm2 = heads, nn.LayerNorm(p), nn.LayerNorm(p)
        self.qkv, self.proj = nn.Linear(p, 3 * p), nn.Linear(p, p)
        self.mlp = nn.Sequential(nn.Linear(p, mlp_ratio * p), nn.GELU(), nn.Linear(mlp_ratio * p, p))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x (V, p)
        V, p = x.shape
        q, k, v = (t.reshape(V, self.heads, p // self.heads) for t in self.qkv(self.norm1(x)).chunk(3, dim=-1))
        q, k = F.elu(q) + 1.0, F.elu(k) + 1.0
        kv = torch.einsum("vhd,vhe->hde", k, v)  # (H, dh, dh)
        normalizer = 1.0 / (torch.einsum("vhd,hd->vh", q, k.sum(0)) + 1e-6)
        attended = torch.einsum("vhd,hde,vh->vhe", q, kv, normalizer).reshape(V, p)
        x = x + self.proj(attended)
        return x + self.mlp(self.norm2(x))


class ScoreEncoder(nn.Module):
    """``u (V, in_dim) -> (S_phi (V, K), tau_phi (V,))``, both float32.

    Four blocks of linear attention over the ``V`` vertex tokens (``V`` up to ~10^4), learned template-node
    embeddings ``g_k (K, p)``, and a learned per-vertex noise scale. ``forward(u, M_s0)`` subtracts
    ``lambda * M_s0`` (the anatomical anchor, ``(V, K)``, ``lambda`` learned) from the score.
    """

    def __init__(self, in_dim: int, p: int, K: int, n_blocks: int = 4, device: torch.device | None = None,
                 heads: int = 4, mlp_ratio: int = 4, lambda_init: float = 1.0, tau_min: float = 1e-3) -> None:
        super().__init__()
        self.p, self.tau_min = p, tau_min
        self.embed = nn.Linear(in_dim, p)
        self.blocks = nn.ModuleList(_LinearAttentionBlock(p, heads, mlp_ratio) for _ in range(n_blocks))
        self.norm = nn.LayerNorm(p)
        self.W = nn.Linear(p, p, bias=False)
        self.g = nn.Parameter(torch.randn(K, p) / math.sqrt(p))  # template-node embeddings
        self.tau_head = nn.Linear(p, 1)
        self.lam = nn.Parameter(torch.tensor(float(lambda_init)))
        if device is not None:
            self.to(device)

    def forward(self, u: torch.Tensor, M_s0: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.embed(u)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        S_phi = self.W(h) @ self.g.T / math.sqrt(self.p)
        if M_s0 is not None:
            S_phi = S_phi - self.lam * M_s0
        tau_phi = F.softplus(self.tau_head(h).squeeze(-1)) + self.tau_min
        return S_phi, tau_phi


class SinkhornPosterior:
    """The reparametrized posterior ``pi = Sink_eps^L(exp[(S_phi + tau_phi * xi) / eps], mu_s, nu)``.

    Everything runs in float64 on CPU. ``xi`` is standard normal, so a draw is a deterministic function of
    ``(S_phi, tau_phi, xi)`` and gradients are pathwise.
    """

    def sample(self, S_phi, tau_phi, mu_s, nu, eps: float, M: int = 4, L: int = 30,
               generator: np.random.Generator | None = None) -> np.ndarray:
        """``(M, V, K)`` float64 draws on the polytope, without gradients (NumPy)."""
        generator = generator or np.random.default_rng()
        S = np.asarray(S_phi, dtype=np.float64)
        xi = generator.standard_normal((M, *S.shape))
        return perturb_then_project(S, tau_phi, xi, np.asarray(mu_s, dtype=np.float64), np.asarray(nu, dtype=np.float64), eps, L)

    def sample_torch(self, S_phi: torch.Tensor, tau_phi: torch.Tensor, mu_s: torch.Tensor, nu: torch.Tensor, eps: float,
                     M: int = 4, L: int = 30, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Differentiable draws for training: ``(pi (M, V, K) float64, xi (M, V, K))``.

        ``S_phi`` / ``tau_phi`` are float32 and may live on MPS, which has no float64. Combined
        ``.to(device="cpu", dtype=float64)`` raises ``TypeError`` on this build, so promote with
        ``.cpu().to(dtype=float64)``. ``.cpu()`` is differentiable and keeps the autograd graph.
        OT arithmetic always runs as float64 on CPU; ``mu_s`` / ``nu`` are forced there the same way.
        """
        S = S_phi.cpu().to(dtype=torch.float64)
        tau = tau_phi.cpu().to(dtype=torch.float64)
        mu = mu_s.cpu().to(dtype=torch.float64) if mu_s.device.type != "cpu" else mu_s.to(dtype=torch.float64)
        nu_c = nu.cpu().to(dtype=torch.float64) if nu.device.type != "cpu" else nu.to(dtype=torch.float64)
        xi = torch.randn((M, *S.shape), generator=generator, dtype=torch.float64, device="cpu")
        pi = sinkhorn_log_torch(S.unsqueeze(0) + tau[None, :, None] * xi, mu, nu_c, eps, n_iter=L)
        return pi, xi

    def log_prob(self, tau_phi, xi):
        """``log q`` of each draw's perturbed score ``S' = S_phi + tau_phi * xi`` (shape ``(M,)``):
        ``log N(xi; 0, I) - K * sum_i log tau_i``. It omits the log-determinant of the Sinkhorn map, so it is a
        diagnostic for the perturbation, not the density of ``pi`` (see :mod:`trajot.inference.entropy`)."""
        K = xi.shape[-1]
        if isinstance(xi, torch.Tensor):
            tau = tau_phi.to(dtype=xi.dtype, device=xi.device)
            return -0.5 * (xi**2).sum((-2, -1)) - 0.5 * xi[0].numel() * math.log(2 * math.pi) - K * torch.log(tau).sum()
        tau = np.asarray(tau_phi, dtype=np.float64)
        return -0.5 * (xi**2).sum((-2, -1)) - 0.5 * xi[0].size * math.log(2 * math.pi) - K * np.log(tau).sum()
