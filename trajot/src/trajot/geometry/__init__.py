"""Geometry primitives used by TrajOT preprocessing and downstream models."""

from .connectivity import (
    connectivity,
    correlation_matrix,
    fisher_z,
    parcellate,
    rank_factorize,
)
from .cost import anatomical_cost, feature_cost, geodesic_cost, vertex_mass
from .diffusion import (
    diffusion_map,
    gauge_features,
    gauge_velocity,
    laplacian_eigenvectors,
)

__all__ = [
    "parcellate",
    "correlation_matrix",
    "fisher_z",
    "connectivity",
    "rank_factorize",
    "diffusion_map",
    "laplacian_eigenvectors",
    "gauge_velocity",
    "gauge_features",
    "geodesic_cost",
    "anatomical_cost",
    "vertex_mass",
    "feature_cost",
]
