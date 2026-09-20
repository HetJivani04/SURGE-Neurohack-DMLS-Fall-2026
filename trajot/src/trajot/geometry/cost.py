from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def geodesic_cost(
    embedding: np.ndarray,
    coords: np.ndarray,
    k: int = 10,
    normalize: bool = True,
) -> np.ndarray:
    """Approximate geodesic distances from a kNN graph in embedding space.

    kNN edges are selected in embedding space and weighted by Euclidean distance
    in anatomical coordinates.
    """

    emb = np.asarray(embedding, dtype=np.float64)
    xyz = np.asarray(coords, dtype=np.float64)

    if emb.ndim != 2:
        raise ValueError(f"embedding must be 2D (V, d), found shape {emb.shape}")
    if xyz.shape != (emb.shape[0], 3):
        raise ValueError(
            f"coords must have shape ({emb.shape[0]}, 3), found {xyz.shape}"
        )

    n_vertices = emb.shape[0]
    if n_vertices == 0:
        raise ValueError("embedding must contain at least one vertex")
    if n_vertices == 1:
        return np.zeros((1, 1), dtype=np.float64)

    k_eff = int(max(1, min(k, n_vertices - 1)))

    emb_dist = cdist(emb, emb, metric="euclidean")
    anat_dist = cdist(xyz, xyz, metric="euclidean")

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for i in range(n_vertices):
        nn = np.argpartition(emb_dist[i], kth=k_eff)[: k_eff + 1]
        for j in nn:
            if i == j:
                continue
            rows.append(i)
            cols.append(j)
            data.append(float(anat_dist[i, j]))

    graph = csr_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices))
    graph = 0.5 * (graph + graph.T)

    dist = dijkstra(csgraph=graph, directed=False, return_predecessors=False)
    dist = np.asarray(dist, dtype=np.float64)

    # Keep the matrix finite and symmetric for downstream OT solvers.
    if not np.isfinite(dist).all():
        max_finite = np.nanmax(dist[np.isfinite(dist)]) if np.isfinite(dist).any() else 1.0
        dist[~np.isfinite(dist)] = max_finite

    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)

    if normalize:
        max_val = float(dist.max())
        if max_val > 0:
            dist /= max_val

    return dist


def anatomical_cost(
    coords: np.ndarray,
    faces: np.ndarray,
    method: str = "geodesic",
) -> np.ndarray:
    """Build anatomical cost M_s^0 as a registered-surface geodesic distance matrix."""

    xyz = np.asarray(coords, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"coords must be (V, 3), found shape {xyz.shape}")

    if method != "geodesic":
        raise ValueError(f"Unsupported method '{method}'. Expected 'geodesic'.")

    tri = np.asarray(faces)
    if tri.size == 0:
        return geodesic_cost(xyz, xyz, k=10, normalize=True)

    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError(f"faces must be (F, 3), found shape {tri.shape}")

    n_vertices = xyz.shape[0]
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for a, b, c in tri.astype(int):
        edges = ((a, b), (b, c), (c, a))
        for u, v in edges:
            if u == v:
                continue
            w = float(np.linalg.norm(xyz[u] - xyz[v]))
            rows.extend([u, v])
            cols.extend([v, u])
            data.extend([w, w])

    graph = csr_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices))
    dist = dijkstra(csgraph=graph, directed=False, return_predecessors=False)
    dist = np.asarray(dist, dtype=np.float64)

    if not np.isfinite(dist).all():
        max_finite = np.nanmax(dist[np.isfinite(dist)]) if np.isfinite(dist).any() else 1.0
        dist[~np.isfinite(dist)] = max_finite

    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)

    max_val = float(dist.max())
    if max_val > 0:
        dist /= max_val
    return dist


def vertex_mass(V: int, mask: np.ndarray | None = None) -> np.ndarray:
    """Return a probability vector over vertices that sums exactly to one."""

    if V <= 0:
        raise ValueError("V must be positive")

    if mask is None:
        mass = np.full(V, 1.0 / V, dtype=np.float64)
    else:
        keep = np.asarray(mask).astype(bool)
        if keep.shape != (V,):
            raise ValueError(f"mask must have shape ({V},), found {keep.shape}")
        n_keep = int(keep.sum())
        if n_keep == 0:
            raise ValueError("mask selects zero vertices")
        mass = np.zeros(V, dtype=np.float64)
        mass[keep] = 1.0 / n_keep

    # Force exact sum-to-one in float64.
    mass[-1] += 1.0 - mass.sum()
    return mass


def feature_cost(Y: np.ndarray, F_bar: np.ndarray) -> np.ndarray:
    """Euclidean feature cost between subject and template feature matrices."""

    left = np.asarray(Y, dtype=np.float64)
    right = np.asarray(F_bar, dtype=np.float64)

    if left.ndim != 2 or right.ndim != 2:
        raise ValueError(
            f"Y and F_bar must be 2D, found shapes {left.shape} and {right.shape}"
        )
    if left.shape[1] != right.shape[1]:
        raise ValueError(
            f"Feature dimensions must match, found {left.shape[1]} and {right.shape[1]}"
        )

    return cdist(left, right, metric="euclidean").astype(np.float64)
