"""Local graph scores for manifold validity diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .diagnostics import GaussianDiagnostics


@dataclass(frozen=True)
class GraphDiagnostics:
    indices: np.ndarray
    neighbor_indices: np.ndarray
    rank2_score: np.ndarray
    normal_variation: np.ndarray
    log_area_variation: np.ndarray
    curvature_scale: np.ndarray


def compute_graph_diagnostics(
    diag: GaussianDiagnostics,
    k: int = 12,
    max_points: int = 200_000,
) -> GraphDiagnostics:
    """Compute local scores on kept surface-like splats.

    Scores are intentionally simple and robust:

    - rank2_score: local PCA smallest eigenvalue divided by trace.
    - normal_variation: mean `1 - |n_i dot n_j|`.
    - log_area_variation: std of log surface mass in the local neighborhood.
    - curvature_scale: max normal-change angle per distance, multiplied by local radius.
    """
    surface_indices = np.flatnonzero(diag.keep_surface)
    if surface_indices.size == 0:
        empty = np.empty((0,), dtype=np.float32)
        return GraphDiagnostics(
            indices=surface_indices,
            neighbor_indices=np.empty((0, 0), dtype=np.int64),
            rank2_score=empty,
            normal_variation=empty,
            log_area_variation=empty,
            curvature_scale=empty,
        )

    if surface_indices.size > max_points:
        rng = np.random.default_rng(0)
        surface_indices = np.sort(rng.choice(surface_indices, size=max_points, replace=False))

    xyz = diag.xyz[surface_indices]
    normals = diag.normals[surface_indices]
    mass = np.maximum(diag.mass[surface_indices], 1e-12)
    radius = np.sqrt(np.maximum(diag.eigenvalues[surface_indices, 0] + diag.eigenvalues[surface_indices, 1], 1e-24))

    if xyz.shape[0] < 2:
        empty_neighbors = np.empty((xyz.shape[0], 0), dtype=np.int64)
        zeros = np.zeros((xyz.shape[0],), dtype=np.float32)
        return GraphDiagnostics(
            indices=surface_indices,
            neighbor_indices=empty_neighbors,
            rank2_score=zeros,
            normal_variation=zeros,
            log_area_variation=zeros,
            curvature_scale=zeros,
        )

    k_eff = min(k + 1, xyz.shape[0])
    tree = cKDTree(xyz)
    distances, local_neighbors = tree.query(xyz, k=k_eff)
    if k_eff == 1:
        local_neighbors = local_neighbors[:, None]
        distances = distances[:, None]

    local_neighbors = local_neighbors[:, 1:]
    distances = distances[:, 1:]
    neighbor_global = surface_indices[local_neighbors]

    rank2 = np.zeros((xyz.shape[0],), dtype=np.float32)
    normal_var = np.zeros_like(rank2)
    log_area_var = np.zeros_like(rank2)
    curv_scale = np.zeros_like(rank2)

    log_mass = np.log(mass)
    for row in range(xyz.shape[0]):
        nb = local_neighbors[row]
        pts = xyz[nb]
        center = pts.mean(axis=0, keepdims=True)
        cov = (pts - center).T @ (pts - center) / max(pts.shape[0], 1)
        eig = np.linalg.eigvalsh(cov)
        rank2[row] = float(eig[0] / max(eig.sum(), 1e-24))

        dots = np.abs(normals[nb] @ normals[row])
        dots = np.clip(dots, 0.0, 1.0)
        normal_var[row] = float(np.mean(1.0 - dots))
        log_area_var[row] = float(np.std(log_mass[nb]))

        angles = np.arccos(dots)
        kappa = angles / np.maximum(distances[row], 1e-12)
        curv_scale[row] = float(np.max(kappa) * radius[row])

    return GraphDiagnostics(
        indices=surface_indices,
        neighbor_indices=neighbor_global.astype(np.int64),
        rank2_score=rank2,
        normal_variation=normal_var,
        log_area_variation=log_area_var,
        curvature_scale=curv_scale.astype(np.float32),
    )


def summarize_graph(graph: GraphDiagnostics) -> dict[str, float]:
    n = int(graph.indices.size)
    if n == 0:
        return {
            "graph_points": 0,
            "rank2_median": 0.0,
            "normal_variation_median": 0.0,
            "log_area_variation_median": 0.0,
            "curvature_scale_median": 0.0,
            "rank2_p90": 0.0,
            "curvature_scale_p90": 0.0,
        }
    return {
        "graph_points": n,
        "rank2_median": float(np.median(graph.rank2_score)),
        "normal_variation_median": float(np.median(graph.normal_variation)),
        "log_area_variation_median": float(np.median(graph.log_area_variation)),
        "curvature_scale_median": float(np.median(graph.curvature_scale)),
        "rank2_p90": float(np.quantile(graph.rank2_score, 0.90)),
        "curvature_scale_p90": float(np.quantile(graph.curvature_scale, 0.90)),
    }


def save_graph_npz(graph: GraphDiagnostics, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        indices=graph.indices,
        neighbor_indices=graph.neighbor_indices,
        rank2_score=graph.rank2_score,
        normal_variation=graph.normal_variation,
        log_area_variation=graph.log_area_variation,
        curvature_scale=graph.curvature_scale,
    )
