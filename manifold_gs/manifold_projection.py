"""Offline projection of Gaussian centers onto manifold-like point patches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ProjectedManifold:
    xyz: np.ndarray
    normals: np.ndarray
    mass: np.ndarray
    confidence: np.ndarray
    accepted: np.ndarray
    radii: np.ndarray
    source_indices: np.ndarray
    neighbors: np.ndarray


def relax_certified_quadrature(
    projected: ProjectedManifold,
    relaxation: float = 0.5,
    radius_cap_quantile: float = 0.5,
) -> np.ndarray:
    """Robustly rebalance certified mass while preserving it exactly.

    Squared kNN radius is a local area estimator. Winsorization prevents an
    isolated certified point from claiming an arbitrarily large surface patch.
    Residual-layer mass is unchanged.
    """
    mass = np.asarray(projected.mass, dtype=np.float64).copy()
    accepted = np.asarray(projected.accepted, dtype=bool)
    if not np.any(accepted) or relaxation <= 0:
        return mass.astype(np.float32)
    eta = float(np.clip(relaxation, 0.0, 1.0))
    radii2 = np.maximum(np.asarray(projected.radii[accepted], dtype=np.float64) ** 2, 1e-16)
    cap = float(np.quantile(radii2, np.clip(radius_cap_quantile, 0.0, 1.0)))
    target = np.minimum(radii2, max(cap, 1e-16))
    accepted_total = float(np.sum(mass[accepted]))
    target *= accepted_total / max(float(np.sum(target)), 1e-16)
    mass[accepted] = (1.0 - eta) * mass[accepted] + eta * target
    # Remove floating-point normalization drift without changing residual mass.
    mass[accepted] *= accepted_total / max(float(np.sum(mass[accepted])), 1e-16)
    return mass.astype(np.float32)


def _fit_local_surfaces(
    xyz: np.ndarray,
    mass: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = xyz.shape[0]
    k_eff = min(max(k, 3) + 1, count)
    tree = cKDTree(xyz)
    distances, neighbors = tree.query(xyz, k=k_eff)
    if neighbors.ndim == 1:
        neighbors = neighbors[:, None]
        distances = distances[:, None]
    local = neighbors[:, 1:]
    local_distances = distances[:, 1:]
    normals = np.zeros_like(xyz, dtype=np.float64)
    centers = np.zeros_like(xyz, dtype=np.float64)
    eigenvalues = np.zeros((count, 3), dtype=np.float64)
    radii = local_distances[:, -1].copy()
    for i in range(count):
        ids = np.concatenate(([i], local[i]))
        pts = xyz[ids]
        bandwidth = max(radii[i], 1e-12)
        spatial = np.exp(-np.sum((pts - xyz[i]) ** 2, axis=1) / (bandwidth * bandwidth))
        weights = spatial * np.maximum(mass[ids], 1e-16)
        weights /= max(np.sum(weights), 1e-16)
        center = np.sum(weights[:, None] * pts, axis=0)
        delta = pts - center
        covariance = (weights[:, None] * delta).T @ delta
        values, vectors = np.linalg.eigh(covariance)
        centers[i] = center
        eigenvalues[i] = values
        plane_normal = vectors[:, 0]
        tangent_u = vectors[:, 2]
        tangent_v = vectors[:, 1]

        # Quadratic height field captures the leading curvature term and avoids
        # the systematic shrinkage caused by repeated projection to flat PCA
        # planes on spheres and other curved patches.
        u = delta @ tangent_u
        v = delta @ tangent_v
        height = delta @ plane_normal
        scaled_u = u / bandwidth
        scaled_v = v / bandwidth
        design = np.column_stack([
            scaled_u * scaled_u,
            scaled_u * scaled_v,
            scaled_v * scaled_v,
            scaled_u,
            scaled_v,
            np.ones_like(scaled_u),
        ])
        sqrt_weights = np.sqrt(weights)
        try:
            coeff, _, _, _ = np.linalg.lstsq(
                design * sqrt_weights[:, None], height * sqrt_weights, rcond=1e-6
            )
            query_delta = xyz[i] - center
            query_u = float(query_delta @ tangent_u)
            query_v = float(query_delta @ tangent_v)
            query_scaled_u = query_u / bandwidth
            query_scaled_v = query_v / bandwidth
            predicted_height = float(np.array([
                query_scaled_u * query_scaled_u,
                query_scaled_u * query_scaled_v,
                query_scaled_v * query_scaled_v,
                query_scaled_u,
                query_scaled_v,
                1.0,
            ]) @ coeff)
            centers[i] = xyz[i] + (
                predicted_height - float(query_delta @ plane_normal)
            ) * plane_normal
            du = (2.0 * coeff[0] * query_scaled_u + coeff[1] * query_scaled_v + coeff[3]) / bandwidth
            dv = (coeff[1] * query_scaled_u + 2.0 * coeff[2] * query_scaled_v + coeff[4]) / bandwidth
            fitted_normal = plane_normal - du * tangent_u - dv * tangent_v
            normals[i] = fitted_normal / max(np.linalg.norm(fitted_normal), 1e-12)
        except np.linalg.LinAlgError:
            centers[i] = xyz[i] - float((xyz[i] - center) @ plane_normal) * plane_normal
            normals[i] = plane_normal
    return centers, normals, eigenvalues, local


def _orient_normals(normals: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    oriented = normals.copy()
    visited = np.zeros(normals.shape[0], dtype=bool)
    for root in range(normals.shape[0]):
        if visited[root]:
            continue
        visited[root] = True
        stack = [root]
        while stack:
            i = stack.pop()
            for j in neighbors[i]:
                j = int(j)
                if visited[j]:
                    continue
                if np.dot(oriented[i], oriented[j]) < 0:
                    oriented[j] *= -1.0
                visited[j] = True
                stack.append(j)
    return oriented


def project_points_to_manifold(
    xyz: np.ndarray,
    mass: np.ndarray | None = None,
    source_indices: np.ndarray | None = None,
    k: int = 16,
    iterations: int = 2,
    max_normal_ratio: float = 0.25,
    min_tangent_ratio: float = 0.12,
    min_confidence: float = 0.25,
    projection_step: float = 0.5,
) -> ProjectedManifold:
    """Apply weighted local-PCA/MLS projection and reject non-surface support.

    The covariance spectrum is estimated from neighboring centers, not from the
    current Gaussian scales. This lets geometry bootstrap even when every
    rendering Gaussian is initially isotropic.
    """
    points = np.asarray(xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")
    if points.shape[0] < 4:
        raise ValueError("at least four points are required")
    if mass is None:
        mass_array = np.ones(points.shape[0], dtype=np.float64)
    else:
        mass_array = np.asarray(mass, dtype=np.float64).reshape(-1)
    if source_indices is None:
        source = np.arange(points.shape[0], dtype=np.int64)
    else:
        source = np.asarray(source_indices, dtype=np.int64).reshape(-1)
    if len(mass_array) != len(points) or len(source) != len(points):
        raise ValueError("mass and source_indices must match xyz")

    projected = points.copy()
    for _ in range(max(iterations, 1)):
        targets, normals, eigenvalues, neighbors = _fit_local_surfaces(projected, mass_array, k)
        projected += projection_step * (targets - projected)

    centers, normals, eigenvalues, neighbors = _fit_local_surfaces(projected, mass_array, k)
    normals = _orient_normals(normals, neighbors)
    normal_ratio = eigenvalues[:, 0] / np.maximum(eigenvalues[:, 1], 1e-16)
    tangent_ratio = eigenvalues[:, 1] / np.maximum(eigenvalues[:, 2], 1e-16)
    confidence = np.clip(1.0 - normal_ratio / max(max_normal_ratio, 1e-12), 0.0, 1.0)
    confidence *= np.clip(tangent_ratio / max(min_tangent_ratio, 1e-12), 0.0, 1.0)
    keep = (normal_ratio <= max_normal_ratio) & (tangent_ratio >= min_tangent_ratio) & (confidence >= min_confidence)

    # Keep the whole measure to preserve coverage and mass. Confidence gates
    # topology construction later; deleting uncertain points here would improve
    # precision by silently sacrificing completeness.
    if projected.shape[0] < 2:
        projected_neighbors = np.empty((projected.shape[0], 0), dtype=np.int64)
        projected_radii = np.zeros(projected.shape[0], dtype=np.float64)
    else:
        k_eff = min(max(k, 3) + 1, projected.shape[0])
        distances, indices = cKDTree(projected).query(projected, k=k_eff)
        if indices.ndim == 1:
            indices = indices[:, None]
            distances = distances[:, None]
        projected_neighbors = indices[:, 1:].astype(np.int64)
        projected_radii = distances[:, -1]
        normals = _orient_normals(normals, projected_neighbors)

    return ProjectedManifold(
        xyz=projected.astype(np.float32),
        normals=normals.astype(np.float32),
        mass=mass_array.astype(np.float32),
        confidence=confidence.astype(np.float32),
        accepted=keep,
        radii=projected_radii.astype(np.float32),
        source_indices=source,
        neighbors=projected_neighbors,
    )
