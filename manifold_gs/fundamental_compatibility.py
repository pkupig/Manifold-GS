"""Discrete compatibility diagnostics for support and Gaussian normal fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class FundamentalCompatibility:
    source_indices: np.ndarray
    support_normals: np.ndarray
    support_shape: np.ndarray
    predicted_shape: np.ndarray
    normal_angle_deg: np.ndarray
    shape_relative: np.ndarray
    symmetry_residual: np.ndarray
    gauss_residual_scaled: np.ndarray
    normal_curl_scaled: np.ndarray
    codazzi_residual_scaled: np.ndarray
    planarity: np.ndarray
    radii: np.ndarray


def _weighted_lstsq(design: np.ndarray, values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    root = np.sqrt(np.maximum(weights, 0.0))
    return np.linalg.lstsq(design * root[:, None], values * root[:, None] if values.ndim == 2 else values * root, rcond=1e-6)[0]


def compute_fundamental_compatibility(
    xyz: np.ndarray,
    predicted_normals: np.ndarray,
    mass: np.ndarray | None = None,
    source_indices: np.ndarray | None = None,
    k: int = 24,
) -> FundamentalCompatibility:
    """Compare support geometry with an independently supplied normal field.

    Support first/second forms come from a quadratic Monge patch fitted to
    centers. The predicted shape operator comes from derivatives of the input
    normal field, normally the covariance normals of Gaussian primitives.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    predicted_normals = np.asarray(predicted_normals, dtype=np.float64)
    if xyz.shape != predicted_normals.shape or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz and predicted_normals must have shape (N, 3)")
    count = xyz.shape[0]
    if count < 7:
        raise ValueError("at least seven points are required")
    predicted_normals /= np.maximum(np.linalg.norm(predicted_normals, axis=1, keepdims=True), 1e-12)
    mass = np.ones(count, dtype=np.float64) if mass is None else np.asarray(mass, dtype=np.float64).reshape(-1)
    source = np.arange(count, dtype=np.int64) if source_indices is None else np.asarray(source_indices, dtype=np.int64)

    k_eff = min(max(k, 8) + 1, count)
    distances, neighbors_full = cKDTree(xyz).query(xyz, k=k_eff)
    neighbors = neighbors_full[:, 1:]
    radii = distances[:, -1]

    support_normals = np.zeros_like(xyz)
    support_shape = np.zeros((count, 2, 2), dtype=np.float64)
    predicted_shape = np.zeros_like(support_shape)
    ambient_predicted_shape = np.zeros((count, 3, 3), dtype=np.float64)
    tangent_frames = np.zeros((count, 3, 2), dtype=np.float64)
    normal_angle = np.zeros(count, dtype=np.float64)
    shape_relative = np.zeros(count, dtype=np.float64)
    symmetry = np.zeros(count, dtype=np.float64)
    gauss = np.zeros(count, dtype=np.float64)
    normal_curl = np.zeros(count, dtype=np.float64)
    planarity = np.zeros(count, dtype=np.float64)

    for i in range(count):
        ids = np.concatenate(([i], neighbors[i]))
        delta = xyz[ids] - xyz[i]
        radius = max(radii[i], 1e-12)
        weights = np.exp(-np.sum(delta * delta, axis=1) / (radius * radius)) * np.maximum(mass[ids], 1e-16)
        weights /= max(np.sum(weights), 1e-16)
        weighted_center = np.sum(weights[:, None] * delta, axis=0)
        centered = delta - weighted_center
        covariance = (weights[:, None] * centered).T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        n0 = eigenvectors[:, 0]
        if np.dot(n0, predicted_normals[i]) < 0:
            n0 *= -1.0
        t1 = eigenvectors[:, 2]
        t2 = np.cross(n0, t1)
        t2 /= max(np.linalg.norm(t2), 1e-12)
        u, v, height = delta @ t1, delta @ t2, delta @ n0
        design2 = np.column_stack([np.ones(len(ids)), u, v, 0.5 * u * u, u * v, 0.5 * v * v])
        coeff = _weighted_lstsq(design2, height, weights)
        gradient = coeff[1:3]
        hessian = np.array([[coeff[3], coeff[4]], [coeff[4], coeff[5]]])
        parametric_basis = np.column_stack([t1 + gradient[0] * n0, t2 + gradient[1] * n0])
        metric = parametric_basis.T @ parametric_basis
        metric_inv = np.linalg.inv(metric)
        support_second = hessian / np.sqrt(1.0 + float(gradient @ gradient))
        support_s = metric_inv @ support_second
        fitted_normal = np.cross(parametric_basis[:, 0], parametric_basis[:, 1])
        fitted_normal /= max(np.linalg.norm(fitted_normal), 1e-12)
        if np.dot(fitted_normal, predicted_normals[i]) < 0:
            fitted_normal *= -1.0
            support_second *= -1.0
            support_s *= -1.0

        local_predicted = predicted_normals[ids].copy()
        signs = np.sign(local_predicted @ predicted_normals[i])
        signs[signs == 0] = 1.0
        local_predicted *= signs[:, None]
        design1 = np.column_stack([np.ones(len(ids)), u, v])
        normal_coeff = _weighted_lstsq(design1, local_predicted, weights)
        normal_derivatives = np.column_stack([normal_coeff[1], normal_coeff[2]])
        predicted_second = -normal_derivatives.T @ parametric_basis
        predicted_s = metric_inv @ predicted_second

        # Ambient operator permits approximate transport into neighboring local
        # frames for the second-pass Codazzi diagnostic.
        ambient_s = parametric_basis @ predicted_s @ metric_inv @ parametric_basis.T
        tangent = np.column_stack([t1, np.cross(fitted_normal, t1)])
        tangent[:, 1] /= max(np.linalg.norm(tangent[:, 1]), 1e-12)

        support_normals[i] = fitted_normal
        tangent_frames[i] = tangent
        support_shape[i] = tangent.T @ (parametric_basis @ support_s @ metric_inv @ parametric_basis.T) @ tangent
        predicted_shape[i] = tangent.T @ ambient_s @ tangent
        ambient_predicted_shape[i] = ambient_s
        normal_angle[i] = np.degrees(np.arccos(np.clip(abs(float(fitted_normal @ predicted_normals[i])), 0.0, 1.0)))
        scale = np.linalg.norm(support_shape[i]) + 1.0 / radius
        shape_relative[i] = np.linalg.norm(predicted_shape[i] - support_shape[i]) / max(scale, 1e-12)
        symmetry[i] = np.linalg.norm(predicted_second - predicted_second.T) / max(np.linalg.norm(predicted_second), 1.0 / radius)
        gauss[i] = abs(np.linalg.det(predicted_shape[i]) - np.linalg.det(support_shape[i])) * radius * radius

        denominator = np.maximum(np.abs(local_predicted @ n0), 1e-6)
        p = -(local_predicted @ t1) / denominator
        q = -(local_predicted @ t2) / denominator
        p_coeff = _weighted_lstsq(design1, p, weights)
        q_coeff = _weighted_lstsq(design1, q, weights)
        normal_curl[i] = abs(p_coeff[2] - q_coeff[1]) * radius
        planarity[i] = eigenvalues[0] / max(eigenvalues.sum(), 1e-16)

    codazzi = np.zeros(count, dtype=np.float64)
    for i in range(count):
        ids = np.concatenate(([i], neighbors[i]))
        frame = tangent_frames[i]
        uv = (xyz[ids] - xyz[i]) @ frame
        radius = max(radii[i], 1e-12)
        weights = np.exp(-np.sum(uv * uv, axis=1) / (radius * radius)) * np.maximum(mass[ids], 1e-16)
        weights /= max(np.sum(weights), 1e-16)
        transported = np.stack([frame.T @ ambient_predicted_shape[j] @ frame for j in ids])
        transported = 0.5 * (transported + transported.transpose(0, 2, 1))
        design = np.column_stack([np.ones(len(ids)), uv])
        coeff11 = _weighted_lstsq(design, transported[:, 0, 0], weights)
        coeff12 = _weighted_lstsq(design, transported[:, 0, 1], weights)
        coeff22 = _weighted_lstsq(design, transported[:, 1, 1], weights)
        residual = np.array([coeff11[2] - coeff12[1], coeff12[2] - coeff22[1]])
        codazzi[i] = np.linalg.norm(residual) * radius * radius

    return FundamentalCompatibility(
        source_indices=source,
        support_normals=support_normals.astype(np.float32),
        support_shape=support_shape.astype(np.float32),
        predicted_shape=predicted_shape.astype(np.float32),
        normal_angle_deg=normal_angle.astype(np.float32),
        shape_relative=shape_relative.astype(np.float32),
        symmetry_residual=symmetry.astype(np.float32),
        gauss_residual_scaled=gauss.astype(np.float32),
        normal_curl_scaled=normal_curl.astype(np.float32),
        codazzi_residual_scaled=codazzi.astype(np.float32),
        planarity=planarity.astype(np.float32),
        radii=radii.astype(np.float32),
    )


def summarize_compatibility(result: FundamentalCompatibility) -> dict[str, float | int]:
    summary: dict[str, float | int] = {"points": int(result.source_indices.size)}
    for name in (
        "normal_angle_deg",
        "shape_relative",
        "symmetry_residual",
        "gauss_residual_scaled",
        "normal_curl_scaled",
        "codazzi_residual_scaled",
        "planarity",
    ):
        values = getattr(result, name)
        summary[f"{name}_median"] = float(np.median(values))
        summary[f"{name}_p90"] = float(np.quantile(values, 0.9))
    return summary

