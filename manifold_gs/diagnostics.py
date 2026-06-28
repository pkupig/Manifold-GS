"""Geometry diagnostics for 3D Gaussian checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ply_io import read_vertex_ply, write_oriented_points_ply


LABEL_BACKGROUND = 0
LABEL_SURFACE = 1
LABEL_CURVE = 2
LABEL_VOLUME = 3


@dataclass(frozen=True)
class GaussianDiagnostics:
    xyz: np.ndarray
    opacity: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    normals: np.ndarray
    tangent_area: np.ndarray
    mass: np.ndarray
    mass_is_explicit: bool
    r12: np.ndarray
    r23: np.ndarray
    labels: np.ndarray
    keep_surface: np.ndarray


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    return q / np.maximum(norm, 1e-12)


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert 3DGS quaternions `[w, x, y, z]` to rotation matrices."""
    q = normalize_quaternion(q)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rot = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - w * z)
    rot[:, 0, 2] = 2 * (x * z + w * y)
    rot[:, 1, 0] = 2 * (x * y + w * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - w * x)
    rot[:, 2, 0] = 2 * (x * z - w * y)
    rot[:, 2, 1] = 2 * (y * z + w * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot.astype(np.float32)


def load_3dgs_ply(path: str | Path) -> dict[str, np.ndarray]:
    ply = read_vertex_ply(path)
    ply.require(["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])
    data = ply.data
    result = {
        "xyz": np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32),
        "opacity_raw": np.asarray(data["opacity"], dtype=np.float32).reshape(-1),
        "scale_raw": np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=1).astype(np.float32),
        "rotation_raw": np.stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], axis=1).astype(np.float32),
    }
    if "geom_mass" in data.dtype.names:
        result["geom_mass"] = np.asarray(data["geom_mass"], dtype=np.float32).reshape(-1)
    return result


def compute_diagnostics(
    path: str | Path,
    surface_r12_min: float = 0.25,
    surface_r23_max: float = 0.08,
    curve_r12_max: float = 0.15,
    curve_r23_max: float = 0.5,
    volume_r23_min: float = 0.2,
    opacity_min: float = 0.02,
) -> GaussianDiagnostics:
    params = load_3dgs_ply(path)
    xyz = params["xyz"]
    opacity = sigmoid(params["opacity_raw"]).astype(np.float32)
    scales = np.exp(params["scale_raw"]).astype(np.float32)
    rotations = quaternion_to_matrix(params["rotation_raw"])

    eigenvalues_unsorted = scales * scales
    order = np.argsort(-eigenvalues_unsorted, axis=1)
    eigenvalues = np.take_along_axis(eigenvalues_unsorted, order, axis=1)

    eigenvectors = np.empty_like(rotations)
    for i in range(3):
        eigenvectors[:, :, i] = np.take_along_axis(
            rotations,
            order[:, None, i : i + 1],
            axis=2,
        )[:, :, 0]

    normals = eigenvectors[:, :, 2]
    tangent_area = np.sqrt(np.maximum(eigenvalues[:, 0] * eigenvalues[:, 1], 1e-24)).astype(np.float32)
    mass_is_explicit = "geom_mass" in params
    mass = params.get("geom_mass", opacity * tangent_area).astype(np.float32)
    r12 = (eigenvalues[:, 1] / np.maximum(eigenvalues[:, 0], 1e-24)).astype(np.float32)
    r23 = (eigenvalues[:, 2] / np.maximum(eigenvalues[:, 1], 1e-24)).astype(np.float32)

    labels = np.full((xyz.shape[0],), LABEL_BACKGROUND, dtype=np.int32)
    surface = (r12 > surface_r12_min) & (r23 < surface_r23_max)
    curve = (r12 < curve_r12_max) & (r23 < curve_r23_max) & ~surface
    volume = (r23 > volume_r23_min) & ~surface & ~curve
    labels[surface] = LABEL_SURFACE
    labels[curve] = LABEL_CURVE
    labels[volume] = LABEL_VOLUME
    keep_surface = surface & (opacity >= opacity_min)

    return GaussianDiagnostics(
        xyz=xyz,
        opacity=opacity,
        scales=scales,
        rotations=rotations,
        eigenvalues=eigenvalues.astype(np.float32),
        eigenvectors=eigenvectors.astype(np.float32),
        normals=normals.astype(np.float32),
        tangent_area=tangent_area,
        mass=mass,
        mass_is_explicit=mass_is_explicit,
        r12=r12,
        r23=r23,
        labels=labels,
        keep_surface=keep_surface,
    )


def summarize(diag: GaussianDiagnostics) -> dict[str, float]:
    n = int(diag.xyz.shape[0])
    counts = {
        "surface": int(np.sum(diag.labels == LABEL_SURFACE)),
        "curve": int(np.sum(diag.labels == LABEL_CURVE)),
        "volume": int(np.sum(diag.labels == LABEL_VOLUME)),
        "background": int(np.sum(diag.labels == LABEL_BACKGROUND)),
        "surface_kept": int(np.sum(diag.keep_surface)),
    }
    summary: dict[str, float] = {
        "num_gaussians": n,
        **counts,
        "surface_ratio": counts["surface"] / max(n, 1),
        "curve_ratio": counts["curve"] / max(n, 1),
        "volume_ratio": counts["volume"] / max(n, 1),
        "opacity_mean": float(np.mean(diag.opacity)) if n else 0.0,
        "mass_sum": float(np.sum(diag.mass)) if n else 0.0,
        "mass_is_explicit": diag.mass_is_explicit,
        "r12_median": float(np.median(diag.r12)) if n else 0.0,
        "r23_median": float(np.median(diag.r23)) if n else 0.0,
        "thinness_median": float(np.median(diag.eigenvalues[:, 2] / np.maximum(diag.eigenvalues[:, :2].sum(axis=1), 1e-24))) if n else 0.0,
    }
    return summary


def export_surface_points(diag: GaussianDiagnostics, path: str | Path) -> None:
    mask = diag.keep_surface
    write_oriented_points_ply(
        path=path,
        xyz=diag.xyz[mask],
        normals=diag.normals[mask],
        weights=diag.mass[mask],
        labels=diag.labels[mask],
    )


def save_npz(diag: GaussianDiagnostics, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        xyz=diag.xyz,
        opacity=diag.opacity,
        scales=diag.scales,
        eigenvalues=diag.eigenvalues,
        normals=diag.normals,
        tangent_area=diag.tangent_area,
        mass=diag.mass,
        mass_is_explicit=np.asarray(diag.mass_is_explicit),
        r12=diag.r12,
        r23=diag.r23,
        labels=diag.labels,
        keep_surface=diag.keep_surface,
    )
