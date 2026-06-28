"""Write manifold-projected geometry back into a full 3DGS PLY."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.lib import recfunctions
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .diagnostics import quaternion_to_matrix
from .manifold_projection import ProjectedManifold
from .ply_io import read_vertex_ply, write_vertex_ply_data


def estimate_knn_area_mass(xyz: np.ndarray, k: int = 3) -> np.ndarray:
    k_eff = min(k, max(xyz.shape[0] - 1, 1))
    if xyz.shape[0] < 2:
        return np.ones(xyz.shape[0], dtype=np.float32)
    distances, _ = cKDTree(xyz).query(xyz, k=k_eff + 1)
    return (np.pi * distances[:, -1] ** 2 / k_eff).astype(np.float32)


def write_projected_gaussians(
    input_ply: str | Path,
    output_ply: str | Path,
    projected: ProjectedManifold,
    normal_scale_ratio: float = 0.08,
) -> None:
    source = read_vertex_ply(input_ply).data
    rows = source[projected.source_indices].copy()
    rows["x"], rows["y"], rows["z"] = projected.xyz.T

    scale_raw = np.stack([rows[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float64)
    scales = np.exp(scale_raw)
    quaternions = np.stack([rows[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float64)
    old_rotations = quaternion_to_matrix(quaternions).astype(np.float64)
    new_rotations = np.empty_like(old_rotations)
    for i, normal in enumerate(projected.normals.astype(np.float64)):
        order = np.argsort(-scales[i])
        normal /= max(np.linalg.norm(normal), 1e-12)
        tangent = old_rotations[i, :, order[0]]
        tangent -= normal * float(tangent @ normal)
        if np.linalg.norm(tangent) < 1e-8:
            axis = np.eye(3)[np.argmin(np.abs(normal))]
            tangent = axis - normal * float(axis @ normal)
        tangent /= max(np.linalg.norm(tangent), 1e-12)
        bitangent = np.cross(normal, tangent)
        frame = np.empty((3, 3), dtype=np.float64)
        frame[:, order[0]] = tangent
        frame[:, order[1]] = bitangent
        frame[:, order[2]] = normal
        if np.linalg.det(frame) < 0:
            frame[:, order[1]] *= -1.0
        new_rotations[i] = frame

        tangent_scale = np.sqrt(scales[i, order[0]] * scales[i, order[1]])
        scales[i, order[2]] = min(scales[i, order[2]], normal_scale_ratio * tangent_scale)

    xyzw = Rotation.from_matrix(new_rotations).as_quat()
    wxyz = xyzw[:, [3, 0, 1, 2]].astype(np.float32)
    for i in range(3):
        rows[f"scale_{i}"] = np.log(np.maximum(scales[:, i], 1e-12)).astype(np.float32)
    for i in range(4):
        rows[f"rot_{i}"] = wxyz[:, i]

    if "geom_mass" in rows.dtype.names:
        rows["geom_mass"] = projected.mass
    else:
        rows = recfunctions.append_fields(rows, "geom_mass", projected.mass, dtypes="f4", usemask=False)
    write_vertex_ply_data(output_ply, rows)
