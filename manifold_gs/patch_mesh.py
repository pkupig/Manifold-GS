"""Patch-graph triangulation from surface-like Gaussian splats."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay, cKDTree

from .diagnostics import GaussianDiagnostics
from .mesh_io import write_triangle_mesh_ply


@dataclass(frozen=True)
class PatchMesh:
    vertices: np.ndarray
    faces: np.ndarray
    source_indices: np.ndarray
    patch_ids: np.ndarray


def _local_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    cov = (points - center).T @ (points - center) / max(points.shape[0], 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]
    t1 = eigvecs[:, 2]
    t2 = np.cross(normal, t1)
    t2 /= max(np.linalg.norm(t2), 1e-12)
    return center, t1, t2


def _triangle_quality(uv: np.ndarray, tri: np.ndarray) -> float:
    pts = uv[tri]
    lengths = np.array([
        np.linalg.norm(pts[1] - pts[0]),
        np.linalg.norm(pts[2] - pts[1]),
        np.linalg.norm(pts[0] - pts[2]),
    ])
    area = 0.5 * abs(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
    denom = np.sum(lengths * lengths)
    if denom <= 1e-24:
        return 0.0
    return float(4.0 * np.sqrt(3.0) * area / denom)


def build_patch_mesh(
    diag: GaussianDiagnostics,
    k: int = 12,
    max_edge_scale: float = 3.0,
    normal_dot_min: float = 0.75,
    tangent_residual_scale: float = 2.5,
    min_patch_size: int = 20,
    max_points: int = 200_000,
    max_triangle_edge_scale: float = 3.5,
    min_triangle_quality: float = 0.03,
) -> PatchMesh:
    """Build a charted patch mesh from kept surface-like splats.

    This is deliberately conservative: bad or ambiguous triangles are dropped.
    The output is a surface backbone, not a watertight reconstruction.
    """
    surface_indices = np.flatnonzero(diag.keep_surface)
    if surface_indices.size == 0:
        return PatchMesh(
            vertices=np.empty((0, 3), dtype=np.float32),
            faces=np.empty((0, 3), dtype=np.int64),
            source_indices=np.empty((0,), dtype=np.int64),
            patch_ids=np.empty((0,), dtype=np.int32),
        )

    if surface_indices.size > max_points:
        rng = np.random.default_rng(0)
        surface_indices = np.sort(rng.choice(surface_indices, size=max_points, replace=False))

    xyz = diag.xyz[surface_indices]
    normals = diag.normals[surface_indices]
    radii = np.sqrt(np.maximum(diag.eigenvalues[surface_indices, 0] + diag.eigenvalues[surface_indices, 1], 1e-24))

    return build_patch_mesh_from_points(
        xyz=xyz,
        normals=normals,
        radii=radii,
        source_indices=surface_indices,
        k=k,
        max_edge_scale=max_edge_scale,
        normal_dot_min=normal_dot_min,
        tangent_residual_scale=tangent_residual_scale,
        min_patch_size=min_patch_size,
        max_points=max_points,
        max_triangle_edge_scale=max_triangle_edge_scale,
        min_triangle_quality=min_triangle_quality,
    )


def build_patch_mesh_from_points(
    xyz: np.ndarray,
    normals: np.ndarray,
    radii: np.ndarray,
    source_indices: np.ndarray | None = None,
    k: int = 12,
    max_edge_scale: float = 3.0,
    normal_dot_min: float = 0.75,
    tangent_residual_scale: float = 2.5,
    min_patch_size: int = 20,
    max_points: int = 200_000,
    max_triangle_edge_scale: float = 3.5,
    min_triangle_quality: float = 0.03,
    chart_normal_dot_min: float = 0.90,
) -> PatchMesh:
    """Triangulate oriented points in bounded-normal-variation charts."""
    xyz = np.asarray(xyz, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    radii = np.asarray(radii, dtype=np.float32).reshape(-1)
    if source_indices is None:
        source_indices = np.arange(xyz.shape[0], dtype=np.int64)
    else:
        source_indices = np.asarray(source_indices, dtype=np.int64)

    if xyz.shape[0] > max_points:
        rng = np.random.default_rng(0)
        selection = np.sort(rng.choice(xyz.shape[0], size=max_points, replace=False))
        xyz, normals, radii, source_indices = xyz[selection], normals[selection], radii[selection], source_indices[selection]

    if xyz.shape[0] < min_patch_size:
        return PatchMesh(xyz.astype(np.float32), np.empty((0, 3), dtype=np.int64), source_indices, np.zeros((xyz.shape[0],), dtype=np.int32))

    tree = cKDTree(xyz)
    k_eff = min(k + 1, xyz.shape[0])
    distances, neighbors = tree.query(xyz, k=k_eff)
    neighbors = neighbors[:, 1:]
    distances = distances[:, 1:]

    row_edges: list[int] = []
    col_edges: list[int] = []
    for i in range(xyz.shape[0]):
        for local_j, dist in zip(neighbors[i], distances[i]):
            j = int(local_j)
            radius_ij = 0.5 * (radii[i] + radii[j])
            if dist > max_edge_scale * max(radius_ij, 1e-12):
                continue
            ndot = abs(float(normals[i] @ normals[j]))
            if ndot < normal_dot_min:
                continue
            tangent_residual = abs(float(normals[i] @ (xyz[j] - xyz[i])))
            if tangent_residual > tangent_residual_scale * max(radius_ij, 1e-12):
                continue
            row_edges.extend([i, j])
            col_edges.extend([j, i])

    if not row_edges:
        return PatchMesh(xyz.astype(np.float32), np.empty((0, 3), dtype=np.int64), source_indices, np.zeros((xyz.shape[0],), dtype=np.int32))

    graph = coo_matrix((np.ones(len(row_edges), dtype=np.uint8), (row_edges, col_edges)), shape=(xyz.shape[0], xyz.shape[0])).tocsr()

    # Pairwise-compatible edges can chain around an entire sphere, which is not
    # a valid single chart. Region growing also bounds variation from the chart
    # seed normal and therefore creates genuinely local parameterizations.
    unassigned = np.ones(xyz.shape[0], dtype=bool)
    charts: list[np.ndarray] = []
    while np.any(unassigned):
        seed = int(np.flatnonzero(unassigned)[0])
        seed_normal = normals[seed]
        members: list[int] = []
        queue = [seed]
        unassigned[seed] = False
        while queue:
            i = queue.pop()
            members.append(i)
            begin, end = graph.indptr[i], graph.indptr[i + 1]
            for j in graph.indices[begin:end]:
                j = int(j)
                if not unassigned[j]:
                    continue
                if abs(float(normals[j] @ seed_normal)) < chart_normal_dot_min:
                    continue
                unassigned[j] = False
                queue.append(j)
        charts.append(np.asarray(members, dtype=np.int64))

    vertices: list[np.ndarray] = []
    source: list[np.ndarray] = []
    patch_ids: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    vertex_offset = 0
    patch_counter = 0

    for local in charts:
        if local.size < min_patch_size:
            continue

        pts = xyz[local]
        center, t1, t2 = _local_frame(pts)
        uv = np.stack([(pts - center) @ t1, (pts - center) @ t2], axis=1)

        try:
            delaunay = Delaunay(uv)
        except Exception:
            continue

        local_faces = []
        median_radius = float(np.median(radii[local]))
        max_edge = max_triangle_edge_scale * max(median_radius, 1e-12)
        for tri in delaunay.simplices:
            tri_pts = pts[tri]
            edge_lengths = np.array([
                np.linalg.norm(tri_pts[1] - tri_pts[0]),
                np.linalg.norm(tri_pts[2] - tri_pts[1]),
                np.linalg.norm(tri_pts[0] - tri_pts[2]),
            ])
            if np.max(edge_lengths) > max_edge:
                continue
            if _triangle_quality(uv, tri) < min_triangle_quality:
                continue
            tri_normals = normals[local[tri]]
            if np.min(np.abs(tri_normals @ tri_normals[0])) < normal_dot_min:
                continue
            face_normal = np.cross(tri_pts[1] - tri_pts[0], tri_pts[2] - tri_pts[0])
            if float(face_normal @ np.mean(tri_normals, axis=0)) < 0:
                tri = tri[[0, 2, 1]]
            local_faces.append(tri + vertex_offset)

        if not local_faces:
            continue

        vertices.append(pts.astype(np.float32))
        source.append(source_indices[local].astype(np.int64))
        patch_ids.append(np.full((local.size,), patch_counter, dtype=np.int32))
        faces.append(np.asarray(local_faces, dtype=np.int64))
        vertex_offset += local.size
        patch_counter += 1

    if not vertices:
        return PatchMesh(
            vertices=np.empty((0, 3), dtype=np.float32),
            faces=np.empty((0, 3), dtype=np.int64),
            source_indices=np.empty((0,), dtype=np.int64),
            patch_ids=np.empty((0,), dtype=np.int32),
        )

    return PatchMesh(
        vertices=np.concatenate(vertices, axis=0),
        faces=np.concatenate(faces, axis=0),
        source_indices=np.concatenate(source, axis=0),
        patch_ids=np.concatenate(patch_ids, axis=0),
    )


def save_patch_mesh(mesh: PatchMesh, mesh_path: str | Path, meta_path: str | Path | None = None) -> None:
    write_triangle_mesh_ply(mesh_path, mesh.vertices, mesh.faces)
    if meta_path is not None:
        meta_path = Path(meta_path)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            meta_path,
            source_indices=mesh.source_indices,
            patch_ids=mesh.patch_ids,
        )
