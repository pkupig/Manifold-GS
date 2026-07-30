"""Mesh-cleanliness and simplification diagnostics for asset evaluation.

The metrics deliberately distinguish open boundaries (often intentional for a
conservative asset) from invalid topology and unsupported geometry.  They are used
with a fixed triangle budget to test whether an extracted asset stays useful after a
standard quadric simplification step.
"""

from __future__ import annotations

import numpy as np


def triangle_quality(vertices: np.ndarray, faces: np.ndarray) -> dict[str, float | int]:
    """Return scale-invariant triangle-shape statistics (1 is equilateral)."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if faces.size == 0:
        return {"triangles": 0, "degenerate_fraction": 0.0, "quality_mean": float("nan"),
                "quality_median": float("nan"), "quality_p05": float("nan"),
                "sliver_fraction_q_lt_0p1": 0.0}
    tri = vertices[faces]
    e0 = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)
    e1 = np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1)
    e2 = np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)
    twice_area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    denom = e0 * e0 + e1 * e1 + e2 * e2
    quality = np.divide(2.0 * np.sqrt(3.0) * twice_area, denom,
                        out=np.zeros_like(denom), where=denom > 1e-18)
    valid = twice_area > 1e-18
    return {
        "triangles": int(faces.shape[0]),
        "degenerate_fraction": float(np.mean(~valid)),
        "quality_mean": float(np.mean(quality)),
        "quality_median": float(np.median(quality)),
        "quality_p05": float(np.quantile(quality, 0.05)),
        "sliver_fraction_q_lt_0p1": float(np.mean(quality < 0.1)),
    }


def mesh_topology(vertices: np.ndarray, faces: np.ndarray) -> dict[str, int | float | bool]:
    """Count components and edge incidences without treating open boundaries as errors."""
    vertices = np.asarray(vertices)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if faces.size == 0:
        return {"vertices": int(vertices.shape[0]), "faces": 0, "components": 0,
                "boundary_edges": 0, "nonmanifold_edges": 0, "nonmanifold_edge_ratio": 0.0,
                "watertight": False}
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = int(np.sum(counts == 1))
    nonmanifold = int(np.sum(counts > 2))

    used = np.unique(faces)
    parent = np.arange(vertices.shape[0], dtype=np.int64)
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x
    def union(a: int, b: int) -> None:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra
    for tri in faces:
        union(tri[0], tri[1]); union(tri[1], tri[2])
    components = len({find(int(v)) for v in used})
    return {
        "vertices": int(vertices.shape[0]), "faces": int(faces.shape[0]),
        "components": int(components), "boundary_edges": boundary,
        "nonmanifold_edges": nonmanifold,
        "nonmanifold_edge_ratio": float(nonmanifold / max(len(counts), 1)),
        "watertight": bool(boundary == 0 and nonmanifold == 0),
    }


def cleanup_and_simplify(vertices: np.ndarray, faces: np.ndarray, target_faces: int) -> tuple[np.ndarray, np.ndarray]:
    """Apply standard Open3D cleanup then quadric decimation at a fixed face budget."""
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32)),
    )
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    if len(mesh.triangles) > target_faces:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=int(target_faces))
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_non_manifold_edges()
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.triangles, dtype=np.int64)
