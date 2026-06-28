"""Minimal mesh PLY writer."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_triangle_mesh_ply(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3)")

    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {vertices.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {faces.shape[0]}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")
        for tri in faces:
            f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")

