"""Minimal triangle-mesh PLY/OBJ I/O used by the asset pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_triangle_mesh_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the ASCII triangle PLY files emitted by this project."""
    path = Path(path)
    with path.open("r", encoding="ascii") as f:
        if f.readline().strip() != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        vertex_count = face_count = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PLY has no end_header: {path}")
            parts = line.strip().split()
            if parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
            elif parts[:2] == ["element", "face"]:
                face_count = int(parts[2])
            elif parts == ["end_header"]:
                break
        if vertex_count is None or face_count is None:
            raise ValueError(f"PLY is missing vertex or face counts: {path}")
        vertices = np.asarray(
            [[float(x) for x in f.readline().split()[:3]] for _ in range(vertex_count)],
            dtype=np.float32,
        )
        faces: list[list[int]] = []
        for _ in range(face_count):
            row = [int(x) for x in f.readline().split()]
            if not row or row[0] != 3 or len(row) < 4:
                raise ValueError("Only triangular faces are supported")
            faces.append(row[1:4])
    return vertices, np.asarray(faces, dtype=np.int64).reshape(-1, 3)


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


def write_grouped_obj(
    path: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    patch_ids: np.ndarray,
) -> None:
    """Write an OBJ whose faces are grouped by per-vertex patch identifiers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    patch_ids = np.asarray(patch_ids, dtype=np.int32).reshape(-1)
    if patch_ids.shape[0] != vertices.shape[0]:
        raise ValueError("patch_ids must have one entry per vertex")
    material_path = path.with_suffix(".mtl")
    patch_values = np.unique(patch_ids)
    with material_path.open("w", encoding="ascii") as mtl:
        for patch_id in patch_values:
            # Deterministic high-contrast pseudo-color for DCC inspection.
            hue = (int(patch_id) * 0.61803398875) % 1.0
            rgb = np.asarray([
                0.35 + 0.55 * abs(np.sin(2 * np.pi * hue)),
                0.35 + 0.55 * abs(np.sin(2 * np.pi * (hue + 1 / 3))),
                0.35 + 0.55 * abs(np.sin(2 * np.pi * (hue + 2 / 3))),
            ])
            mtl.write(f"newmtl patch_{int(patch_id):04d}\n")
            mtl.write(f"Kd {rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}\n")
            mtl.write("Ka 0.05 0.05 0.05\nKs 0 0 0\n\n")
    with path.open("w", encoding="ascii") as f:
        f.write("# ManifoldGS certified open-patch asset\n")
        f.write(f"mtllib {material_path.name}\n")
        for vertex in vertices:
            f.write(f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n")
        face_patch = np.asarray(
            [int(np.bincount(patch_ids[tri].clip(min=0)).argmax()) for tri in faces],
            dtype=np.int32,
        ) if faces.size else np.empty((0,), dtype=np.int32)
        for patch_id in np.unique(face_patch):
            f.write(f"o patch_{int(patch_id):04d}\n")
            f.write(f"g patch_{int(patch_id):04d}\n")
            f.write(f"usemtl patch_{int(patch_id):04d}\n")
            for tri in faces[face_patch == patch_id]:
                a, b, c = tri + 1
                f.write(f"f {a} {b} {c}\n")
