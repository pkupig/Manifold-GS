"""Small PLY reader/writer for 3DGS point_cloud.ply files.

The official 3DGS checkpoints store Gaussian parameters as vertex properties
in binary little-endian PLY files. Pulling in the full 3DGS code path requires
CUDA extensions, so this module implements only the subset needed for offline
geometry diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


_PLY_TO_DTYPE = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "i2",
    "ushort": "u2",
    "int16": "i2",
    "uint16": "u2",
    "int": "i4",
    "uint": "u4",
    "int32": "i4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


@dataclass(frozen=True)
class PlyVertexData:
    properties: list[str]
    data: np.ndarray

    def require(self, names: Iterable[str]) -> None:
        missing = [name for name in names if name not in self.data.dtype.names]
        if missing:
            raise ValueError(f"PLY is missing required vertex properties: {missing}")


def read_vertex_ply(path: str | Path) -> PlyVertexData:
    path = Path(path)
    with path.open("rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY without end_header: {path}")
            decoded = line.decode("ascii").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break

        if header_lines[0] != "ply":
            raise ValueError(f"Not a PLY file: {path}")

        fmt = None
        vertex_count = None
        vertex_props: list[tuple[str, str]] = []
        in_vertex = False

        for line in header_lines[1:]:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
                in_vertex = True
            elif parts[0] == "element":
                in_vertex = False
            elif in_vertex and parts[0] == "property":
                if parts[1] == "list":
                    raise ValueError("List vertex properties are not supported")
                ply_type, name = parts[1], parts[2]
                if ply_type not in _PLY_TO_DTYPE:
                    raise ValueError(f"Unsupported PLY property type: {ply_type}")
                vertex_props.append((name, ply_type))

        if fmt is None or vertex_count is None:
            raise ValueError(f"PLY header missing format or vertex element: {path}")

        dtype = np.dtype([(name, _PLY_TO_DTYPE[ply_type]) for name, ply_type in vertex_props])

        if fmt == "binary_little_endian":
            data = np.fromfile(f, dtype=dtype.newbyteorder("<"), count=vertex_count)
        elif fmt == "binary_big_endian":
            data = np.fromfile(f, dtype=dtype.newbyteorder(">"), count=vertex_count)
        elif fmt == "ascii":
            rows = []
            for _ in range(vertex_count):
                rows.append(tuple(float(x) for x in f.readline().decode("ascii").split()))
            data = np.array(rows, dtype=dtype)
        else:
            raise ValueError(f"Unsupported PLY format: {fmt}")

    return PlyVertexData(properties=[name for name, _ in vertex_props], data=data)


def write_oriented_points_ply(
    path: str | Path,
    xyz: np.ndarray,
    normals: np.ndarray,
    weights: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    xyz = np.asarray(xyz, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    if xyz.shape != normals.shape or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz and normals must both have shape (N, 3)")

    if weights is None:
        weights = np.ones((xyz.shape[0],), dtype=np.float32)
    else:
        weights = np.asarray(weights, dtype=np.float32).reshape(-1)

    if labels is None:
        labels = np.zeros((xyz.shape[0],), dtype=np.int32)
    else:
        labels = np.asarray(labels, dtype=np.int32).reshape(-1)

    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        for name in ("x", "y", "z", "nx", "ny", "nz", "weight"):
            f.write(f"property float {name}\n")
        f.write("property int label\n")
        f.write("end_header\n")
        for p, n, w, label in zip(xyz, normals, weights, labels):
            f.write(
                f"{p[0]:.9g} {p[1]:.9g} {p[2]:.9g} "
                f"{n[0]:.9g} {n[1]:.9g} {n[2]:.9g} "
                f"{w:.9g} {int(label)}\n"
            )


def write_vertex_ply_data(path: str | Path, data: np.ndarray) -> None:
    """Write a flat structured vertex array as binary little-endian PLY."""
    if data.dtype.names is None:
        raise ValueError("data must be a structured array")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype_to_ply = {
        ("i", 1): "char", ("u", 1): "uchar",
        ("i", 2): "short", ("u", 2): "ushort",
        ("i", 4): "int", ("u", 4): "uint",
        ("f", 4): "float", ("f", 8): "double",
    }
    with path.open("wb") as f:
        lines = ["ply", "format binary_little_endian 1.0", f"element vertex {len(data)}"]
        output_dtype = []
        for name in data.dtype.names:
            field = data.dtype.fields[name][0]
            key = (field.kind, field.itemsize)
            if key not in dtype_to_ply:
                raise ValueError(f"Unsupported dtype for PLY property {name}: {field}")
            lines.append(f"property {dtype_to_ply[key]} {name}")
            output_dtype.append((name, field.str))
        lines.append("end_header")
        f.write(("\n".join(lines) + "\n").encode("ascii"))
        np.asarray(data, dtype=np.dtype(output_dtype).newbyteorder("<")).tofile(f)

