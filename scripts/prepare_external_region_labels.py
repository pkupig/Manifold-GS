#!/usr/bin/env python3
"""Convert manual PLY vertex indices into source IDs for external-region evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--vertex-indices", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    vertex_indices = np.asarray([int(x) for x in args.vertex_indices.read_text(encoding="utf-8").split()], dtype=np.int64)
    if vertex_indices.size == 0:
        raise ValueError("vertex-index file is empty")
    source_indices = np.asarray(np.load(args.bundle / "asset_mapping.npz")["source_indices"], dtype=np.int64).reshape(-1)
    if vertex_indices.min() < 0 or vertex_indices.max() >= source_indices.size:
        raise ValueError("vertex index is outside certified_patches.ply range")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, source_indices=source_indices[vertex_indices], mesh_vertex_indices=vertex_indices)
    print(f"wrote {vertex_indices.size} labels to {args.out}")


if __name__ == "__main__":
    main()
