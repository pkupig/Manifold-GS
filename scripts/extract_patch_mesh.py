#!/usr/bin/env python3
"""Extract conservative charted patch mesh from a 3DGS point_cloud.ply."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manifold_gs.diagnostics import compute_diagnostics
from manifold_gs.patch_mesh import build_patch_mesh, save_patch_mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True, help="Path to 3DGS point_cloud.ply")
    parser.add_argument("--mesh", required=True, help="Output patch mesh PLY")
    parser.add_argument("--meta", default=None, help="Optional output npz for patch ids and source Gaussian ids")
    parser.add_argument("--knn", type=int, default=12)
    parser.add_argument("--min-patch-size", type=int, default=20)
    parser.add_argument("--normal-dot-min", type=float, default=0.75)
    parser.add_argument("--max-edge-scale", type=float, default=3.0)
    parser.add_argument("--max-triangle-edge-scale", type=float, default=3.5)
    parser.add_argument("--min-triangle-quality", type=float, default=0.03)
    parser.add_argument("--max-points", type=int, default=200_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diag = compute_diagnostics(args.ply)
    mesh = build_patch_mesh(
        diag,
        k=args.knn,
        max_edge_scale=args.max_edge_scale,
        normal_dot_min=args.normal_dot_min,
        min_patch_size=args.min_patch_size,
        max_points=args.max_points,
        max_triangle_edge_scale=args.max_triangle_edge_scale,
        min_triangle_quality=args.min_triangle_quality,
    )
    meta = args.meta
    if meta is None:
        meta = str(Path(args.mesh).with_suffix(".patches.npz"))
    save_patch_mesh(mesh, args.mesh, meta)
    summary = {
        "vertices": int(mesh.vertices.shape[0]),
        "faces": int(mesh.faces.shape[0]),
        "patches": int(mesh.patch_ids.max() + 1) if mesh.patch_ids.size else 0,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {args.mesh}")
    print(f"Wrote {meta}")


if __name__ == "__main__":
    main()

