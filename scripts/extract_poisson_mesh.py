#!/usr/bin/env python3
"""Extract a Poisson mesh from manifold-GS oriented points."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", required=True, help="Oriented point PLY from analyze_gaussians.py")
    parser.add_argument("--mesh", required=True, help="Output mesh path, usually .ply")
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--density-quantile", type=float, default=0.02)
    parser.add_argument("--voxel-size", type=float, default=0.0, help="Optional point-cloud downsample voxel size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    point_cloud = o3d.io.read_point_cloud(args.points)
    if not point_cloud.has_points():
        raise ValueError(f"No points found in {args.points}")
    if not point_cloud.has_normals():
        raise ValueError(f"Point cloud has no normals: {args.points}")

    if args.voxel_size > 0:
        point_cloud = point_cloud.voxel_down_sample(args.voxel_size)
        point_cloud.normalize_normals()

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud,
        depth=args.depth,
    )
    densities = np.asarray(densities)
    if densities.size and args.density_quantile > 0:
        threshold = np.quantile(densities, args.density_quantile)
        mesh.remove_vertices_by_mask(densities < threshold)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    out = Path(args.mesh)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(out), mesh):
        raise RuntimeError(f"Failed to write mesh: {out}")

    print(f"Wrote {out}")
    print(f"vertices={len(mesh.vertices)} triangles={len(mesh.triangles)}")


if __name__ == "__main__":
    main()

