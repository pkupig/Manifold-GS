#!/usr/bin/env python3
"""Close the offline geometry path from a Gaussian PLY to manifold patches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manifold_gs.diagnostics import compute_diagnostics
from manifold_gs.gt_metrics import geometry_metrics, normalized_kernel_varifold_distance
from manifold_gs.gaussian_projection import estimate_knn_area_mass, write_projected_gaussians
from manifold_gs.manifold_projection import project_points_to_manifold, relax_certified_quadrature
from manifold_gs.patch_mesh import build_patch_mesh_from_points, save_patch_mesh
from manifold_gs.ply_io import write_oriented_points_ply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gt", default=None, help="Optional analytic gt/surface.npz")
    parser.add_argument("--opacity-min", type=float, default=0.02)
    parser.add_argument("--knn", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--max-normal-ratio", type=float, default=0.25)
    parser.add_argument("--min-tangent-ratio", type=float, default=0.12)
    parser.add_argument("--min-confidence", type=float, default=0.25)
    parser.add_argument("--projection-step", type=float, default=0.5)
    parser.add_argument("--min-patch-size", type=int, default=12)
    parser.add_argument("--chart-normal-dot-min", type=float, default=0.90)
    parser.add_argument("--max-points", type=int, default=100000)
    parser.add_argument("--normal-scale-ratio", type=float, default=0.08)
    parser.add_argument("--mass-relaxation", type=float, default=0.75)
    parser.add_argument("--mass-radius-cap-quantile", type=float, default=0.5)
    parser.add_argument("--mesh-radius-cap-quantile", type=float, default=0.5)
    return parser.parse_args()


def mesh_statistics(vertices: np.ndarray, faces: np.ndarray) -> dict[str, int | float]:
    if faces.size == 0:
        return {"vertices": int(vertices.shape[0]), "faces": 0, "boundary_edges": 0, "nonmanifold_edges": 0}
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return {
        "vertices": int(vertices.shape[0]),
        "faces": int(faces.shape[0]),
        "boundary_edges": int(np.sum(counts == 1)),
        "nonmanifold_edges": int(np.sum(counts > 2)),
        "nonmanifold_edge_ratio": float(np.mean(counts > 2)),
    }


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    diag = compute_diagnostics(args.ply, opacity_min=args.opacity_min)
    selected = np.flatnonzero(diag.opacity >= args.opacity_min)
    if selected.size > args.max_points:
        # Preserve the largest geometric masses when bounding offline cost.
        order = np.argsort(-diag.mass[selected])[: args.max_points]
        selected = np.sort(selected[order])
    geometric_mass = diag.mass[selected] if diag.mass_is_explicit else estimate_knn_area_mass(diag.xyz[selected])
    projected = project_points_to_manifold(
        xyz=diag.xyz[selected],
        mass=geometric_mass,
        source_indices=selected,
        k=args.knn,
        iterations=args.iterations,
        max_normal_ratio=args.max_normal_ratio,
        min_tangent_ratio=args.min_tangent_ratio,
        min_confidence=args.min_confidence,
        projection_step=args.projection_step,
    )
    asset_mass = relax_certified_quadrature(
        projected, args.mass_relaxation, args.mass_radius_cap_quantile
    )
    projected = projected.__class__(**{**projected.__dict__, "mass": asset_mass})
    write_oriented_points_ply(
        out / "projected_points.ply", projected.xyz, projected.normals, projected.mass
    )
    accepted = projected.accepted
    write_oriented_points_ply(
        out / "accepted_points.ply",
        projected.xyz[accepted],
        projected.normals[accepted],
        projected.mass[accepted],
    )
    np.savez_compressed(
        out / "projected_manifold.npz",
        xyz=projected.xyz,
        normals=projected.normals,
        mass=projected.mass,
        confidence=projected.confidence,
        accepted=projected.accepted,
        radii=projected.radii,
        source_indices=projected.source_indices,
        neighbors=projected.neighbors,
    )
    write_projected_gaussians(
        args.ply,
        out / "projected_gaussians.ply",
        projected,
        normal_scale_ratio=args.normal_scale_ratio,
    )
    accepted_radii = projected.radii[accepted]
    if accepted_radii.size:
        radius_cap = float(np.quantile(
            accepted_radii, np.clip(args.mesh_radius_cap_quantile, 0.0, 1.0)
        ))
        accepted_radii = np.minimum(accepted_radii, radius_cap)
    mesh = build_patch_mesh_from_points(
        projected.xyz[accepted],
        projected.normals[accepted],
        accepted_radii,
        projected.source_indices[accepted],
        k=args.knn,
        min_patch_size=args.min_patch_size,
        chart_normal_dot_min=args.chart_normal_dot_min,
    )
    save_patch_mesh(mesh, out / "patch_mesh.ply", out / "patch_mesh_meta.npz")

    summary: dict[str, object] = {
        "input_points": int(selected.size),
        "projected_points": int(projected.xyz.shape[0]),
        "accepted_points": int(np.sum(accepted)),
        "acceptance_ratio": float(np.mean(accepted)) if accepted.size else 0.0,
        "input_mass": float(np.sum(geometric_mass)),
        "input_mass_source": "explicit geom_mass" if diag.mass_is_explicit else "kNN area quadrature",
        "projected_mass": float(np.sum(projected.mass)),
        "accepted_mass": float(np.sum(projected.mass[accepted])),
        "confidence_median": float(np.median(projected.confidence)) if projected.confidence.size else 0.0,
        "patches": int(mesh.patch_ids.max() + 1) if mesh.patch_ids.size else 0,
        "mesh": mesh_statistics(mesh.vertices, mesh.faces),
    }
    if args.gt is not None and projected.xyz.shape[0] > 0:
        gt = np.load(args.gt)
        gt_xyz = np.asarray(gt["xyz"], dtype=np.float32)
        gt_normals = np.asarray(gt["normals"], dtype=np.float32)
        gt_weights = np.asarray(gt["weights"], dtype=np.float32)
        bbox_diagonal = float(np.linalg.norm(np.ptp(gt_xyz, axis=0)))
        summary["gt_geometry_all"] = geometry_metrics(
            projected.xyz, projected.normals, gt_xyz, gt_normals, bbox_diagonal
        )
        summary["gt_geometry_accepted"] = geometry_metrics(
            projected.xyz[accepted], projected.normals[accepted], gt_xyz, gt_normals, bbox_diagonal
        )
        summary["gt_normalized_varifold"] = normalized_kernel_varifold_distance(
            projected.xyz,
            projected.normals,
            projected.mass,
            gt_xyz,
            gt_normals,
            gt_weights,
            sigma=0.05 * bbox_diagonal,
            max_points=2048,
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote manifold projection to {out}")


if __name__ == "__main__":
    main()
