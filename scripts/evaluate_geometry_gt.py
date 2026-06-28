#!/usr/bin/env python3
"""Evaluate a 3DGS checkpoint against analytic surface GT."""

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
from manifold_gs.manifold_projection import project_points_to_manifold, relax_certified_quadrature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--gt", required=True, help="Analytic gt/surface.npz")
    parser.add_argument("--out", required=True, help="Output JSON")
    parser.add_argument("--opacity-min", type=float, default=0.02)
    parser.add_argument("--surface-r12-min", type=float, default=0.25)
    parser.add_argument("--surface-r23-max", type=float, default=0.08)
    parser.add_argument("--kernel-sigma-fraction", type=float, default=0.05)
    parser.add_argument("--max-varifold-points", type=int, default=4096)
    parser.add_argument("--certified-knn", type=int, default=20)
    parser.add_argument("--certified-mass-relaxation", type=float, default=0.75)
    parser.add_argument("--certified-radius-cap-quantile", type=float, default=0.5)
    return parser.parse_args()


def evaluate_subset(
    name: str,
    mask: np.ndarray,
    diag,
    gt_xyz: np.ndarray,
    gt_normals: np.ndarray,
    gt_weights: np.ndarray,
    bbox_diagonal: float,
    sigma: float,
    max_points: int,
    measure_mass: np.ndarray | None = None,
) -> dict[str, object]:
    metrics: dict[str, object] = geometry_metrics(
        diag.xyz[mask], diag.normals[mask], gt_xyz, gt_normals, bbox_diagonal
    )
    if np.any(mask):
        subset_mass = diag.mass[mask] if measure_mass is None else measure_mass[mask]
        metrics["normalized_kernel_varifold"] = normalized_kernel_varifold_distance(
            diag.xyz[mask],
            diag.normals[mask],
            np.maximum(subset_mass, 1e-12),
            gt_xyz,
            gt_normals,
            gt_weights,
            sigma=sigma,
            max_points=max_points,
        )
        metrics["surrogate_mass_sum"] = float(np.sum(subset_mass))
    else:
        metrics["normalized_kernel_varifold"] = None
        metrics["surrogate_mass_sum"] = 0.0
    metrics["selection"] = name
    return metrics


def main() -> None:
    args = parse_args()
    gt = np.load(args.gt)
    gt_xyz = np.asarray(gt["xyz"], dtype=np.float32)
    gt_normals = np.asarray(gt["normals"], dtype=np.float32)
    gt_weights = np.asarray(gt["weights"], dtype=np.float32)
    bbox_diagonal = float(np.linalg.norm(np.ptp(gt_xyz, axis=0)))
    sigma = args.kernel_sigma_fraction * bbox_diagonal
    diag = compute_diagnostics(
        args.ply,
        surface_r12_min=args.surface_r12_min,
        surface_r23_max=args.surface_r23_max,
        opacity_min=args.opacity_min,
    )
    opaque = diag.opacity >= args.opacity_min
    certified = np.zeros_like(opaque, dtype=bool)
    opaque_indices = np.flatnonzero(opaque)
    certified_mass_fraction = 0.0
    certified_measure_mass = diag.mass.copy()
    if opaque_indices.size >= 7:
        projected = project_points_to_manifold(
            diag.xyz[opaque_indices], mass=diag.mass[opaque_indices],
            source_indices=opaque_indices, k=args.certified_knn, iterations=2,
            min_confidence=0.25, projection_step=0.5,
        )
        certified[opaque_indices[projected.accepted]] = True
        certified_measure_mass[opaque_indices] = relax_certified_quadrature(
            projected,
            relaxation=args.certified_mass_relaxation,
            radius_cap_quantile=args.certified_radius_cap_quantile,
        )
        opaque_mass = float(np.sum(diag.mass[opaque_indices]))
        certified_mass_fraction = float(np.sum(diag.mass[certified]) / max(opaque_mass, 1e-16))
    gt_mass = float(np.sum(gt_weights))
    estimated_mass = float(np.sum(diag.mass))
    gt_first_moment = np.sum(gt_weights[:, None] * gt_xyz, axis=0)
    estimated_first_moment = np.sum(diag.mass[:, None] * diag.xyz, axis=0)
    explicit_measure = None
    if diag.mass_is_explicit:
        explicit_measure = {
            "total_mass": estimated_mass,
            "gt_total_mass": gt_mass,
            "relative_mass_error": abs(estimated_mass - gt_mass) / max(gt_mass, 1e-12),
            "first_moment_l2_error": float(np.linalg.norm(estimated_first_moment - gt_first_moment)),
            "barycenter_l2_error": float(np.linalg.norm(
                estimated_first_moment / max(estimated_mass, 1e-12)
                - gt_first_moment / max(gt_mass, 1e-12)
            )),
        }
    report = {
        "gt": str(Path(args.gt).resolve()),
        "ply": str(Path(args.ply).resolve()),
        "bbox_diagonal": bbox_diagonal,
        "kernel_sigma": sigma,
        "gt_area": gt_mass,
        "explicit_measure": explicit_measure,
        "mass_note": "mass uses explicit geom_mass when present; legacy PLY files fall back to opacity*tangent_area",
        "all_opaque": evaluate_subset(
            "opacity >= threshold", opaque, diag, gt_xyz, gt_normals, gt_weights,
            bbox_diagonal, sigma, args.max_varifold_points,
        ),
        "surface_only": evaluate_subset(
            "opacity threshold + covariance surface classification", diag.keep_surface, diag,
            gt_xyz, gt_normals, gt_weights, bbox_diagonal, sigma, args.max_varifold_points,
        ),
        "certified_charts": evaluate_subset(
            "opacity threshold + center-support chart certification", certified, diag,
            gt_xyz, gt_normals, gt_weights, bbox_diagonal, sigma, args.max_varifold_points,
            measure_mass=certified_measure_mass,
        ),
        "certified_chart_mass_fraction_of_opaque": certified_mass_fraction,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
