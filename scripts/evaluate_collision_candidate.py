#!/usr/bin/env python3
"""Score an exported collision candidate against a GT surface (P0.4).

Reports supported coverage, floater/false surface area, Hausdorff and normal error
for ``collision_candidate.ply``. If a probe file is given, also reports the
contact-proxy false/missed collision confusion. All metrics are CPU-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manifold_gs.mesh_io import read_triangle_mesh_ply
from manifold_gs.collision_metrics import (
    collision_confusion,
    coverage_tolerance_sweep,
    sample_mesh_surface,
    surface_coverage_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, help="collision_candidate.ply")
    parser.add_argument("--gt", required=True, help="GT surface npz with xyz/normals")
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tolerance-fraction", type=float, default=0.01,
        help="support tolerance as a fraction of the GT bbox diagonal",
    )
    parser.add_argument(
        "--probes", default=None,
        help="optional npz with 'points' and optional 'labels' for collision confusion",
    )
    args = parser.parse_args()

    vertices, faces = read_triangle_mesh_ply(args.candidate)
    gt = np.load(args.gt)
    gt_xyz = np.asarray(gt["xyz"], dtype=np.float64)
    gt_normals = np.asarray(gt["normals"], dtype=np.float64)
    bbox_diagonal = float(np.linalg.norm(np.ptp(gt_xyz, axis=0)))
    tolerance = args.tolerance_fraction * max(bbox_diagonal, 1e-12)

    report: dict[str, object] = {
        "candidate": str(Path(args.candidate).resolve()),
        "gt": str(Path(args.gt).resolve()),
        "bbox_diagonal": bbox_diagonal,
        "tolerance": tolerance,
        "coverage": surface_coverage_metrics(
            vertices, faces, gt_xyz, gt_normals,
            tolerance=tolerance, samples=args.samples, seed=args.seed,
        ),
        "coverage_sweep": coverage_tolerance_sweep(
            vertices, faces, gt_xyz,
            tolerances=np.array([0.005, 0.01, 0.02, 0.03, 0.05, 0.08]) * max(bbox_diagonal, 1e-12),
            samples=args.samples, seed=args.seed,
        ),
    }
    if args.probes is not None:
        probe_data = np.load(args.probes, allow_pickle=True)
        probe_points = np.asarray(probe_data["points"], dtype=np.float64)
        labels = np.asarray(probe_data["labels"]) if "labels" in probe_data else None
        candidate_points, _, _ = sample_mesh_surface(vertices, faces, args.samples, seed=args.seed)
        report["collision"] = collision_confusion(
            candidate_points, gt_xyz, probe_points,
            contact_tolerance=tolerance, probe_labels=labels,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
