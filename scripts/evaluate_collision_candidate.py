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


def evaluate_collision(
    candidate: Path,
    gt: Path,
    *,
    samples: int = 50000,
    seed: int = 0,
    tolerance_fraction: float = 0.01,
    probes: Path | None = None,
) -> dict:
    """Score a collision candidate against a GT surface npz (P0.4).

    Returns the report dict; does not touch disk. Shared by the standalone script
    and ``scripts/run_asset_benchmark.py``.
    """
    vertices, faces = read_triangle_mesh_ply(str(candidate))
    gt_data = np.load(str(gt))
    gt_xyz = np.asarray(gt_data["xyz"], dtype=np.float64)
    gt_normals = np.asarray(gt_data["normals"], dtype=np.float64)
    bbox_diagonal = float(np.linalg.norm(np.ptp(gt_xyz, axis=0)))
    tolerance = tolerance_fraction * max(bbox_diagonal, 1e-12)

    report: dict[str, object] = {
        "candidate": str(Path(candidate).resolve()),
        "gt": str(Path(gt).resolve()),
        "bbox_diagonal": bbox_diagonal,
        "tolerance": tolerance,
        "tolerance_fraction": tolerance_fraction,
        "coverage": surface_coverage_metrics(
            vertices, faces, gt_xyz, gt_normals,
            tolerance=tolerance, samples=samples, seed=seed,
        ),
        "coverage_sweep": coverage_tolerance_sweep(
            vertices, faces, gt_xyz,
            tolerances=np.array([0.005, 0.01, 0.02, 0.03, 0.05, 0.08]) * max(bbox_diagonal, 1e-12),
            samples=samples, seed=seed,
        ),
    }
    if probes is not None:
        probe_data = np.load(str(probes), allow_pickle=True)
        probe_points = np.asarray(probe_data["points"], dtype=np.float64)
        labels = np.asarray(probe_data["labels"]) if "labels" in probe_data else None
        candidate_points, _, _ = sample_mesh_surface(vertices, faces, samples, seed=seed)
        report["collision"] = collision_confusion(
            candidate_points, gt_xyz, probe_points,
            contact_tolerance=tolerance, probe_labels=labels,
        )
    return report


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

    report = evaluate_collision(
        Path(args.candidate),
        Path(args.gt),
        samples=args.samples,
        seed=args.seed,
        tolerance_fraction=args.tolerance_fraction,
        probes=Path(args.probes) if args.probes is not None else None,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
