#!/usr/bin/env python3
"""Evaluate fixed-budget quadric-simplification robustness of candidate asset meshes.

This is intentionally a *simplification* benchmark, not an isotropic-remeshing
claim.  Each input receives the same Open3D cleanup and target-face budget, then is
scored for triangle quality, topology and collision-vs-GT precision/coverage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manifold_gs.collision_metrics import surface_coverage_metrics
from manifold_gs.mesh_io import read_triangle_mesh_ply
from manifold_gs.mesh_quality import cleanup_and_simplify, mesh_topology, triangle_quality


def score(vertices: np.ndarray, faces: np.ndarray, gt: dict[str, np.ndarray], tolerance: float,
          samples: int, seed: int) -> dict:
    return {
        "topology": mesh_topology(vertices, faces),
        "triangle_quality": triangle_quality(vertices, faces),
        "geometry": surface_coverage_metrics(vertices, faces, gt["xyz"], gt["normals"],
                                              tolerance=tolerance, samples=samples, seed=seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="GT npz with xyz/normals")
    parser.add_argument("--method", nargs=2, action="append", metavar=("NAME", "MESH"), required=True,
                        help="repeatable method label and triangle PLY path")
    parser.add_argument("--target-faces", type=int, nargs="+", default=[5000, 10000])
    parser.add_argument("--transform-scale-mat-inv", nargs=2, action="append", default=[],
                        metavar=("METHOD", "CAMERAS_NPZ"),
                        help="transform a method from DTU-mm to Gaussian frame using scale_mat_inv_0")
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance-fraction", type=float, default=0.01)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if any(x < 4 for x in args.target_faces):
        raise ValueError("target face budgets must be >= 4")
    gt_raw = np.load(args.gt)
    gt = {"xyz": np.asarray(gt_raw["xyz"], dtype=np.float64),
          "normals": np.asarray(gt_raw["normals"], dtype=np.float64)}
    bbox = float(np.linalg.norm(np.ptp(gt["xyz"], axis=0)))
    tol = args.tolerance_fraction * max(bbox, 1e-12)
    transforms = {name: Path(path) for name, path in args.transform_scale_mat_inv}
    unknown = set(transforms) - {name for name, _ in args.method}
    if unknown:
        raise ValueError(f"transforms name unknown methods: {sorted(unknown)}")
    rows = []
    for name, mesh_path in args.method:
        v, f = read_triangle_mesh_ply(mesh_path)
        transform_note = None
        if name in transforms:
            matrix = np.asarray(np.load(transforms[name])["scale_mat_inv_0"], dtype=np.float64)
            v = (np.c_[v, np.ones(v.shape[0])] @ matrix.T)[:, :3]
            transform_note = f"DTU-mm -> Gaussian frame via {transforms[name].resolve()} scale_mat_inv_0"
        item = {"method": name, "mesh": str(Path(mesh_path).resolve()),
                "coordinate_transform": transform_note,
                "original": score(v, f, gt, tol, args.samples, args.seed), "budgets": {}}
        for budget in sorted(set(args.target_faces)):
            started = time.perf_counter()
            sv, sf = cleanup_and_simplify(v, f, budget)
            elapsed = time.perf_counter() - started
            item["budgets"][str(budget)] = {"simplification_seconds": elapsed,
                                             **score(sv, sf, gt, tol, args.samples, args.seed)}
        rows.append(item)
    report = {"protocol": "asset-simplification/v1", "operation": "Open3D cleanup + quadric decimation",
              "gt": str(Path(args.gt).resolve()), "bbox_diagonal": bbox,
              "tolerance_fraction": args.tolerance_fraction, "target_faces": sorted(set(args.target_faces)),
              "methods": rows}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
