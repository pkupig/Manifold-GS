#!/usr/bin/env python3
"""Analyze 3DGS/SuGaR Gaussian PLY geometry.

Example:
    python scripts/analyze_gaussians.py \
        --ply outputs/scene/point_cloud/iteration_30000/point_cloud.ply \
        --out experiments/scene_diagnostics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manifold_gs.diagnostics import compute_diagnostics, export_surface_points, save_npz, summarize
from manifold_gs.graph_diagnostics import compute_graph_diagnostics, save_graph_npz, summarize_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True, help="Path to 3DGS point_cloud.ply")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--surface-r12-min", type=float, default=0.25)
    parser.add_argument("--surface-r23-max", type=float, default=0.08)
    parser.add_argument("--curve-r12-max", type=float, default=0.15)
    parser.add_argument("--curve-r23-max", type=float, default=0.5)
    parser.add_argument("--volume-r23-min", type=float, default=0.2)
    parser.add_argument("--opacity-min", type=float, default=0.02)
    parser.add_argument("--knn", type=int, default=12, help="Neighbors for local manifold graph diagnostics")
    parser.add_argument("--max-graph-points", type=int, default=200_000)
    parser.add_argument("--skip-graph", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    diag = compute_diagnostics(
        path=args.ply,
        surface_r12_min=args.surface_r12_min,
        surface_r23_max=args.surface_r23_max,
        curve_r12_max=args.curve_r12_max,
        curve_r23_max=args.curve_r23_max,
        volume_r23_min=args.volume_r23_min,
        opacity_min=args.opacity_min,
    )
    summary = summarize(diag)
    if not args.skip_graph:
        graph = compute_graph_diagnostics(diag, k=args.knn, max_points=args.max_graph_points)
        save_graph_npz(graph, out / "graph_diagnostics.npz")
        summary.update(summarize_graph(graph))

    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    save_npz(diag, out / "diagnostics.npz")
    export_surface_points(diag, out / "surface_oriented_points.ply")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {out / 'diagnostics.npz'}")
    if not args.skip_graph:
        print(f"Wrote {out / 'graph_diagnostics.npz'}")
    print(f"Wrote {out / 'surface_oriented_points.ply'}")


if __name__ == "__main__":
    main()
