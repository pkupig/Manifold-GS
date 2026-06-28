#!/usr/bin/env python3
"""Analyze whether Gaussian tangent/curvature fields fit their center support."""

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
from manifold_gs.fundamental_compatibility import compute_fundamental_compatibility, summarize_compatibility
from manifold_gs.manifold_projection import project_points_to_manifold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--opacity-min", type=float, default=0.02)
    parser.add_argument("--knn", type=int, default=24)
    parser.add_argument("--max-points", type=int, default=20000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    diag = compute_diagnostics(args.ply, opacity_min=args.opacity_min)
    selected = np.flatnonzero(diag.opacity >= args.opacity_min)
    if selected.size > args.max_points:
        order = np.argsort(-diag.mass[selected])[: args.max_points]
        selected = np.sort(selected[order])
    result = compute_fundamental_compatibility(
        diag.xyz[selected],
        diag.normals[selected],
        mass=diag.mass[selected],
        source_indices=selected,
        k=args.knn,
    )
    summary = summarize_compatibility(result)
    projected = project_points_to_manifold(
        diag.xyz[selected], mass=diag.mass[selected], source_indices=selected,
        k=args.knn, iterations=2, min_confidence=0.25, projection_step=0.5,
    )
    selected_mass = np.maximum(diag.mass[selected], 0.0)
    summary["chart_acceptance_fraction"] = float(np.mean(projected.accepted))
    summary["chart_accepted_mass_fraction"] = float(
        np.sum(selected_mass[projected.accepted]) / max(np.sum(selected_mass), 1e-16)
    )
    summary["chart_mass_weighted_confidence"] = float(
        np.sum(selected_mass * projected.confidence) / max(np.sum(selected_mass), 1e-16)
    )
    np.savez_compressed(out / "fundamental_compatibility.npz", **result.__dict__)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote compatibility diagnostics to {out}")


if __name__ == "__main__":
    main()
