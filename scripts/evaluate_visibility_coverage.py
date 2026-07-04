#!/usr/bin/env python3
"""Measure analytic GT surface mass visible as a first camera-ray hit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Generated analytic scene directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=("train", "heldout", "all"), default="train")
    parser.add_argument(
        "--tolerance-fractions", type=float, nargs="+", default=(0.0025, 0.005, 0.01, 0.02),
        help="First-hit depth tolerances as fractions of the GT bounding-box diagonal",
    )
    return parser.parse_args()


def visible_first_hits(
    xyz: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    intrinsic: np.ndarray,
    depth: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    camera = xyz @ rotation.T + translation
    z = camera[:, 2]
    fx, fy, cx, cy = intrinsic
    px = fx * camera[:, 0] / np.maximum(z, 1e-12) + cx
    py = fy * camera[:, 1] / np.maximum(z, 1e-12) + cy
    ix = np.floor(px).astype(np.int64)
    iy = np.floor(py).astype(np.int64)
    height, width = depth.shape
    in_front = z > 1e-8
    min_error = np.full(len(xyz), np.inf, dtype=np.float64)
    # A 3x3 footprint makes the estimate robust to pixel-center convention and
    # analytic-sample versus rasterized-mesh discretization.
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            xx = ix + ox
            yy = iy + oy
            valid = in_front & (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
            indices = np.flatnonzero(valid)
            if indices.size == 0:
                continue
            observed = depth[yy[indices], xx[indices]]
            has_hit = observed > 0
            hit_indices = indices[has_hit]
            min_error[hit_indices] = np.minimum(
                min_error[hit_indices], np.abs(z[hit_indices] - observed[has_hit])
            )
    return min_error <= tolerance


def main() -> None:
    args = parse_args()
    data = Path(args.data)
    gt_dir = data / "gt"
    surface = np.load(gt_dir / "surface.npz")
    xyz = np.asarray(surface["xyz"], dtype=np.float64)
    mass = np.maximum(np.asarray(surface["weights"], dtype=np.float64), 0.0)
    cameras = np.load(gt_dir / "cameras.npz")
    bbox_diagonal = float(np.linalg.norm(np.ptp(xyz, axis=0)))
    splits = np.asarray(cameras["splits"]).astype(str)
    selected = np.arange(len(splits)) if args.split == "all" else np.flatnonzero(splits == args.split)
    if selected.size == 0:
        raise ValueError(f"no cameras found for split {args.split!r}")

    reports = {}
    total_mass = max(float(np.sum(mass)), 1e-16)
    for fraction in args.tolerance_fractions:
        tolerance = float(fraction) * bbox_diagonal
        union = np.zeros(len(xyz), dtype=bool)
        views = []
        for index in selected:
            name = str(cameras["names"][index])
            depth = np.load(gt_dir / "depth" / f"{Path(name).stem}.npy")
            visible = visible_first_hits(
                xyz,
                cameras["rotations"][index],
                cameras["translations"][index],
                cameras["intrinsic"],
                depth,
                tolerance,
            )
            union |= visible
            views.append({
                "name": name,
                "point_fraction": float(np.mean(visible)),
                "mass_fraction": float(np.sum(mass[visible]) / total_mass),
            })
        reports[f"{fraction:.6g}"] = {
            "absolute_tolerance": tolerance,
            "union_point_fraction": float(np.mean(union)),
            "union_mass_fraction": float(np.sum(mass[union]) / total_mass),
            "views": views,
        }

    report = {
        "data": str(data.resolve()),
        "split": args.split,
        "camera_count": int(selected.size),
        "gt_points": int(len(xyz)),
        "bbox_diagonal": bbox_diagonal,
        "tolerances": reports,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
