#!/usr/bin/env python3
"""Compare certified patch binding vs a proximity baseline on a patch edit (P0.3).

Loads an exported bundle's attached/residual Gaussians and ``asset_mapping.npz``,
applies a rigid edit to the selected patch(es), propagates it with (a) certified
patch binding and (b) a nearest-radius baseline, and reports edit error, boundary
leakage and residual contamination for each. CPU-only, deterministic.
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

from manifold_gs.ply_io import read_vertex_ply
from manifold_gs.edit_metrics import (
    certified_patch_binding,
    edit_propagation_metrics,
    propagate_edit,
    radius_binding,
    rigid_deformation,
)


def _centers(path: Path) -> np.ndarray:
    rows = read_vertex_ply(path).data
    return np.stack([rows["x"], rows["y"], rows["z"]], axis=1).astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, help="asset bundle directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--patches", type=int, nargs="*", default=None,
                        help="patch ids to edit (default: the largest attached patch)")
    parser.add_argument("--translate-frac", type=float, default=0.1,
                        help="edit translation along +z as a fraction of the attached bbox diagonal")
    parser.add_argument("--rotate-deg", type=float, default=0.0,
                        help="edit rotation about +z through the selected centroid")
    parser.add_argument("--radius-frac", type=float, default=0.05,
                        help="baseline binding radius as a fraction of the attached bbox diagonal")
    args = parser.parse_args()

    bundle = Path(args.bundle)
    mapping = np.load(bundle / "asset_mapping.npz")
    if "attached_patch_ids" not in mapping:
        raise ValueError("asset_mapping.npz predates attached_patch_ids; re-export the bundle")
    attached = _centers(bundle / "attached_gaussians.ply")
    attached_patch_ids = np.asarray(mapping["attached_patch_ids"], dtype=np.int32).reshape(-1)
    if attached_patch_ids.shape[0] != attached.shape[0]:
        raise ValueError("attached_patch_ids and attached_gaussians.ply are misaligned")
    residual_path = bundle / "residual_gaussians.ply"
    residual = _centers(residual_path) if residual_path.is_file() else np.empty((0, 3))

    points = np.vstack([attached, residual]) if residual.shape[0] else attached
    patch_ids = np.concatenate([attached_patch_ids, np.full(residual.shape[0], -1, dtype=np.int32)])
    residual_mask = patch_ids < 0

    if args.patches:
        selected = np.asarray(args.patches, dtype=np.int32)
    else:
        values, counts = np.unique(attached_patch_ids, return_counts=True)
        selected = np.asarray([int(values[counts.argmax()])], dtype=np.int32)
    edit_region = certified_patch_binding(patch_ids, selected)
    if not edit_region.any():
        raise ValueError(f"selected patches {selected.tolist()} match no attached Gaussians")

    diagonal = float(np.linalg.norm(np.ptp(attached, axis=0)))
    centroid = attached[np.isin(attached_patch_ids, selected)].mean(axis=0)
    angle = np.radians(args.rotate_deg)
    rotation = np.asarray([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    translation = np.asarray([0.0, 0.0, args.translate_frac * diagonal])
    deform = rigid_deformation(rotation, translation, pivot=centroid)

    target = propagate_edit(points, deform, edit_region)
    certified = propagate_edit(points, deform, certified_patch_binding(patch_ids, selected))
    radius = args.radius_frac * max(diagonal, 1e-12)
    baseline = propagate_edit(points, deform, radius_binding(points, points[edit_region], radius))

    report = {
        "bundle": str(bundle.resolve()),
        "selected_patches": selected.tolist(),
        "attached_bbox_diagonal": diagonal,
        "edit": {"translate_frac": args.translate_frac, "rotate_deg": args.rotate_deg},
        "baseline_radius": radius,
        "certified_binding": edit_propagation_metrics(
            points, certified, target, edit_region, residual_mask=residual_mask
        ),
        "radius_baseline": edit_propagation_metrics(
            points, baseline, target, edit_region, residual_mask=residual_mask
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
