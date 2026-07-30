#!/usr/bin/env python3
"""Evaluate patch editing against an externally defined source-ID region.

Unlike ``evaluate_edit_propagation.py``, the target region here is *not* selected
by patch ID.  A label NPZ supplies ``source_indices`` from an external annotation
(or a GT transfer).  Certified editing must approximate that target by selecting
patches with sufficient overlap, so the report exposes both region IoU and edit
leakage instead of proving a patch against a region it defined itself.
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

from evaluate_edit_propagation import _centers
from manifold_gs.edit_metrics import edit_propagation_metrics, propagate_edit, radius_binding, rigid_deformation


def evaluate_external_region(bundle: Path, labels: Path, *, patch_overlap_min: float = 0.5,
                             translate_frac: float = 0.1, radius_frac: float = 0.05) -> dict:
    if not 0.0 < patch_overlap_min <= 1.0:
        raise ValueError("patch_overlap_min must be in (0, 1]")
    bundle = Path(bundle)
    mapping = np.load(bundle / "asset_mapping.npz")
    attached = _centers(bundle / "attached_gaussians.ply")
    attached_source = np.asarray(mapping["attached_source_indices"], dtype=np.int64).reshape(-1)
    attached_patch = np.asarray(mapping["attached_patch_ids"], dtype=np.int32).reshape(-1)
    external_sources = np.asarray(np.load(labels)["source_indices"], dtype=np.int64).reshape(-1)
    external = np.isin(attached_source, external_sources)
    if not external.any():
        raise ValueError("external labels select no attached Gaussians in this bundle")

    patch_ids, inverse = np.unique(attached_patch, return_inverse=True)
    total = np.bincount(inverse, minlength=patch_ids.size)
    inside = np.bincount(inverse, weights=external.astype(float), minlength=patch_ids.size)
    overlap = inside / np.maximum(total, 1)
    selected = patch_ids[overlap >= patch_overlap_min]
    certified_attached = np.isin(attached_patch, selected)
    if not certified_attached.any():
        raise ValueError("no patch meets patch-overlap threshold; lower --patch-overlap-min")

    residual_path = bundle / "residual_gaussians.ply"
    residual = _centers(residual_path) if residual_path.is_file() else np.empty((0, 3))
    points = np.vstack([attached, residual]) if residual.size else attached
    target_region = np.concatenate([external, np.zeros(residual.shape[0], dtype=bool)])
    certified = np.concatenate([certified_attached, np.zeros(residual.shape[0], dtype=bool)])
    residual_mask = np.zeros(points.shape[0], dtype=bool); residual_mask[attached.shape[0]:] = True
    diagonal = float(np.linalg.norm(np.ptp(attached, axis=0)))
    deform = rigid_deformation(np.eye(3), np.array([0.0, 0.0, translate_frac * diagonal]))
    target = propagate_edit(points, deform, target_region)
    moved_certified = propagate_edit(points, deform, certified)
    radius = radius_frac * max(diagonal, 1e-12)
    seed = points[target_region]
    moved_radius = propagate_edit(points, deform, radius_binding(points, seed, radius))
    intersection = int(np.sum(certified & target_region))
    union = int(np.sum(certified | target_region))
    return {
        "protocol": "external-region-edit/v1",
        "bundle": str(bundle.resolve()), "labels": str(Path(labels).resolve()),
        "external_region": {"attached_points": int(external.sum()), "source_ids": int(external_sources.size)},
        "patch_approximation": {
            "overlap_threshold": patch_overlap_min, "selected_patch_ids": selected.astype(int).tolist(),
            "selected_patches": int(selected.size), "iou": intersection / max(union, 1),
            "recall": intersection / max(int(target_region.sum()), 1),
            "precision": intersection / max(int(certified.sum()), 1),
        },
        "edit": {"translate_frac": translate_frac, "radius": radius},
        "certified_patch_binding": edit_propagation_metrics(points, moved_certified, target, target_region,
                                                              residual_mask=residual_mask),
        "radius_baseline": edit_propagation_metrics(points, moved_radius, target, target_region,
                                                      residual_mask=residual_mask),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True); p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True); p.add_argument("--patch-overlap-min", type=float, default=0.5)
    p.add_argument("--translate-frac", type=float, default=0.1); p.add_argument("--radius-frac", type=float, default=0.05)
    a = p.parse_args()
    report = evaluate_external_region(Path(a.bundle), Path(a.labels), patch_overlap_min=a.patch_overlap_min,
                                      translate_frac=a.translate_frac, radius_frac=a.radius_frac)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
