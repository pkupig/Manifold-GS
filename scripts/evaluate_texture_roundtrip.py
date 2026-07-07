#!/usr/bin/env python3
"""Per-patch texture baking round-trip and seam error on an asset bundle (P0.5).

Observed colour defaults to each attached Gaussian's SH DC term, so this runs on any
exported bundle without extra inputs. Pass ``--evidence`` to use the multi-view
``photometric_mean_color`` from an observation-evidence cache instead. CPU-only.
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
from manifold_gs.texture_metrics import baking_roundtrip_metrics, seam_error_metrics

_SH_C0 = 0.28209479177387814


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, help="asset bundle directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--min-patch-samples", type=int, default=8)
    parser.add_argument("--boundary-radius-frac", type=float, default=0.02,
                        help="seam boundary radius as a fraction of the attached bbox diagonal")
    parser.add_argument("--evidence", default=None,
                        help="observation evidence npz; uses photometric_mean_color instead of SH DC")
    args = parser.parse_args()

    bundle = Path(args.bundle)
    mapping = np.load(bundle / "asset_mapping.npz")
    if "attached_patch_ids" not in mapping:
        raise ValueError("asset_mapping.npz predates attached_patch_ids; re-export the bundle")
    rows = read_vertex_ply(bundle / "attached_gaussians.ply").data
    points = np.stack([rows["x"], rows["y"], rows["z"]], axis=1).astype(np.float64)
    patch_ids = np.asarray(mapping["attached_patch_ids"], dtype=np.int32).reshape(-1)

    if args.evidence is not None:
        cache = np.load(args.evidence)
        if "photometric_mean_color" not in cache:
            raise ValueError("evidence cache has no photometric_mean_color; run build with --images")
        row_by_source = {int(s): i for i, s in enumerate(np.asarray(cache["source_indices"]).reshape(-1))}
        attached_sources = np.asarray(mapping["attached_source_indices"], dtype=np.int64).reshape(-1)
        colors = np.asarray(cache["photometric_mean_color"], dtype=np.float64)[
            [row_by_source[int(s)] for s in attached_sources]
        ]
        color_source = "photometric_mean_color"
    else:
        colors = 0.5 + _SH_C0 * np.stack(
            [rows["f_dc_0"], rows["f_dc_1"], rows["f_dc_2"]], axis=1
        ).astype(np.float64)
        colors = np.clip(colors, 0.0, 1.0)
        color_source = "sh_dc"

    diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
    per_patch = []
    for patch in np.unique(patch_ids):
        rows_p = np.flatnonzero(patch_ids == patch)
        if rows_p.size < args.min_patch_samples:
            continue
        metrics = baking_roundtrip_metrics(points[rows_p], colors[rows_p], args.resolution)
        metrics["patch_id"] = int(patch)
        per_patch.append(metrics)

    finite = [p["reprojection_psnr"] for p in per_patch if np.isfinite(p["reprojection_psnr"])]
    seam = seam_error_metrics(
        points, patch_ids, colors, args.resolution,
        boundary_radius=args.boundary_radius_frac * max(diagonal, 1e-12),
    )
    report = {
        "bundle": str(bundle.resolve()),
        "color_source": color_source,
        "resolution": args.resolution,
        "attached_bbox_diagonal": diagonal,
        "evaluated_patches": len(per_patch),
        "reprojection_psnr_mean": float(np.mean(finite)) if finite else float("inf"),
        "reprojection_error_mean": float(np.mean([p["reprojection_error_mean"] for p in per_patch])) if per_patch else 0.0,
        "seam": seam,
        "per_patch": per_patch,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_patch"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
