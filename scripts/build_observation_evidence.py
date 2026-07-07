#!/usr/bin/env python3
"""Build the shared COLMAP sparse-support evidence cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manifold_gs.observation_evidence import (
    build_sparse_support_evidence,
    compute_photometric_evidence,
    read_colmap_training_cameras,
    save_sparse_support_evidence,
)
from manifold_gs.ply_io import read_vertex_ply


def _load_camera_images(images_dir: Path, cameras: list) -> list[np.ndarray]:
    from PIL import Image

    images: list[np.ndarray] = []
    for camera in cameras:
        with Image.open(images_dir / camera.name) as handle:
            image = handle.convert("RGB")
            if image.size != (camera.width, camera.height):
                image = image.resize((camera.width, camera.height), Image.BILINEAR)
            images.append(np.asarray(image, dtype=np.float64) / 255.0)
    return images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaussians", required=True)
    parser.add_argument("--colmap-points", required=True, help="COLMAP sparse points3D.ply")
    parser.add_argument("--source-map", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", default=None, help="COLMAP scene root; adds training-view evidence")
    parser.add_argument("--images", default=None,
                        help="training images dir (requires --scene); adds photometric consistency")
    parser.add_argument("--support-k", type=int, default=3)
    parser.add_argument("--radius-multiplier", type=float, default=2.0)
    args = parser.parse_args()
    if args.images and not args.scene:
        parser.error("--images requires --scene for the matching training cameras")
    source_indices = None
    if args.source_map:
        source_indices = np.asarray(np.load(args.source_map)["source_indices"], dtype=np.int64)
    evidence = build_sparse_support_evidence(
        args.gaussians, args.colmap_points, source_indices=source_indices,
        support_k=args.support_k, radius_multiplier=args.radius_multiplier,
        scene=args.scene,
    )
    if args.images:
        cameras = read_colmap_training_cameras(args.scene)
        gaussian_rows = read_vertex_ply(args.gaussians).data
        gaussian_xyz = np.stack(
            [gaussian_rows["x"], gaussian_rows["y"], gaussian_rows["z"]], axis=1
        ).astype(np.float64)
        images = _load_camera_images(Path(args.images), cameras)
        evidence.update(compute_photometric_evidence(gaussian_xyz, cameras, images))
    save_sparse_support_evidence(args.out, evidence)
    print(json.dumps({
        "output": args.out,
        "gaussians": int(evidence["source_indices"].size),
        "sparse_supported": int(evidence["sparse_supported"].sum()),
        "sparse_supported_fraction": float(evidence["sparse_supported"].mean()),
        "training_camera_count": (
            int(evidence["training_camera_count"]) if "training_camera_count" in evidence else None
        ),
        "photometric_multiview_fraction": (
            float(np.mean(evidence["photometric_view_count"] >= 2))
            if "photometric_view_count" in evidence else None
        ),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
