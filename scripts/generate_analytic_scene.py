#!/usr/bin/env python3
"""Generate a COLMAP scene and exact GT from one analytic surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
GS_ROOT = ROOT / "third_party" / "gaussian-splatting"
for path in (ROOT, GS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from manifold_gs.analytic_scene import (
    camera_eyes,
    create_scene_mesh,
    look_at_w2c,
    rasterize_mesh,
    sample_analytic_surface,
    surface_color,
)
from manifold_gs.ply_io import write_oriented_points_ply
from utils.read_write_model import Camera, Image as ColmapImage, Point3D, rotmat2qvec, write_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", choices=("plane", "sphere", "torus"), default="sphere")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--train-views", type=int, default=6)
    parser.add_argument("--heldout-views", type=int, default=12)
    parser.add_argument("--gt-points", type=int, default=20000)
    parser.add_argument("--init-points", type=int, default=2000)
    parser.add_argument("--init-noise", type=float, default=0.01, help="Isotropic SfM point noise in world units")
    parser.add_argument("--mesh-resolution", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_views < 1 or args.heldout_views < 1:
        raise ValueError("train-views and heldout-views must both be positive")
    out = Path(args.out)
    image_dir = out / "images"
    heldout_dir = out / "heldout_images"
    sparse_dir = out / "sparse" / "0"
    gt_dir = out / "gt"
    for directory in (image_dir, heldout_dir, sparse_dir, gt_dir / "depth", gt_dir / "normal", gt_dir / "mask"):
        directory.mkdir(parents=True, exist_ok=True)

    mesh = create_scene_mesh(args.scene, args.mesh_resolution)
    mesh.export(gt_dir / "mesh.ply")
    gt_xyz, gt_normals, gt_weights = sample_analytic_surface(args.scene, args.gt_points, args.seed)
    np.savez_compressed(gt_dir / "surface.npz", xyz=gt_xyz, normals=gt_normals, weights=gt_weights)
    write_oriented_points_ply(gt_dir / "surface.ply", gt_xyz, gt_normals, gt_weights)

    init_xyz, init_normals, _ = sample_analytic_surface(args.scene, args.init_points, args.seed + 1)
    if args.init_noise > 0:
        rng = np.random.default_rng(args.seed + 2)
        init_xyz = init_xyz + rng.normal(0.0, args.init_noise, init_xyz.shape).astype(np.float32)
    init_rgb = np.clip(np.round(surface_color(init_xyz, init_normals) * 255.0), 0, 255).astype(np.uint8)

    total_views = args.train_views + args.heldout_views
    eyes = camera_eyes(args.scene, total_views)
    focal = 0.95 * max(args.width, args.height)
    cameras = {
        1: Camera(
            id=1,
            model="PINHOLE",
            width=args.width,
            height=args.height,
            params=np.array([focal, focal, args.width / 2.0, args.height / 2.0]),
        )
    }
    colmap_images = {}
    rotations, translations, splits, names = [], [], [], []
    for idx, eye in enumerate(eyes):
        rotation, translation = look_at_w2c(eye)
        rendered = rasterize_mesh(mesh, rotation, translation, args.width, args.height, focal)
        is_train = idx < args.train_views
        split = "train" if is_train else "heldout"
        name = f"{split}_{idx:03d}.png"
        Image.fromarray(rendered.rgb).save(image_dir / name)
        if not is_train:
            Image.fromarray(rendered.rgb).save(heldout_dir / name)
        np.save(gt_dir / "depth" / f"{split}_{idx:03d}.npy", rendered.depth)
        np.save(gt_dir / "normal" / f"{split}_{idx:03d}.npy", rendered.normal_world)
        Image.fromarray((rendered.mask.astype(np.uint8) * 255)).save(gt_dir / "mask" / name)
        rotations.append(rotation)
        translations.append(translation)
        splits.append(split)
        names.append(name)
        image_id = len(colmap_images) + 1
        colmap_images[image_id] = ColmapImage(
            id=image_id,
            qvec=rotmat2qvec(rotation),
            tvec=translation,
            camera_id=1,
            name=name,
            xys=np.empty((0, 2), dtype=np.float64),
            point3D_ids=np.empty((0,), dtype=np.int64),
        )

    points = {}
    for idx, (xyz, rgb) in enumerate(zip(init_xyz, init_rgb), start=1):
        points[idx] = Point3D(
            id=idx,
            xyz=xyz.astype(np.float64),
            rgb=rgb,
            error=0.0,
            image_ids=np.empty((0,), dtype=np.int32),
            point2D_idxs=np.empty((0,), dtype=np.int32),
        )
    write_model(cameras, colmap_images, points, sparse_dir, ext=".bin")
    (sparse_dir / "test.txt").write_text(
        "".join(f"{name}\n" for name, split in zip(names, splits) if split == "heldout"),
        encoding="utf-8",
    )
    np.savez_compressed(
        gt_dir / "cameras.npz",
        rotations=np.asarray(rotations),
        translations=np.asarray(translations),
        eyes=eyes,
        splits=np.asarray(splits),
        names=np.asarray(names),
        intrinsic=np.array([focal, focal, args.width / 2.0, args.height / 2.0]),
        width=args.width,
        height=args.height,
    )
    metadata = {
        "scene": args.scene,
        "seed": args.seed,
        "train_views": args.train_views,
        "heldout_views": args.heldout_views,
        "gt_points": args.gt_points,
        "init_points": args.init_points,
        "init_noise": args.init_noise,
        "surface_area": float(np.sum(gt_weights)),
        "gt_consistency": "RGB/depth/normal/mask/mesh/samples share one analytic surface",
    }
    (gt_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Wrote analytic scene to {out}")


if __name__ == "__main__":
    main()
