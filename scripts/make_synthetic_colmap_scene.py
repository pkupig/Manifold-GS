#!/usr/bin/env python3
"""Create a tiny COLMAP-style scene for 3DGS smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
GS_ROOT = ROOT / "third_party" / "gaussian-splatting"
if str(GS_ROOT) not in sys.path:
    sys.path.insert(0, str(GS_ROOT))

from utils.read_write_model import Camera, Image as ColmapImage, Point3D, rotmat2qvec, write_model


def look_at_w2c(eye: np.ndarray, target: np.ndarray = np.zeros(3)) -> tuple[np.ndarray, np.ndarray]:
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    rot = np.stack([right, down, forward], axis=0)
    tvec = -rot @ eye
    return rot, tvec


def make_image(width: int, height: int, idx: int) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    cx = width * (0.45 + 0.05 * np.cos(idx))
    cy = height * (0.50 + 0.04 * np.sin(idx))
    r = np.sqrt(((x - cx) / width) ** 2 + ((y - cy) / height) ** 2)
    img = np.zeros((height, width, 3), dtype=np.float32)
    img[..., 0] = 0.15 + 0.75 * np.exp(-60 * r * r)
    img[..., 1] = 0.20 + 0.50 * (x / max(width - 1, 1))
    img[..., 2] = 0.25 + 0.45 * (y / max(height - 1, 1))
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "experiments" / "synthetic_colmap"))
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--num-cameras", type=int, default=4)
    parser.add_argument("--num-points-side", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    images_dir = out / "images"
    sparse_dir = out / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    fx = fy = 130.0
    cx = args.width / 2.0
    cy = args.height / 2.0
    cameras = {
        1: Camera(
            id=1,
            model="PINHOLE",
            width=args.width,
            height=args.height,
            params=np.array([fx, fy, cx, cy], dtype=np.float64),
        )
    }

    images = {}
    radius = 1.8
    for idx in range(args.num_cameras):
        theta = 2.0 * np.pi * idx / args.num_cameras
        eye = np.array([radius * np.cos(theta), 0.35, radius * np.sin(theta)], dtype=np.float64)
        rot, tvec = look_at_w2c(eye)
        qvec = rotmat2qvec(rot)
        name = f"image_{idx:03d}.png"
        Image.fromarray(make_image(args.width, args.height, idx)).save(images_dir / name)
        images[idx + 1] = ColmapImage(
            id=idx + 1,
            qvec=qvec,
            tvec=tvec,
            camera_id=1,
            name=name,
            xys=np.empty((0, 2), dtype=np.float64),
            point3D_ids=np.empty((0,), dtype=np.int64),
        )

    points = {}
    side = args.num_points_side
    pid = 1
    for i in range(side):
        for j in range(side):
            x = (i / (side - 1) - 0.5) * 0.9
            z = (j / (side - 1) - 0.5) * 0.9
            y = 0.03 * np.sin(5 * x) * np.cos(5 * z)
            rgb = np.array([
                int(60 + 160 * i / (side - 1)),
                int(80 + 120 * j / (side - 1)),
                180,
            ], dtype=np.uint8)
            points[pid] = Point3D(
                id=pid,
                xyz=np.array([x, y, z], dtype=np.float64),
                rgb=rgb,
                error=0.0,
                image_ids=np.empty((0,), dtype=np.int32),
                point2D_idxs=np.empty((0,), dtype=np.int32),
            )
            pid += 1

    write_model(cameras, images, points, sparse_dir, ext=".bin")
    print(f"Wrote synthetic COLMAP scene to {out}")


if __name__ == "__main__":
    main()

