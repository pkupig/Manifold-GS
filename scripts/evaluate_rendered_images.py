#!/usr/bin/env python3
"""Evaluate paired rendered/GT PNG directories without external model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
GS_ROOT = ROOT / "third_party" / "gaussian-splatting"
if str(GS_ROOT) not in sys.path:
    sys.path.insert(0, str(GS_ROOT))

from utils.loss_utils import ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--renders", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_rgb(path: Path) -> torch.Tensor:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)


def main() -> None:
    args = parse_args()
    renders_dir = Path(args.renders)
    gt_dir = Path(args.gt)
    names = sorted(path.name for path in renders_dir.glob("*.png"))
    if not names:
        raise ValueError(f"No PNG renders found in {renders_dir}")
    missing = [name for name in names if not (gt_dir / name).is_file()]
    if missing:
        raise ValueError(f"Missing {len(missing)} GT images, first: {missing[0]}")

    per_view = {}
    for name in names:
        render = load_rgb(renders_dir / name)
        gt = load_rgb(gt_dir / name)
        if render.shape != gt.shape:
            raise ValueError(f"Shape mismatch for {name}: {tuple(render.shape)} vs {tuple(gt.shape)}")
        mse = torch.mean((render - gt).square()).clamp_min(1e-12)
        per_view[name] = {
            "psnr": float(-10.0 * torch.log10(mse)),
            "ssim": float(ssim(render, gt)),
        }

    report = {
        "views": len(names),
        "mean_psnr": float(np.mean([entry["psnr"] for entry in per_view.values()])),
        "mean_ssim": float(np.mean([entry["ssim"] for entry in per_view.values()])),
        "per_view": per_view,
        "note": "RGB PSNR/SSIM only; no external perceptual-model weights are required",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
