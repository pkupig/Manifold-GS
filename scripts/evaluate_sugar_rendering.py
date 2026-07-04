#!/usr/bin/env python3
"""Evaluate refined SuGaR rendering on the registered held-out split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import open3d as o3d
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
SUGAR_ROOT = ROOT / "third_party" / "SuGaR"
sys.path.insert(0, str(SUGAR_ROOT))
sys.path.insert(0, str(SUGAR_ROOT / "gaussian_splatting"))

from gaussian_splatting.utils.image_utils import psnr  # noqa: E402
from gaussian_splatting.utils.loss_utils import ssim  # noqa: E402
from sugar_scene.gs_model import GaussianSplattingWrapper  # noqa: E402
from sugar_scene.sugar_model import SuGaR  # noqa: E402
from sugar_utils.spherical_harmonics import SH2RGB  # noqa: E402


def newest(paths: list[Path], description: str) -> Path:
    if not paths:
        raise FileNotFoundError(f"No {description} found")
    return max(paths, key=lambda path: path.stat().st_mtime)


def save_rgb(path: Path, image: torch.Tensor) -> None:
    array = (image.detach().cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def alpha_mask(source: Path, image_name: str, shape: tuple[int, int], device) -> torch.Tensor:
    path = source / "images" / image_name
    if not path.exists():
        matches = list((source / "images").glob(f"{Path(image_name).stem}.*"))
        if len(matches) != 1:
            raise FileNotFoundError(f"cannot resolve alpha image for {image_name}")
        path = matches[0]
    with Image.open(path) as image:
        if "A" not in image.getbands():
            return torch.ones(shape, dtype=torch.float32, device=device)
        alpha = image.getchannel("A").resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
        array = np.asarray(alpha, dtype=np.float32) / 255.0
    return torch.from_numpy(array).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "experiments/manifests/sugar_plane_torus_pilot.json",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    protocol = manifest["protocol"]
    aggregate: dict[str, object] = {"protocol": "fixed test.txt held-out split", "scenes": {}}

    for scene in manifest["scenes"]:
        source = resolve(manifest["dataset_root"]) / scene
        suffix = manifest.get("checkpoint_suffix", "_vanilla")
        vanilla = resolve(manifest["checkpoint_root"]) / f"{scene}{suffix}"
        output = resolve(manifest["output_root"]) / scene
        refined = newest(
            list((SUGAR_ROOT / "output/refined" / scene).glob("**/*.pt")),
            f"refined SuGaR checkpoint for {scene}",
        )
        mesh = newest(
            list((SUGAR_ROOT / "output/coarse_mesh" / scene).glob("*.ply")),
            f"coarse SuGaR mesh for {scene}",
        )

        wrapper = GaussianSplattingWrapper(
            source_path=str(source), output_path=str(vanilla),
            iteration_to_load=protocol["iteration_to_load"], load_gt_images=True,
            eval_split=True,
        )
        checkpoint = torch.load(refined, map_location=wrapper.device)
        state = checkpoint["state_dict"]
        model = SuGaR(
            nerfmodel=wrapper, points=state["_points"],
            colors=SH2RGB(state["_sh_coordinates_dc"][:, 0, :]), initialize=False,
            sh_levels=wrapper.gaussians.active_sh_degree + 1,
            keep_track_of_knn=False, knn_to_track=0, beta_mode="average",
            surface_mesh_to_bind=o3d.io.read_triangle_mesh(str(mesh)),
            n_gaussians_per_surface_triangle=protocol["gaussians_per_triangle"],
        )
        model.load_state_dict(state)
        model.eval()

        render_dir = output / "rendering"
        render_dir.mkdir(parents=True, exist_ok=True)
        views = []
        with torch.no_grad():
            for index in range(len(wrapper.test_cameras)):
                gt = wrapper.get_test_gt_image(index).permute(2, 0, 1).unsqueeze(0)
                prediction_hwc = model.render_image_gaussian_rasterizer(
                    nerf_cameras=wrapper.test_cameras, camera_indices=index,
                    verbose=False, bg_color=None,
                    sh_deg=wrapper.gaussians.active_sh_degree,
                    compute_color_in_rasterizer=True,
                ).clamp(0, 1)
                if protocol.get("mask_predictions_with_alpha", False):
                    name = wrapper.test_cam_list[index].image_name
                    mask = alpha_mask(source, name, prediction_hwc.shape[:2], prediction_hwc.device)
                    prediction_hwc = prediction_hwc * mask[..., None]
                    gt = gt * mask[None, None]
                prediction = prediction_hwc.permute(2, 0, 1).unsqueeze(0)
                view_psnr = float(psnr(prediction, gt).item())
                view_ssim = float(ssim(prediction, gt).item())
                views.append({"index": index, "psnr": view_psnr, "ssim": view_ssim})

                gt_hwc = gt[0].permute(1, 2, 0)
                error = (prediction_hwc - gt_hwc).abs()
                save_rgb(render_dir / f"{index:03d}_prediction.png", prediction_hwc)
                save_rgb(render_dir / f"{index:03d}_gt.png", gt_hwc)
                save_rgb(render_dir / f"{index:03d}_error_x4.png", error * 4)

        result = {
            "scene": scene, "num_test_views": len(views),
            "psnr": float(np.mean([view["psnr"] for view in views])),
            "ssim": float(np.mean([view["ssim"] for view in views])),
            "refined_checkpoint": str(refined.relative_to(ROOT)),
            "coarse_mesh": str(mesh.relative_to(ROOT)), "views": views,
        }
        (output / "render_metrics.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        aggregate["scenes"][scene] = result
        print(f"{scene}: PSNR {result['psnr']:.4f}, SSIM {result['ssim']:.4f}")

    aggregate_path = ROOT / manifest["output_root"] / "render_metrics.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {aggregate_path}")


if __name__ == "__main__":
    main()
