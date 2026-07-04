#!/usr/bin/env python3
"""Check external baseline repositories and inputs without importing GPU code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def exists(path: Path) -> bool:
    return path.exists()


def inspect_sugar(scene: Path | None, checkpoint: Path | None) -> dict[str, object]:
    repo = ROOT / "third_party" / "SuGaR"
    checks = {
        "repository": exists(repo / "train.py"),
        "submodule_rasterizer": exists(
            repo / "gaussian_splatting" / "submodules" / "diff-gaussian-rasterization"
        ),
        "submodule_knn": exists(repo / "gaussian_splatting" / "submodules" / "simple-knn"),
        "scene": scene is not None and scene.is_dir(),
        "scene_colmap_sparse": scene is not None and (scene / "sparse").is_dir(),
        "checkpoint": checkpoint is not None and checkpoint.is_dir(),
        "checkpoint_iteration_7000": checkpoint is not None
        and (checkpoint / "point_cloud" / "iteration_7000" / "point_cloud.ply").is_file(),
    }
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "note": (
            "SuGaR consumes a COLMAP scene and a vanilla 3DGS checkpoint at iteration 7000. "
            "Its exported refined PLY follows the standard 3DGS schema and can be passed "
            "directly to scripts/evaluate_geometry_gt.py."
        ),
    }


def inspect_2dgs() -> dict[str, object]:
    repo = ROOT / "third_party" / "2d-gaussian-splatting"
    checks = {
        "repository": exists(repo / "train.py"),
        "submodule_rasterizer": exists(repo / "submodules" / "diff-surfel-rasterization"),
        "submodule_knn": exists(repo / "submodules" / "simple-knn"),
    }
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "note": (
            "Official repository and CUDA submodules are present. Native PLY stores two tangent "
            "scales; run scripts/convert_2dgs_ply.py before the shared Gaussian evaluator."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    report = {
        "sugar": inspect_sugar(args.scene, args.checkpoint),
        "2dgs": inspect_2dgs(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
