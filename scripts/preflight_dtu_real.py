#!/usr/bin/env python3
"""Validate DTU data, official GT, split, and external work storage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


PILOT_SCANS = (24, 65, 105)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=Path("/tmp/emgs-real"))
    parser.add_argument("--scan", type=int, action="append")
    args = parser.parse_args()
    scans = args.scan or list(PILOT_SCANS)
    free_gb = shutil.disk_usage(args.work_root.parent).free / 1024**3
    report: dict[str, object] = {
        "work_root": str(args.work_root), "work_free_gb": free_gb,
        "work_storage_ok": free_gb >= 20, "scans": {},
    }
    for scan in scans:
        scene = args.data_root / f"scan{scan}"
        test_file = scene / "sparse" / "0" / "test.txt"
        checks = {
            "images": (scene / "images").is_dir(),
            "colmap_cameras": (scene / "sparse" / "0" / "cameras.bin").is_file(),
            "colmap_images": (scene / "sparse" / "0" / "images.bin").is_file(),
            "colmap_points": (scene / "sparse" / "0" / "points3D.ply").is_file(),
            "explicit_test_split": test_file.is_file(),
            "official_observation_mask": (
                args.official_root / "ObsMask" / f"ObsMask{scan}_10.mat"
            ).is_file(),
            "official_ground_plane": (
                args.official_root / "ObsMask" / f"Plane{scan}.mat"
            ).is_file(),
            "official_gt_points": (
                args.official_root / "Points" / "stl" / f"stl{scan:03}_total.ply"
            ).is_file(),
        }
        report["scans"][f"scan{scan}"] = {"ready": all(checks.values()), "checks": checks}
    report["ready"] = report["work_storage_ok"] and all(
        item["ready"] for item in report["scans"].values()
    )
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ready"] else 1)


if __name__ == "__main__":
    main()
