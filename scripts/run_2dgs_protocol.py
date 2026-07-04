#!/usr/bin/env python3
"""Run or print the registered 2DGS external-baseline protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party" / "2d-gaussian-splatting"


def run(command: list[str], cwd: Path, execute: bool, done: Path | None = None) -> None:
    print(f"[cwd={cwd}] {shlex.join(command)}")
    if execute and not (done is not None and done.exists()):
        subprocess.run(command, cwd=cwd, check=True)
    elif execute and done is not None:
        print(f"[resume] exists: {done}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "experiments/manifests/2dgs_plane_torus_protocol.json",
    )
    parser.add_argument("--protocol", default="official_30k")
    parser.add_argument("--stage", choices=("train", "render", "evaluate", "all"), default="all")
    parser.add_argument("--conda-env", default="surfel_splatting")
    parser.add_argument("--execute", action="store_true", help="Execute commands; default is dry-run")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = manifest["protocols"][args.protocol]
    iterations = int(config["iterations"])
    data_root = ROOT / manifest["source_dataset"]
    output_root = ROOT / manifest["output_root"] / args.protocol
    conda_python = ["conda", "run", "-n", args.conda_env, "python"]

    for scene in manifest["scenes"]:
        for seed in manifest["seeds"]:
            tag = f"{scene}_s{seed}"
            source = data_root / tag
            output = output_root / tag
            native_ply = output / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
            adapted_ply = output / "evaluation" / "point_cloud_eval.ply"
            gt = source / "gt" / "surface.npz"

            if args.stage in ("train", "all"):
                command = conda_python + [
                    "train.py", "-s", str(source), "-m", str(output), "--eval",
                    "--iterations", str(iterations), "--test_iterations", str(iterations),
                    "--save_iterations", str(iterations), "--seed", str(seed),
                    "--lambda_normal", str(config["lambda_normal"]),
                    "--lambda_dist", str(config["lambda_dist"]),
                    "--depth_ratio", str(config["depth_ratio"]), "--quiet",
                ]
                run(command, REPO, args.execute, native_ply)

            if args.stage in ("render", "all"):
                command = conda_python + [
                    "render.py", "-s", str(source), "-m", str(output),
                    "--iteration", str(iterations), "--skip_train", "--skip_mesh", "--quiet",
                ]
                rendered = output / "test" / f"ours_{iterations}" / "renders"
                run(command, REPO, args.execute, rendered)
                run(conda_python + ["metrics.py", "-m", str(output)], REPO, args.execute,
                    output / "results.json")

            if args.stage in ("evaluate", "all"):
                run(
                    [sys.executable, "scripts/convert_2dgs_ply.py", "--input", str(native_ply),
                     "--output", str(adapted_ply)],
                    ROOT, args.execute, adapted_ply,
                )
                run(
                    [sys.executable, "scripts/evaluate_geometry_gt.py", "--ply", str(adapted_ply),
                     "--gt", str(gt), "--out", str(output / "evaluation" / "geometry_metrics.json")],
                    ROOT, args.execute, output / "evaluation" / "geometry_metrics.json",
                )


if __name__ == "__main__":
    main()
