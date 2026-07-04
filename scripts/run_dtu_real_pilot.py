#!/usr/bin/env python3
"""Run the registered DTU real-scene pilot without writing large outputs to the repo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "third_party/2d-gaussian-splatting/scripts/eval_dtu/evaluate_single_scene.py"


def run(command: list[str], *, execute: bool, output: Path | None = None, resume: bool = False) -> None:
    if resume and output is not None and output.exists():
        print(f"[skip] {output}")
        return
    print("+ " + shlex.join(command), flush=True)
    if execute:
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", type=Path,
        default=ROOT / "experiments/protocols/dtu_real_pilot_v1.json",
    )
    parser.add_argument("--scan", type=int, action="append")
    parser.add_argument(
        "--method",
        choices=(
            "vanilla", "vanilla_matched", "manifold_full", "manifold_colmap_anchor",
        ),
        action="append",
    )
    parser.add_argument("--stage", choices=("train", "evaluate", "all"), default="all")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-python", default=sys.executable)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    scans = args.scan or protocol["scans"]
    methods = args.method or ["vanilla", "manifold_full"]
    data_root = Path(protocol["data_root"])
    output_root = Path(protocol["output_root"]) / protocol["name"]
    official_root = Path(protocol["official_gt_root"])
    stages = {args.stage} if args.stage != "all" else {"train", "evaluate"}

    manifold_args = [
        "--mcgs_preserve_pruned_mass", "--mcgs_warmup", "500",
        "--mcgs_refresh_interval", "250", "--mcgs_knn", "12",
        "--mcgs_max_points", "2048",
        "--mcgs_lambda_support", "0.001", "--mcgs_lambda_tangent", "0.005",
        "--mcgs_lambda_shape", "0.001", "--mcgs_lambda_symmetry", "0.001",
        "--mcgs_compatibility_start", "2500", "--mcgs_compatibility_ramp", "1500",
        "--mcgs_compatibility_confidence_floor", "0.10",
    ]
    colmap_anchor_args = [
        "--mcgs_lambda_static_support", "0.01",
        "--mcgs_static_support_start", "500",
        "--mcgs_static_support_interval", "10",
        "--mcgs_static_support_max_points", "2048",
        "--mcgs_static_support_reference_max_points", "8192",
    ]
    for scan in scans:
        source = data_root / f"scan{scan}"
        for method in methods:
            output = output_root / f"scan{scan}_{method}"
            ply = output / "point_cloud/iteration_7000/point_cloud.ply"
            if "train" in stages:
                train_command = [
                    sys.executable, "third_party/gaussian-splatting/train.py",
                    "-s", str(source), "-m", str(output), "-r", "2",
                    "--iterations", "7000", "--save_iterations", "7000",
                    "--test_iterations", "7000", "--eval", "--disable_viewer",
                    "--checkpoint_iterations", "1000", "2000", "3000", "4000", "5000", "6000",
                    *(
                        ["--densify_until_iter", "3001"]
                        if method in {
                            "vanilla_matched", "manifold_full", "manifold_colmap_anchor",
                        }
                        else []
                    ),
                    *(
                        manifold_args
                        if method in {"manifold_full", "manifold_colmap_anchor"}
                        else []
                    ),
                    *(colmap_anchor_args if method == "manifold_colmap_anchor" else []),
                ]
                if args.resume:
                    checkpoints = list(output.glob("chkpnt*.pth"))
                    if checkpoints:
                        latest = max(
                            checkpoints,
                            key=lambda path: int(path.stem.replace("chkpnt", "")),
                        )
                        train_command.extend(["--start_checkpoint", str(latest)])
                        print(f"[resume checkpoint] {latest}")
                run(train_command, execute=args.execute, output=ply, resume=args.resume)
            if "evaluate" not in stages:
                continue
            asset = output / "asset"
            mesh = asset / "patch_mesh.ply"
            run([
                sys.executable, "scripts/project_manifold.py", "--ply", str(ply),
                "--out", str(asset), "--knn", "20",
            ], execute=args.execute, output=mesh, resume=args.resume)
            rendered = output / "test/ours_7000"
            run([
                sys.executable, "third_party/gaussian-splatting/render.py",
                "-s", str(source), "-m", str(output), "--eval", "--iteration", "7000",
                "--skip_train",
            ], execute=args.execute, output=rendered / "renders", resume=args.resume)
            run([
                sys.executable, "scripts/evaluate_rendered_images.py",
                "--renders", str(rendered / "renders"), "--gt", str(rendered / "gt"),
                "--out", str(output / "heldout_metrics.json"),
            ], execute=args.execute, output=output / "heldout_metrics.json", resume=args.resume)
            dtu_output = output / "dtu_evaluation"
            run([
                args.eval_python, str(EVAL), "--input_mesh", str(mesh),
                "--scan_id", str(scan), "--output_dir", str(dtu_output),
                "--mask_dir", str(data_root), "--DTU", str(official_root),
            ], execute=args.execute, output=dtu_output / "results.json", resume=args.resume)


if __name__ == "__main__":
    main()
