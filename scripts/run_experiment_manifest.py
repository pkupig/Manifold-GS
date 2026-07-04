#!/usr/bin/env python3
"""Run reproducible analytic data, training, and evaluation jobs from a manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stage", choices=("prepare", "train", "evaluate", "all"), default="all")
    parser.add_argument("--scene", action="append", help="Restrict to a scene; repeatable")
    parser.add_argument("--seed", action="append", type=int, help="Restrict to a seed; repeatable")
    parser.add_argument("--method", action="append", help="Restrict to a method; repeatable")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip stages whose declared outputs already exist")
    return parser.parse_args()


def run(command: list[str], dry_run: bool, outputs: tuple[Path, ...] = (), resume: bool = False) -> None:
    if resume and outputs and all(path.exists() for path in outputs):
        print("= skip " + ", ".join(str(path) for path in outputs), flush=True)
        return
    print("+ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    benchmark = ROOT / "experiments" / "benchmarks" / manifest["name"]
    data_benchmark = manifest.get("dataset_benchmark", manifest["name"])
    data_root = ROOT / "experiments" / "benchmarks" / data_benchmark / "data"
    run_root = benchmark / "runs"
    scenes = [s for s in manifest["scenes"] if args.scene is None or s in args.scene]
    seeds = [s for s in manifest["seeds"] if args.seed is None or s in args.seed]
    methods = {
        name: method_args for name, method_args in manifest["methods"].items()
        if args.method is None or name in args.method
    }
    if not scenes or not seeds or not methods:
        raise ValueError("The requested scene/seed/method filters select no jobs")

    stages = {args.stage} if args.stage != "all" else {"prepare", "train", "evaluate"}
    dataset = manifest["dataset"]
    iterations = int(manifest["training"]["iterations"])
    for scene in scenes:
        for seed in seeds:
            data_dir = data_root / f"{scene}_s{seed}"
            if "prepare" in stages:
                command = [
                    sys.executable, "scripts/generate_analytic_scene.py",
                    "--out", str(data_dir), "--scene", scene,
                    "--width", str(dataset["width"]), "--height", str(dataset["height"]),
                    "--train-views", str(dataset["train_views"]),
                    "--heldout-views", str(dataset["heldout_views"]),
                    "--gt-points", str(dataset["gt_points"]),
                    "--init-points", str(dataset["init_points"]),
                    "--init-noise", str(dataset["init_noise"]),
                    "--mesh-resolution", str(dataset["mesh_resolution"]), "--seed", str(seed),
                ]
                run(
                    command, args.dry_run,
                    outputs=(data_dir / "gt" / "metadata.json", data_dir / "sparse" / "0" / "test.txt"),
                    resume=args.resume,
                )

            if "evaluate" in stages:
                run([
                    sys.executable, "scripts/evaluate_visibility_coverage.py",
                    "--data", str(data_dir),
                    "--out", str(data_dir / "gt" / "train_visibility_coverage.json"),
                    "--split", "train",
                ], args.dry_run, outputs=(
                    data_dir / "gt" / "train_visibility_coverage.json",
                ), resume=args.resume)

            for method, method_args in methods.items():
                output = run_root / f"{scene}_s{seed}_{method}"
                if "train" in stages:
                    command = [
                        sys.executable, "third_party/gaussian-splatting/train.py",
                        "-s", str(data_dir), "-m", str(output),
                        "--iterations", str(iterations),
                        "--save_iterations", str(iterations),
                        "--test_iterations", str(iterations),
                        *manifest["training"].get("common_args", []), *method_args,
                    ]
                    run(command, args.dry_run, outputs=(
                        output / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply",
                    ), resume=args.resume)

                if "evaluate" in stages:
                    ply = output / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
                    run([
                        sys.executable, "scripts/evaluate_geometry_gt.py",
                        "--ply", str(ply), "--gt", str(data_dir / "gt" / "surface.npz"),
                        "--out", str(output / "geometry_metrics.json"),
                    ], args.dry_run, outputs=(output / "geometry_metrics.json",), resume=args.resume)
                    run([
                        sys.executable, "scripts/analyze_fundamental_compatibility.py",
                        "--ply", str(ply), "--out", str(output / "fundamental"), "--knn", "20",
                    ], args.dry_run, outputs=(output / "fundamental" / "summary.json",), resume=args.resume)
                    run([
                        sys.executable, "scripts/project_manifold.py",
                        "--ply", str(ply), "--out", str(output / "asset"),
                        "--gt", str(data_dir / "gt" / "surface.npz"), "--knn", "20",
                    ], args.dry_run, outputs=(output / "asset" / "patch_mesh.ply",), resume=args.resume)
                    run([
                        sys.executable, "scripts/evaluate_mesh_gt.py",
                        "--mesh", str(output / "asset" / "patch_mesh.ply"),
                        "--gt", str(data_dir / "gt" / "surface.npz"),
                        "--out", str(output / "asset" / "mesh_metrics.json"),
                    ], args.dry_run, outputs=(output / "asset" / "mesh_metrics.json",), resume=args.resume)
                    run([
                        sys.executable, "third_party/gaussian-splatting/render.py",
                        "-s", str(data_dir), "-m", str(output), "--eval",
                        "--iteration", str(iterations), "--skip_train",
                    ], args.dry_run, outputs=(
                        output / "test" / f"ours_{iterations}" / "renders",
                        output / "test" / f"ours_{iterations}" / "gt",
                    ), resume=args.resume)
                    rendered = output / "test" / f"ours_{iterations}"
                    run([
                        sys.executable, "scripts/evaluate_rendered_images.py",
                        "--renders", str(rendered / "renders"), "--gt", str(rendered / "gt"),
                        "--out", str(output / "heldout_metrics.json"),
                    ], args.dry_run, outputs=(output / "heldout_metrics.json",), resume=args.resume)


if __name__ == "__main__":
    main()
