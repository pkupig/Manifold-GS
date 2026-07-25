#!/usr/bin/env python3
"""Run the frozen DTU 2DGS native-mesh baseline in resumable stages.

This runner deliberately exports 2DGS's own TSDF mesh.  It does not invent
Manifold-GS patch/source bindings, so downstream comparison must label edit
semantics as native mesh connectivity rather than certified patch binding.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TWO_DGS = ROOT / "third_party" / "2d-gaussian-splatting"


def run(command: list[str], *, cwd: Path, execute: bool, done: Path, resume: bool) -> None:
    if resume and done.exists():
        print(f"[skip] {done}")
        return
    print("+ " + shlex.join(command))
    if execute:
        subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=int, action="append", choices=(24, 65, 105))
    parser.add_argument("--stage", choices=("train", "render", "mesh", "all"), default="all")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--conda-env", default="surfel_splatting")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--mesh-res", type=int, default=1024)
    parser.add_argument("--num-cluster", type=int, default=50)
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/emgs-real/dtu-preprocessed/DTU"))
    parser.add_argument("--output-root", type=Path, default=Path("/root/autodl-tmp/emgs-real/outputs/2dgs_dtu_asset_v1"))
    args = parser.parse_args()
    if args.iterations != 30000 or args.mesh_res != 1024 or args.num_cluster != 50:
        raise ValueError("2dgs-dtu-asset/v1 is frozen: iterations=30000, mesh-res=1024, num-cluster=50")
    scans = args.scan or [24, 65, 105]
    stages = {args.stage} if args.stage != "all" else {"train", "render", "mesh"}
    python = ["conda", "run", "--no-capture-output", "-n", args.conda_env, "python"]
    for scan in scans:
        source = args.data_root / f"scan{scan}"
        split = source / "sparse" / "0" / "test.txt"
        if not split.is_file():
            raise FileNotFoundError(f"frozen split missing: {split}")
        output = args.output_root / f"scan{scan}_official_2dgs"
        checkpoint = output / "point_cloud" / f"iteration_{args.iterations}" / "point_cloud.ply"
        if "train" in stages:
            run(python + ["train.py", "-s", str(source), "-m", str(output), "--eval",
                          "--iterations", str(args.iterations), "--save_iterations", str(args.iterations),
                          "--test_iterations", str(args.iterations), "--quiet"],
                cwd=TWO_DGS, execute=args.execute, done=checkpoint, resume=args.resume)
        rendered = output / "test" / f"ours_{args.iterations}" / "renders"
        if "render" in stages:
            run(python + ["render.py", "-s", str(source), "-m", str(output), "--eval",
                          "--iteration", str(args.iterations), "--skip_train", "--skip_mesh", "--quiet"],
                cwd=TWO_DGS, execute=args.execute, done=rendered, resume=args.resume)
            run(python + ["metrics.py", "-m", str(output)], cwd=TWO_DGS, execute=args.execute,
                done=output / "results.json", resume=args.resume)
        mesh = output / "train" / f"ours_{args.iterations}" / "fuse_post.ply"
        if "mesh" in stages:
            run(python + ["render.py", "-s", str(source), "-m", str(output), "--eval",
                          "--iteration", str(args.iterations), "--skip_test", "--mesh_res", str(args.mesh_res),
                          "--num_cluster", str(args.num_cluster), "--quiet"],
                cwd=TWO_DGS, execute=args.execute, done=mesh, resume=args.resume)


if __name__ == "__main__":
    main()
