#!/usr/bin/env python3
"""Run or print the registered two-scene SuGaR mechanism pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from dtu_official_layout import ensure_dtu_official_layout

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party" / "SuGaR"
DTU_EVAL = ROOT / "third_party/2d-gaussian-splatting/scripts/eval_dtu/evaluate_single_scene.py"


def invoke(command: list[str], cwd: Path, execute: bool, log_path: Path | None = None) -> None:
    print(f"[cwd={cwd}] {shlex.join(command)}")
    if execute:
        if log_path is None:
            subprocess.run(command, cwd=cwd, check=True)
            return
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log.write(line)
                log.flush()
            if process.wait() != 0:
                raise subprocess.CalledProcessError(process.returncode, command)


def newest(pattern: str) -> Path:
    matches = list(REPO.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"SuGaR did not produce expected output: {pattern}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def bbox_arg(values: list[float]) -> str:
    return ",".join(f"{value:.9g}" for value in values)


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def prepare_external_output_links(scene: str, output: Path) -> None:
    external = output / "sugar_internal"
    for kind in ("coarse", "coarse_mesh", "refined", "refined_ply"):
        parent = REPO / "output" / kind
        parent.mkdir(parents=True, exist_ok=True)
        target = external / kind / scene
        target.mkdir(parents=True, exist_ok=True)
        link = parent / scene
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise RuntimeError(f"SuGaR output link points elsewhere: {link}")
        elif link.exists():
            raise FileExistsError(f"refusing to replace existing SuGaR output: {link}")
        else:
            link.symlink_to(target, target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "experiments/manifests/sugar_plane_torus_pilot.json",
    )
    parser.add_argument("--conda-env", default="sugar")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = manifest["protocol"]

    for scene in manifest["scenes"]:
        source = resolve(manifest["dataset_root"]) / scene
        checkpoint_suffix = manifest.get("checkpoint_suffix", "_vanilla")
        checkpoint = resolve(manifest["checkpoint_root"]) / f"{scene}{checkpoint_suffix}"
        output = resolve(manifest["output_root"]) / scene
        # scene_bbox is optional: when a scene has no recorded foreground box we
        # let SuGaR compute its own camera-based bbox (official default), which
        # keeps the derivation identical and reproducible across scans.
        bbox = manifest.get("scene_bbox", {}).get(scene)
        geometry_out = output / "geometry_metrics.json"
        evaluation_type = manifest.get("evaluation", {}).get("type")
        completion = (
            output / "dtu_patch_mesh/results.json"
            if evaluation_type == "dtu"
            else geometry_out
        )
        if args.execute and completion.exists():
            print(f"[resume] exists: {completion}")
            continue

        copied_ply = output / "refined_gaussians.ply"
        copied_mesh = output / "coarse_mesh.ply"
        assets_ready = copied_ply.is_file() and copied_mesh.is_file()

        if args.execute and assets_ready:
            print(f"[resume] reusing extracted assets: {copied_mesh}, {copied_ply}")
        elif args.execute and config.get("externalize_sugar_outputs", False):
            prepare_external_output_links(scene, output)
        # NOTE: the old PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 (an 8GB-era
        # setting) fragments the allocator on the larger matched-3DGS scenes
        # (~330k gaussians, e.g. scan24/65) and makes mesh extraction OOM inside
        # the PyTorch3D rasterizer even though peak usage is only ~3GB. On the
        # 24GB 3090 the default allocator extracts fine, so we no longer pin it.
        command = [
            "conda", "run", "--no-capture-output", "-n", args.conda_env, "python", "train.py",
            "-s", str(source), "-c", str(checkpoint),
            "-i", str(config["iteration_to_load"]),
            "-r", str(config["regularization_type"]),
            "--sdf_samples", str(config["sdf_samples"]),
            "-l", str(config["surface_level"]),
            "-v", str(config["mesh_vertices"]),
            "--mesh_surface_points", str(config.get("mesh_surface_points", 10_000_000)),
            "--poisson_depth", str(config.get("poisson_depth", 10)),
            "--poisson_threads", str(config.get("poisson_threads", -1)),
            "-g", str(config["gaussians_per_triangle"]),
            "-f", str(config["refinement_iterations"]),
            "--export_uv_textured_mesh", str(config["export_uv_textured_mesh"]),
            "--export_ply", "True", "--eval", "True",
            "--seed", str(config["seed"]),
        ]
        if bbox is not None:
            # Explicit recorded box (e.g. scan105); used as-is, not re-centered.
            command += [
                f"--bboxmin={bbox_arg(bbox['min'])}",
                f"--bboxmax={bbox_arg(bbox['max'])}",
                "--center_bbox", "False",
            ]
        else:
            # No recorded box: SuGaR builds a camera-based foreground box
            # (radius = 1.1 * max||cam_center - mean||, centered on the camera
            # average). Omitting --bboxmin/--bboxmax selects that default path.
            command += ["--center_bbox", "True"]
        coarse_checkpoint = (
            REPO / "output/coarse" / scene
            / "sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt"
        )
        if coarse_checkpoint.is_file():
            command.extend(["--coarse_model_path", str(coarse_checkpoint)])
        if not assets_ready:
            invoke(command, REPO, args.execute, output / "sugar.log")
        if not args.execute:
            continue

        if not assets_ready:
            refined_ply = newest(f"output/refined_ply/{scene}/*.ply")
            coarse_mesh = newest(f"output/coarse_mesh/{scene}/*.ply")
            output.mkdir(parents=True, exist_ok=True)
            shutil.copy2(refined_ply, copied_ply)
            shutil.copy2(coarse_mesh, copied_mesh)
        if manifest.get("evaluation", {}).get("type") == "dtu":
            evaluation = manifest["evaluation"]
            scan = int(scene.removeprefix("scan"))
            official_root = Path(evaluation["official_gt_root"])
            ensure_dtu_official_layout(official_root)
            data_root = str(resolve(manifest["dataset_root"]))
            native_out = output / "dtu_native_mesh"
            invoke([
                sys.executable, str(DTU_EVAL), "--input_mesh", str(copied_mesh),
                "--scan_id", str(scan), "--output_dir", str(native_out),
                "--mask_dir", data_root, "--DTU", str(official_root),
            ], ROOT, True)
            patch_asset = output / "patch_asset"
            invoke([
                sys.executable, "scripts/project_manifold.py", "--ply", str(copied_ply),
                "--out", str(patch_asset), "--knn", "20",
            ], ROOT, True)
            invoke([
                sys.executable, str(DTU_EVAL),
                "--input_mesh", str(patch_asset / "patch_mesh.ply"),
                "--scan_id", str(scan),
                "--output_dir", str(output / "dtu_patch_mesh"),
                "--mask_dir", data_root, "--DTU", str(official_root),
            ], ROOT, True)
        else:
            invoke([
                sys.executable, "scripts/evaluate_geometry_gt.py", "--ply", str(copied_ply),
                "--gt", str(source / "gt/surface.npz"), "--out", str(geometry_out),
            ], ROOT, True)
            invoke([
                sys.executable, "scripts/evaluate_mesh_gt.py", "--mesh", str(copied_mesh),
                "--gt", str(source / "gt/surface.npz"), "--out", str(output / "mesh_metrics.json"),
            ], ROOT, True)


if __name__ == "__main__":
    main()
