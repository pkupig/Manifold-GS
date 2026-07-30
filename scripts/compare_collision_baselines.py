#!/usr/bin/env python3
"""Compare collision-precision of our certified candidate vs external mesh baselines.

Scores three meshes against the same DTU GT surface (collision-vs-GT, §4.7):
- ours:    the observation-certified ``collision_candidate.ply``;
- sugar:   SuGaR's DTU-eval ``culled_mesh.ply`` (DTU mm frame -> Gaussian frame
           via the preprocessing ``cameras.npz`` scale_mat, same transform used
           to build the GT npz);
- poisson: Poisson-from-3DGS over the SAME oriented points ``projected_points.ply``
           (open3d depth=9 + density-quantile trim + input-bbox crop).
- 2DGS:    official 2DGS native TSDF mesh, scored by the same frozen P0.4
           collision evaluator during ``run_2dgs_dtu_asset.py``.

All CPU. GT npz (``gt_surface_stlNNN.npz``) must already exist beside the bundle
(built by the P0.4 alignment step). Writes a JSON + Markdown table next to OUT.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manifold_gs.mesh_io import read_triangle_mesh_ply
from manifold_gs.collision_metrics import surface_coverage_metrics


def poisson_from_points(points_ply: str, out_ply: str, depth: int = 9,
                        density_quantile: float = 0.1, dilate: float = 0.05):
    """Watertight Poisson over oriented points, density-trimmed and bbox-cropped."""
    pc = o3d.io.read_point_cloud(points_ply)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=depth)
    dens = np.asarray(dens)
    if dens.size and density_quantile > 0:
        mesh.remove_vertices_by_mask(dens < np.quantile(dens, density_quantile))
    p = np.asarray(pc.points)
    lo, hi = p.min(0), p.max(0)
    center, ext = (lo + hi) / 2, (hi - lo) * (1 + dilate)
    mesh = mesh.crop(o3d.geometry.AxisAlignedBoundingBox(center - ext / 2, center + ext / 2))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    Path(out_ply).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(out_ply, mesh)
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles)


def _precision(metrics: dict, bbox_diag: float) -> dict:
    return {
        "faces_or_samples": int(metrics["candidate_samples"]),
        "floater_fraction": float(metrics["false_surface_fraction"]),
        "coverage_at_1pct": float(metrics["coverage"]),
        "candidate_to_reference_p95_pctbbox": float(metrics["candidate_to_reference_p95"] / bbox_diag * 100),
        "normal_median_deg": float(metrics["supported_normal_median_deg"]),
    }


def compare_scene(bundle: Path, gt_npz: Path, projected_points: Path,
                  sugar_culled: Path | None, cameras_npz: Path | None,
                  two_dgs_collision: Path | None, two_dgs_mesh: Path | None,
                  samples: int = 50000, seed: int = 0) -> dict:
    gt = np.load(str(gt_npz))
    gx = np.asarray(gt["xyz"], dtype=np.float64)
    gn = np.asarray(gt["normals"], dtype=np.float64)
    dg = float(np.linalg.norm(np.ptp(gx, axis=0)))
    tol = 0.01 * dg
    row: dict = {}

    ov, of = read_triangle_mesh_ply(str(bundle / "collision_candidate.ply"))
    m = surface_coverage_metrics(ov.astype(np.float64), of, gx, gn, tolerance=tol, samples=samples, seed=seed)
    row["ours"] = {**_precision(m, dg), "faces": int(of.shape[0])}

    if sugar_culled is not None and sugar_culled.exists() and cameras_npz is not None:
        s_inv = np.load(str(cameras_npz))["scale_mat_inv_0"]
        sm = o3d.io.read_triangle_mesh(str(sugar_culled))
        sv = np.asarray(sm.vertices)
        sf = np.asarray(sm.triangles)
        svn = (np.c_[sv, np.ones(len(sv))] @ s_inv.T)[:, :3]
        m = surface_coverage_metrics(svn, sf, gx, gn, tolerance=tol, samples=samples, seed=seed)
        row["sugar_culled"] = {**_precision(m, dg), "faces": int(sf.shape[0])}

    pv, pf = poisson_from_points(str(projected_points), str(bundle / "baselines" / "poisson_fair.ply"))
    m = surface_coverage_metrics(pv, pf, gx, gn, tolerance=tol, samples=samples, seed=seed)
    row["poisson"] = {**_precision(m, dg), "faces": int(pf.shape[0])}
    # Native 2DGS is already scored by the frozen shared evaluator.
    if two_dgs_collision is not None and two_dgs_collision.is_file() and two_dgs_mesh is not None:
        report = json.loads(two_dgs_collision.read_text(encoding="utf-8"))
        coverage = report["coverage"]
        _, faces = read_triangle_mesh_ply(str(two_dgs_mesh))
        row["2dgs_native"] = {
            "faces_or_samples": int(coverage["candidate_samples"]),
            "floater_fraction": float(coverage["false_surface_fraction"]),
            "coverage_at_1pct": float(coverage["coverage"]),
            "candidate_to_reference_p95_pctbbox": float(coverage["candidate_to_reference_p95"] / dg * 100),
            "normal_median_deg": float(coverage["supported_normal_median_deg"]),
            "faces": int(faces.shape[0]),
        }

    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot-root",
        default="/root/autodl-tmp/emgs-real/outputs/dtu_real_pilot_v1",
        help="root holding <scan>_vanilla_matched/{asset,hybrid_asset}",
    )
    parser.add_argument("--sugar-root", default="/root/autodl-tmp/emgs-real/outputs/sugar_dtu_pilot_v1")
    parser.add_argument("--dtu-root", default="/root/autodl-tmp/emgs-real/dtu-preprocessed/DTU")
    parser.add_argument("--two-dgs-root", default="/root/autodl-tmp/emgs-real/outputs/2dgs_dtu_asset_v1")
    parser.add_argument("--scenes", nargs="+", default=["scan24", "scan65", "scan105"])
    parser.add_argument("--out", default=None, help="output json (default <pilot-root>/collision_precision_comparison.json)")
    args = parser.parse_args()

    pilot = Path(args.pilot_root)
    out = Path(args.out) if args.out else pilot / "collision_precision_comparison.json"
    scenes = []
    for scan in args.scenes:
        n = scan.replace("scan", "").zfill(3)
        bundle = pilot / f"{scan}_vanilla_matched" / "hybrid_asset"
        row = {"scene": scan, **compare_scene(
            bundle,
            bundle / "asset_eval" / f"gt_surface_stl{n}.npz",
            pilot / f"{scan}_vanilla_matched" / "asset" / "projected_points.ply",
            Path(args.sugar_root) / scan / "dtu_native_mesh" / "culled_mesh.ply",
            Path(args.dtu_root) / scan / "cameras.npz",
            Path(args.two_dgs_root) / f"{scan}_official_2dgs" / "asset_eval" / "native_mesh_collision.json",
            Path(args.two_dgs_root) / f"{scan}_official_2dgs" / "train" / "ours_30000" / "fuse_post.ply",
        )}
        scenes.append(row)

    report = {
        "protocol": "collision-vs-GT, tol=1% GT bbox, 50k samples, seed 0",
        "gt": "DTU official stl -> Gaussian frame via cameras.npz scale_mat_inv",
        "methods": {
            "ours": "observation-certified collision_candidate.ply",
            "sugar_culled": "SuGaR DTU-eval culled_mesh (mm->normalized)",
            "poisson": "Poisson-from-3DGS over projected_points (depth9, dens-q0.1, bbox crop)",
            "2dgs_native": "official 2DGS 30k TSDF mesh, shared collision evaluator",
        },
        "scenes": scenes,
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["| scene | method | faces | floater% | coverage@1% | normal° |", "|---|---|---:|---:|---:|---:|"]
    for r in scenes:
        for mth in ("ours", "sugar_culled", "poisson", "2dgs_native"):
            if mth not in r:
                continue
            d = r[mth]
            lines.append(
                f"| {r['scene']} | {mth} | {d['faces']} | {d['floater_fraction']*100:.2f} "
                f"| {d['coverage_at_1pct']*100:.1f}% | {d['normal_median_deg']:.1f} |"
            )
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
