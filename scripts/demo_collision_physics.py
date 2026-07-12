#!/usr/bin/env python3
"""Physics-utility demo: does a collision mesh block only where the object is?

For each candidate mesh we (1) score a *phantom-collision rate* — the fraction of
free-space probes (points far from the true GT surface) that nonetheless lie within
contact tolerance of the mesh, i.e. where a rigid body would bump into empty space —
and (2) render a cross-section so the phantom surface is visible. CPU / headless
(numpy + scipy + matplotlib Agg). Compares ours vs Poisson vs SuGaR on DTU scenes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manifold_gs.mesh_io import read_triangle_mesh_ply
from manifold_gs.collision_metrics import sample_mesh_surface

try:
    import open3d as o3d
except Exception:  # pragma: no cover
    o3d = None


def _load_mesh(path: Path, cameras_npz: Path | None = None):
    """Return (vertices, faces). Applies scale_mat_inv (mm->Gaussian) when given."""
    if path.suffix == ".ply" and cameras_npz is None:
        try:
            return read_triangle_mesh_ply(str(path))
        except Exception:
            pass
    mesh = o3d.io.read_triangle_mesh(str(path))
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.triangles)
    if cameras_npz is not None:
        s_inv = np.load(str(cameras_npz))["scale_mat_inv_0"]
        v = (np.c_[v, np.ones(len(v))] @ s_inv.T)[:, :3]
    return v, f


def phantom_collision(cand_points, gt_tree, free, probes, tol):
    """Fraction of free-space probes (far from GT) within tol of the candidate surface."""
    d_cand = cKDTree(cand_points).query(probes, workers=-1)[0]
    blocked = d_cand <= tol                      # a probe here would register contact
    free_blocked = free & blocked
    return {
        "free_probes": int(free.sum()),
        "phantom_collision_rate": float(free_blocked.sum() / max(free.sum(), 1)),
    }, free, blocked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-root", default="/root/autodl-tmp/emgs-real/outputs/dtu_real_pilot_v1")
    ap.add_argument("--sugar-root", default="/root/autodl-tmp/emgs-real/outputs/sugar_dtu_pilot_v1")
    ap.add_argument("--dtu-root", default="/root/autodl-tmp/emgs-real/dtu-preprocessed/DTU")
    ap.add_argument("--scenes", nargs="+", default=["scan24", "scan65", "scan105"])
    ap.add_argument("--out-dir", default="/root/autodl-tmp/emgs-real/outputs/dtu_real_pilot_v1/physics_demo")
    ap.add_argument("--n-probes", type=int, default=200000)
    ap.add_argument("--samples", type=int, default=80000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pilot = Path(args.pilot_root)
    rng = np.random.default_rng(0)
    report = []

    for scan in args.scenes:
        n = scan.replace("scan", "").zfill(3)
        B = pilot / f"{scan}_vanilla_matched" / "hybrid_asset"
        gt = np.load(B / "asset_eval" / f"gt_surface_stl{n}.npz")["xyz"].astype(np.float64)
        dg = float(np.linalg.norm(np.ptp(gt, axis=0)))
        tol = 0.01 * dg

        meshes = {
            "ours": _load_mesh(B / "collision_candidate.ply"),
            "poisson": _load_mesh(B / "baselines" / "poisson_fair.ply"),
            "sugar": _load_mesh(Path(args.sugar_root) / scan / "dtu_native_mesh" / "culled_mesh.ply",
                                Path(args.dtu_root) / scan / "cameras.npz"),
        }
        # probes: uniform in a box around the object (union of ours+GT extent)
        allv = np.vstack([meshes["ours"][0], gt])
        lo, hi = allv.min(0) - tol, allv.max(0) + tol
        probes = rng.uniform(lo, hi, size=(args.n_probes, 3))
        gt_tree = cKDTree(gt)
        free = gt_tree.query(probes, workers=-1)[0] > 2 * tol   # genuinely empty space

        row = {"scene": scan, "tol_pctbbox": 1.0}
        surf = {}
        for name, (v, f) in meshes.items():
            pts, _, _ = sample_mesh_surface(v.astype(np.float64), f, args.samples, seed=0)
            surf[name] = pts
            stats, _, _ = phantom_collision(pts, gt_tree, free, probes, tol)
            row[name] = stats
        report.append(row)

        # cross-section figure: thinnest bbox axis as the slab normal
        ax_slab = int(np.argmin(np.ptp(gt, axis=0)))
        axes2 = [i for i in range(3) if i != ax_slab]
        c = gt[:, ax_slab].mean(); half = 0.03 * dg
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        for ax, name in zip(axs, ["ours", "poisson", "sugar"]):
            gmask = np.abs(gt[:, ax_slab] - c) < half
            ax.scatter(gt[gmask][:, axes2[0]], gt[gmask][:, axes2[1]], s=1, c="0.7", label="GT surface")
            p = surf[name]; pmask = np.abs(p[:, ax_slab] - c) < half
            ax.scatter(p[pmask][:, axes2[0]], p[pmask][:, axes2[1]], s=2, c="crimson", label=f"{name} surface")
            ax.set_title(f"{scan} — {name}\nphantom-collision {row[name]['phantom_collision_rate']*100:.1f}%")
            ax.set_aspect("equal"); ax.legend(loc="upper right", markerscale=4, fontsize=8)
        fig.suptitle(f"Cross-section slab (GT grey vs candidate red); red beyond grey = phantom collision surface")
        fig.tight_layout()
        fig.savefig(out_dir / f"{scan}_cross_section.png", dpi=110)
        plt.close(fig)

    (out_dir / "phantom_collision.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out_dir}/phantom_collision.json + <scene>_cross_section.png")
    print(f"{'scene':8} {'ours':>8} {'sugar':>8} {'poisson':>8}   (phantom-collision rate, lower=better)")
    for r in report:
        print(f"{r['scene']:8} {r['ours']['phantom_collision_rate']*100:7.2f}% "
              f"{r['sugar']['phantom_collision_rate']*100:7.2f}% {r['poisson']['phantom_collision_rate']*100:7.2f}%")


if __name__ == "__main__":
    main()
