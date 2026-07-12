#!/usr/bin/env python3
"""Asset-utility demo: edit a semantic part and show certified binding doesn't leak.

Select the certified patches inside a ball (a limb-like part), rigidly rotate them,
and propagate two ways: (a) certified patch binding — moves exactly those patches;
(b) a mesh-free proximity binding — grabs everything within a radius, dragging
adjacent geometry. Reports boundary leakage for each and renders a before/after
figure. CPU / headless (numpy + matplotlib Agg).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manifold_gs.mesh_io import read_triangle_mesh_ply
from manifold_gs.edit_metrics import (
    certified_patch_binding, radius_binding, propagate_edit,
    rigid_deformation, edit_propagation_metrics,
)


def _rot_about(axis, deg):
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    t = np.radians(deg); K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="/root/autodl-tmp/emgs-real/outputs/dtu_real_pilot_v1/scan105_vanilla_matched/hybrid_asset")
    ap.add_argument("--meta", default="/root/autodl-tmp/emgs-real/outputs/dtu_real_pilot_v1/scan105_vanilla_matched/asset/patch_mesh_meta.npz")
    ap.add_argument("--seed-xyz", type=float, nargs=3, default=[0.45, -0.10, 0.43])
    ap.add_argument("--select-radius", type=float, default=0.22)
    ap.add_argument("--rotate-deg", type=float, default=35.0)
    ap.add_argument("--prox-radius-frac", type=float, default=0.05,
                    help="proximity selection radius as a fraction of bbox diagonal")
    ap.add_argument("--out-dir", default="/root/autodl-tmp/emgs-real/outputs/dtu_real_pilot_v1/edit_demo")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    v, f = read_triangle_mesh_ply(str(Path(args.bundle) / "certified_patches.ply"))
    v = v.astype(np.float64)
    pid = np.asarray(np.load(args.meta)["patch_ids"], np.int64).reshape(-1)
    dg = float(np.linalg.norm(np.ptp(v, axis=0)))

    seed = np.asarray(args.seed_xyz)
    pc = {int(p): v[pid == p].mean(0) for p in np.unique(pid)}
    selected = [p for p, c in pc.items() if np.linalg.norm(c - seed) < args.select_radius]
    edit_region = certified_patch_binding(pid, selected)          # the intended semantic part
    pivot = v[edit_region].mean(0)
    deform = rigid_deformation(_rot_about([0, 0, 1], args.rotate_deg), np.zeros(3), pivot=pivot)

    # ground-truth edit: move exactly the selected part
    target = propagate_edit(v, deform, edit_region)
    # (a) certified binding -> moves exactly edit_region
    moved_cert = propagate_edit(v, deform, certified_patch_binding(pid, selected))
    # (b) proximity binding -> grabs everything within radius of the part's points
    seeds = v[edit_region]
    prox_mask = radius_binding(v, seeds, radius=args.prox_radius_frac * dg)
    moved_prox = propagate_edit(v, deform, prox_mask)

    m_cert = edit_propagation_metrics(v, moved_cert, target, edit_region)
    m_prox = edit_propagation_metrics(v, moved_prox, target, edit_region)
    report = {
        "bundle": args.bundle, "selected_patches": sorted(selected),
        "edited_vertices": int(edit_region.sum()), "total_vertices": int(v.shape[0]),
        "rotate_deg": args.rotate_deg, "bbox_diag": dg,
        "certified": {"leaked_point_fraction": m_cert["leaked_point_fraction"],
                      "boundary_leakage_mean_pctbbox": m_cert["boundary_leakage_mean"] / dg * 100},
        "proximity": {"leaked_point_fraction": m_prox["leaked_point_fraction"],
                      "boundary_leakage_mean_pctbbox": m_prox["boundary_leakage_mean"] / dg * 100,
                      "leaked_vertices": int(round(m_prox["leaked_point_fraction"] * (~edit_region).sum()))},
    }
    (out_dir / "edit_propagation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # figure: project to the 2 widest axes
    ax2 = np.argsort(-np.ptp(v, axis=0))[:2]
    disp_c = np.linalg.norm(moved_cert - v, axis=1)
    disp_p = np.linalg.norm(moved_prox - v, axis=1)
    eps = 1e-9
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    for ax, moved, disp, title in [
        (axs[0], moved_cert, disp_c, f"certified binding\nleak {report['certified']['leaked_point_fraction']*100:.2f}%"),
        (axs[1], moved_prox, disp_p, f"proximity binding\nleak {report['proximity']['leaked_point_fraction']*100:.2f}% ({report['proximity']['leaked_vertices']} verts)"),
    ]:
        fixed = ~edit_region & (disp <= eps)
        leaked = ~edit_region & (disp > eps)
        ax.scatter(moved[fixed][:, ax2[0]], moved[fixed][:, ax2[1]], s=3, c="0.75", label="stayed fixed")
        ax.scatter(moved[edit_region][:, ax2[0]], moved[edit_region][:, ax2[1]], s=6, c="royalblue", label="intended edit part")
        if leaked.any():
            ax.scatter(moved[leaked][:, ax2[0]], moved[leaked][:, ax2[1]], s=14, c="crimson", label="LEAKED (should be fixed)")
        ax.set_title(title); ax.set_aspect("equal"); ax.legend(loc="upper right", markerscale=2, fontsize=8)
    fig.suptitle(f"scan105 — rotate a {report['edited_vertices']}-vertex part by {args.rotate_deg:.0f}°  (blue moves; red = collateral leakage)")
    fig.tight_layout(); fig.savefig(out_dir / "edit_before_after.png", dpi=120); plt.close(fig)

    print(json.dumps(report, indent=2))
    print(f"wrote {out_dir}/edit_propagation.json + edit_before_after.png")


if __name__ == "__main__":
    main()
