#!/usr/bin/env python3
"""Package a hybrid asset bundle into a single engine-ready .glb.

Builds a glTF scene with two nodes: ``certified_patches`` (the certified surface,
face-coloured per patch with the same deterministic palette as the grouped OBJ) and
``collision_candidate`` (the observation-certified physics proxy). The result is a
standard .glb that drags straight into Blender / Unity / Godot. CPU (trimesh).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manifold_gs.mesh_io import read_triangle_mesh_ply


def _patch_palette(patch_values: np.ndarray) -> dict[int, np.ndarray]:
    """Deterministic high-contrast RGB per patch (matches write_grouped_obj)."""
    palette = {}
    for p in patch_values:
        hue = (int(p) * 0.61803398875) % 1.0
        palette[int(p)] = np.array([
            0.35 + 0.55 * abs(np.sin(2 * np.pi * hue)),
            0.35 + 0.55 * abs(np.sin(2 * np.pi * (hue + 1 / 3))),
            0.35 + 0.55 * abs(np.sin(2 * np.pi * (hue + 2 / 3))),
        ])
    return palette


def build_scene(bundle: Path, meta: Path) -> trimesh.Scene:
    v, f = read_triangle_mesh_ply(str(bundle / "certified_patches.ply"))
    v = v.astype(np.float64)
    pid = np.asarray(np.load(str(meta))["patch_ids"], np.int64).reshape(-1)
    face_patch = np.array(
        [int(np.bincount(pid[t].clip(min=0)).argmax()) for t in f], dtype=np.int64
    ) if f.size else np.empty((0,), np.int64)
    palette = _patch_palette(np.unique(face_patch))
    face_colors = np.zeros((f.shape[0], 4), np.uint8)
    face_colors[:, 3] = 255
    for i, p in enumerate(face_patch):
        face_colors[i, :3] = np.clip(palette[int(p)] * 255, 0, 255).astype(np.uint8)
    certified = trimesh.Trimesh(vertices=v, faces=f, face_colors=face_colors, process=False)

    scene = trimesh.Scene()
    scene.add_geometry(certified, geom_name="certified_patches", node_name="certified_patches")

    cand = bundle / "collision_candidate.ply"
    if cand.exists():
        cv, cf = read_triangle_mesh_ply(str(cand))
        col = trimesh.Trimesh(vertices=cv.astype(np.float64), faces=cf, process=False)
        col.visual.face_colors = np.array([120, 200, 255, 120], np.uint8)  # translucent proxy
        scene.add_geometry(col, geom_name="collision_candidate", node_name="collision_candidate")
    return scene


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-root", default="/root/autodl-tmp/emgs-real/outputs/dtu_real_pilot_v1")
    ap.add_argument("--scenes", nargs="+", default=["scan24", "scan65", "scan105"])
    args = ap.parse_args()
    pilot = Path(args.pilot_root)
    for scan in args.scenes:
        bundle = pilot / f"{scan}_vanilla_matched" / "hybrid_asset"
        meta = pilot / f"{scan}_vanilla_matched" / "asset" / "patch_mesh_meta.npz"
        scene = build_scene(bundle, meta)
        out = bundle / "asset_glb" / f"{scan}_hybrid_asset.glb"
        out.parent.mkdir(parents=True, exist_ok=True)
        scene.export(str(out))
        # verify round-trip
        back = trimesh.load(str(out))
        geoms = list(back.geometry) if hasattr(back, "geometry") else ["<single>"]
        total_f = sum(int(g.faces.shape[0]) for g in back.geometry.values()) if hasattr(back, "geometry") else int(back.faces.shape[0])
        print(f"{scan}: wrote {out}  ({out.stat().st_size//1024} KB, nodes={geoms}, faces={total_f})")


if __name__ == "__main__":
    main()
