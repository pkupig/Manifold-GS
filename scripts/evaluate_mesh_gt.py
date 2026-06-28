#!/usr/bin/env python3
"""Evaluate sampled mesh surface geometry and basic topology against GT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manifold_gs.gt_metrics import geometry_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def topology(mesh: trimesh.Trimesh) -> dict[str, int | float | bool]:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    if faces.shape[0]:
        adjacency = mesh.face_adjacency
        graph = coo_matrix(
            (np.ones(2 * len(adjacency)),
             (np.concatenate((adjacency[:, 0], adjacency[:, 1])),
              np.concatenate((adjacency[:, 1], adjacency[:, 0])))),
            shape=(len(faces), len(faces)),
        ).tocsr()
        components = int(connected_components(graph, directed=False, return_labels=False))
    else:
        components = 0
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(faces)),
        "components": components,
        "boundary_edges": int(np.sum(counts == 1)),
        "nonmanifold_edges": int(np.sum(counts > 2)),
        "nonmanifold_edge_ratio": float(np.mean(counts > 2)) if len(counts) else 0.0,
        "watertight": bool(mesh.is_watertight),
        "euler_number": int(mesh.euler_number),
        "surface_area": float(mesh.area),
        "unique_edges": int(len(unique_edges)),
    }


def main() -> None:
    args = parse_args()
    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("Input must be a non-empty triangle mesh")
    np.random.seed(args.seed)
    sampled_xyz, face_ids = trimesh.sample.sample_surface(mesh, args.samples)
    sampled_normals = np.asarray(mesh.face_normals[face_ids], dtype=np.float32)
    gt = np.load(args.gt)
    gt_xyz = np.asarray(gt["xyz"], dtype=np.float32)
    gt_normals = np.asarray(gt["normals"], dtype=np.float32)
    bbox_diagonal = float(np.linalg.norm(np.ptp(gt_xyz, axis=0)))
    report = {
        "mesh": str(Path(args.mesh).resolve()),
        "gt": str(Path(args.gt).resolve()),
        "samples": int(args.samples),
        "geometry": geometry_metrics(
            sampled_xyz.astype(np.float32), sampled_normals, gt_xyz, gt_normals, bbox_diagonal
        ),
        "topology": topology(mesh),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
