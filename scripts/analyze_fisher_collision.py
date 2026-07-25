#!/usr/bin/env python3
"""Read-only Fisher-vs-GT-collision diagnostic for a hybrid asset bundle."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manifold_gs.collision_metrics import sample_mesh_surface


def read_grouped_obj(path: Path) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Read vertices and `o patch_NNNN` face groups from our grouped OBJ exporter."""
    vertices: list[list[float]] = []
    faces: dict[int, list[list[int]]] = {}
    patch: int | None = None
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append([float(v) for v in fields[1:4]])
        elif fields[0] == "o" and len(fields) == 2 and fields[1].startswith("patch_"):
            patch = int(fields[1].split("_", 1)[1])
        elif fields[0] == "f" and patch is not None:
            face = [int(v.split("/", 1)[0]) for v in fields[1:4]]
            if min(face) < 1:
                raise ValueError("only positive OBJ face indices are supported")
            faces.setdefault(patch, []).append([v - 1 for v in face])
    return np.asarray(vertices, dtype=np.float64), {
        patch: np.asarray(group, dtype=np.int64) for patch, group in faces.items()
    }


def median_or_none(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--fisher", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples-per-patch", type=int, default=2000)
    parser.add_argument("--tolerance-fraction", type=float, default=0.01)
    args = parser.parse_args()
    if args.samples_per_patch < 1 or args.tolerance_fraction <= 0:
        raise ValueError("samples-per-patch and tolerance-fraction must be positive")

    bundle = Path(args.bundle)
    vertices, face_groups = read_grouped_obj(bundle / "certified_patches.obj")
    mapping = np.load(bundle / "asset_mapping.npz")
    collision_patches = set(map(int, np.asarray(mapping["collision_patch_ids"], np.int64)))
    fisher = json.loads(Path(args.fisher).read_text())
    records = {int(r["patch_id"]): r for r in fisher["records"]}
    gt = np.load(args.gt)
    gt_xyz = np.asarray(gt["xyz"], dtype=np.float64)
    bbox = float(np.linalg.norm(np.ptp(gt_xyz, axis=0)))
    tolerance = args.tolerance_fraction * bbox
    tree = cKDTree(gt_xyz)

    rows = []
    for patch in sorted(collision_patches & set(records) & set(face_groups)):
        points, _, area = sample_mesh_surface(
            vertices, face_groups[patch], args.samples_per_patch, seed=patch
        )
        distances, _ = tree.query(points, k=1)
        false_fraction = float(np.mean(distances > tolerance))
        label = "floater" if false_fraction > 0.90 else "clean" if false_fraction < 0.10 else "ambiguous"
        record = records[patch]
        rows.append({
            "patch_id": patch, "gt_label": label,
            "false_surface_fraction": false_fraction, "patch_area": area,
            "fisher": record.get("fisher"), "fisher_status": record["status"],
            "views": record["views"], "pixels": record["pixels"],
        })
    by_label = {}
    for label in ("floater", "clean", "ambiguous"):
        part = [r for r in rows if r["gt_label"] == label]
        status = Counter(r["fisher_status"] for r in part)
        values = [float(r["fisher"]) for r in part if r["fisher"] is not None]
        by_label[label] = {
            "patches": len(part), "patch_area": float(sum(r["patch_area"] for r in part)),
            "fisher_status": dict(sorted(status.items())), "fisher_median": median_or_none(values),
        }
    report = {
        "protocol_version": fisher["protocol_version"], "diagnostic": "read-only collision GT comparison; does not alter Fisher threshold or status",
        "bbox_diagonal": bbox, "tolerance_fraction": args.tolerance_fraction,
        "tolerance": tolerance, "samples_per_patch": args.samples_per_patch,
        "collision_patches_with_fisher": len(rows), "by_gt_label": by_label, "records": rows,
    }
    output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": str(output), "by_gt_label": by_label}, indent=2))


if __name__ == "__main__":
    main()
