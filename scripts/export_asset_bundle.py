#!/usr/bin/env python3
"""Package certified patches and Gaussian layers as a traceable hybrid asset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manifold_gs.asset_bundle import export_asset_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaussians", required=True, help="Full-attribute source/projected 3DGS PLY")
    parser.add_argument("--mesh", required=True, help="Certified patch mesh PLY")
    parser.add_argument("--meta", required=True, help="patch_mesh_meta.npz")
    parser.add_argument(
        "--source-map",
        default=None,
        help="projected_manifold.npz containing original source_indices for a filtered projected PLY",
    )
    parser.add_argument("--out", required=True, help="Output asset bundle directory")
    parser.add_argument("--collision-min-faces", type=int, default=8)
    parser.add_argument("--collision-max-patch-diameter-ratio", type=float, default=3.0)
    args = parser.parse_args()
    manifest = export_asset_bundle(
        args.gaussians,
        args.mesh,
        args.meta,
        args.out,
        args.collision_min_faces,
        args.source_map,
        args.collision_max_patch_diameter_ratio,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
