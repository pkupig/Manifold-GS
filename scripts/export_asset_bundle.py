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
    parser.add_argument("--observation-evidence", default=None)
    parser.add_argument("--min-observation-supported-fraction", type=float, default=0.5)
    parser.add_argument("--min-observation-training-views", type=int, default=0)
    parser.add_argument("--min-observation-parallax-deg", type=float, default=0.0)
    parser.add_argument("--min-observation-first-hit-views", type=int, default=0)
    parser.add_argument("--max-observation-photometric-std", type=float, default=float("inf"))
    parser.add_argument(
        "--max-observation-photometric-std-percentile", type=float, default=None,
        help="per-scene relative gate: reject patches above this percentile of the scene's "
             "finite per-patch median photometric std (e.g. 90 rejects the worst 10%%)",
    )
    parser.add_argument("--min-observation-photometric-views", type=int, default=0)
    args = parser.parse_args()
    manifest = export_asset_bundle(
        args.gaussians,
        args.mesh,
        args.meta,
        args.out,
        args.collision_min_faces,
        args.source_map,
        args.collision_max_patch_diameter_ratio,
        args.observation_evidence,
        args.min_observation_supported_fraction,
        args.min_observation_training_views,
        args.min_observation_parallax_deg,
        args.min_observation_first_hit_views,
        args.max_observation_photometric_std,
        max_observation_photometric_std_percentile=args.max_observation_photometric_std_percentile,
        min_observation_photometric_views=args.min_observation_photometric_views,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
