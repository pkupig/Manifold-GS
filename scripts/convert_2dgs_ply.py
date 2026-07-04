#!/usr/bin/env python3
"""Adapt a 2DGS two-scale PLY for the repository's three-scale evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from numpy.lib import recfunctions

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manifold_gs.ply_io import read_vertex_ply, write_vertex_ply_data


def convert(source: Path, output: Path, normal_scale_ratio: float) -> None:
    if not 0 < normal_scale_ratio < 1:
        raise ValueError("normal_scale_ratio must be in (0, 1)")
    rows = read_vertex_ply(source).data
    names = set(rows.dtype.names or ())
    required = {"scale_0", "scale_1", "rot_0", "rot_1", "rot_2", "rot_3"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"Not a compatible 2DGS PLY; missing properties: {missing}")
    if "scale_2" in names:
        raise ValueError("Input already has scale_2; refusing to overwrite a 3DGS PLY")

    # 2DGS stores log tangent scales. Its renderer appends a unit third axis;
    # the adapter instead appends a small relative thickness for diagnostics.
    log_thickness = (
        0.5 * (rows["scale_0"].astype(np.float64) + rows["scale_1"].astype(np.float64))
        + np.log(normal_scale_ratio)
    ).astype(np.float32)
    adapted = recfunctions.append_fields(
        rows, "scale_2", log_thickness, dtypes="f4", usemask=False
    )
    write_vertex_ply_data(output, adapted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--normal-scale-ratio", type=float, default=1e-3)
    args = parser.parse_args()
    convert(args.input, args.output, args.normal_scale_ratio)
    print(f"Wrote evaluator-compatible 2DGS PLY: {args.output}")
    print("Note: scale_2 is an evaluation adapter; do not report it as a learned 2DGS thickness.")


if __name__ == "__main__":
    main()
