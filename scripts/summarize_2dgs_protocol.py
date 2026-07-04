#!/usr/bin/env python3
"""Summarize completed 2DGS protocol outputs and detect collapsed runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("chamfer_l1", "normal_accuracy_median_deg", "normalized_kernel_varifold")


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "experiments/external/2dgs_plane_torus/official_30k",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--collapse-min-points", type=int, default=100)
    args = parser.parse_args()
    out = args.out or args.root / "summary.json"

    runs: dict[str, dict[str, object]] = {}
    for run_dir in sorted(path for path in args.root.iterdir() if path.is_dir()):
        geometry_path = run_dir / "evaluation" / "geometry_metrics.json"
        rendering_path = run_dir / "results.json"
        if not geometry_path.is_file() or not rendering_path.is_file():
            continue
        geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
        rendering = next(iter(json.loads(rendering_path.read_text(encoding="utf-8")).values()))
        all_opaque = geometry["all_opaque"]
        certified = geometry["certified_charts"]
        point_count = int(all_opaque.get("num_estimated", 0))
        runs[run_dir.name] = {
            "scene": run_dir.name.rsplit("_s", 1)[0],
            "collapsed": point_count < args.collapse_min_points,
            "opaque_points": point_count,
            "certified_points": int(certified.get("num_estimated", 0)),
            "certified_mass_fraction": geometry["certified_chart_mass_fraction_of_opaque"],
            "all_opaque": {key: all_opaque.get(key) for key in METRICS},
            "certified_charts": {key: certified.get(key) for key in METRICS},
            "rendering": {key.lower(): rendering[key] for key in ("PSNR", "SSIM", "LPIPS")},
        }

    scene_means: dict[str, dict[str, object]] = {}
    for scene in sorted({str(run["scene"]) for run in runs.values()}):
        selected = [run for run in runs.values() if run["scene"] == scene]
        scene_means[scene] = {
            "runs": len(selected),
            "collapsed_runs": sum(bool(run["collapsed"]) for run in selected),
            "opaque_points": mean([float(run["opaque_points"]) for run in selected]),
            "certified_mass_fraction": mean(
                [float(run["certified_mass_fraction"]) for run in selected]
            ),
            "all_opaque": {
                key: mean([float(run["all_opaque"][key]) for run in selected
                           if run["all_opaque"][key] is not None])
                for key in METRICS
            },
            "certified_charts": {
                key: mean([float(run["certified_charts"][key]) for run in selected
                           if run["certified_charts"][key] is not None])
                for key in METRICS
            },
            "rendering": {
                key: mean([float(run["rendering"][key]) for run in selected])
                for key in ("psnr", "ssim", "lpips")
            },
        }

    report = {
        "method": "2D Gaussian Splatting official 30k",
        "collapse_min_points": args.collapse_min_points,
        "runs": runs,
        "scene_means": scene_means,
        "valid_for_headline": all(not run["collapsed"] for run in runs.values()),
        "note": "Collapsed runs remain in the report and must not be silently excluded.",
    }
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
