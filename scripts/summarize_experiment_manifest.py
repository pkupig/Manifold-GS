#!/usr/bin/env python3
"""Summarize paired manifest runs with confidence intervals and decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]


def interval(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    if len(array) < 2:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean, "n": len(array)}
    radius = float(t.ppf(0.975, len(array) - 1) * np.std(array, ddof=1) / np.sqrt(len(array)))
    return {"mean": mean, "ci95_low": mean - radius, "ci95_high": mean + radius, "n": len(array)}


def lower_bound_decision(stats: dict[str, float], threshold: float) -> str:
    if stats["ci95_low"] >= threshold:
        return "PASS"
    if stats["ci95_high"] < threshold:
        return "FAIL"
    return "INCONCLUSIVE"


def upper_bound_decision(stats: dict[str, float], threshold: float) -> str:
    if stats["ci95_high"] <= threshold:
        return "PASS"
    if stats["ci95_low"] > threshold:
        return "FAIL"
    return "INCONCLUSIVE"


def load_metrics(path: Path, geometry_subset: str) -> dict[str, float]:
    geometry_report = json.loads((path / "geometry_metrics.json").read_text(encoding="utf-8"))
    geometry = geometry_report[geometry_subset]
    rendering = json.loads((path / "heldout_metrics.json").read_text(encoding="utf-8"))
    compatibility = json.loads((path / "fundamental" / "summary.json").read_text(encoding="utf-8"))
    mesh = json.loads((path / "asset" / "mesh_metrics.json").read_text(encoding="utf-8"))
    return {
        "chamfer_l1": geometry["chamfer_l1"],
        "normal_accuracy_median_deg": geometry["normal_accuracy_median_deg"],
        "normalized_kernel_varifold": geometry["normalized_kernel_varifold"],
        "psnr": rendering["mean_psnr"],
        "ssim": rendering["mean_ssim"],
        "symmetry": compatibility["symmetry_residual_median"],
        "chart_accepted_mass_fraction": geometry_report["certified_chart_mass_fraction_of_opaque"],
        "mesh_chamfer_l1": mesh["geometry"]["chamfer_l1"],
        "mesh_normal_accuracy_median_deg": mesh["geometry"]["normal_accuracy_median_deg"],
        "mesh_components": mesh["topology"]["components"],
        "mesh_boundary_edges": mesh["topology"]["boundary_edges"],
        "mesh_nonmanifold_edges": mesh["topology"]["nonmanifold_edges"],
    }


def compute_checks(rows: list[dict], comparison: dict) -> tuple[dict, str]:
    """Compute paired decisions for one statistically coherent row group."""
    baseline = comparison["baseline"]
    matched = comparison["matched_geometry_baseline"]
    candidate = comparison["candidate"]
    checks = {}
    for metric, threshold in comparison["minimum_relative_improvement"].items():
        values = [(row[baseline][metric] - row[candidate][metric]) / row[baseline][metric] for row in rows]
        stats = interval(values)
        checks[f"candidate_vs_baseline/{metric}"] = {
            **stats, "threshold": threshold, "decision": lower_bound_decision(stats, threshold)
        }
    for metric, threshold in comparison.get("asset_minimum_relative_improvement", {}).items():
        values = [(row[baseline][metric] - row[candidate][metric]) / row[baseline][metric] for row in rows]
        stats = interval(values)
        checks[f"asset_candidate_vs_baseline/{metric}"] = {
            **stats, "threshold": threshold, "decision": lower_bound_decision(stats, threshold)
        }
    for metric, limit in (("psnr", comparison["guardrails"]["maximum_mean_psnr_drop_db"]),
                          ("ssim", comparison["guardrails"]["maximum_mean_ssim_drop"])):
        values = [row[baseline][metric] - row[candidate][metric] for row in rows]
        stats = interval(values)
        checks[f"guardrail/{metric}_drop"] = {
            **stats, "threshold": limit, "decision": upper_bound_decision(stats, limit)
        }

    incremental = comparison["incremental_vs_matched_baseline"]
    values = [(row[matched]["symmetry"] - row[candidate]["symmetry"]) / row[matched]["symmetry"] for row in rows]
    stats = interval(values)
    checks["candidate_vs_tangent/symmetry"] = {
        **stats, "threshold": incremental["minimum_symmetry_improvement"],
        "decision": lower_bound_decision(stats, incremental["minimum_symmetry_improvement"]),
    }
    for metric in comparison["minimum_relative_improvement"]:
        values = [(row[candidate][metric] - row[matched][metric]) / row[matched][metric] for row in rows]
        stats = interval(values)
        checks[f"candidate_vs_tangent/{metric}_regression"] = {
            **stats, "threshold": incremental["maximum_primary_metric_regression"],
            "decision": upper_bound_decision(stats, incremental["maximum_primary_metric_regression"]),
        }
    values = [
        row[matched]["chart_accepted_mass_fraction"] - row[candidate]["chart_accepted_mass_fraction"]
        for row in rows
    ]
    stats = interval(values)
    checks["candidate_vs_tangent/chart_accepted_mass_drop"] = {
        **stats, "threshold": incremental["maximum_chart_accepted_mass_drop"],
        "decision": upper_bound_decision(stats, incremental["maximum_chart_accepted_mass_drop"]),
    }
    decisions = [check["decision"] for check in checks.values()]
    status = "FAIL" if "FAIL" in decisions else "INCONCLUSIVE" if "INCONCLUSIVE" in decisions else "PASS"
    return checks, status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", help="Defaults to the benchmark summary.json")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    benchmark = ROOT / "experiments" / "benchmarks" / manifest["name"]
    run_root = benchmark / "runs"
    comparison = manifest["comparison"]
    baseline = comparison["baseline"]
    matched = comparison["matched_geometry_baseline"]
    candidate = comparison["candidate"]
    geometry_subset = comparison.get("geometry_subset", "all_opaque")

    rows = []
    missing = []
    for scene in manifest["scenes"]:
        for seed in manifest["seeds"]:
            row = {"scene": scene, "seed": seed}
            for method in (baseline, matched, candidate):
                path = run_root / f"{scene}_s{seed}_{method}"
                try:
                    row[method] = load_metrics(path, geometry_subset)
                except FileNotFoundError as error:
                    missing.append(str(error.filename))
            rows.append(row)
    if missing:
        report = {"status": "INCOMPLETE", "missing": sorted(set(missing)), "rows": rows}
    else:
        checks, status = compute_checks(rows, comparison)
        scene_results = {}
        if len(manifest["scenes"]) > 1:
            for scene in manifest["scenes"]:
                scene_rows = [row for row in rows if row["scene"] == scene]
                scene_checks, scene_status = compute_checks(scene_rows, comparison)
                scene_results[scene] = {"status": scene_status, "checks": scene_checks}
        report = {"status": status, "checks": checks, "scene_results": scene_results, "rows": rows}

    out = Path(args.out) if args.out else benchmark / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
