#!/usr/bin/env python3
"""Summarize a targeted sweep against its pre-registered reference run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(run: Path) -> dict[str, float]:
    geometry_report = load(run / "geometry_metrics.json")
    geometry = geometry_report["certified_charts"]
    rendering = load(run / "heldout_metrics.json")
    return {
        "chamfer_l1": geometry["chamfer_l1"],
        "normal_median_deg": geometry["normal_accuracy_median_deg"],
        "normalized_kernel_varifold": geometry["normalized_kernel_varifold"],
        "certified_mass_fraction": geometry_report["certified_chart_mass_fraction_of_opaque"],
        "psnr": rendering["mean_psnr"],
        "ssim": rendering["mean_ssim"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = load(args.manifest.resolve())
    benchmark = ROOT / "experiments/benchmarks" / manifest["name"]
    thresholds = manifest["comparison"]["acceptance"]
    reference_template = manifest["comparison"].get("reference_run_template")
    fixed_reference = manifest["comparison"].get("reference_run")
    summary = {"references": {}, "methods": {}}

    for scene in manifest["scenes"]:
        for seed in manifest["seeds"]:
            reference_run = (
                reference_template.format(scene=scene, seed=seed)
                if reference_template else fixed_reference
            )
            if reference_run is None:
                raise ValueError("comparison requires reference_run or reference_run_template")
            baseline = metrics(ROOT / "experiments/benchmarks" / reference_run)
            summary["references"][f"{scene}_s{seed}"] = baseline
            for method in manifest["methods"]:
                key = f"{scene}_s{seed}_{method}"
                current = metrics(benchmark / "runs" / key)
                chamfer_gain = (baseline["chamfer_l1"] - current["chamfer_l1"]) / baseline["chamfer_l1"]
                normal_gain = (baseline["normal_median_deg"] - current["normal_median_deg"]) / baseline["normal_median_deg"]
                checks = {
                    "psnr_guardrail": current["psnr"] >= baseline["psnr"] - thresholds["maximum_psnr_drop_db"],
                    "ssim_guardrail": current["ssim"] >= baseline["ssim"] - thresholds["maximum_ssim_drop"],
                    "chamfer_improvement": chamfer_gain >= thresholds["minimum_chamfer_improvement"],
                    "normal_improvement": normal_gain >= thresholds["minimum_normal_improvement"],
                }
                geometry_pass = checks["chamfer_improvement"] and checks["normal_improvement"]
                rendering_pass = checks["psnr_guardrail"] and checks["ssim_guardrail"]
                if geometry_pass and rendering_pass:
                    decision = "PASS"
                elif geometry_pass:
                    decision = "TRADEOFF"
                else:
                    decision = "FAIL"
                summary["methods"][key] = {
                    "metrics": current, "relative_chamfer_improvement": chamfer_gain,
                    "relative_normal_improvement": normal_gain, "checks": checks,
                    "decision": decision,
                }

    aggregates = {}
    for method in manifest["methods"]:
        selected = [value for key, value in summary["methods"].items() if key.endswith(f"_{method}")]
        aggregates[method] = {
            "runs": len(selected),
            "decision_counts": {
                decision: sum(item["decision"] == decision for item in selected)
                for decision in ("PASS", "TRADEOFF", "FAIL")
            },
            "mean_relative_chamfer_improvement": mean(
                item["relative_chamfer_improvement"] for item in selected
            ),
            "mean_relative_normal_improvement": mean(
                item["relative_normal_improvement"] for item in selected
            ),
            "mean_psnr": mean(item["metrics"]["psnr"] for item in selected),
            "mean_ssim": mean(item["metrics"]["ssim"] for item in selected),
        }
    summary["aggregates"] = aggregates
    out = benchmark / "targeted_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for method, result in summary["methods"].items():
        values = result["metrics"]
        print(
            f"{method}: {result['decision']} | PSNR {values['psnr']:.3f} | "
            f"SSIM {values['ssim']:.3f} | Chamfer {values['chamfer_l1']:.5f} | "
            f"Normal {values['normal_median_deg']:.2f} deg"
        )
    for method, aggregate in aggregates.items():
        print(f"aggregate {method}: {aggregate}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
