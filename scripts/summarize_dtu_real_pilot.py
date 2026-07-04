#!/usr/bin/env python3
"""Summarize official DTU geometry and held-out rendering metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ply_vertex_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        for raw_line in handle:
            line = raw_line.decode("ascii", errors="strict").strip()
            if line.startswith("element vertex "):
                return int(line.rsplit(" ", 1)[1])
            if line == "end_header":
                break
    return None


def paired_delta(control: dict, method: dict) -> dict:
    return {
        "relative_accuracy_improvement_percent":
            100.0 * (control["accuracy"] - method["accuracy"]) / control["accuracy"],
        "relative_completeness_improvement_percent":
            100.0 * (control["completeness"] - method["completeness"])
            / control["completeness"],
        "relative_overall_improvement_percent":
            100.0 * (control["overall"] - method["overall"]) / control["overall"],
        "psnr_delta_db": method["psnr"] - control["psnr"],
        "ssim_delta": method["ssim"] - control["ssim"],
        "relative_primitive_count_difference_percent":
            100.0 * abs(method["gaussians"] - control["gaussians"])
            / control["gaussians"],
    }


def decide(pairs: list[dict], rule: dict) -> dict:
    required = rule["required_scans"]
    result = {
        "status": "INCOMPLETE",
        "completed_pairs": len(pairs),
        "required_pairs": required,
    }
    if not pairs:
        return result
    result.update({
        "mean_relative_overall_improvement_percent": mean(
            pair["relative_overall_improvement_percent"] for pair in pairs
        ),
        "positive_overall_scans": sum(
            pair["relative_overall_improvement_percent"] > 0 for pair in pairs
        ),
        "worst_psnr_delta_db": min(pair["psnr_delta_db"] for pair in pairs),
        "worst_relative_primitive_count_difference_percent": max(
            pair["relative_primitive_count_difference_percent"] for pair in pairs
        ),
    })
    if len(pairs) < required:
        return result
    checks = {
        "mean_overall": result["mean_relative_overall_improvement_percent"]
            >= rule["minimum_mean_relative_overall_improvement_percent"],
        "positive_scans": result["positive_overall_scans"]
            >= rule["minimum_positive_overall_scans"],
        "psnr_guardrail": result["worst_psnr_delta_db"]
            >= rule["minimum_psnr_delta_db_per_scan"],
        "primitive_count_guardrail":
            result["worst_relative_primitive_count_difference_percent"]
            <= rule["maximum_relative_primitive_count_difference_percent"],
    }
    result["checks"] = checks
    result["status"] = "PASS" if all(checks.values()) else "FAIL"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", type=Path,
        default=ROOT / "experiments/protocols/dtu_real_pilot_v1.json",
    )
    parser.add_argument("--scan", type=int, action="append")
    args = parser.parse_args()
    protocol = load(args.protocol)
    scans = args.scan or protocol["scans"]
    root = Path(protocol["output_root"]) / protocol["name"]
    summary = {
        "protocol": protocol["name"],
        "scans": {},
        "paired_comparisons": {},
        "posthoc_diagnostics": {},
    }
    pairs = []
    anchor_replication_pairs = []
    anchor_replication_scans = set(
        protocol.get("anchor_replication_rule", {}).get("replication_scans", [])
    )
    for scan in scans:
        scene = {}
        for method in (
            "vanilla", "vanilla_matched", "manifold_full", "manifold_colmap_anchor",
        ):
            output = root / f"scan{scan}_{method}"
            geometry_path = output / "dtu_evaluation/results.json"
            rendering_path = output / "heldout_metrics.json"
            if not geometry_path.exists() or not rendering_path.exists():
                continue
            geometry = load(geometry_path)
            rendering = load(rendering_path)
            scene[method] = {
                "gaussians": ply_vertex_count(
                    output / "point_cloud/iteration_7000/point_cloud.ply"
                ),
                "accuracy": geometry["mean_d2s"],
                "completeness": geometry["mean_s2d"],
                "overall": geometry["overall"],
                "psnr": rendering["mean_psnr"],
                "ssim": rendering["mean_ssim"],
            }
        summary["scans"][f"scan{scan}"] = scene
        for method, values in scene.items():
            print(
                f"scan{scan} {method}: {values['gaussians']} Gaussians | "
                f"DTU {values['overall']:.4f} "
                f"(acc {values['accuracy']:.4f}, comp {values['completeness']:.4f}) | "
                f"PSNR {values['psnr']:.3f} SSIM {values['ssim']:.3f}"
            )
        if "vanilla_matched" in scene and "manifold_full" in scene:
            pair = paired_delta(scene["vanilla_matched"], scene["manifold_full"])
            summary["paired_comparisons"][f"scan{scan}"] = pair
            pairs.append(pair)
            print(
                f"  paired: overall {pair['relative_overall_improvement_percent']:+.3f}% | "
                f"PSNR {pair['psnr_delta_db']:+.3f} dB | "
                f"points {pair['relative_primitive_count_difference_percent']:.2f}% apart"
            )
        if "manifold_colmap_anchor" in scene:
            anchor_comparisons = {}
            for control_name in ("vanilla_matched", "manifold_full"):
                if control_name in scene:
                    anchor_comparisons[f"vs_{control_name}"] = paired_delta(
                        scene[control_name], scene["manifold_colmap_anchor"]
                    )
            summary["posthoc_diagnostics"][f"scan{scan}_colmap_anchor"] = anchor_comparisons
            if scan in anchor_replication_scans and "vs_vanilla_matched" in anchor_comparisons:
                anchor_replication_pairs.append(anchor_comparisons["vs_vanilla_matched"])
            for control_name, values in anchor_comparisons.items():
                print(
                    f"  posthoc anchor {control_name}: "
                    f"overall {values['relative_overall_improvement_percent']:+.3f}% | "
                    f"completeness {values['relative_completeness_improvement_percent']:+.3f}% | "
                    f"PSNR {values['psnr_delta_db']:+.3f} dB"
                )
    rule = protocol["prospective_decision_rule"]
    summary["prospective_decision_rule"] = rule
    summary["decision"] = decide(pairs, rule)
    print(
        f"Frozen RGB-only decision: {summary['decision']['status']} "
        f"({summary['decision']['completed_pairs']}/{summary['decision']['required_pairs']} pairs)"
    )
    if summary["posthoc_diagnostics"]:
        print("Posthoc diagnostics are reported separately and do not alter the frozen decision.")
    if "anchor_replication_rule" in protocol:
        anchor_rule = protocol["anchor_replication_rule"]
        summary["anchor_replication_rule"] = anchor_rule
        summary["anchor_replication_decision"] = decide(
            anchor_replication_pairs, anchor_rule
        )
        anchor_decision = summary["anchor_replication_decision"]
        print(
            f"Anchor replication decision: {anchor_decision['status']} "
            f"({anchor_decision['completed_pairs']}/{anchor_decision['required_pairs']} pairs)"
        )
    out = root / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
