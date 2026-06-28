#!/usr/bin/env python3
"""Compare ManifoldGS diagnostic summary files.

Example:
    python scripts/compare_diagnostics.py \
      --summary vanilla=experiments/synthetic_vanilla_2k_diag/summary.json \
      --summary mcgs=experiments/synthetic_mcgs_2k_diag/summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_METRICS = [
    "num_gaussians",
    "surface_ratio",
    "curve_ratio",
    "volume_ratio",
    "surface_kept",
    "graph_points",
    "thinness_median",
    "r12_median",
    "r23_median",
    "rank2_median",
    "normal_variation_median",
    "log_area_variation_median",
    "curvature_scale_median",
    "opacity_mean",
    "mass_sum",
]


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.parent.name or path.stem, path
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError(f"empty name in {value!r}")
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        type=parse_named_path,
        help="Named summary path, e.g. vanilla=experiments/foo/summary.json",
    )
    parser.add_argument(
        "--metric",
        action="append",
        default=None,
        help="Metric to print. Defaults to the core geometry metrics.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Name of the baseline summary for delta columns. Defaults to the first summary.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): v for k, v in data.items()}


def format_value(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:.1f}"
        if abs(value) >= 1:
            return f"{value:.4f}"
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    args = parse_args()
    metrics = args.metric or DEFAULT_METRICS
    named = [(name, load_summary(path)) for name, path in args.summary]
    names = [name for name, _ in named]
    baseline_name = args.baseline or names[0]
    baseline = dict(named).get(baseline_name)
    if baseline is None:
        raise SystemExit(f"Unknown baseline {baseline_name!r}. Available: {', '.join(names)}")

    header = ["metric", *names]
    if len(named) > 1:
        header.extend([f"delta:{name}-{baseline_name}" for name in names if name != baseline_name])
    rows = [header]

    for metric in metrics:
        row = [metric]
        values = []
        for _, summary in named:
            value = summary.get(metric, "")
            values.append(value)
            row.append(format_value(value) if value != "" else "")
        if len(named) > 1:
            base_value = baseline.get(metric)
            for name, summary in named:
                if name == baseline_name:
                    continue
                value = summary.get(metric)
                if isinstance(value, (int, float)) and isinstance(base_value, (int, float)):
                    row.append(format_value(float(value) - float(base_value)))
                else:
                    row.append("")
        rows.append(row)

    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for idx, row in enumerate(rows):
        print("| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |")
        if idx == 0:
            print("| " + " | ".join("-" * widths[i] for i in range(len(row))) + " |")


if __name__ == "__main__":
    main()

