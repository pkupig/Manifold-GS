#!/usr/bin/env python3
"""One command for the downstream asset benchmark (A5: P0.3/P0.4/P0.5).

Runs the edit-propagation (P0.3) and texture round-trip (P0.5) tracks on an exported
asset bundle, plus the collision track (P0.4) when a GT surface is supplied, applies
the frozen PASS/FAIL protocol in :mod:`manifold_gs.asset_benchmark`, writes a single
``asset_benchmark.json``, prints a compact table, and exits non-zero on FAIL.

Edit and texture need no external input beyond the bundle, so the minimal invocation is

    python scripts/run_asset_benchmark.py --bundle <bundle_dir>

Collision requires a GT surface npz (``xyz`` + ``normals``) aligned to the Gaussian
frame; pass ``--gt`` (and optionally ``--probes``) to enable it. Everything is CPU-only
and deterministic given the seed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_edit_propagation import evaluate_edit
from evaluate_texture_roundtrip import evaluate_texture
from evaluate_collision_candidate import evaluate_collision
from manifold_gs.asset_benchmark import (
    PROTOCOL_VERSION,
    THRESHOLDS,
    collision_verdict,
    edit_verdict,
    summarize,
    texture_verdict,
)


def _fmt(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value != value:  # nan
            return "nan"
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt(v) for v in value)
    return str(value)


def _render_table(verdicts: list[dict]) -> str:
    icon = {"pass": "PASS", "fail": "FAIL", "skipped": "skip", "uninformative": "uninform"}
    lines = ["| track | status | headline |", "| --- | --- | --- |"]
    for v in verdicts:
        headline = "; ".join(f"{k}={_fmt(val)}" for k, val in v["headline"].items())
        lines.append(f"| {v['track']} | {icon.get(v['status'], v['status'])} | {headline} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, help="asset bundle directory")
    parser.add_argument("--out", default=None,
                        help="output JSON (default: <bundle>/asset_eval/asset_benchmark.json)")
    # Edit knobs (P0.3)
    parser.add_argument("--patches", type=int, nargs="*", default=None,
                        help="patch ids to edit (default: the largest attached patch)")
    parser.add_argument("--translate-frac", type=float, default=0.1)
    parser.add_argument("--rotate-deg", type=float, default=0.0)
    parser.add_argument("--radius-frac", type=float, default=0.05)
    # Texture knobs (P0.5)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--min-patch-samples", type=int, default=8)
    parser.add_argument("--boundary-radius-frac", type=float, default=0.02)
    parser.add_argument("--evidence", default=None,
                        help="observation evidence npz; use photometric_mean_color instead of SH DC")
    # Collision knobs (P0.4) — only run when --gt is given
    parser.add_argument("--gt", default=None, help="GT surface npz with xyz/normals (enables P0.4)")
    parser.add_argument("--probes", default=None, help="optional probe npz for collision confusion")
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance-fraction", type=float, default=0.01)
    args = parser.parse_args()

    bundle = Path(args.bundle)
    out = Path(args.out) if args.out else bundle / "asset_eval" / "asset_benchmark.json"

    edit_report = evaluate_edit(
        bundle, patches=args.patches, translate_frac=args.translate_frac,
        rotate_deg=args.rotate_deg, radius_frac=args.radius_frac,
    )
    texture_report = evaluate_texture(
        bundle, resolution=args.resolution, min_patch_samples=args.min_patch_samples,
        boundary_radius_frac=args.boundary_radius_frac, evidence=args.evidence,
    )
    collision_report = None
    if args.gt is not None:
        candidate = bundle / "collision_candidate.ply"
        collision_report = evaluate_collision(
            candidate, Path(args.gt), samples=args.samples, seed=args.seed,
            tolerance_fraction=args.tolerance_fraction,
            probes=Path(args.probes) if args.probes is not None else None,
        )

    verdicts = [
        edit_verdict(edit_report),
        texture_verdict(texture_report),
        collision_verdict(collision_report),
    ]
    rollup = summarize(verdicts)

    document = {
        "protocol_version": PROTOCOL_VERSION,
        "thresholds": THRESHOLDS,
        "bundle": str(bundle.resolve()),
        "overall_status": rollup["overall_status"],
        "track_status": rollup["track_status"],
        "verdicts": verdicts,
        "reports": {
            "edit": edit_report,
            "texture": {k: v for k, v in texture_report.items() if k != "per_patch"},
            "collision": collision_report,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")

    print(_render_table(verdicts))
    print(f"\nprotocol: {PROTOCOL_VERSION}")
    print(f"overall: {rollup['overall_status'].upper()}  ->  {out}")
    return 0 if rollup["overall_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
