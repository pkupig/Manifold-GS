#!/usr/bin/env python3
"""Aggregate per-scene ``asset_benchmark.json`` into the paper asset-utility table.

Point it at one or more scenes (either an ``asset_benchmark.json`` file, or a bundle
directory / ``asset_eval`` directory containing one) and it emits a single Markdown
table plus a CSV with one row per scene, covering the three asset-utility tracks and
the overall PASS/FAIL. It reads only the frozen benchmark JSON, so it never re-runs any
metric and stays deterministic.

    python scripts/summarize_asset_benchmark.py \
        outputs/dtu_real_pilot_v1/scan105_vanilla_matched/hybrid_asset \
        outputs/dtu_real_pilot_v1/scan24_vanilla_matched/hybrid_asset \
        --markdown asset_table.md --csv asset_table.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manifold_gs.asset_benchmark import PROTOCOL_VERSION

# (column header, verdict track, headline key). Missing values render as "-", so a run
# without GT still produces a clean table (the collision cells read "-" rather than
# inventing numbers).
COLUMNS: list[tuple[str, str, str]] = [
    ("edit certified leak", "edit", "certified_leak_fraction"),
    ("edit baseline leak", "edit", "baseline_leak_fraction"),
    ("edit leak reduction", "edit", "leak_reduction"),
    ("tex round-trip PSNR (dB)", "texture", "roundtrip_psnr_db"),
    ("tex baking excess", "texture", "baking_excess"),
    ("col coverage@floor", "collision", "coverage_at_floor"),
    ("col tol-frac@floor", "collision", "coverage_tolerance_fraction_at_floor"),
    ("col false-collision", "collision", "false_collision_rate"),
]


def _find_benchmark_json(path: Path) -> Path:
    if path.is_file():
        return path
    for candidate in (path / "asset_benchmark.json", path / "asset_eval" / "asset_benchmark.json"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no asset_benchmark.json under {path}")


def _headline(doc: dict, track: str) -> dict:
    for verdict in doc.get("verdicts", []):
        if verdict["track"] == track:
            return verdict.get("headline", {})
    return {}


def _scene_name(doc: dict, path: Path) -> str:
    bundle = doc.get("bundle")
    if bundle:
        # <scene>/hybrid_asset -> <scene>
        parts = Path(bundle).parts
        return parts[-2] if parts[-1] == "hybrid_asset" and len(parts) >= 2 else Path(bundle).name
    return path.parent.name


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt(v) for v in value)
    if isinstance(value, float):
        if value != value:
            return "-"
        return f"{value:.4g}"
    return str(value)


def _row(doc: dict, path: Path) -> dict[str, object]:
    row: dict[str, object] = {"scene": _scene_name(doc, path)}
    for header, track, key in COLUMNS:
        row[header] = _headline(doc, track).get(key) if key else None
    row["overall"] = doc.get("overall_status", "?")
    row["status(edit/tex/col)"] = "/".join(
        (doc.get("track_status", {}) or {}).get(t, "?") for t in ("edit", "texture", "collision")
    )
    return row


def _to_markdown(rows: list[dict[str, object]]) -> str:
    headers = ["scene"] + [c[0] for c in COLUMNS] + ["status(edit/tex/col)", "overall"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(h)) for h in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", nargs="+",
                        help="asset_benchmark.json files or bundle / asset_eval directories")
    parser.add_argument("--markdown", default=None, help="write the Markdown table here")
    parser.add_argument("--csv", default=None, help="write the CSV table here")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for scene in args.scenes:
        path = _find_benchmark_json(Path(scene))
        doc = json.loads(path.read_text(encoding="utf-8"))
        version = doc.get("protocol_version")
        if version != PROTOCOL_VERSION:
            print(f"warning: {path} is protocol {version!r}, current is {PROTOCOL_VERSION!r}",
                  file=sys.stderr)
        rows.append(_row(doc, path))

    markdown = _to_markdown(rows)
    print(markdown)

    if args.markdown:
        Path(args.markdown).write_text(markdown + "\n", encoding="utf-8")
    if args.csv:
        headers = ["scene"] + [c[0] for c in COLUMNS] + ["status(edit/tex/col)", "overall"]
        with Path(args.csv).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow({h: _fmt(row.get(h)) for h in headers})


if __name__ == "__main__":
    main()
