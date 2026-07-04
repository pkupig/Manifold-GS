#!/usr/bin/env python3
"""Create a deterministic explicit held-out split for a COLMAP scene."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.interval < 2:
        raise ValueError("interval must be at least 2")
    images = args.scene / "images"
    names = sorted(
        path.name for path in images.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if len(names) < args.interval:
        raise ValueError(f"not enough images in {images}: {len(names)}")
    output = args.scene / "sparse" / "0" / "test.txt"
    if output.exists() and not args.force:
        existing = [line.strip() for line in output.read_text().splitlines() if line.strip()]
        print(f"preserving existing {output} ({len(existing)} test views)")
        return
    selected = names[::args.interval]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(selected) + "\n", encoding="utf-8")
    print(f"wrote {output}: {len(selected)} test / {len(names)} total")


if __name__ == "__main__":
    main()
