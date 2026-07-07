#!/usr/bin/env python3
"""Stage the official DTU layout expected by the DTUeval scripts."""

from __future__ import annotations

from pathlib import Path


DEFAULT_DTU_OFFICIAL_SOURCE = Path("/root/autodl-tmp/emgs-real/dtu-official")


def ensure_dtu_official_layout(
    staged_root: Path,
    source_root: Path = DEFAULT_DTU_OFFICIAL_SOURCE,
) -> None:
    """Create a minimal DTU official dataset layout under ``staged_root``.

    The evaluation scripts expect:
    - ``ObsMask/ObsMask{scan}_10.mat``
    - ``ObsMask/Plane{scan}.mat``
    - ``Points/stl/stl{scan:03}_total.ply``

    Our local archive stores those directories under two different top-level
    locations, so we expose them through symlinks.
    """

    obs_mask_source = source_root / "SampleSet" / "SampleSet" / "MVS Data" / "ObsMask"
    stl_source = source_root / "Points" / "Points" / "stl"
    if not obs_mask_source.is_dir():
        raise FileNotFoundError(f"missing DTU ObsMask source: {obs_mask_source}")
    if not stl_source.is_dir():
        raise FileNotFoundError(f"missing DTU STL source: {stl_source}")

    links = {
        staged_root / "ObsMask": obs_mask_source,
        staged_root / "Points" / "stl": stl_source,
    }
    for link, target in links.items():
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise RuntimeError(f"DTU layout link points elsewhere: {link}")
            continue
        if link.exists():
            raise FileExistsError(f"refusing to replace existing DTU path: {link}")
        link.symlink_to(target, target_is_directory=True)
