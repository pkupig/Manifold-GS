#!/usr/bin/env python3
"""Check whether the local environment can run patched 3DGS training."""

from __future__ import annotations

import importlib
import json

import torch


def check_import(name: str) -> dict[str, object]:
    try:
        importlib.import_module(name)
        return {"ok": True, "error": ""}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    report = {
        "torch": {
            "version": torch.__version__,
            "cuda_compiled": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
        "imports": {
            "diff_gaussian_rasterization": check_import("diff_gaussian_rasterization"),
            "simple_knn._C": check_import("simple_knn._C"),
            "plyfile": check_import("plyfile"),
        },
    }
    if torch.cuda.is_available():
        report["torch"]["device_name"] = torch.cuda.get_device_name(0)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

