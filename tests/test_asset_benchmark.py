"""Tests for the frozen asset-benchmark protocol (A5: P0.3/P0.4/P0.5).

Covers the verdict logic on synthetic reports and an end-to-end run of the single
command on a real (synthetic) exported bundle, including the summary table generator.
Everything is CPU-only and deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manifold_gs.asset_bundle import export_asset_bundle
from manifold_gs.asset_benchmark import (
    PROTOCOL_VERSION,
    collision_verdict,
    edit_verdict,
    overall_status,
    texture_verdict,
)
from manifold_gs.diagnostics import compute_diagnostics
from manifold_gs.patch_mesh import build_patch_mesh, save_patch_mesh


def _write_plane_ply(path: Path, n: int = 8) -> None:
    props = [
        "x", "y", "z", "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity", "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    rows = []
    rng = np.random.default_rng(0)
    for i in range(n):
        for j in range(n):
            rows.append((
                (i - (n - 1) / 2) * 0.05, (j - (n - 1) / 2) * 0.05, 0.0,
                0, 0, 0,
                float(rng.normal()), float(rng.normal()), float(rng.normal()),
                2.0, np.log(0.045), np.log(0.045), np.log(0.003), 1, 0, 0, 0,
            ))
    with path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(rows)}\n")
        for name in props:
            f.write(f"property float {name}\n")
        f.write("end_header\n")
        for row in rows:
            f.write(" ".join(map(str, row)) + "\n")


def _build_bundle(tmp_path: Path) -> Path:
    ply = tmp_path / "plane.ply"
    _write_plane_ply(ply)
    diag = compute_diagnostics(ply)
    mesh = build_patch_mesh(diag, min_patch_size=5, k=8)
    mesh_path = tmp_path / "patch.ply"
    meta_path = tmp_path / "patch.npz"
    save_patch_mesh(mesh, mesh_path, meta_path)
    bundle = tmp_path / "hybrid_asset"
    export_asset_bundle(ply, mesh_path, meta_path, bundle, 1)
    return bundle


# --- Verdict unit tests ------------------------------------------------------

def test_edit_verdict_pass_when_baseline_leaks_and_certified_clean() -> None:
    report = {
        "certified_binding": {"leaked_point_fraction": 0.0, "residual_contaminated_fraction": 0.0,
                              "edit_error_mean": 0.0},
        "radius_baseline": {"leaked_point_fraction": 0.13},
    }
    verdict = edit_verdict(report)
    assert verdict["status"] == "pass"
    assert verdict["headline"]["leak_reduction"] == 0.13


def test_edit_verdict_uninformative_when_baseline_does_not_leak() -> None:
    report = {
        "certified_binding": {"leaked_point_fraction": 0.0, "residual_contaminated_fraction": 0.0},
        "radius_baseline": {"leaked_point_fraction": 0.0},
    }
    assert edit_verdict(report)["status"] == "uninformative"


def test_edit_verdict_fails_when_certified_leaks() -> None:
    report = {
        "certified_binding": {"leaked_point_fraction": 0.02, "residual_contaminated_fraction": 0.0},
        "radius_baseline": {"leaked_point_fraction": 0.13},
    }
    assert edit_verdict(report)["status"] == "fail"


def test_texture_verdict_gates_excess_not_absolute_seam() -> None:
    # High absolute seam but near-zero baking excess -> pass (seam is raw colour variance).
    good = {"reprojection_psnr_mean": 34.0,
            "seam": {"boundary_pairs": 100, "baking_excess_error_mean": 0.001,
                     "seam_error_mean": 0.3, "raw_seam_error_mean": 0.3}}
    assert texture_verdict(good)["status"] == "pass"
    # Baking itself adds a lot of seam -> fail even if PSNR is fine.
    bad = {"reprojection_psnr_mean": 34.0,
           "seam": {"boundary_pairs": 100, "baking_excess_error_mean": 0.2}}
    assert texture_verdict(bad)["status"] == "fail"


def test_texture_verdict_fails_on_low_roundtrip_psnr() -> None:
    report = {"reprojection_psnr_mean": 12.0,
              "seam": {"boundary_pairs": 100, "baking_excess_error_mean": 0.0}}
    assert texture_verdict(report)["status"] == "fail"


def test_texture_verdict_seamless_asset_skips_seam_gate() -> None:
    report = {"reprojection_psnr_mean": float("inf"), "seam": {"boundary_pairs": 0}}
    verdict = texture_verdict(report)
    assert verdict["status"] == "pass"
    assert any(c["name"] == "baking_excess" and c["ok"] for c in verdict["checks"])


def test_collision_verdict_skipped_without_gt() -> None:
    assert collision_verdict(None)["status"] == "skipped"


def test_collision_verdict_pass_when_coverage_met_at_reasonable_tolerance() -> None:
    report = {
        "bbox_diagonal": 1.0,
        "coverage": {"coverage": 0.9, "hausdorff": 0.05, "supported_normal_median_deg": 3.0},
        "coverage_sweep": {"sweep": [
            {"tolerance": 0.005, "coverage": 0.5},
            {"tolerance": 0.02, "coverage": 0.85},
            {"tolerance": 0.05, "coverage": 0.99},
        ]},
    }
    verdict = collision_verdict(report)
    assert verdict["status"] == "pass"
    assert verdict["headline"]["coverage_tolerance_fraction_at_floor"] == 0.02


def test_collision_verdict_fails_when_floor_needs_loose_tolerance() -> None:
    report = {
        "bbox_diagonal": 1.0,
        "coverage": {"coverage": 0.3},
        "coverage_sweep": {"sweep": [
            {"tolerance": 0.02, "coverage": 0.4},
            {"tolerance": 0.08, "coverage": 0.85},  # only reaches floor at 8% bbox
        ]},
    }
    assert collision_verdict(report)["status"] == "fail"


def test_collision_verdict_fails_on_false_collisions() -> None:
    report = {
        "bbox_diagonal": 1.0,
        "coverage": {"coverage": 0.9},
        "coverage_sweep": {"sweep": [{"tolerance": 0.02, "coverage": 0.9}]},
        "collision": {"false_collision_rate": 0.4, "missed_collision_rate": 0.0},
    }
    assert collision_verdict(report)["status"] == "fail"


def test_overall_status_fails_if_any_track_fails() -> None:
    assert overall_status([{"track": "a", "status": "pass"}, {"track": "b", "status": "skipped"}]) == "pass"
    assert overall_status([{"track": "a", "status": "pass"}, {"track": "b", "status": "fail"}]) == "fail"
    assert overall_status([{"track": "a", "status": "uninformative"}]) == "pass"


# --- End-to-end tests --------------------------------------------------------

def test_run_asset_benchmark_cli_passes_on_synthetic_bundle(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    out = tmp_path / "benchmark.json"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_asset_benchmark.py"),
         "--bundle", str(bundle), "--out", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    doc = json.loads(out.read_text())
    assert doc["protocol_version"] == PROTOCOL_VERSION
    assert doc["overall_status"] == "pass"
    # The single-patch plane cannot make the proximity baseline leak, so the edit track
    # is correctly "uninformative" here; the informative pass/fail paths are unit-tested.
    assert doc["track_status"]["edit"] in {"pass", "uninformative"}
    assert doc["track_status"]["texture"] == "pass"
    assert doc["track_status"]["collision"] == "skipped"


def test_summarize_asset_benchmark_cli_tabulates_scene(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_asset_benchmark.py"), "--bundle", str(bundle)],
        capture_output=True, text=True, check=True,
    )
    csv_path = tmp_path / "table.csv"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "summarize_asset_benchmark.py"),
         str(bundle), "--csv", str(csv_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "overall" in result.stdout
    body = csv_path.read_text()
    assert "/pass/skipped" in body  # texture pass, collision skipped
    assert body.strip().endswith("pass")  # overall pass
