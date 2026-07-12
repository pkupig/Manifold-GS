"""Frozen PASS/FAIL protocol for the downstream asset benchmark (A5: P0.3/P0.4/P0.5).

This module is the single source of truth for the asset-utility gates. Given the raw
sub-reports produced by the three evaluators
(:mod:`scripts.evaluate_edit_propagation`, :mod:`scripts.evaluate_texture_roundtrip`,
:mod:`scripts.evaluate_collision_candidate`) it returns a per-track verdict with an
explicit list of checks, so both the CLI driver and the paper table generator apply
the same thresholds.

Design notes (why these gates and not the "obvious" ones):

- **Edit (P0.3).** Certified binding is exact *by construction* — its edit error and
  boundary leakage are zero whenever the exporter is sound. So the certified side is
  a correctness sanity check, and the *informative* signal is that the proximity
  baseline actually leaks on this scene. A scene where the baseline does not leak
  (well-separated patches) cannot demonstrate the certified advantage, so we mark it
  ``uninformative`` rather than a spurious PASS.
- **Texture (P0.5).** The absolute seam magnitude is dominated by genuine cross-patch
  colour variance (see the 2026-07-08 CPU diagnosis: raw seam ~= baked seam). A shared
  atlas cannot fix that, so we do *not* gate on absolute seam. We gate on (a) the
  round-trip fidelity and (b) ``baking_excess`` — how much seam the per-patch baking
  *adds* on top of the raw colour variance. Excess near zero means the charting is not
  the problem.
- **Collision (P0.4).** A single tight tolerance (1% bbox) reads almost every sample as
  "false surface" when the tolerance is below the method's own accuracy. So coverage is
  gated adaptively: PASS if the surface reaches the coverage floor at *some* tolerance
  no looser than ``max_coverage_tolerance_fraction``, and we report the achieved
  tolerance alongside it. This track is only evaluated when a GT surface is supplied.

Bump ``PROTOCOL_VERSION`` whenever a threshold changes so cached benchmark JSON can be
told apart from a re-run under a new protocol.
"""

from __future__ import annotations

import math
from typing import Any

PROTOCOL_VERSION = "asset-benchmark/1.0"

# --- Frozen thresholds -------------------------------------------------------
# Do not tune these to make a particular scene pass; a threshold change is a
# protocol change and must bump PROTOCOL_VERSION and be recorded in the docs.
THRESHOLDS: dict[str, float] = {
    # Edit (P0.3)
    "edit_max_certified_leak_fraction": 0.0,        # certified must not leak at all
    "edit_max_certified_residual_fraction": 0.0,    # certified must not touch residuals
    "edit_min_baseline_leak_fraction": 0.01,        # baseline must leak for the test to be informative
    # Texture (P0.5)
    "texture_min_roundtrip_psnr_db": 30.0,          # baking round-trip must preserve colour
    "texture_max_baking_excess": 0.02,              # baking must not add seam beyond raw variance
    # Collision (P0.4)
    "collision_min_coverage": 0.80,                 # surface coverage floor
    "collision_max_coverage_tolerance_fraction": 0.03,  # coverage floor must be met at <= this tol
    "collision_max_false_collision_rate": 0.05,     # contact-proxy false collisions (needs probes)
}


def _check(name: str, value: float | None, op: str, threshold: float) -> dict[str, Any]:
    """Build one check record. ``op`` is one of ``<=``, ``>=``, ``==``, ``>``, ``<``."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        ok = False
    elif op == "<=":
        ok = value <= threshold
    elif op == ">=":
        ok = value >= threshold
    elif op == "==":
        ok = value == threshold
    elif op == ">":
        ok = value > threshold
    elif op == "<":
        ok = value < threshold
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown op {op!r}")
    return {"name": name, "value": value, "op": op, "threshold": threshold, "ok": bool(ok)}


def edit_verdict(report: dict[str, Any], thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    """Verdict for a P0.3 edit-propagation report."""
    t = {**THRESHOLDS, **(thresholds or {})}
    certified = report["certified_binding"]
    baseline = report["radius_baseline"]

    cert_leak = float(certified.get("leaked_point_fraction", 0.0))
    cert_resid = float(certified.get("residual_contaminated_fraction", 0.0))
    base_leak = float(baseline.get("leaked_point_fraction", 0.0))

    checks = [
        _check("certified_leak_fraction", cert_leak, "<=", t["edit_max_certified_leak_fraction"]),
        _check("certified_residual_fraction", cert_resid, "<=", t["edit_max_certified_residual_fraction"]),
        _check("baseline_leak_fraction", base_leak, ">=", t["edit_min_baseline_leak_fraction"]),
    ]
    certified_clean = checks[0]["ok"] and checks[1]["ok"]
    baseline_informative = checks[2]["ok"]
    if not certified_clean:
        status = "fail"
    elif not baseline_informative:
        status = "uninformative"
    else:
        status = "pass"
    return {
        "track": "edit",
        "status": status,
        "checks": checks,
        "headline": {
            "certified_leak_fraction": cert_leak,
            "baseline_leak_fraction": base_leak,
            "leak_reduction": base_leak - cert_leak,
            "certified_edit_error_mean": float(certified.get("edit_error_mean", 0.0)),
        },
    }


def texture_verdict(report: dict[str, Any], thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    """Verdict for a P0.5 texture round-trip report."""
    t = {**THRESHOLDS, **(thresholds or {})}
    psnr = float(report.get("reprojection_psnr_mean", float("nan")))
    seam = report.get("seam", {})
    boundary_pairs = int(seam.get("boundary_pairs", 0))
    excess = float(seam.get("baking_excess_error_mean", float("nan")))

    checks = [_check("roundtrip_psnr_db", psnr, ">=", t["texture_min_roundtrip_psnr_db"])]
    if boundary_pairs > 0:
        # A multi-chart asset has seams; gate on how much baking *adds* over the raw
        # cross-boundary colour variance (absolute seam is raw-variance-dominated).
        checks.append(_check("baking_excess", excess, "<=", t["texture_max_baking_excess"]))
    else:
        # Single-chart / seamless asset: no cross-patch boundary exists to gate.
        checks.append({
            "name": "baking_excess", "value": None, "op": "<=",
            "threshold": t["texture_max_baking_excess"], "ok": True,
            "note": "no cross-patch boundary pairs; seam gate not applicable",
        })
    status = "pass" if all(c["ok"] for c in checks) else "fail"
    return {
        "track": "texture",
        "status": status,
        "checks": checks,
        "headline": {
            "roundtrip_psnr_db": psnr,
            "baking_excess": excess,
            "seam_error_mean": float(seam.get("seam_error_mean", float("nan"))),
            "raw_seam_error_mean": float(seam.get("raw_seam_error_mean", float("nan"))),
            "evaluated_patches": int(report.get("evaluated_patches", 0)),
            "color_source": report.get("color_source"),
        },
    }


def _coverage_at_floor(report: dict[str, Any], floor: float) -> tuple[float | None, float | None]:
    """Smallest tolerance fraction whose coverage reaches ``floor`` and that coverage.

    Reads the coverage sweep, converting each absolute tolerance back to a bbox
    fraction. Returns ``(None, None)`` if the floor is never reached in the swept range.
    """
    sweep = report.get("coverage_sweep", {})
    rows = sweep.get("sweep", [])
    bbox = float(report.get("bbox_diagonal", 0.0)) or 1e-12
    best: tuple[float, float] | None = None
    for row in rows:
        coverage = float(row["coverage"])
        if coverage >= floor:
            frac = float(row["tolerance"]) / bbox
            if best is None or frac < best[0]:
                best = (frac, coverage)
    return best if best is not None else (None, None)


def collision_verdict(report: dict[str, Any] | None, thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    """Verdict for a P0.4 collision report. ``None`` -> skipped (no GT supplied)."""
    t = {**THRESHOLDS, **(thresholds or {})}
    if report is None:
        return {"track": "collision", "status": "skipped", "checks": [], "headline": {}}

    floor = t["collision_min_coverage"]
    tol_frac, coverage_at_floor = _coverage_at_floor(report, floor)
    coverage_ok = tol_frac is not None and tol_frac <= t["collision_max_coverage_tolerance_fraction"]
    checks = [{
        "name": "coverage_tolerance_fraction_at_floor",
        "value": tol_frac,
        "op": "<=",
        "threshold": t["collision_max_coverage_tolerance_fraction"],
        "ok": bool(coverage_ok),
        "note": f"tolerance fraction at which coverage first reaches {floor}",
    }]

    collision = report.get("collision")
    if collision is not None:
        false_rate = float(collision.get("false_collision_rate", float("nan")))
        checks.append(_check("false_collision_rate", false_rate, "<=", t["collision_max_false_collision_rate"]))

    status = "pass" if all(c["ok"] for c in checks) else "fail"
    coverage = report.get("coverage", {})
    headline = {
        "coverage_floor": floor,
        "coverage_tolerance_fraction_at_floor": tol_frac,
        "coverage_at_floor": coverage_at_floor,
        "coverage_at_1pct": float(coverage.get("coverage", float("nan"))),
        "hausdorff": float(coverage.get("hausdorff", float("nan"))),
        "supported_normal_median_deg": float(coverage.get("supported_normal_median_deg", float("nan"))),
    }
    if collision is not None:
        headline["false_collision_rate"] = float(collision.get("false_collision_rate", float("nan")))
        headline["missed_collision_rate"] = float(collision.get("missed_collision_rate", float("nan")))
        if "unknown_marked_free_fraction" in collision:
            headline["unknown_marked_free_fraction"] = float(collision["unknown_marked_free_fraction"])
    return {"track": "collision", "status": status, "checks": checks, "headline": headline}


def overall_status(verdicts: list[dict[str, Any]]) -> str:
    """``fail`` if any track failed, else ``pass``.

    ``skipped`` and ``uninformative`` tracks do not fail the run — they are surfaced in
    the report so a reader knows the gate was not actually exercised.
    """
    return "fail" if any(v["status"] == "fail" for v in verdicts) else "pass"


def summarize(verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll up per-track verdicts into an overall record."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "overall_status": overall_status(verdicts),
        "track_status": {v["track"]: v["status"] for v in verdicts},
    }
