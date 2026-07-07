"""Evaluate a collision-candidate mesh against a reference surface.

These metrics answer P0.4 of ``PROJECT-GAPS-ZH.md``: a collision candidate must be
scored, not just exported. Everything here is pure geometry (numpy/scipy) so it runs
on CPU and is deterministic given a seed.

Conventions:
- ``coverage`` is recall of the reference surface by the candidate (how much true
  surface is represented within tolerance);
- ``false surface`` is the candidate area whose nearest reference point is beyond
  tolerance (floaters / hallucinated collision surface);
- collision confusion is a proximity proxy on probe points, not a watertight
  occupancy test: open candidate meshes have no interior, so "false/missed
  collision" is defined by contact distance to each surface, and callers must read
  it as a contact proxy.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _normal_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Unoriented: patches carry sign-free tangent planes, so compare |n . m|.
    dots = np.abs(np.sum(a * b, axis=1))
    return np.degrees(np.arccos(np.clip(dots, 0.0, 1.0)))


def sample_mesh_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_samples: int,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Area-weighted deterministic surface sampling.

    Returns ``(points, face_normals, total_area)``. Degenerate (zero-area) faces are
    dropped before sampling so the area distribution stays finite.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if faces.size == 0:
        return np.empty((0, 3)), np.empty((0, 3)), 0.0
    tris = vertices[faces]
    edge1 = tris[:, 1] - tris[:, 0]
    edge2 = tris[:, 2] - tris[:, 0]
    cross = np.cross(edge1, edge2)
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > 1e-18
    if not valid.any():
        return np.empty((0, 3)), np.empty((0, 3)), 0.0
    tris, edge1, edge2, cross, double_area = (
        tris[valid], edge1[valid], edge2[valid], cross[valid], double_area[valid]
    )
    normals = cross / double_area[:, None]
    areas = 0.5 * double_area
    total_area = float(areas.sum())

    rng = np.random.default_rng(seed)
    face_idx = rng.choice(areas.shape[0], size=n_samples, p=areas / areas.sum())
    u = rng.random(n_samples)
    v = rng.random(n_samples)
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    points = tris[face_idx, 0] + u[:, None] * edge1[face_idx] + v[:, None] * edge2[face_idx]
    return points, normals[face_idx], total_area


def surface_coverage_metrics(
    candidate_vertices: np.ndarray,
    candidate_faces: np.ndarray,
    reference_xyz: np.ndarray,
    reference_normals: np.ndarray,
    *,
    tolerance: float,
    samples: int = 50000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Score a candidate mesh against a reference surface point set.

    ``coverage`` = fraction of reference points within ``tolerance`` of the candidate
    (recall). ``false_surface_fraction`` = fraction of candidate area beyond
    ``tolerance`` from the reference (floater surface). Normal errors are measured on
    the candidate samples that are supported by the reference.
    """
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    reference_xyz = np.asarray(reference_xyz, dtype=np.float64)
    reference_normals = np.asarray(reference_normals, dtype=np.float64)
    cand_points, cand_normals, cand_area = sample_mesh_surface(
        candidate_vertices, candidate_faces, samples, seed=seed
    )
    if cand_points.shape[0] == 0 or reference_xyz.shape[0] == 0:
        return {
            "candidate_samples": int(cand_points.shape[0]),
            "candidate_area": float(cand_area),
            "coverage": 0.0,
            "false_surface_fraction": 1.0 if cand_points.shape[0] else 0.0,
            "false_surface_area": float(cand_area),
            "supported_surface_area": 0.0,
        }
    reference_tree = cKDTree(reference_xyz)
    candidate_tree = cKDTree(cand_points)

    reference_to_candidate, _ = candidate_tree.query(reference_xyz, k=1)
    coverage = float(np.mean(reference_to_candidate <= tolerance))

    candidate_to_reference, match = reference_tree.query(cand_points, k=1)
    supported = candidate_to_reference <= tolerance
    false_fraction = float(np.mean(~supported))
    normal_error = _normal_angle_deg(cand_normals[supported], reference_normals[match[supported]])

    return {
        "candidate_samples": int(cand_points.shape[0]),
        "candidate_area": float(cand_area),
        "coverage": coverage,
        "false_surface_fraction": false_fraction,
        "false_surface_area": float(false_fraction * cand_area),
        "supported_surface_area": float((1.0 - false_fraction) * cand_area),
        "supported_normal_median_deg": float(np.median(normal_error)) if normal_error.size else float("nan"),
        "supported_normal_mean_deg": float(np.mean(normal_error)) if normal_error.size else float("nan"),
        "hausdorff": float(max(candidate_to_reference.max(), reference_to_candidate.max())),
        "hausdorff_candidate_to_reference": float(candidate_to_reference.max()),
        "hausdorff_reference_to_candidate": float(reference_to_candidate.max()),
        "candidate_to_reference_p95": float(np.quantile(candidate_to_reference, 0.95)),
        "reference_to_candidate_p95": float(np.quantile(reference_to_candidate, 0.95)),
        "tolerance": float(tolerance),
    }


def coverage_tolerance_sweep(
    candidate_vertices: np.ndarray,
    candidate_faces: np.ndarray,
    reference_xyz: np.ndarray,
    *,
    tolerances: np.ndarray,
    samples: int = 50000,
    seed: int = 0,
) -> dict[str, object]:
    """Coverage and false-surface as a function of tolerance.

    A single tolerance is misleading: if it is tighter than the method's own surface
    accuracy, almost every candidate sample reads as "false surface" even when the
    geometry is uniformly close-but-not-exact. This returns the full trade-off plus
    the candidate/reference distance distributions so a caller can compare the tested
    tolerance against the method's median error (``candidate_to_reference_median``)
    instead of pinning one arbitrary threshold.
    """
    reference_xyz = np.asarray(reference_xyz, dtype=np.float64)
    cand_points, _, cand_area = sample_mesh_surface(
        candidate_vertices, candidate_faces, samples, seed=seed
    )
    tolerances = np.asarray(tolerances, dtype=np.float64).reshape(-1)
    if cand_points.shape[0] == 0 or reference_xyz.shape[0] == 0:
        rows = [
            {"tolerance": float(t), "coverage": 0.0,
             "false_surface_fraction": 1.0 if cand_points.shape[0] else 0.0}
            for t in tolerances
        ]
        return {"candidate_area": float(cand_area), "sweep": rows}
    reference_tree = cKDTree(reference_xyz)
    candidate_tree = cKDTree(cand_points)
    reference_to_candidate, _ = candidate_tree.query(reference_xyz, k=1)
    candidate_to_reference, _ = reference_tree.query(cand_points, k=1)
    rows = []
    for t in tolerances:
        rows.append({
            "tolerance": float(t),
            "coverage": float(np.mean(reference_to_candidate <= t)),
            "false_surface_fraction": float(np.mean(candidate_to_reference > t)),
        })
    return {
        "candidate_area": float(cand_area),
        "candidate_to_reference_median": float(np.median(candidate_to_reference)),
        "candidate_to_reference_p90": float(np.quantile(candidate_to_reference, 0.90)),
        "reference_to_candidate_median": float(np.median(reference_to_candidate)),
        "sweep": rows,
    }


def collision_confusion(
    candidate_xyz: np.ndarray,
    reference_xyz: np.ndarray,
    probe_points: np.ndarray,
    *,
    contact_tolerance: float,
    probe_labels: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Contact-proxy confusion between a candidate and reference surface.

    A probe is "in contact" with a surface when it is within ``contact_tolerance``.
    ``false_collision`` = probes touching the candidate but not the reference (the
    candidate would block empty space). ``missed_collision`` = probes touching the
    reference but not the candidate (a real surface the candidate does not cover).

    When ``probe_labels`` is given (strings ``"free"`` / ``"occupied"`` / ``"unknown"``),
    ``unknown_marked_free_fraction`` reports how many unknown-region probes the
    candidate leaves untouched — i.e. implicitly treats as free space, the failure
    mode P0.4 warns about.
    """
    if contact_tolerance <= 0:
        raise ValueError("contact_tolerance must be positive")
    probe_points = np.asarray(probe_points, dtype=np.float64)
    if probe_points.shape[0] == 0:
        raise ValueError("at least one probe point is required")
    candidate_xyz = np.asarray(candidate_xyz, dtype=np.float64)
    reference_xyz = np.asarray(reference_xyz, dtype=np.float64)

    near_candidate = (
        cKDTree(candidate_xyz).query(probe_points, k=1)[0] <= contact_tolerance
        if candidate_xyz.shape[0] else np.zeros(probe_points.shape[0], dtype=bool)
    )
    near_reference = (
        cKDTree(reference_xyz).query(probe_points, k=1)[0] <= contact_tolerance
        if reference_xyz.shape[0] else np.zeros(probe_points.shape[0], dtype=bool)
    )
    false_collision = near_candidate & ~near_reference
    missed_collision = ~near_candidate & near_reference
    agree = near_candidate == near_reference

    result: dict[str, float | int] = {
        "probes": int(probe_points.shape[0]),
        "candidate_contact": int(near_candidate.sum()),
        "reference_contact": int(near_reference.sum()),
        "false_collision": int(false_collision.sum()),
        "missed_collision": int(missed_collision.sum()),
        "false_collision_rate": float(np.mean(false_collision)),
        "missed_collision_rate": float(np.mean(missed_collision)),
        "agreement": float(np.mean(agree)),
        "contact_tolerance": float(contact_tolerance),
    }
    if probe_labels is not None:
        probe_labels = np.asarray(probe_labels).reshape(-1)
        if probe_labels.shape[0] != probe_points.shape[0]:
            raise ValueError("probe_labels must have one label per probe point")
        unknown = probe_labels == "unknown"
        if unknown.any():
            result["unknown_probes"] = int(unknown.sum())
            result["unknown_marked_free_fraction"] = float(np.mean(~near_candidate[unknown]))
    return result
