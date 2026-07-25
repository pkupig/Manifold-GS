"""Protocol-v1 bookkeeping for restricted-rendering finite-difference Fisher."""
from __future__ import annotations

import numpy as np

PROTOCOL_VERSION = "restricted-fisher/v1"


def patch_normals(xyz: np.ndarray, patch_ids: np.ndarray) -> dict[int, np.ndarray]:
    """Return deterministic PCA normals for each non-degenerate patch.

    Signs are deliberately arbitrary: v1 uses symmetric +/- normal displacements.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    patch_ids = np.asarray(patch_ids, dtype=np.int64).reshape(-1)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) != len(patch_ids):
        raise ValueError("xyz must be Nx3 with one patch id per row")
    out: dict[int, np.ndarray] = {}
    for patch in np.unique(patch_ids):
        points = xyz[patch_ids == patch]
        if len(points) < 3:
            continue
        _, values, vectors = np.linalg.svd(points - points.mean(0), full_matrices=False)
        # A surface patch is rank-2 by design: only the two in-plane singular
        # values must be non-degenerate; the smallest one should be near zero.
        if values[-2] <= 1e-12:
            continue
        normal = vectors[-1]
        # Deterministic orientation solely for serialisation/reproducibility.
        pivot = np.flatnonzero(np.abs(normal) > 1e-12)
        if pivot.size and normal[pivot[0]] < 0:
            normal = -normal
        out[int(patch)] = normal / np.linalg.norm(normal)
    return out


def finite_difference_fisher(plus: np.ndarray, minus: np.ndarray, epsilon: float, valid: np.ndarray | None = None) -> tuple[float, int]:
    """Mean RGB squared-Jacobian norm and number of valid pixels for v1."""
    plus = np.asarray(plus, dtype=np.float64)
    minus = np.asarray(minus, dtype=np.float64)
    if plus.shape != minus.shape or plus.ndim < 2 or plus.shape[-1] != 3:
        raise ValueError("plus/minus must share shape (..., 3)")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    finite = np.isfinite(plus).all(-1) & np.isfinite(minus).all(-1)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != finite.shape:
            raise ValueError("valid mask shape must match image without RGB channel")
        finite &= valid
    count = int(finite.sum())
    if count == 0:
        return float("nan"), 0
    jacobian = (plus[finite] - minus[finite]) / (2.0 * epsilon)
    return float(np.mean(np.sum(jacobian * jacobian, axis=-1))), count


def classify_patch(fisher: float, views: int, pixels: int, threshold: float | None, *, min_views: int = 3, min_pixels: int = 256) -> str:
    """Protocol-v1 status; unresolved never silently becomes low Fisher."""
    if views < min_views:
        return "insufficient_views"
    if pixels < min_pixels or not np.isfinite(fisher):
        return "numerically_unresolved"
    if threshold is None or not np.isfinite(threshold):
        return "pending_threshold"
    return "fisher_supported" if fisher >= threshold else "weakly_identified"


def scene_threshold(records: list[dict]) -> float:
    """v1 p10 threshold over only numerically valid, sufficiently observed patches."""
    values = [
        float(r["fisher"])
        for r in records
        if r.get("fisher") is not None
        and int(r.get("views", 0)) >= 3
        and int(r.get("pixels", 0)) >= 256
        and np.isfinite(float(r["fisher"]))
    ]
    return float(np.quantile(values, 0.10)) if values else float("nan")
