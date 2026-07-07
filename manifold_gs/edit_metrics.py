"""Quantify how cleanly a patch edit propagates to bound Gaussians (P0.3).

The certified asset binds every attached Gaussian to exactly one patch. Editing a
selected patch should move only that patch's Gaussians and leave everything else
fixed. This module builds the synthetic-GT harness that scores an edit:

- ``edit_error``: how far the propagated result is from the intended target inside
  the edited region (under-propagation / distortion);
- ``boundary_leakage``: motion applied to points that should have stayed fixed
  (a proximity baseline leaks across patch boundaries, certified binding does not);
- ``residual_contamination``: motion applied to unattached residual Gaussians that
  were never part of any patch.

Everything is pure numpy/scipy so it runs on CPU with a deterministic result. It is
a synthetic-GT benchmark, not a substitute for the real DCC human-inspection task.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.spatial import cKDTree


Deformation = Callable[[np.ndarray], np.ndarray]


def rigid_deformation(
    rotation: np.ndarray,
    translation: np.ndarray,
    pivot: np.ndarray | None = None,
) -> Deformation:
    """Return a rigid transform ``x -> R (x - pivot) + pivot + t``."""
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    pivot = None if pivot is None else np.asarray(pivot, dtype=np.float64).reshape(3)

    def deform(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if pivot is None:
            return points @ rotation.T + translation
        return (points - pivot) @ rotation.T + pivot + translation

    return deform


def certified_patch_binding(patch_ids: np.ndarray, selected_patches) -> np.ndarray:
    """Bind exactly the points whose patch id is in ``selected_patches``."""
    patch_ids = np.asarray(patch_ids).reshape(-1)
    selected = np.asarray(list(selected_patches)).reshape(-1)
    return np.isin(patch_ids, selected)


def radius_binding(points: np.ndarray, seed_points: np.ndarray, radius: float) -> np.ndarray:
    """Bind points within ``radius`` of any seed — a geometric-proximity baseline.

    This is the naive "nearest attachment" a mesh-free edit would use; it has no
    patch identity, so it leaks across boundaries whenever patches are close.
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
    points = np.asarray(points, dtype=np.float64)
    seed_points = np.asarray(seed_points, dtype=np.float64)
    if seed_points.shape[0] == 0:
        return np.zeros(points.shape[0], dtype=bool)
    return cKDTree(seed_points).query(points, k=1)[0] <= radius


def propagate_edit(
    points: np.ndarray,
    deformation: Deformation,
    bound_mask: np.ndarray,
) -> np.ndarray:
    """Apply ``deformation`` to the bound points and leave the rest fixed."""
    points = np.asarray(points, dtype=np.float64)
    bound_mask = np.asarray(bound_mask, dtype=bool).reshape(-1)
    moved = points.copy()
    if bound_mask.any():
        moved[bound_mask] = deformation(points[bound_mask])
    return moved


def edit_propagation_metrics(
    original: np.ndarray,
    moved: np.ndarray,
    target: np.ndarray,
    edit_region: np.ndarray,
    *,
    residual_mask: np.ndarray | None = None,
    leak_eps: float = 1e-9,
) -> dict[str, float | int]:
    """Score a propagated edit against the intended target.

    ``target`` is the ground-truth result: the deformation applied to exactly
    ``edit_region`` and nothing else (build it with ``propagate_edit`` using
    ``edit_region`` as the bound mask). ``moved`` is the result under test.
    """
    original = np.asarray(original, dtype=np.float64)
    moved = np.asarray(moved, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    edit_region = np.asarray(edit_region, dtype=bool).reshape(-1)
    if not (original.shape == moved.shape == target.shape):
        raise ValueError("original, moved and target must share shape")
    if edit_region.shape[0] != original.shape[0]:
        raise ValueError("edit_region must have one flag per point")

    displacement = np.linalg.norm(moved - original, axis=1)
    error = np.linalg.norm(moved - target, axis=1)
    target_shift = np.linalg.norm(target - original, axis=1)
    outside = ~edit_region

    result: dict[str, float | int] = {
        "points": int(original.shape[0]),
        "edited_points": int(edit_region.sum()),
        "target_shift_mean": float(np.mean(target_shift[edit_region])) if edit_region.any() else 0.0,
        "edit_error_mean": float(np.mean(error[edit_region])) if edit_region.any() else 0.0,
        "edit_error_max": float(np.max(error[edit_region])) if edit_region.any() else 0.0,
        "boundary_leakage_mean": float(np.mean(displacement[outside])) if outside.any() else 0.0,
        "boundary_leakage_max": float(np.max(displacement[outside])) if outside.any() else 0.0,
        "leaked_point_fraction": float(np.mean(displacement[outside] > leak_eps)) if outside.any() else 0.0,
    }
    if residual_mask is not None:
        residual_mask = np.asarray(residual_mask, dtype=bool).reshape(-1)
        if residual_mask.shape[0] != original.shape[0]:
            raise ValueError("residual_mask must have one flag per point")
        if residual_mask.any():
            result["residual_points"] = int(residual_mask.sum())
            result["residual_contamination_mean"] = float(np.mean(displacement[residual_mask]))
            result["residual_contamination_max"] = float(np.max(displacement[residual_mask]))
            result["residual_contaminated_fraction"] = float(
                np.mean(displacement[residual_mask] > leak_eps)
            )
    return result
