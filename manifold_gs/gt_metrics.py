"""Ground-truth geometry and discrete-varifold metrics."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _normal_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dots = np.abs(np.sum(a * b, axis=1))
    return np.degrees(np.arccos(np.clip(dots, 0.0, 1.0)))


def geometry_metrics(
    estimated_xyz: np.ndarray,
    estimated_normals: np.ndarray,
    gt_xyz: np.ndarray,
    gt_normals: np.ndarray,
    bbox_diagonal: float,
) -> dict[str, float | int]:
    estimated_xyz = np.asarray(estimated_xyz, dtype=np.float64)
    estimated_normals = np.asarray(estimated_normals, dtype=np.float64)
    gt_xyz = np.asarray(gt_xyz, dtype=np.float64)
    gt_normals = np.asarray(gt_normals, dtype=np.float64)
    if estimated_xyz.shape[0] == 0:
        return {"num_estimated": 0}

    gt_tree = cKDTree(gt_xyz)
    est_tree = cKDTree(estimated_xyz)
    accuracy, accuracy_match = gt_tree.query(estimated_xyz, k=1)
    completeness, completeness_match = est_tree.query(gt_xyz, k=1)
    accuracy_normal = _normal_angle_deg(estimated_normals, gt_normals[accuracy_match])
    completeness_normal = _normal_angle_deg(gt_normals, estimated_normals[completeness_match])
    scale = max(float(bbox_diagonal), 1e-12)

    result: dict[str, float | int] = {
        "num_estimated": int(estimated_xyz.shape[0]),
        "accuracy_mean": float(np.mean(accuracy)),
        "accuracy_median": float(np.median(accuracy)),
        "completeness_mean": float(np.mean(completeness)),
        "completeness_median": float(np.median(completeness)),
        "chamfer_l1": float(0.5 * (np.mean(accuracy) + np.mean(completeness))),
        "chamfer_l1_normalized": float(0.5 * (np.mean(accuracy) + np.mean(completeness)) / scale),
        "normal_accuracy_median_deg": float(np.median(accuracy_normal)),
        "normal_completeness_median_deg": float(np.median(completeness_normal)),
    }
    for fraction in (0.005, 0.01, 0.02):
        threshold = fraction * scale
        precision = float(np.mean(accuracy <= threshold))
        recall = float(np.mean(completeness <= threshold))
        fscore = 2.0 * precision * recall / max(precision + recall, 1e-12)
        key = f"{fraction:.3f}".replace(".", "p")
        result[f"precision_{key}_bbox"] = precision
        result[f"recall_{key}_bbox"] = recall
        result[f"fscore_{key}_bbox"] = fscore
    return result


def _kernel_sum(
    xyz_a: np.ndarray,
    normals_a: np.ndarray,
    weights_a: np.ndarray,
    xyz_b: np.ndarray,
    normals_b: np.ndarray,
    weights_b: np.ndarray,
    sigma: float,
    tangent_sigma: float,
    block_size: int = 512,
) -> float:
    total = 0.0
    inv_two_sigma2 = 0.5 / max(sigma * sigma, 1e-24)
    for start in range(0, xyz_a.shape[0], block_size):
        stop = min(start + block_size, xyz_a.shape[0])
        delta = xyz_a[start:stop, None, :] - xyz_b[None, :, :]
        spatial = np.exp(-np.sum(delta * delta, axis=-1) * inv_two_sigma2)
        dot_squared = (normals_a[start:stop] @ normals_b.T) ** 2
        # For unoriented codimension-one planes, ||P-Q||_F^2 = 2(1-(n.m)^2).
        # An RBF on embedded projectors is characteristic; the previous linear
        # `(n.m)^2` kernel compared only a finite tangent moment.
        tangent = np.exp(-(1.0 - dot_squared) / max(tangent_sigma * tangent_sigma, 1e-24))
        kernel = spatial * tangent
        total += float(np.sum(weights_a[start:stop, None] * weights_b[None, :] * kernel))
    return total


def _subsample(
    xyz: np.ndarray,
    normals: np.ndarray,
    weights: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if xyz.shape[0] <= max_points:
        return xyz, normals, weights
    # Deterministic systematic resampling preserves weighted support better than
    # selecting the largest splats only.
    probabilities = weights / np.maximum(np.sum(weights), 1e-24)
    cdf = np.cumsum(probabilities)
    targets = (np.arange(max_points) + 0.5) / max_points
    indices = np.searchsorted(cdf, targets, side="left")
    unique, counts = np.unique(np.minimum(indices, xyz.shape[0] - 1), return_counts=True)
    sampled_weights = counts.astype(np.float64) / max_points
    return xyz[unique], normals[unique], sampled_weights


def normalized_kernel_varifold_distance(
    estimated_xyz: np.ndarray,
    estimated_normals: np.ndarray,
    estimated_weights: np.ndarray,
    gt_xyz: np.ndarray,
    gt_normals: np.ndarray,
    gt_weights: np.ndarray,
    sigma: float,
    tangent_sigma: float = 0.5,
    max_points: int = 4096,
) -> float:
    """Kernel MMD between unit-mass unoriented discrete varifolds.

    Unit-mass normalization intentionally removes global area scale. Until the
    model contains explicit geometric masses q_i, this is a support/tangent
    metric and not evidence of total-area conservation.
    """
    if len(estimated_xyz) == 0:
        return float("inf")
    estimated_weights = np.maximum(np.asarray(estimated_weights, dtype=np.float64), 0.0)
    gt_weights = np.maximum(np.asarray(gt_weights, dtype=np.float64), 0.0)
    estimated_weights /= np.maximum(np.sum(estimated_weights), 1e-24)
    gt_weights /= np.maximum(np.sum(gt_weights), 1e-24)
    estimated_xyz, estimated_normals, estimated_weights = _subsample(
        np.asarray(estimated_xyz, dtype=np.float64),
        np.asarray(estimated_normals, dtype=np.float64),
        estimated_weights,
        max_points,
    )
    gt_xyz, gt_normals, gt_weights = _subsample(
        np.asarray(gt_xyz, dtype=np.float64),
        np.asarray(gt_normals, dtype=np.float64),
        gt_weights,
        max_points,
    )
    estimated_weights /= np.maximum(np.sum(estimated_weights), 1e-24)
    gt_weights /= np.maximum(np.sum(gt_weights), 1e-24)
    aa = _kernel_sum(estimated_xyz, estimated_normals, estimated_weights, estimated_xyz, estimated_normals, estimated_weights, sigma, tangent_sigma)
    bb = _kernel_sum(gt_xyz, gt_normals, gt_weights, gt_xyz, gt_normals, gt_weights, sigma, tangent_sigma)
    ab = _kernel_sum(estimated_xyz, estimated_normals, estimated_weights, gt_xyz, gt_normals, gt_weights, sigma, tangent_sigma)
    return float(np.sqrt(max(aa + bb - 2.0 * ab, 0.0)))
