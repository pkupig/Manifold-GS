"""Diagnostic exact-depth losses for analytic identifiability audits."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def perturb_depth_target(
    depth: np.ndarray,
    noise_fraction: float = 0.0,
    dropout: float = 0.0,
    seed: int = 0,
    scale: float = 1.0,
    bias_fraction: float = 0.0,
    low_frequency_fraction: float = 0.0,
) -> np.ndarray:
    """Apply deterministic affine, low-frequency, iid, and missing-depth errors."""
    result = np.asarray(depth, dtype=np.float32).copy()
    valid = result > 0
    rng = np.random.default_rng(seed)
    if np.any(valid):
        median = float(np.median(result[valid]))
        result[valid] = scale * result[valid] + bias_fraction * median
        if low_frequency_fraction > 0:
            yy, xx = np.mgrid[0:result.shape[0], 0:result.shape[1]]
            phase_x, phase_y = rng.uniform(0.0, 2.0 * np.pi, size=2)
            field = (
                np.sin(2.0 * np.pi * xx / max(result.shape[1], 1) + phase_x)
                + np.cos(2.0 * np.pi * yy / max(result.shape[0], 1) + phase_y)
            )
            field_valid = field[valid]
            field = (field - float(np.mean(field_valid))) / max(float(np.std(field_valid)), 1e-8)
            result[valid] += low_frequency_fraction * median * field[valid]
    if noise_fraction > 0 and np.any(valid):
        noise_scale = float(np.median(result[valid]))
        result[valid] += rng.normal(
            0.0, noise_fraction * noise_scale, int(np.sum(valid))
        ).astype(np.float32)
    result[valid] = np.maximum(result[valid], 1e-6)
    if dropout > 0:
        dropped = rng.random(result.shape) < dropout
        result[valid & dropped] = 0.0
    return result


def prepare_depth_target(depth_z: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    if mode not in {"z", "inverse"}:
        raise ValueError(f"unsupported depth mode: {mode}")
    valid = depth_z > 0
    if mode == "inverse":
        target = torch.where(valid, depth_z.clamp_min(1e-8).reciprocal(), torch.zeros_like(depth_z))
    else:
        target = depth_z
    return target, valid


def oracle_depth_losses(
    rendered: torch.Tensor,
    depth_z: torch.Tensor,
    mode: str = "z",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return visible-pixel value and finite-difference gradient L1 losses."""
    target, valid = prepare_depth_target(depth_z, mode)
    rendered = rendered.reshape_as(target)
    value = torch.sum(torch.abs(rendered - target) * valid) / valid.sum().clamp_min(1)

    dx_valid = valid[..., :, 1:] & valid[..., :, :-1]
    dy_valid = valid[..., 1:, :] & valid[..., :-1, :]
    dx_error = (rendered[..., :, 1:] - rendered[..., :, :-1]) - (
        target[..., :, 1:] - target[..., :, :-1]
    )
    dy_error = (rendered[..., 1:, :] - rendered[..., :-1, :]) - (
        target[..., 1:, :] - target[..., :-1, :]
    )
    dx_loss = torch.sum(torch.abs(dx_error) * dx_valid) / dx_valid.sum().clamp_min(1)
    dy_loss = torch.sum(torch.abs(dy_error) * dy_valid) / dy_valid.sum().clamp_min(1)
    return value, 0.5 * (dx_loss + dy_loss)


def oracle_center_depth_loss(
    xyz: torch.Tensor,
    depth_z: torch.Tensor,
    camera_rotation_transposed: torch.Tensor,
    camera_translation: torch.Tensor,
    fov_x: float,
    fov_y: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project centers and penalize camera-z error to the analytic first hit."""
    z, sampled, valid = sample_center_depths(
        xyz, depth_z, camera_rotation_transposed, camera_translation, fov_x, fov_y
    )
    residual = torch.abs(z - sampled)
    loss = torch.sum(residual * valid) / valid.sum().clamp_min(1)
    return loss, valid.float().mean()


def sample_center_depths(
    xyz: torch.Tensor,
    depth_z: torch.Tensor,
    camera_rotation_transposed: torch.Tensor,
    camera_translation: torch.Tensor,
    fov_x: float,
    fov_y: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a camera-z depth map at projected center locations."""
    camera = xyz @ camera_rotation_transposed + camera_translation
    z = camera[:, 2]
    height, width = depth_z.shape[-2:]
    fx = 0.5 * width / torch.tan(xyz.new_tensor(0.5 * fov_x))
    fy = 0.5 * height / torch.tan(xyz.new_tensor(0.5 * fov_y))
    px = fx * camera[:, 0] / z.clamp_min(1e-8) + 0.5 * width
    py = fy * camera[:, 1] / z.clamp_min(1e-8) + 0.5 * height
    grid = torch.stack([2.0 * px / width - 1.0, 2.0 * py / height - 1.0], dim=1)
    sampled_numerator = F.grid_sample(
        depth_z[None, None], grid[None, :, None], mode="bilinear",
        padding_mode="zeros", align_corners=False,
    ).reshape(-1)
    depth_mask = (depth_z > 0).to(depth_z.dtype)
    sampled_weight = F.grid_sample(
        depth_mask[None, None], grid[None, :, None], mode="bilinear",
        padding_mode="zeros", align_corners=False,
    ).reshape(-1)
    sampled = sampled_numerator / sampled_weight.clamp_min(1e-8)
    valid = (
        (z > 1e-8)
        & (grid[:, 0] >= -1.0) & (grid[:, 0] <= 1.0)
        & (grid[:, 1] >= -1.0) & (grid[:, 1] <= 1.0)
        & (sampled_weight > 1e-4)
    )
    return z, sampled, valid


def robust_affine_fit(
    source: torch.Tensor,
    target: torch.Tensor,
    trim_quantile: float = 0.9,
    iterations: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit target ~= scale * source + shift with iterative residual trimming."""
    keep = torch.isfinite(source) & torch.isfinite(target)
    scale = source.new_tensor(1.0)
    shift = source.new_tensor(0.0)
    for _ in range(max(iterations, 1)):
        if int(keep.sum()) < 2:
            break
        design = torch.stack([source[keep], torch.ones_like(source[keep])], dim=1)
        solution = torch.linalg.lstsq(design, target[keep, None]).solution[:, 0]
        scale, shift = solution[0], solution[1]
        residual = torch.abs(scale * source + shift - target)
        threshold = torch.quantile(residual[keep], trim_quantile)
        keep = keep & (residual <= threshold)
    return scale, shift, keep


@torch.no_grad()
def calibrate_depth_to_centers(
    depth_z: torch.Tensor,
    xyz: torch.Tensor,
    camera_rotation_transposed: torch.Tensor,
    camera_translation: torch.Tensor,
    fov_x: float,
    fov_y: float,
    min_points: int = 32,
) -> tuple[torch.Tensor, float, float, int]:
    """Calibrate a depth prior to fixed initial centers using robust affine fit."""
    z, sampled, valid = sample_center_depths(
        xyz, depth_z, camera_rotation_transposed, camera_translation, fov_x, fov_y
    )
    if int(valid.sum()) < min_points:
        return depth_z, 1.0, 0.0, int(valid.sum())
    scale, shift, inliers = robust_affine_fit(sampled[valid], z[valid])
    calibrated = depth_z.clone()
    depth_valid = calibrated > 0
    calibrated[depth_valid] = (scale * calibrated[depth_valid] + shift).clamp_min(1e-6)
    return calibrated, float(scale.cpu()), float(shift.cpu()), int(inliers.sum())
