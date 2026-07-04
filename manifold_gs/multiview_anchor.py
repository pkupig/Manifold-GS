"""Image-coercive multi-view constraints for Gaussian centers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def project_centers(points: torch.Tensor, camera) -> tuple[torch.Tensor, torch.Tensor]:
    """Project world points to a grid_sample grid and camera-space z depth."""
    homogeneous = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    clip = homogeneous @ camera.full_proj_transform
    grid = clip[:, :2] / clip[:, 3:4].clamp_min(1e-8)
    view = homogeneous @ camera.world_view_transform
    return grid, view[:, 2]


def _sample(image: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    values = F.grid_sample(
        image.unsqueeze(0), grid.view(1, -1, 1, 2), mode="bilinear",
        padding_mode="zeros", align_corners=False,
    )
    return values[0, :, :, 0].transpose(0, 1)


def _texture_strength(image: torch.Tensor) -> torch.Tensor:
    dx = F.pad((image[:, :, 1:] - image[:, :, :-1]).abs().mean(0, keepdim=True), (0, 1, 0, 0))
    dy = F.pad((image[:, 1:, :] - image[:, :-1, :]).abs().mean(0, keepdim=True), (0, 0, 0, 1))
    return dx + dy


def _center_zbuffer_visibility(
    grid: torch.Tensor, depth: torch.Tensor, height: int, width: int, tolerance: float,
) -> torch.Tensor:
    """Approximate visibility with a detached nearest-center z-buffer."""
    x = ((grid[:, 0].detach() + 1) * 0.5 * width).long().clamp(0, width - 1)
    y = ((grid[:, 1].detach() + 1) * 0.5 * height).long().clamp(0, height - 1)
    bins = y * width + x
    nearest = torch.full((height * width,), torch.inf, device=depth.device, dtype=depth.dtype)
    nearest.scatter_reduce_(0, bins, depth.detach(), reduce="amin", include_self=True)
    return depth.detach() <= nearest[bins] * (1.0 + tolerance)


def multiview_center_loss(
    points: torch.Tensor,
    current_camera,
    paired_camera,
    paired_inverse_depth: torch.Tensor,
    candidate_indices: torch.Tensor,
    opacity: torch.Tensor,
    *,
    occlusion_tolerance: float = 0.08,
    texture_floor: float = 0.01,
) -> tuple[torch.Tensor, float]:
    """Match GT colors across views while using rendered depth only as a detached mask."""
    if candidate_indices.numel() == 0:
        return points.new_zeros(()), 0.0
    selected = points[candidate_indices]
    current_grid, current_z = project_centers(selected, current_camera)
    paired_grid, paired_z = project_centers(selected, paired_camera)

    in_current = (current_grid.abs() < 0.98).all(dim=1) & (current_z > 1e-4)
    in_paired = (paired_grid.abs() < 0.98).all(dim=1) & (paired_z > 1e-4)
    if bool(torch.any(paired_inverse_depth.detach() > 0)):
        sampled_inverse_depth = _sample(paired_inverse_depth.detach(), paired_grid).squeeze(1)
        expected_inverse_depth = paired_z.detach().reciprocal()
        depth_relative_error = (
            (sampled_inverse_depth - expected_inverse_depth).abs()
            / expected_inverse_depth.abs().clamp_min(1e-4)
        )
        visible = (sampled_inverse_depth > 0) & (depth_relative_error < occlusion_tolerance)
    else:
        height, width = paired_camera.original_image.shape[-2:]
        visible = _center_zbuffer_visibility(
            paired_grid, paired_z, height, width, occlusion_tolerance,
        )

    current_color = _sample(current_camera.original_image, current_grid)
    paired_color = _sample(paired_camera.original_image, paired_grid)
    current_texture = _sample(_texture_strength(current_camera.original_image), current_grid).squeeze(1)
    paired_texture = _sample(_texture_strength(paired_camera.original_image), paired_grid).squeeze(1)
    texture = torch.maximum(current_texture, paired_texture).detach()
    valid = in_current & in_paired & visible & (texture >= texture_floor)
    if not torch.any(valid):
        return points.new_zeros(()), 0.0

    residual = torch.sqrt((current_color - paired_color).square() + 1e-6).mean(dim=1)
    weights = opacity[candidate_indices].reshape(-1).detach() * texture
    weights = weights[valid] / weights[valid].sum().clamp_min(1e-8)
    return torch.sum(weights * residual[valid]), float(valid.float().mean().detach().cpu())
