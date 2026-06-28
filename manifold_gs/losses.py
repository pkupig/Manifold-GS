"""Prototype manifold-conservative losses.

These functions are deliberately renderer-agnostic. They are meant to be wired
into a 3DGS/2DGS/SuGaR training loop after the offline diagnostics validate the
signal.
"""

from __future__ import annotations

import torch


def gaussian_eigenvalues_from_scales(scales: torch.Tensor) -> torch.Tensor:
    """Return sorted covariance eigenvalues from activated 3DGS scales.

    Args:
        scales: Tensor of shape `(N, 3)` after the official exp activation.
    """
    eigenvalues = scales.square()
    return torch.sort(eigenvalues, dim=-1, descending=True).values


def thinness_loss(eigenvalues: torch.Tensor, threshold: float = 0.04) -> torch.Tensor:
    """Penalize surface splats whose normal thickness is too large."""
    ratio = eigenvalues[:, 2] / (eigenvalues[:, 0] + eigenvalues[:, 1]).clamp_min(1e-12)
    return torch.relu(ratio - threshold).square().mean()


def area_measure_loss(
    eigenvalues: torch.Tensor,
    opacity: torch.Tensor,
    edges: torch.Tensor,
) -> torch.Tensor:
    """Smooth local log area-mass over accepted manifold graph edges.

    Args:
        eigenvalues: Sorted covariance eigenvalues `(N, 3)`.
        opacity: Activated opacity `(N,)` or `(N, 1)`.
        edges: Long tensor `(E, 2)`.
    """
    if edges.numel() == 0:
        return eigenvalues.new_zeros(())
    opacity = opacity.reshape(-1).clamp_min(1e-8)
    tangent_area = (eigenvalues[:, 0] * eigenvalues[:, 1]).clamp_min(1e-24).sqrt()
    log_mass = torch.log(opacity * tangent_area + 1e-12)
    diff = log_mass[edges[:, 0]] - log_mass[edges[:, 1]]
    return torch.nn.functional.smooth_l1_loss(diff, torch.zeros_like(diff))


def curvature_scale_loss(
    xyz: torch.Tensor,
    normals: torch.Tensor,
    eigenvalues: torch.Tensor,
    edges: torch.Tensor,
    threshold: float = 0.35,
) -> torch.Tensor:
    """Large splats should not span high normal variation."""
    if edges.numel() == 0:
        return xyz.new_zeros(())
    i = edges[:, 0]
    j = edges[:, 1]
    dots = torch.sum(normals[i] * normals[j], dim=-1).abs().clamp(0.0, 1.0)
    angles = torch.acos(dots)
    dist = torch.linalg.norm(xyz[i] - xyz[j], dim=-1).clamp_min(1e-8)
    radius = (eigenvalues[i, 0] + eigenvalues[i, 1]).clamp_min(1e-12).sqrt()
    score = angles / dist * radius
    return torch.relu(score - threshold).square().mean()


def rank2_neighborhood_loss(xyz: torch.Tensor, neighbor_indices: torch.Tensor) -> torch.Tensor:
    """Penalize local neighborhoods that are not approximately 2D.

    This is a prototype. In full training, neighbor indices should be refreshed
    periodically instead of every iteration.
    """
    if neighbor_indices.numel() == 0:
        return xyz.new_zeros(())
    pts = xyz[neighbor_indices]  # (N, K, 3)
    centered = pts - pts.mean(dim=1, keepdim=True)
    cov = centered.transpose(1, 2) @ centered / max(1, pts.shape[1])
    eig = torch.linalg.eigvalsh(cov)
    score = eig[:, 0] / eig.sum(dim=-1).clamp_min(1e-12)
    return score.mean()


def normal_consistency_loss(normals: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    if edges.numel() == 0:
        return normals.new_zeros(())
    dots = torch.sum(normals[edges[:, 0]] * normals[edges[:, 1]], dim=-1).abs().clamp(0.0, 1.0)
    return (1.0 - dots).mean()

