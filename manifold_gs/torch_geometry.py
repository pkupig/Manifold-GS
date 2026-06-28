"""Torch geometry helpers for 3DGS tensors."""

from __future__ import annotations

import torch


def normalize_quaternion(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert 3DGS quaternions `[w, x, y, z]` to rotation matrices."""
    q = normalize_quaternion(q)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rot = torch.empty((q.shape[0], 3, 3), dtype=q.dtype, device=q.device)
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - w * z)
    rot[:, 0, 2] = 2 * (x * z + w * y)
    rot[:, 1, 0] = 2 * (x * y + w * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - w * x)
    rot[:, 2, 0] = 2 * (x * z - w * y)
    rot[:, 2, 1] = 2 * (y * z + w * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def sorted_gaussian_geometry(scales: torch.Tensor, rotations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sorted covariance eigenvalues, eigenvectors, and normals.

    Args:
        scales: activated 3DGS scales `(N, 3)`.
        rotations: raw or normalized quaternions `(N, 4)`.
    """
    eigenvalues_unsorted = scales.square()
    order = torch.argsort(eigenvalues_unsorted, dim=-1, descending=True)
    eigenvalues = torch.gather(eigenvalues_unsorted, 1, order)

    rot = quaternion_to_matrix(rotations)
    gather_index = order[:, None, :].expand(-1, 3, -1)
    eigenvectors = torch.gather(rot, 2, gather_index)
    normals = eigenvectors[:, :, 2]
    return eigenvalues, eigenvectors, normals


def surface_mask_from_eigenvalues(
    eigenvalues: torch.Tensor,
    opacity: torch.Tensor,
    surface_r12_min: float = 0.25,
    surface_r23_max: float = 0.08,
    opacity_min: float = 0.02,
) -> torch.Tensor:
    r12 = eigenvalues[:, 1] / eigenvalues[:, 0].clamp_min(1e-24)
    r23 = eigenvalues[:, 2] / eigenvalues[:, 1].clamp_min(1e-24)
    opacity = opacity.reshape(-1)
    return (r12 > surface_r12_min) & (r23 < surface_r23_max) & (opacity > opacity_min)

