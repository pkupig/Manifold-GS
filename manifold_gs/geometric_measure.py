"""Conservative operations on discrete unoriented surface measures."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MeasureMoments:
    mass: torch.Tensor
    first_moment: torch.Tensor
    tangent_moment: torch.Tensor


def measure_moments(xyz: torch.Tensor, normals: torch.Tensor, mass: torch.Tensor) -> MeasureMoments:
    mass = mass.reshape(-1)
    normal_projectors = normals[:, :, None] @ normals[:, None, :]
    return MeasureMoments(
        mass=mass.sum(),
        first_moment=torch.sum(mass[:, None] * xyz, dim=0),
        tangent_moment=torch.sum(mass[:, None, None] * normal_projectors, dim=0),
    )


def conservative_split(
    parent_xyz: torch.Tensor,
    parent_normals: torch.Tensor,
    parent_mass: torch.Tensor,
    offsets: torch.Tensor,
    fractions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split each parent while preserving mass and first moment exactly.

    Args:
        parent_xyz: `(P, 3)` centers.
        parent_normals: `(P, 3)` unoriented normals.
        parent_mass: `(P,)` positive geometric masses.
        offsets: proposed child offsets `(P, C, 3)`.
        fractions: optional positive child mass fractions `(P, C)`.

    Proposed offsets are recentered using the child mass fractions. This turns a
    stochastic 3DGS split into a pointwise conservative operation rather than a
    conservation statement that holds only in expectation.
    """
    if offsets.ndim != 3 or offsets.shape[0] != parent_xyz.shape[0] or offsets.shape[2] != 3:
        raise ValueError("offsets must have shape (P, C, 3)")
    child_count = offsets.shape[1]
    if fractions is None:
        fractions = torch.full(
            offsets.shape[:2], 1.0 / child_count, dtype=parent_xyz.dtype, device=parent_xyz.device
        )
    fractions = fractions.clamp_min(0.0)
    fractions = fractions / fractions.sum(dim=1, keepdim=True).clamp_min(1e-12)
    centered_offsets = offsets - torch.sum(fractions[:, :, None] * offsets, dim=1, keepdim=True)
    child_xyz = parent_xyz[:, None, :] + centered_offsets
    child_normals = parent_normals[:, None, :].expand_as(child_xyz)
    child_mass = parent_mass.reshape(-1, 1) * fractions
    return child_xyz, child_normals, child_mass


def conservative_merge(
    child_xyz: torch.Tensor,
    child_normals: torch.Tensor,
    child_mass: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge groups `(P, C, ...)` while preserving zeroth/first moments."""
    if child_xyz.ndim != 3 or child_xyz.shape != child_normals.shape:
        raise ValueError("child_xyz and child_normals must have shape (P, C, 3)")
    mass = child_mass.reshape(child_xyz.shape[:2]).clamp_min(0.0)
    merged_mass = mass.sum(dim=1)
    merged_xyz = torch.sum(mass[:, :, None] * child_xyz, dim=1) / merged_mass[:, None].clamp_min(1e-12)

    projectors = child_normals[:, :, :, None] @ child_normals[:, :, None, :]
    normal_moment = torch.sum(mass[:, :, None, None] * projectors, dim=1)
    _, eigenvectors = torch.linalg.eigh(normal_moment)
    merged_normals = eigenvectors[:, :, -1]
    return merged_xyz, merged_normals, merged_mass


def redistribute_pruned_mass(
    xyz: torch.Tensor,
    mass: torch.Tensor,
    prune_mask: torch.Tensor,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move pruned mass to nearest retained samples and return transport cost.

    Zeroth mass is exact. First-moment error is bounded by the returned
    mass-weighted transport distance.
    """
    keep_mask = ~prune_mask.reshape(-1)
    mass = mass.reshape(-1)
    if not torch.any(keep_mask):
        raise ValueError("Cannot redistribute mass when every sample is pruned")
    kept_mass = mass[keep_mask].clone()
    removed = torch.nonzero(~keep_mask, as_tuple=False).reshape(-1)
    if removed.numel() == 0:
        return kept_mass, mass.new_zeros(())

    kept_xyz = xyz[keep_mask].detach()
    transferred = torch.zeros_like(kept_mass)
    transport_cost = mass.new_zeros(())
    for start in range(0, removed.numel(), chunk_size):
        ids = removed[start:start + chunk_size]
        distances = torch.cdist(xyz[ids].detach(), kept_xyz)
        nearest_distance, nearest = torch.min(distances, dim=1)
        transferred.scatter_add_(0, nearest, mass[ids])
        transport_cost = transport_cost + torch.sum(mass[ids] * nearest_distance)
    return kept_mass + transferred, transport_cost


def conservation_residual(
    before: MeasureMoments,
    after: MeasureMoments,
) -> dict[str, torch.Tensor]:
    return {
        "mass": torch.abs(after.mass - before.mass),
        "first_moment": torch.linalg.norm(after.first_moment - before.first_moment),
        "tangent_moment": torch.linalg.norm(after.tangent_moment - before.tangent_moment),
    }
