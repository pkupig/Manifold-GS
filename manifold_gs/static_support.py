"""Fixed data support built from the input COLMAP point cloud."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .manifold_projection import project_points_to_manifold
from .ply_io import read_vertex_ply


class StaticPointSupport:
    """A non-moving MLS support surface estimated once from COLMAP points."""

    def __init__(
        self,
        ply_path: str | Path,
        device: torch.device,
        k: int = 20,
        max_support_points: int = 0,
    ) -> None:
        rows = read_vertex_ply(ply_path).data
        xyz = np.stack((rows["x"], rows["y"], rows["z"]), axis=1).astype(np.float32)
        projected = project_points_to_manifold(
            xyz, k=k, iterations=2, min_confidence=0.0, projection_step=1.0,
        )
        accepted = projected.accepted
        if int(accepted.sum()) < 4:
            raise ValueError(f"insufficient fixed support in {ply_path}")
        points = projected.xyz[accepted]
        normals = projected.normals[accepted]
        radii = projected.radii[accepted]
        if max_support_points > 0 and points.shape[0] > max_support_points:
            rng = np.random.default_rng(0)
            selected = np.sort(rng.choice(points.shape[0], max_support_points, replace=False))
            points = points[selected]
            normals = normals[selected]
            radii = radii[selected]
        self.points = torch.as_tensor(points, device=device)
        self.normals = torch.as_tensor(normals, device=device)
        self.radii = torch.as_tensor(radii, device=device).clamp_min(1e-4)
        self.scale = float(max(np.median(radii), 1e-4))

    def loss(
        self,
        xyz: torch.Tensor,
        opacity: torch.Tensor,
        max_points: int = 8192,
        opacity_min: float = 0.02,
        tangent_radius_cap: float = 0.0,
    ) -> tuple[torch.Tensor, float]:
        candidates = torch.nonzero(opacity.reshape(-1) > opacity_min, as_tuple=False).reshape(-1)
        if candidates.numel() == 0:
            return xyz.new_zeros(()), 0.0
        if candidates.numel() > max_points:
            values = opacity[candidates].reshape(-1)
            candidates = candidates[torch.topk(values, max_points, sorted=False).indices]
        centers = xyz[candidates]
        # Nearest-patch assignment is discrete; the point-to-plane residual remains differentiable.
        nearest = torch.cdist(centers.detach(), self.points).argmin(dim=1)
        displacement = centers - self.points[nearest]
        signed = torch.sum(displacement * self.normals[nearest], dim=1) / self.scale
        tangent = displacement - torch.sum(
            displacement * self.normals[nearest], dim=1, keepdim=True
        ) * self.normals[nearest]
        tangent_excess = signed.new_zeros(signed.shape)
        if tangent_radius_cap > 0:
            tangent_ratio = torch.linalg.norm(tangent, dim=1) / self.radii[nearest]
            tangent_excess = torch.relu(tangent_ratio - tangent_radius_cap)
        weights = opacity[candidates].reshape(-1).detach()
        weights = weights / weights.sum().clamp_min(1e-8)
        value = torch.sum(weights * torch.sqrt(signed.square() + tangent_excess.square() + 1e-6))
        return value, float(candidates.numel() / max(xyz.shape[0], 1))
