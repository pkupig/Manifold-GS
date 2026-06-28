"""Training hooks for manifold-conservative 3DGS optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree

from .manifold_projection import project_points_to_manifold

from .losses import (
    area_measure_loss,
    curvature_scale_loss,
    normal_consistency_loss,
    rank2_neighborhood_loss,
    thinness_loss,
)
from .torch_geometry import sorted_gaussian_geometry, surface_mask_from_eigenvalues


@dataclass
class CompatibilityCache:
    indices: torch.Tensor
    neighbor_indices: torch.Tensor
    derivative_operators: torch.Tensor
    orientation_signs: torch.Tensor
    target_neighbor_normals: torch.Tensor
    frames: torch.Tensor
    support_gauss: torch.Tensor
    confidence: torch.Tensor
    radii: torch.Tensor


@dataclass
class ManifoldGraphState:
    num_points: int
    edges: torch.Tensor
    neighbor_indices: torch.Tensor
    active_indices: torch.Tensor
    proximal_indices: torch.Tensor
    proximal_xyz: torch.Tensor
    proximal_normals: torch.Tensor
    proximal_confidence: torch.Tensor
    proximal_radii: torch.Tensor
    compatibility: CompatibilityCache | None


class ManifoldLossController:
    """Periodic graph refresh and manifold loss computation.

    This class is intentionally small so it can be imported from the official
    3DGS training loop without changing the renderer.
    """

    def __init__(
        self,
        warmup: int = 2000,
        refresh_interval: int = 250,
        knn: int = 12,
        max_points: int = 30000,
        surface_r12_min: float = 0.25,
        surface_r23_max: float = 0.08,
        opacity_min: float = 0.02,
        edge_normal_dot_min: float = 0.7,
        lambda_thin: float = 0.02,
        lambda_area: float = 0.005,
        lambda_curv: float = 0.005,
        lambda_rank2: float = 0.002,
        lambda_normal: float = 0.002,
        lambda_support: float = 0.0,
        lambda_tangent: float = 0.0,
        lambda_symmetry: float = 0.0,
        lambda_gauss: float = 0.0,
        compatibility_start: int | None = None,
        compatibility_ramp: int = 0,
        compatibility_confidence_floor: float = 0.05,
        proximal_iterations: int = 2,
        proximal_step: float = 0.5,
        proximal_min_confidence: float = 0.25,
        bootstrap_thin: bool = False,
    ) -> None:
        self.warmup = warmup
        self.refresh_interval = refresh_interval
        self.knn = knn
        self.max_points = max_points
        self.surface_r12_min = surface_r12_min
        self.surface_r23_max = surface_r23_max
        self.opacity_min = opacity_min
        self.edge_normal_dot_min = edge_normal_dot_min
        self.lambda_thin = lambda_thin
        self.lambda_area = lambda_area
        self.lambda_curv = lambda_curv
        self.lambda_rank2 = lambda_rank2
        self.lambda_normal = lambda_normal
        self.lambda_support = lambda_support
        self.lambda_tangent = lambda_tangent
        self.lambda_symmetry = lambda_symmetry
        self.lambda_gauss = lambda_gauss
        self.compatibility_start = warmup if compatibility_start is None else compatibility_start
        self.compatibility_ramp = compatibility_ramp
        self.compatibility_confidence_floor = compatibility_confidence_floor
        self.proximal_iterations = proximal_iterations
        self.proximal_step = proximal_step
        self.proximal_min_confidence = proximal_min_confidence
        self.bootstrap_thin = bootstrap_thin
        self.state: ManifoldGraphState | None = None
        self.last_refresh = -1
        self.last_terms: dict[str, float] = {}

    def _compatibility_scale(self, iteration: int) -> float:
        if iteration < self.compatibility_start:
            return 0.0
        if self.compatibility_ramp <= 0:
            return 1.0
        return min(1.0, (iteration - self.compatibility_start) / self.compatibility_ramp)

    @property
    def enabled(self) -> bool:
        return any(
            weight > 0
            for weight in (
                self.lambda_thin,
                self.lambda_area,
                self.lambda_curv,
                self.lambda_rank2,
                self.lambda_normal,
                self.lambda_support,
                self.lambda_tangent,
                self.lambda_symmetry,
                self.lambda_gauss,
            )
        )

    def maybe_refresh(
        self,
        iteration: int,
        xyz: torch.Tensor,
        eigenvalues: torch.Tensor,
        normals: torch.Tensor,
        opacity: torch.Tensor,
        geometric_mass: torch.Tensor,
    ) -> None:
        if not self.enabled or iteration < self.warmup:
            return
        point_count_changed = self.state is not None and self.state.num_points != xyz.shape[0]
        if self.state is not None and not point_count_changed and (iteration - self.last_refresh) < self.refresh_interval:
            return
        self.state = self._build_graph(xyz, eigenvalues, normals, opacity, geometric_mass)
        self.last_refresh = iteration

    def _build_graph(
        self,
        xyz: torch.Tensor,
        eigenvalues: torch.Tensor,
        normals: torch.Tensor,
        opacity: torch.Tensor,
        geometric_mass: torch.Tensor,
    ) -> ManifoldGraphState:
        with torch.no_grad():
            mask = surface_mask_from_eigenvalues(
                eigenvalues,
                opacity,
                surface_r12_min=self.surface_r12_min,
                surface_r23_max=self.surface_r23_max,
                opacity_min=self.opacity_min,
            )
            active = torch.nonzero(mask, as_tuple=False).reshape(-1)
            if active.numel() > self.max_points:
                perm = torch.randperm(active.numel(), device=active.device)[: self.max_points]
                active = active[perm].sort().values

            edges = torch.empty((0, 2), dtype=torch.long, device=xyz.device)
            neighbor_indices = torch.empty((0, 0), dtype=torch.long, device=xyz.device)
            if active.numel() >= 2:
                xyz_cpu = xyz[active].detach().float().cpu().numpy()
                normals_cpu = normals[active].detach().float().cpu().numpy()
                k_eff = min(self.knn + 1, active.numel())
                tree = cKDTree(xyz_cpu)
                _, neighbors = tree.query(xyz_cpu, k=k_eff)
                if neighbors.ndim == 1:
                    neighbors = neighbors[:, None]
                neighbors = neighbors[:, 1:]

                rows = []
                cols = []
                for i in range(neighbors.shape[0]):
                    js = neighbors[i]
                    dots = np.abs(normals_cpu[js] @ normals_cpu[i])
                    keep = dots >= self.edge_normal_dot_min
                    for j in js[keep]:
                        rows.append(i)
                        cols.append(int(j))

                if rows:
                    edge_local = torch.tensor(np.stack([rows, cols], axis=1), dtype=torch.long, device=xyz.device)
                    edges = active[edge_local]
                neighbor_indices = active[torch.as_tensor(neighbors, dtype=torch.long, device=xyz.device)]
            proximal_indices = torch.empty((0,), dtype=torch.long, device=xyz.device)
            proximal_xyz = torch.empty((0, 3), dtype=xyz.dtype, device=xyz.device)
            proximal_normals = torch.empty((0, 3), dtype=xyz.dtype, device=xyz.device)
            proximal_confidence = torch.empty((0,), dtype=xyz.dtype, device=xyz.device)
            proximal_radii = torch.empty((0,), dtype=xyz.dtype, device=xyz.device)
            compatibility = None
            if any(weight > 0 for weight in (self.lambda_support, self.lambda_tangent, self.lambda_symmetry, self.lambda_gauss)):
                candidates = torch.nonzero(opacity.reshape(-1) > self.opacity_min, as_tuple=False).reshape(-1)
                if candidates.numel() > self.max_points:
                    candidate_mass = geometric_mass[candidates].reshape(-1)
                    candidates = candidates[torch.topk(candidate_mass, self.max_points, sorted=False).indices]
                if candidates.numel() >= 7:
                    projected = project_points_to_manifold(
                        xyz[candidates].detach().float().cpu().numpy(),
                        mass=geometric_mass[candidates].detach().float().cpu().numpy(),
                        source_indices=np.arange(candidates.numel()),
                        k=self.knn,
                        iterations=self.proximal_iterations,
                        min_confidence=self.proximal_min_confidence,
                        projection_step=self.proximal_step,
                    )
                    accepted = np.flatnonzero(projected.accepted)
                    if accepted.size:
                        accepted_t = torch.as_tensor(accepted, dtype=torch.long, device=xyz.device)
                        proximal_indices = candidates[accepted_t]
                        proximal_xyz = torch.as_tensor(projected.xyz[accepted], dtype=xyz.dtype, device=xyz.device)
                        proximal_normals = torch.as_tensor(projected.normals[accepted], dtype=xyz.dtype, device=xyz.device)
                        proximal_confidence = torch.as_tensor(projected.confidence[accepted], dtype=xyz.dtype, device=xyz.device)
                        proximal_radii = torch.as_tensor(projected.radii[accepted], dtype=xyz.dtype, device=xyz.device)
                        if self.lambda_symmetry > 0 or self.lambda_gauss > 0:
                            compatibility = self._build_compatibility_cache(
                                projected,
                                candidates,
                                geometric_mass[candidates].detach().float().cpu().numpy(),
                                np.arange(candidates.numel()),
                                xyz[candidates].detach().float().cpu().numpy(),
                                xyz.device,
                                xyz.dtype,
                            )

            return ManifoldGraphState(
                num_points=xyz.shape[0],
                edges=edges,
                neighbor_indices=neighbor_indices,
                active_indices=active,
                proximal_indices=proximal_indices,
                proximal_xyz=proximal_xyz,
                proximal_normals=proximal_normals,
                proximal_confidence=proximal_confidence,
                proximal_radii=proximal_radii,
                compatibility=compatibility,
            )

    def _build_compatibility_cache(
        self,
        projected,
        candidates: torch.Tensor,
        mass: np.ndarray,
        selected: np.ndarray,
        support_points: np.ndarray,
        device: torch.device,
        dtype: torch.dtype,
    ) -> CompatibilityCache:
        # Derivatives must live on the trainable center support evaluated at
        # test time. Projected points provide robust normals/confidence only.
        points = np.asarray(support_points, dtype=np.float64)
        normals = projected.normals.astype(np.float64)
        k_eff = min(max(self.knn, 8) + 1, points.shape[0])
        distances, neighbors = cKDTree(points).query(points, k=k_eff)
        if neighbors.ndim == 1:
            neighbors = neighbors[:, None]
            distances = distances[:, None]

        derivative_ops = []
        orientation_signs = []
        frames = []
        support_gauss = []
        radii = []
        for center_index in selected:
            ids = neighbors[center_index]
            delta = points[ids] - points[center_index]
            normal = normals[center_index]
            tangent_delta = delta - (delta @ normal)[:, None] * normal[None, :]
            tangent_norm = np.linalg.norm(tangent_delta, axis=1)
            t1 = tangent_delta[int(np.argmax(tangent_norm))]
            if np.linalg.norm(t1) < 1e-8:
                axis = np.eye(3)[np.argmin(np.abs(normal))]
                t1 = axis - normal * float(axis @ normal)
            t1 /= max(np.linalg.norm(t1), 1e-12)
            t2 = np.cross(normal, t1)
            t2 /= max(np.linalg.norm(t2), 1e-12)
            frame = np.column_stack([t1, t2])
            uv = delta @ frame
            height = delta @ normal
            radius = max(float(distances[center_index, -1]), 1e-8)
            weights = np.exp(-np.sum(uv * uv, axis=1) / (radius * radius)) * np.maximum(mass[ids], 1e-16)
            weights /= max(np.sum(weights), 1e-16)
            root = np.sqrt(weights)

            design1 = np.column_stack([np.ones(len(ids)), uv])
            operator = np.linalg.pinv(design1 * root[:, None], rcond=1e-6) * root[None, :]
            derivative_ops.append(operator[1:3])

            design2 = np.column_stack([
                np.ones(len(ids)), uv[:, 0], uv[:, 1],
                0.5 * uv[:, 0] ** 2, uv[:, 0] * uv[:, 1], 0.5 * uv[:, 1] ** 2,
            ])
            coeff = np.linalg.lstsq(design2 * root[:, None], height * root, rcond=1e-6)[0]
            gradient = coeff[1:3]
            hessian = np.array([[coeff[3], coeff[4]], [coeff[4], coeff[5]]])
            metric = np.eye(2) + np.outer(gradient, gradient)
            second = hessian / np.sqrt(1.0 + float(gradient @ gradient))
            support_gauss.append(float(np.linalg.det(np.linalg.solve(metric, second))))

            signs = np.sign(normals[ids] @ normal)
            signs[signs == 0] = 1.0
            orientation_signs.append(signs)
            frames.append(frame)
            radii.append(radius)

        selected_t = torch.as_tensor(selected, dtype=torch.long, device=device)
        neighbor_local = torch.as_tensor(neighbors[selected], dtype=torch.long, device=device)
        return CompatibilityCache(
            indices=candidates[selected_t],
            neighbor_indices=candidates[neighbor_local],
            derivative_operators=torch.as_tensor(np.stack(derivative_ops), dtype=dtype, device=device),
            orientation_signs=torch.as_tensor(np.stack(orientation_signs), dtype=dtype, device=device),
            target_neighbor_normals=torch.as_tensor(normals[neighbors[selected]], dtype=dtype, device=device),
            frames=torch.as_tensor(np.stack(frames), dtype=dtype, device=device),
            support_gauss=torch.as_tensor(np.asarray(support_gauss), dtype=dtype, device=device),
            confidence=torch.as_tensor(projected.confidence[selected], dtype=dtype, device=device),
            radii=torch.as_tensor(np.asarray(radii), dtype=dtype, device=device),
        )

    def loss(
        self,
        iteration: int,
        xyz: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacity: torch.Tensor,
        geometric_mass: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.enabled or iteration < self.warmup:
            self.last_terms = {}
            return xyz.new_zeros(())

        eigenvalues, _, normals = sorted_gaussian_geometry(scales, rotations)
        if geometric_mass is None:
            geometric_mass = torch.ones((xyz.shape[0],), dtype=xyz.dtype, device=xyz.device)
        self.maybe_refresh(
            iteration,
            xyz,
            eigenvalues.detach(),
            normals.detach(),
            opacity.detach(),
            geometric_mass.detach(),
        )
        state = self.state
        if state is None:
            self.last_terms = {}
            return xyz.new_zeros(())

        total = xyz.new_zeros(())
        terms: dict[str, torch.Tensor] = {}

        active = state.active_indices
        if active.numel() == 0 and self.bootstrap_thin:
            active = torch.nonzero(opacity.reshape(-1) > self.opacity_min, as_tuple=False).reshape(-1)
            if active.numel() > self.max_points:
                active = active[: self.max_points]
        if active.numel() > 0 and self.lambda_thin > 0:
            terms["mcgs_thin"] = thinness_loss(eigenvalues[active])
        if state.edges.numel() > 0:
            if self.lambda_area > 0:
                terms["mcgs_area"] = area_measure_loss(eigenvalues, opacity, state.edges)
            if self.lambda_curv > 0:
                terms["mcgs_curv"] = curvature_scale_loss(xyz, normals, eigenvalues, state.edges)
            if self.lambda_normal > 0:
                terms["mcgs_normal"] = normal_consistency_loss(normals, state.edges)
        if state.neighbor_indices.numel() > 0 and self.lambda_rank2 > 0:
            terms["mcgs_rank2"] = rank2_neighborhood_loss(xyz, state.neighbor_indices)
        if state.proximal_indices.numel() > 0:
            prox = state.proximal_indices
            weights = geometric_mass[prox].reshape(-1) * state.proximal_confidence
            weights = weights / weights.sum().clamp_min(1e-12)
            if self.lambda_support > 0:
                radius = state.proximal_radii.clamp_min(1e-6)
                displacement = (xyz[prox] - state.proximal_xyz) / radius[:, None]
                terms["mcgs_support"] = torch.sum(weights * torch.sum(displacement.square(), dim=1))
            if self.lambda_tangent > 0:
                dots = torch.sum(normals[prox] * state.proximal_normals, dim=1)
                terms["mcgs_tangent"] = torch.sum(weights * (1.0 - dots.square()))
        compatibility = state.compatibility
        compatibility_scale = self._compatibility_scale(iteration)
        if compatibility is not None and compatibility.indices.numel() > 0:
            local_normals = normals[compatibility.neighbor_indices]
            signs = torch.sign(torch.sum(local_normals * compatibility.target_neighbor_normals, dim=2)).detach()
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            local_normals = local_normals * signs[:, :, None]
            normal_derivatives = torch.einsum(
                "pdk,pkc->pdc", compatibility.derivative_operators, local_normals
            )
            second_form = -torch.einsum(
                "pdc,pcb->pdb", normal_derivatives, compatibility.frames
            )
            confidence = compatibility.confidence.clamp_min(self.compatibility_confidence_floor)
            compat_weights = geometric_mass[compatibility.indices].reshape(-1) * confidence
            compat_weights = compat_weights / compat_weights.sum().clamp_min(1e-12)
            if self.lambda_symmetry > 0 and compatibility_scale > 0:
                asymmetry = (second_form[:, 0, 1] - second_form[:, 1, 0]) * compatibility.radii
                terms["mcgs_symmetry"] = torch.sum(compat_weights * asymmetry.square())
            if self.lambda_gauss > 0 and compatibility_scale > 0:
                symmetric_second = 0.5 * (second_form + second_form.transpose(1, 2))
                predicted_gauss = torch.linalg.det(symmetric_second)
                residual = (predicted_gauss - compatibility.support_gauss) * compatibility.radii.square()
                terms["mcgs_gauss"] = torch.sum(compat_weights * residual.square())

        weights = {
            "mcgs_thin": self.lambda_thin,
            "mcgs_area": self.lambda_area,
            "mcgs_curv": self.lambda_curv,
            "mcgs_rank2": self.lambda_rank2,
            "mcgs_normal": self.lambda_normal,
            "mcgs_support": self.lambda_support,
            "mcgs_tangent": self.lambda_tangent,
            "mcgs_symmetry": self.lambda_symmetry * compatibility_scale,
            "mcgs_gauss": self.lambda_gauss * compatibility_scale,
        }
        for name, value in terms.items():
            total = total + weights[name] * value

        self.last_terms = {name: float(value.detach().cpu()) for name, value in terms.items()}
        self.last_terms["mcgs_loss"] = float(total.detach().cpu())
        self.last_terms["mcgs_graph_points"] = float(active.numel())
        self.last_terms["mcgs_graph_edges"] = float(state.edges.shape[0])
        self.last_terms["mcgs_compatibility_scale"] = compatibility_scale
        return total


def add_manifold_args(parser) -> None:
    group = parser.add_argument_group("Manifold-conservative GS")
    group.add_argument("--mcgs_warmup", type=int, default=2000)
    group.add_argument("--mcgs_refresh_interval", type=int, default=250)
    group.add_argument("--mcgs_knn", type=int, default=12)
    group.add_argument("--mcgs_max_points", type=int, default=30000)
    group.add_argument("--mcgs_lambda_thin", type=float, default=0.0)
    group.add_argument("--mcgs_lambda_area", type=float, default=0.0)
    group.add_argument("--mcgs_lambda_curv", type=float, default=0.0)
    group.add_argument("--mcgs_lambda_rank2", type=float, default=0.0)
    group.add_argument("--mcgs_lambda_normal", type=float, default=0.0)
    group.add_argument("--mcgs_lambda_support", type=float, default=0.0)
    group.add_argument("--mcgs_lambda_tangent", type=float, default=0.0)
    group.add_argument("--mcgs_lambda_symmetry", type=float, default=0.0)
    group.add_argument("--mcgs_lambda_gauss", type=float, default=0.0)
    group.add_argument("--mcgs_compatibility_start", type=int, default=None)
    group.add_argument("--mcgs_compatibility_ramp", type=int, default=0)
    group.add_argument("--mcgs_compatibility_confidence_floor", type=float, default=0.05)
    group.add_argument("--mcgs_proximal_iterations", type=int, default=2)
    group.add_argument("--mcgs_proximal_step", type=float, default=0.5)
    group.add_argument("--mcgs_proximal_min_confidence", type=float, default=0.25)
    group.add_argument("--mcgs_bootstrap_thin", action="store_true", default=False)
    group.add_argument("--mcgs_preserve_pruned_mass", action="store_true", default=False)
    group.add_argument(
        "--mcgs_freeze_geometry", action="store_true", default=False,
        help="Optimize appearance only: freeze centers, covariance, and opacity",
    )
    group.add_argument("--mcgs_initial_ply", type=str, default=None, help="Initialize trainable Gaussians from a projected full-attribute PLY")


def controller_from_args(args) -> ManifoldLossController:
    return ManifoldLossController(
        warmup=args.mcgs_warmup,
        refresh_interval=args.mcgs_refresh_interval,
        knn=args.mcgs_knn,
        max_points=args.mcgs_max_points,
        lambda_thin=args.mcgs_lambda_thin,
        lambda_area=args.mcgs_lambda_area,
        lambda_curv=args.mcgs_lambda_curv,
        lambda_rank2=args.mcgs_lambda_rank2,
        lambda_normal=args.mcgs_lambda_normal,
        lambda_support=args.mcgs_lambda_support,
        lambda_tangent=args.mcgs_lambda_tangent,
        lambda_symmetry=args.mcgs_lambda_symmetry,
        lambda_gauss=args.mcgs_lambda_gauss,
        compatibility_start=args.mcgs_compatibility_start,
        compatibility_ramp=args.mcgs_compatibility_ramp,
        compatibility_confidence_floor=args.mcgs_compatibility_confidence_floor,
        proximal_iterations=args.mcgs_proximal_iterations,
        proximal_step=args.mcgs_proximal_step,
        proximal_min_confidence=args.mcgs_proximal_min_confidence,
        bootstrap_thin=args.mcgs_bootstrap_thin,
    )
