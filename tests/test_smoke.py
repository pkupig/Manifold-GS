from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manifold_gs.diagnostics import compute_diagnostics
from manifold_gs.asset_bundle import export_asset_bundle
from manifold_gs.analytic_scene import create_scene_mesh, rasterize_mesh, sample_analytic_surface, look_at_w2c
from manifold_gs.gt_metrics import geometry_metrics, normalized_kernel_varifold_distance
from manifold_gs.geometric_measure import (
    conservation_residual,
    conservative_merge,
    conservative_split,
    measure_moments,
    redistribute_pruned_mass,
)
from manifold_gs.manifold_projection import project_points_to_manifold, relax_certified_quadrature
from manifold_gs.fundamental_compatibility import compute_fundamental_compatibility, summarize_compatibility
from manifold_gs.losses import (
    area_measure_loss,
    curvature_scale_loss,
    gaussian_eigenvalues_from_scales,
    normal_consistency_loss,
    rank2_neighborhood_loss,
    thinness_loss,
)
from manifold_gs.patch_mesh import build_patch_mesh, save_patch_mesh
from manifold_gs.training_hooks import ManifoldLossController
from manifold_gs.oracle_depth import (
    calibrate_depth_to_centers,
    oracle_center_depth_loss,
    oracle_depth_losses,
    perturb_depth_target,
    robust_affine_fit,
)
from manifold_gs.multiview_anchor import multiview_center_loss, project_centers
from manifold_gs.static_support import StaticPointSupport


def write_plane_ply(path: Path, n: int = 8) -> None:
    props = [
        "x", "y", "z", "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity", "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    rows = []
    for i in range(n):
        for j in range(n):
            rows.append((
                (i - (n - 1) / 2) * 0.05,
                (j - (n - 1) / 2) * 0.05,
                0.0,
                0, 0, 0,
                0, 0, 0,
                2.0,
                np.log(0.045), np.log(0.045), np.log(0.003),
                1, 0, 0, 0,
            ))
    with path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(rows)}\n")
        for name in props:
            f.write(f"property float {name}\n")
        f.write("end_header\n")
        for row in rows:
            f.write(" ".join(map(str, row)) + "\n")


def test_patch_mesh_smoke(tmp_path: Path) -> None:
    ply = tmp_path / "plane.ply"
    write_plane_ply(ply)
    diag = compute_diagnostics(ply)
    mesh = build_patch_mesh(diag, min_patch_size=5, k=8)
    assert mesh.vertices.shape[0] == 64
    assert mesh.faces.shape[0] > 0

    mesh_path = tmp_path / "patch.ply"
    meta_path = tmp_path / "patch.npz"
    save_patch_mesh(mesh, mesh_path, meta_path)
    manifest = export_asset_bundle(ply, mesh_path, meta_path, tmp_path / "asset", 1)
    assert manifest["source_gaussians"] == 64
    assert manifest["attached_gaussians"] == 64
    assert manifest["residual_gaussians"] == 0
    assert manifest["nonmanifold_edges"] == 0
    assert (tmp_path / "asset" / "certified_patches.obj").is_file()
    assert (tmp_path / "asset" / "certified_patches.mtl").is_file()
    assert (tmp_path / "asset" / "asset_mapping.npz").is_file()


def test_losses_are_differentiable() -> None:
    xyz = torch.randn(6, 3, requires_grad=True)
    scales = torch.exp(torch.randn(6, 3) * 0.1).requires_grad_(True)
    eigenvalues = gaussian_eigenvalues_from_scales(scales)
    normals = torch.nn.functional.normalize(torch.randn(6, 3), dim=-1)
    opacity = torch.sigmoid(torch.randn(6, 1))
    edges = torch.tensor([[0, 1], [1, 2], [3, 4]], dtype=torch.long)
    neighbors = torch.tensor([
        [1, 2, 3],
        [0, 2, 3],
        [0, 1, 3],
        [0, 1, 2],
        [0, 1, 5],
        [0, 1, 4],
    ], dtype=torch.long)

    loss = (
        thinness_loss(eigenvalues)
        + area_measure_loss(eigenvalues, opacity, edges)
        + curvature_scale_loss(xyz, normals, eigenvalues, edges)
        + normal_consistency_loss(normals, edges)
        + rank2_neighborhood_loss(xyz, neighbors)
    )
    loss.backward()
    assert xyz.grad is not None
    assert scales.grad is not None


def test_oracle_depth_value_and_gradient_losses() -> None:
    target = torch.tensor([[[1.0, 2.0], [0.0, 4.0]]])
    rendered = target.clone().requires_grad_(True)
    value, gradient = oracle_depth_losses(rendered, target, mode="z")
    assert value.item() == 0.0
    assert gradient.item() == 0.0

    inverse = torch.where(target > 0, target.clamp_min(1e-8).reciprocal(), torch.zeros_like(target))
    value, gradient = oracle_depth_losses(inverse, target, mode="inverse")
    assert value.item() == 0.0
    assert gradient.item() == 0.0

    perturbed = (target + torch.tensor([[[0.1, -0.2], [0.0, 0.3]]])).requires_grad_(True)
    value, gradient = oracle_depth_losses(perturbed, target, mode="z")
    (value + gradient).backward()
    assert value.item() > 0
    assert gradient.item() > 0
    assert perturbed.grad is not None


def test_oracle_center_depth_loss_projects_camera_z() -> None:
    xyz = torch.tensor([[0.0, 0.0, 2.2], [0.2, -0.1, 1.8]], requires_grad=True)
    target = torch.full((16, 16), 2.0)
    loss, coverage = oracle_center_depth_loss(
        xyz, target, torch.eye(3), torch.zeros(3), 1.0, 1.0
    )
    assert torch.isclose(loss, torch.tensor(0.2), atol=1e-6)
    assert coverage.item() == 1.0
    loss.backward()
    assert xyz.grad is not None
    assert torch.linalg.norm(xyz.grad) > 0


def test_center_depth_sampling_normalizes_missing_pixels() -> None:
    xyz = torch.tensor([[0.0, 0.0, 2.0]], requires_grad=True)
    depth = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
    loss, coverage = oracle_center_depth_loss(
        xyz, depth, torch.eye(3), torch.zeros(3), 1.0, 1.0
    )
    assert loss.item() < 1e-6
    assert coverage.item() == 1.0


def test_depth_target_perturbation_is_reproducible() -> None:
    depth = np.array([[0.0, 2.0, 2.1], [1.9, 2.2, 0.0]], dtype=np.float32)
    first = perturb_depth_target(depth, noise_fraction=0.01, dropout=0.25, seed=17)
    second = perturb_depth_target(depth, noise_fraction=0.01, dropout=0.25, seed=17)
    assert np.array_equal(first, second)
    assert np.all(first[depth == 0] == 0)
    assert np.all(first >= 0)

    affine = perturb_depth_target(depth, scale=1.1, bias_fraction=0.05, seed=3)
    median = np.median(depth[depth > 0])
    assert np.allclose(affine[depth > 0], 1.1 * depth[depth > 0] + 0.05 * median)
    structured = perturb_depth_target(depth, low_frequency_fraction=0.02, seed=4)
    assert not np.allclose(structured[depth > 0], depth[depth > 0])
    assert np.all(structured[depth == 0] == 0)


def test_robust_affine_depth_fit_rejects_outlier() -> None:
    source = torch.linspace(1.0, 3.0, 100)
    target = 0.8 * source - 0.15
    target[-1] = 20.0
    scale, shift, inliers = robust_affine_fit(source, target)
    assert torch.isclose(scale, torch.tensor(0.8), atol=1e-4)
    assert torch.isclose(shift, torch.tensor(-0.15), atol=1e-4)
    assert not inliers[-1]


def test_training_controller_smoke() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = 20
    grid_x, grid_y = torch.meshgrid(
        torch.linspace(-0.2, 0.2, 5, device=device),
        torch.linspace(-0.2, 0.2, 4, device=device),
        indexing="ij",
    )
    xyz = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), torch.zeros(n, device=device)], dim=-1).requires_grad_(True)
    scales = torch.tensor([0.05, 0.05, 0.003], device=device).repeat(n, 1).requires_grad_(True)
    rotations = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n, 1).requires_grad_(True)
    opacity = torch.full((n, 1), 0.5, device=device)

    controller = ManifoldLossController(
        warmup=0,
        refresh_interval=1,
        knn=4,
        lambda_thin=0.1,
        lambda_area=0.1,
        lambda_curv=0.1,
        lambda_rank2=0.1,
        lambda_normal=0.1,
    )
    loss = controller.loss(1, xyz, scales, rotations, opacity)
    loss.backward()
    assert controller.state is not None
    assert controller.state.edges.shape[0] > 0
    assert xyz.grad is not None
    assert scales.grad is not None


def test_training_controller_bootstrap_thin() -> None:
    xyz = torch.randn(6, 3, requires_grad=True)
    scales = torch.ones(6, 3, requires_grad=True)
    rotations = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(6, 1).requires_grad_(True)
    opacity = torch.full((6, 1), 0.5)
    controller = ManifoldLossController(
        warmup=0,
        lambda_thin=0.1,
        bootstrap_thin=True,
    )
    loss = controller.loss(1, xyz, scales, rotations, opacity)
    loss.backward()
    assert loss.item() > 0
    assert scales.grad is not None


def test_analytic_sphere_is_self_consistent() -> None:
    xyz, normals, weights = sample_analytic_surface("sphere", 1000, seed=3)
    assert np.isclose(weights.sum(), 4.0 * np.pi * 0.7**2, rtol=1e-5)
    assert np.allclose(np.linalg.norm(xyz, axis=1), 0.7, atol=1e-6)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)

    metrics = geometry_metrics(xyz, normals, xyz, normals, bbox_diagonal=2.0 * 0.7 * np.sqrt(3.0))
    assert metrics["chamfer_l1"] == 0.0
    distance = normalized_kernel_varifold_distance(
        xyz, normals, weights, xyz, normals, weights, sigma=0.1, max_points=256
    )
    assert distance < 1e-7


def test_cpu_rasterizer_produces_geometry() -> None:
    mesh = create_scene_mesh("sphere", resolution=16)
    rotation, translation = look_at_w2c(np.array([0.0, 0.0, 2.4]))
    view = rasterize_mesh(mesh, rotation, translation, width=48, height=48, focal=48.0)
    assert view.mask.sum() > 100
    assert np.all(view.depth[view.mask] > 0.0)
    assert np.allclose(np.linalg.norm(view.normal_world[view.mask], axis=1), 1.0, atol=1e-5)


def test_split_merge_conserves_discrete_measure() -> None:
    xyz = torch.tensor([[0.2, -0.1, 0.4], [-0.3, 0.5, 0.1]], dtype=torch.float64)
    normals = torch.nn.functional.normalize(
        torch.tensor([[0.1, 0.2, 1.0], [0.0, 1.0, 0.2]], dtype=torch.float64), dim=1
    )
    mass = torch.tensor([0.7, 1.3], dtype=torch.float64)
    offsets = torch.tensor(
        [[[0.2, 0.0, 0.0], [-0.1, 0.1, 0.0], [0.0, -0.3, 0.1]],
         [[0.0, 0.2, 0.0], [0.1, -0.1, 0.2], [-0.2, 0.0, -0.1]]],
        dtype=torch.float64,
    )
    fractions = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]], dtype=torch.float64)
    child_xyz, child_normals, child_mass = conservative_split(xyz, normals, mass, offsets, fractions)
    before = measure_moments(xyz, normals, mass)
    after_split = measure_moments(
        child_xyz.reshape(-1, 3), child_normals.reshape(-1, 3), child_mass.reshape(-1)
    )
    split_residual = conservation_residual(before, after_split)
    assert split_residual["mass"] < 1e-12
    assert split_residual["first_moment"] < 1e-12
    assert split_residual["tangent_moment"] < 1e-12

    merged_xyz, merged_normals, merged_mass = conservative_merge(child_xyz, child_normals, child_mass)
    after_merge = measure_moments(merged_xyz, merged_normals, merged_mass)
    merge_residual = conservation_residual(before, after_merge)
    assert merge_residual["mass"] < 1e-12
    assert merge_residual["first_moment"] < 1e-12
    assert merge_residual["tangent_moment"] < 1e-12


def test_prune_redistribution_preserves_mass_and_bounds_first_moment() -> None:
    xyz = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mass = torch.tensor([1.0, 2.0, 3.0])
    prune = torch.tensor([False, True, False])
    retained_mass, transport_cost = redistribute_pruned_mass(xyz, mass, prune)
    retained_xyz = xyz[~prune]
    before_moment = torch.sum(mass[:, None] * xyz, dim=0)
    after_moment = torch.sum(retained_mass[:, None] * retained_xyz, dim=0)

    assert torch.allclose(retained_mass.sum(), mass.sum())
    assert torch.linalg.norm(after_moment - before_moment) <= transport_cost + 1e-7
    assert torch.allclose(transport_cost, torch.tensor(0.2))


def test_mls_projection_reduces_sphere_noise() -> None:
    clean, _, weights = sample_analytic_surface("sphere", 1200, seed=11)
    rng = np.random.default_rng(12)
    radial = clean / np.linalg.norm(clean, axis=1, keepdims=True)
    noise = rng.normal(0.0, 0.025, (clean.shape[0], 1))
    noisy = clean + noise * radial
    before = np.mean(np.abs(np.linalg.norm(noisy, axis=1) - 0.7))
    projected = project_points_to_manifold(
        noisy,
        mass=weights,
        k=16,
        iterations=2,
        max_normal_ratio=0.35,
        min_tangent_ratio=0.08,
        min_confidence=0.1,
        projection_step=0.5,
    )
    after = np.mean(np.abs(np.linalg.norm(projected.xyz, axis=1) - 0.7))
    assert projected.xyz.shape[0] == noisy.shape[0]
    assert projected.accepted.mean() > 0.9
    assert after < before


def test_quadratic_projection_reduces_plane_and_torus_noise() -> None:
    for scene in ("plane", "torus"):
        clean, normals, weights = sample_analytic_surface(scene, 800, seed=21)
        rng = np.random.default_rng(22)
        noisy = clean + rng.normal(0.0, 0.02, (clean.shape[0], 1)) * normals
        dense_gt, _, _ = sample_analytic_surface(scene, 5000, seed=23)
        tree = __import__("scipy").spatial.cKDTree(dense_gt)
        before = np.mean(tree.query(noisy, k=1)[0])
        projected = project_points_to_manifold(
            noisy,
            mass=weights,
            k=16,
            iterations=2,
            max_normal_ratio=0.4,
            min_tangent_ratio=0.06,
            min_confidence=0.05,
            projection_step=0.5,
        )
        after = np.mean(tree.query(projected.xyz, k=1)[0])
        assert after < before


def test_fundamental_compatibility_detects_invalid_normal_field() -> None:
    xyz, normals, weights = sample_analytic_surface("sphere", 900, seed=31)
    good = summarize_compatibility(compute_fundamental_compatibility(xyz, normals, weights, k=20))
    shuffled = normals[np.random.default_rng(32).permutation(len(normals))]
    bad = summarize_compatibility(compute_fundamental_compatibility(xyz, shuffled, weights, k=20))
    assert good["normal_curl_scaled_median"] < 0.05 * bad["normal_curl_scaled_median"]
    assert good["gauss_residual_scaled_median"] < 0.1 * bad["gauss_residual_scaled_median"]
    assert good["codazzi_residual_scaled_median"] < 0.1 * bad["codazzi_residual_scaled_median"]


def test_fundamental_compatibility_is_normal_sign_invariant() -> None:
    xyz, normals, weights = sample_analytic_surface("sphere", 700, seed=33)
    signs = np.random.default_rng(34).choice([-1.0, 1.0], size=(len(normals), 1))
    base = summarize_compatibility(compute_fundamental_compatibility(xyz, normals, weights, k=20))
    flipped = summarize_compatibility(compute_fundamental_compatibility(xyz, normals * signs, weights, k=20))
    assert np.isclose(base["gauss_residual_scaled_median"], flipped["gauss_residual_scaled_median"], rtol=1e-5)
    assert np.isclose(base["normal_curl_scaled_median"], flipped["normal_curl_scaled_median"], rtol=1e-5)


def test_fundamental_compatibility_is_scale_invariant() -> None:
    xyz, normals, weights = sample_analytic_surface("sphere", 600, seed=17)
    summaries = []
    for scale in (1e-3, 1.0, 1e3):
        result = compute_fundamental_compatibility(
            xyz * scale, normals.copy(), weights * scale * scale, k=20
        )
        summaries.append(summarize_compatibility(result))

    dimensionless = (
        "normal_angle_deg_median",
        "shape_relative_median",
        "symmetry_residual_median",
        "gauss_residual_scaled_median",
        "normal_curl_scaled_median",
        "codazzi_residual_scaled_median",
        "planarity_median",
        "normal_alignment_abs_p10",
        "linear_gram_min_p10",
        "quadratic_gram_min_p10",
        "knn_gap_ratio_p10",
        "normal_eigengap_ratio_p10",
    )
    for metric in dimensionless:
        values = np.asarray([summary[metric] for summary in summaries])
        assert np.allclose(values, values[1], rtol=2e-3, atol=2e-5), (metric, values)


def test_proximal_controller_bootstraps_without_surface_splats() -> None:
    clean, _, weights_np = sample_analytic_surface("sphere", 180, seed=41)
    xyz = torch.tensor(clean, dtype=torch.float32, requires_grad=True)
    scales = torch.full((len(clean), 3), 0.08, requires_grad=True)
    rotations = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(len(clean), 1).requires_grad_(True)
    opacity = torch.full((len(clean), 1), 0.5)
    geometric_mass = torch.tensor(weights_np)
    controller = ManifoldLossController(
        warmup=0,
        refresh_interval=10,
        knn=16,
        lambda_thin=0.0,
        lambda_area=0.0,
        lambda_curv=0.0,
        lambda_rank2=0.0,
        lambda_normal=0.0,
        lambda_support=0.1,
        lambda_tangent=0.1,
    )
    loss = controller.loss(1, xyz, scales, rotations, opacity, geometric_mass)
    loss.backward()
    assert controller.state is not None
    assert controller.state.active_indices.numel() == 0
    assert controller.state.proximal_indices.numel() > 0.8 * len(clean)
    assert "mcgs_support" in controller.last_terms
    assert "mcgs_tangent" in controller.last_terms
    assert xyz.grad is not None
    assert rotations.grad is not None


def test_fundamental_compatibility_losses_are_differentiable() -> None:
    clean, _, weights_np = sample_analytic_surface("sphere", 160, seed=43)
    xyz = torch.tensor(clean, dtype=torch.float32, requires_grad=True)
    scales = torch.tensor([0.08, 0.07, 0.02]).repeat(len(clean), 1).requires_grad_(True)
    rng = torch.Generator().manual_seed(44)
    rotations = torch.randn((len(clean), 4), generator=rng).requires_grad_(True)
    opacity = torch.full((len(clean), 1), 0.5)
    geometric_mass = torch.tensor(weights_np)
    controller = ManifoldLossController(
        warmup=0,
        knn=16,
        lambda_thin=0.0,
        lambda_area=0.0,
        lambda_curv=0.0,
        lambda_rank2=0.0,
        lambda_normal=0.0,
        lambda_tangent=0.01,
        lambda_shape=0.01,
        lambda_symmetry=0.01,
        lambda_gauss=0.01,
    )
    loss = controller.loss(1, xyz, scales, rotations, opacity, geometric_mass)
    loss.backward()
    assert torch.isfinite(loss)
    assert "mcgs_shape" in controller.last_terms
    assert "mcgs_symmetry" in controller.last_terms
    assert "mcgs_gauss" in controller.last_terms
    assert rotations.grad is not None
    assert torch.isfinite(rotations.grad).all()
    assert torch.linalg.norm(rotations.grad) > 0


def test_shape_only_builds_compatibility_cache() -> None:
    clean, _, weights = sample_analytic_surface("sphere", 120, seed=31)
    xyz = torch.tensor(clean, dtype=torch.float32, requires_grad=True)
    scales = torch.tensor([0.08, 0.07, 0.02]).repeat(len(clean), 1).requires_grad_(True)
    rotations = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(len(clean), 1).requires_grad_(True)
    opacity = torch.full((len(clean), 1), 0.8)
    mass = torch.tensor(weights, dtype=torch.float32)
    controller = ManifoldLossController(
        warmup=0,
        refresh_interval=1,
        knn=12,
        lambda_shape=0.001,
        compatibility_start=0,
    )
    loss = controller.loss(1, xyz, scales, rotations, opacity, mass)
    loss.backward()
    assert controller.state is not None
    assert controller.state.compatibility is not None
    assert "mcgs_shape" in controller.last_terms
    assert "mcgs_alignment_min" in controller.last_terms
    assert "mcgs_gram_min" in controller.last_terms
    assert controller.last_terms["mcgs_cache_drift_max"] < 1e-6
    assert rotations.grad is not None
    assert torch.isfinite(rotations.grad).all()
    assert torch.linalg.norm(rotations.grad) > 0


def test_compatibility_cache_can_refresh_on_relative_drift(tmp_path: Path) -> None:
    clean, _, weights = sample_analytic_surface("sphere", 120, seed=32)
    xyz = torch.tensor(clean, dtype=torch.float32, requires_grad=True)
    scales = torch.tensor([0.08, 0.07, 0.02]).repeat(len(clean), 1)
    rotations = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(len(clean), 1)
    opacity = torch.full((len(clean), 1), 0.8)
    mass = torch.tensor(weights, dtype=torch.float32)
    controller = ManifoldLossController(
        warmup=0,
        refresh_interval=100,
        knn=12,
        lambda_shape=0.001,
        compatibility_start=0,
        compatibility_max_cache_drift=0.01,
    )
    controller.loss(1, xyz, scales, rotations, opacity, mass)
    controller.record_diagnostics(1)
    assert controller.last_refresh == 1
    with torch.no_grad():
        xyz[0, 0] += 0.05
    controller.loss(2, xyz, scales, rotations, opacity, mass)
    controller.record_diagnostics(2)
    assert controller.last_refresh == 2
    output = tmp_path / "mcgs_diagnostics.json"
    controller.export_diagnostics(output)
    payload = __import__("json").loads(output.read_text())
    assert payload["refresh_count"] == 2
    assert payload["history"][-1]["iteration"] == 2


def test_fundamental_compatibility_schedule() -> None:
    clean, _, weights_np = sample_analytic_surface("sphere", 160, seed=45)
    xyz = torch.tensor(clean, dtype=torch.float32, requires_grad=True)
    scales = torch.tensor([0.08, 0.07, 0.02]).repeat(len(clean), 1).requires_grad_(True)
    rotations = torch.randn((len(clean), 4), generator=torch.Generator().manual_seed(46)).requires_grad_(True)
    opacity = torch.full((len(clean), 1), 0.5)
    controller = ManifoldLossController(
        warmup=0,
        refresh_interval=100,
        knn=16,
        lambda_thin=0.0,
        lambda_area=0.0,
        lambda_curv=0.0,
        lambda_rank2=0.0,
        lambda_normal=0.0,
        lambda_tangent=0.01,
        lambda_symmetry=0.01,
        compatibility_start=10,
        compatibility_ramp=20,
    )
    mass = torch.tensor(weights_np)

    controller.loss(5, xyz, scales, rotations, opacity, mass)
    assert "mcgs_symmetry" not in controller.last_terms
    assert controller.last_terms["mcgs_compatibility_scale"] == 0.0

    controller.loss(20, xyz, scales, rotations, opacity, mass)
    assert "mcgs_symmetry" in controller.last_terms
    assert controller.last_terms["mcgs_compatibility_scale"] == 0.5

    controller.loss(35, xyz, scales, rotations, opacity, mass)
    assert controller.last_terms["mcgs_compatibility_scale"] == 1.0


def test_graph_refreshes_immediately_after_point_count_change() -> None:
    clean, _, weights_np = sample_analytic_surface("sphere", 80, seed=47)
    xyz = torch.tensor(clean, dtype=torch.float32)
    scales = torch.tensor([0.08, 0.07, 0.02]).repeat(len(clean), 1)
    rotations = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(len(clean), 1)
    opacity = torch.full((len(clean), 1), 0.5)
    mass = torch.tensor(weights_np)
    controller = ManifoldLossController(
        warmup=0,
        refresh_interval=100,
        knn=12,
        lambda_thin=0.0,
        lambda_area=0.0,
        lambda_curv=0.0,
        lambda_rank2=0.0,
        lambda_normal=0.0,
        lambda_tangent=0.01,
    )
    controller.loss(1, xyz, scales, rotations, opacity, mass)
    assert controller.state is not None
    assert controller.state.num_points == 80

    controller.loss(
        2,
        xyz[:60],
        scales[:60],
        rotations[:60],
        opacity[:60],
        mass[:60],
    )
    assert controller.state is not None
    assert controller.state.num_points == 60
    assert controller.last_refresh == 2


def test_certified_quadrature_relaxation_preserves_layer_masses() -> None:
    clean, _, weights = sample_analytic_surface("sphere", 80, seed=48)
    projected = project_points_to_manifold(clean, mass=weights, k=12, min_confidence=0.0)
    original = projected.mass.copy()
    relaxed = relax_certified_quadrature(projected, relaxation=0.5, radius_cap_quantile=0.5)
    accepted = projected.accepted
    assert np.isclose(relaxed[accepted].sum(), original[accepted].sum(), rtol=1e-6)
    assert np.allclose(relaxed[~accepted], original[~accepted])
    assert np.all(relaxed >= 0)


class _TestCamera:
    def __init__(self, image: torch.Tensor, translate_x: float = 0.0) -> None:
        self.original_image = image
        self.world_view_transform = torch.eye(4)
        self.full_proj_transform = torch.eye(4)
        self.world_view_transform[3, 0] = translate_x
        self.full_proj_transform[3, 0] = translate_x


def test_multiview_anchor_projects_and_backpropagates() -> None:
    width = 16
    x_ramp = torch.linspace(0, 1, width).view(1, 1, width).expand(3, width, width)
    current = _TestCamera(x_ramp)
    paired = _TestCamera(torch.flip(x_ramp, dims=(2,)), translate_x=0.05)
    points = torch.tensor([[-0.4, 0.0, 1.0], [0.2, 0.1, 1.0]], requires_grad=True)
    grid, depth = project_centers(points, current)
    assert torch.allclose(grid, points[:, :2])
    assert torch.allclose(depth, points[:, 2])

    inverse_depth = torch.zeros((1, width, width))
    loss, coverage = multiview_center_loss(
        points, current, paired, inverse_depth, torch.arange(2), torch.ones((2, 1)),
        occlusion_tolerance=0.1, texture_floor=0.0,
    )
    loss.backward()
    assert coverage == 1.0
    assert torch.isfinite(loss)
    assert points.grad is not None
    assert torch.isfinite(points.grad).all()
    assert points.grad.abs().sum() > 0


def test_static_support_penalizes_normal_not_tangent_motion(tmp_path: Path) -> None:
    ply = tmp_path / "support.ply"
    write_plane_ply(ply, n=8)
    support = StaticPointSupport(ply, torch.device("cpu"), k=8)
    base = support.points[:16].detach()
    opacity = torch.ones((16, 1))

    normal_shift = (base + torch.tensor([0.0, 0.0, 0.05])).requires_grad_(True)
    normal_loss, coverage = support.loss(normal_shift, opacity)
    normal_loss.backward()
    tangent_loss, _ = support.loss(base + torch.tensor([0.01, 0.0, 0.0]), opacity)

    assert coverage == 1.0
    assert normal_loss > tangent_loss * 5
    assert normal_shift.grad is not None
    assert normal_shift.grad[:, 2].abs().mean() > normal_shift.grad[:, :2].abs().mean()

    far_tangent = base + torch.tensor([2.0, 0.0, 0.0])
    trusted_loss, _ = support.loss(far_tangent, opacity, tangent_radius_cap=2.0)
    assert trusted_loss > tangent_loss * 5


def test_static_support_reference_cap_is_deterministic(tmp_path: Path) -> None:
    ply = tmp_path / "support.ply"
    write_plane_ply(ply, n=12)
    first = StaticPointSupport(
        ply, torch.device("cpu"), k=8, max_support_points=32,
    )
    second = StaticPointSupport(
        ply, torch.device("cpu"), k=8, max_support_points=32,
    )

    assert first.points.shape == (32, 3)
    assert torch.equal(first.points, second.points)
    assert torch.equal(first.normals, second.normals)
