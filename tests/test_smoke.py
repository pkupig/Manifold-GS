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
from manifold_gs.observation_evidence import (
    ColmapCamera,
    aggregate_patch_evidence,
    build_sparse_support_evidence,
    compute_camera_evidence,
    compute_photometric_evidence,
    compute_visibility_evidence,
    save_sparse_support_evidence,
)


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


def test_sparse_observation_evidence_rejects_floater_patch(tmp_path: Path) -> None:
    gaussians = tmp_path / "gaussians.ply"
    sparse = tmp_path / "sparse.ply"
    write_plane_ply(gaussians, n=8)
    write_plane_ply(sparse, n=8)
    evidence = build_sparse_support_evidence(
        gaussians, sparse, support_k=3, radius_multiplier=2.0,
    )
    assert evidence["sparse_supported"].all()

    # Treat four supported vertices as patch 0 and override four vertices as a
    # synthetic unsupported floater patch to test aggregation independently.
    evidence["sparse_supported"][-4:] = False
    cache = tmp_path / "evidence.npz"
    save_sparse_support_evidence(cache, evidence)
    aggregate = aggregate_patch_evidence(
        cache,
        np.asarray([0, 1, 2, 3, 60, 61, 62, 63]),
        np.asarray([0, 0, 0, 0, 1, 1, 1, 1]),
        min_supported_fraction=0.5,
    )
    assert aggregate["observationally_supported"].tolist() == [True, False]
    assert aggregate["reject_reason"].tolist() == ["accepted", "insufficient_sparse_support"]


def test_camera_evidence_reports_views_parallax_and_footprint() -> None:
    cameras = [
        ColmapCamera("left.png", 100, 100, 100, 100, 50, 50, np.eye(3), np.asarray([0.5, 0, 0])),
        ColmapCamera("right.png", 100, 100, 100, 100, 50, 50, np.eye(3), np.asarray([-0.5, 0, 0])),
    ]
    evidence = compute_camera_evidence(
        np.asarray([[0.0, 0.0, 5.0], [100.0, 0.0, 5.0]]),
        np.asarray([0.1, 0.1]),
        cameras,
    )
    assert evidence["training_view_count"].tolist() == [2, 0]
    assert evidence["max_parallax_deg"][0] > 10
    assert np.isclose(evidence["mean_projection_radius_px"][0], 2.0)
    assert evidence["camera_support_kind"].item() == "frustum_no_occlusion"


def test_visibility_evidence_counts_first_hit_over_occluded() -> None:
    camera = ColmapCamera("front.png", 100, 100, 100, 100, 50, 50, np.eye(3), np.zeros(3))
    # A occludes B along the same pixel ray; C sits in its own pixel bin.
    xyz = np.asarray([[0.0, 0.0, 2.0], [0.0, 0.0, 5.0], [1.0, 0.0, 5.0]])
    evidence = compute_visibility_evidence(xyz, [camera], pixel_bin=4.0)
    assert evidence["first_hit_view_count"].tolist() == [1, 0, 1]
    assert evidence["occluded_view_count"].tolist() == [0, 1, 0]
    assert evidence["visibility_support_kind"].item() == "first_hit_occlusion"


def test_first_hit_threshold_rejects_occluded_patch(tmp_path: Path) -> None:
    gaussians = tmp_path / "gaussians.ply"
    sparse = tmp_path / "sparse.ply"
    write_plane_ply(gaussians, n=8)
    write_plane_ply(sparse, n=8)
    evidence = build_sparse_support_evidence(gaussians, sparse, support_k=3, radius_multiplier=2.0)
    # Inject first-hit visibility: patch 0 is seen, patch 1 is fully occluded.
    evidence["first_hit_view_count"] = np.full(64, 3, dtype=np.int16)
    evidence["first_hit_view_count"][-4:] = 0
    cache = tmp_path / "evidence.npz"
    save_sparse_support_evidence(cache, evidence)
    aggregate = aggregate_patch_evidence(
        cache,
        np.asarray([0, 1, 2, 3, 60, 61, 62, 63]),
        np.asarray([0, 0, 0, 0, 1, 1, 1, 1]),
        min_supported_fraction=0.5,
        min_first_hit_views=1,
    )
    assert aggregate["observationally_supported"].tolist() == [True, False]
    assert aggregate["reject_reason"].tolist() == ["accepted", "insufficient_first_hit_visibility"]
    assert aggregate["median_first_hit_view_count"].tolist() == [3.0, 0.0]


def test_mesh_surface_sampling_is_area_weighted_and_deterministic() -> None:
    from manifold_gs.collision_metrics import sample_mesh_surface

    vertices = np.asarray([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]])
    first = sample_mesh_surface(vertices, faces, 2000, seed=5)
    second = sample_mesh_surface(vertices, faces, 2000, seed=5)
    assert np.array_equal(first[0], second[0])
    assert np.isclose(first[2], 1.0)
    assert np.all(first[0][:, 2] == 0.0)
    assert np.allclose(np.abs(first[1][:, 2]), 1.0)


def test_collision_coverage_flags_floater_surface() -> None:
    from manifold_gs.collision_metrics import surface_coverage_metrics

    square_v = np.asarray([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    square_f = np.asarray([[0, 1, 2], [0, 2, 3]])
    # Candidate = the true square plus a far floater triangle (area 0.5, distance 10).
    candidate_v = np.vstack([square_v, [[0, 0, 10], [1, 0, 10], [0, 1, 10]]])
    candidate_f = np.vstack([square_f, [[4, 5, 6]]])
    reference_xyz, reference_normals, _ = __import__(
        "manifold_gs.collision_metrics", fromlist=["sample_mesh_surface"]
    ).sample_mesh_surface(square_v, square_f, 8000, seed=1)

    metrics = surface_coverage_metrics(
        candidate_v, candidate_f, reference_xyz, reference_normals,
        tolerance=0.05, samples=8000, seed=2,
    )
    assert metrics["coverage"] > 0.99
    assert 0.25 < metrics["false_surface_fraction"] < 0.42
    assert metrics["false_surface_area"] > 0.3
    assert metrics["supported_normal_median_deg"] < 1.0
    assert metrics["hausdorff"] > 9.0


def test_coverage_sweep_is_monotonic_and_reports_error_distribution() -> None:
    from manifold_gs.collision_metrics import coverage_tolerance_sweep, sample_mesh_surface

    # A unit square shifted 0.03 off its reference plane: the surface is uniformly
    # "close but not exact", so at a tolerance below 0.03 almost all of it reads as
    # false surface, and above 0.03 almost none does -- the single-tolerance trap.
    ref_v = np.asarray([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    ref_f = np.asarray([[0, 1, 2], [0, 2, 3]])
    cand_v = ref_v + np.asarray([0.0, 0.0, 0.03])
    reference_xyz, _, _ = sample_mesh_surface(ref_v, ref_f, 8000, seed=1)

    result = coverage_tolerance_sweep(
        cand_v, ref_f, reference_xyz,
        tolerances=np.asarray([0.01, 0.02, 0.05, 0.1]), samples=8000, seed=2,
    )
    rows = result["sweep"]
    coverage = [r["coverage"] for r in rows]
    false_frac = [r["false_surface_fraction"] for r in rows]
    # Coverage rises and false surface falls as tolerance grows.
    assert coverage == sorted(coverage)
    assert false_frac == sorted(false_frac, reverse=True)
    # Tolerance 0.01 < offset 0.03 => nearly all false; 0.05 > offset => nearly none.
    assert false_frac[0] > 0.9
    assert false_frac[2] < 0.05
    assert abs(result["candidate_to_reference_median"] - 0.03) < 5e-3


def test_collision_confusion_separates_false_and_missed() -> None:
    from manifold_gs.collision_metrics import collision_confusion

    candidate = np.asarray([[0.0, 0, 0], [1, 0, 0]])
    reference = np.asarray([[0.0, 0, 0], [5, 0, 0]])
    probes = np.asarray([[0, 0, 0.01], [1, 0, 0.01], [5, 0, 0.01], [10, 10, 10]])
    labels = np.asarray(["occupied", "occupied", "occupied", "unknown"])
    result = collision_confusion(
        candidate, reference, probes, contact_tolerance=0.05, probe_labels=labels
    )
    assert result["false_collision"] == 1
    assert result["missed_collision"] == 1
    assert result["candidate_contact"] == 2
    assert result["reference_contact"] == 2
    assert np.isclose(result["agreement"], 0.5)
    assert result["unknown_probes"] == 1
    assert np.isclose(result["unknown_marked_free_fraction"], 1.0)


def test_texture_roundtrip_improves_with_resolution() -> None:
    from manifold_gs.texture_metrics import baking_roundtrip_metrics

    grid = np.linspace(0.0, 1.0, 40)
    gx, gy = np.meshgrid(grid, grid, indexing="ij")
    points = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])

    constant = np.full((points.shape[0], 3), 0.4)
    constant_metrics = baking_roundtrip_metrics(points, constant, resolution=32)
    assert constant_metrics["reprojection_error_max"] < 1e-9
    assert not np.isfinite(constant_metrics["reprojection_psnr"]) or constant_metrics["reprojection_psnr"] > 80

    # A high-frequency stripe pattern: low-res baking averages it away, high-res keeps it.
    stripes = 0.5 + 0.5 * np.sign(np.sin(20.0 * np.pi * gx.ravel()))
    stripe_colors = np.repeat(stripes[:, None], 3, axis=1)
    low = baking_roundtrip_metrics(points, stripe_colors, resolution=4)
    high = baking_roundtrip_metrics(points, stripe_colors, resolution=64)
    assert high["reprojection_psnr"] > low["reprojection_psnr"]
    assert high["reprojection_error_mean"] < low["reprojection_error_mean"]


def test_seam_error_grows_when_patches_disagree() -> None:
    from manifold_gs.texture_metrics import seam_error_metrics

    ys = np.linspace(0.0, 1.0, 5)
    left = np.array([[x, y, 0.0] for x in (0.9, 1.0) for y in ys])
    right = np.array([[x, y, 0.0] for x in (1.1, 1.2) for y in ys])
    points = np.vstack([left, right])
    patch_ids = np.array([0] * len(left) + [1] * len(right))
    gray = lambda pts: np.repeat(pts[:, 1:2], 3, axis=1)  # colour depends on y only

    matched = np.vstack([gray(left), gray(right)])
    matched_seam = seam_error_metrics(points, patch_ids, matched, 8, boundary_radius=0.15)
    assert matched_seam["boundary_pairs"] > 0
    assert matched_seam["seam_error_mean"] < 0.05

    mismatched = np.vstack([gray(left), gray(right) + 0.5])
    mismatched_seam = seam_error_metrics(points, patch_ids, mismatched, 8, boundary_radius=0.15)
    assert mismatched_seam["boundary_pairs"] == matched_seam["boundary_pairs"]
    assert mismatched_seam["seam_error_mean"] > 0.4


def test_seam_raw_ceiling_flags_genuine_colour_variance() -> None:
    from manifold_gs.texture_metrics import seam_error_metrics

    ys = np.linspace(0.0, 1.0, 5)
    left = np.array([[x, y, 0.0] for x in (0.9, 1.0) for y in ys])
    right = np.array([[x, y, 0.0] for x in (1.1, 1.2) for y in ys])
    points = np.vstack([left, right])
    patch_ids = np.array([0] * len(left) + [1] * len(right))

    # The two patches carry genuinely different constant colours. Then the baked seam
    # cannot be better than the raw cross-patch colour disagreement, and the baking
    # itself adds essentially nothing -- a shared atlas would not help.
    colors = np.vstack([
        np.full((len(left), 3), 0.2),
        np.full((len(right), 3), 0.7),
    ])
    seam = seam_error_metrics(points, patch_ids, colors, 8, boundary_radius=0.15)
    assert seam["boundary_pairs"] > 0
    assert np.isclose(seam["raw_seam_error_mean"], seam["seam_error_mean"], atol=1e-6)
    assert abs(seam["baking_excess_error_mean"]) < 1e-6
    assert np.isclose(seam["raw_seam_psnr"], seam["seam_psnr"], atol=1e-6)


def test_photometric_evidence_flags_view_disagreement() -> None:
    left = ColmapCamera("left.png", 100, 100, 100, 100, 50, 50, np.eye(3), np.asarray([0.5, 0, 0]))
    right = ColmapCamera("right.png", 100, 100, 100, 100, 50, 50, np.eye(3), np.asarray([-0.5, 0, 0]))
    # p0 seen consistently, p1 seen with disagreeing colour, p2 seen in one view only.
    xyz = np.asarray([[0.0, 0, 5], [1.0, 0, 5], [-2.5, 0, 5]])
    image_left = np.zeros((100, 100, 3))
    image_right = np.zeros((100, 100, 3))
    image_left[50, 60] = [0.5, 0.5, 0.5]   # p0 in left
    image_right[50, 40] = [0.5, 0.5, 0.5]  # p0 in right
    image_left[50, 80] = [1.0, 0.0, 0.0]   # p1 in left (red)
    image_right[50, 60] = [0.0, 0.0, 1.0]  # p1 in right (blue)
    image_left[50, 10] = [0.3, 0.3, 0.3]   # p2 in left only
    evidence = compute_photometric_evidence(
        xyz, [left, right], [image_left, image_right], pixel_bin=4.0
    )
    assert evidence["photometric_view_count"].tolist() == [2, 2, 1]
    assert evidence["photometric_std"][0] < 1e-6
    assert evidence["photometric_std"][1] > 0.3
    assert not np.isfinite(evidence["photometric_std"][2])
    assert evidence["photometric_support_kind"].item() == "first_hit_pixel_sample"


def test_photometric_thresholds_reject_inconsistent_and_underseen(tmp_path: Path) -> None:
    gaussians = tmp_path / "gaussians.ply"
    sparse = tmp_path / "sparse.ply"
    write_plane_ply(gaussians, n=8)
    write_plane_ply(sparse, n=8)
    evidence = build_sparse_support_evidence(gaussians, sparse, support_k=3, radius_multiplier=2.0)
    evidence["photometric_std"] = np.full(64, 0.01, dtype=np.float32)
    evidence["photometric_view_count"] = np.full(64, 4, dtype=np.int16)
    evidence["photometric_std"][[20, 21, 22, 23]] = 0.9      # patch 1: inconsistent colour
    evidence["photometric_view_count"][[40, 41, 42, 43]] = 1  # patch 2: too few views
    cache = tmp_path / "evidence.npz"
    save_sparse_support_evidence(cache, evidence)
    aggregate = aggregate_patch_evidence(
        cache,
        np.asarray([0, 1, 2, 3, 20, 21, 22, 23, 40, 41, 42, 43]),
        np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),
        min_supported_fraction=0.5,
        max_photometric_std=0.1,
        min_photometric_views=2,
    )
    assert aggregate["observationally_supported"].tolist() == [True, False, False]
    assert aggregate["reject_reason"].tolist() == [
        "accepted", "inconsistent_photometry", "insufficient_photometric_views"
    ]


def test_relative_photometric_percentile_gate_is_per_scene(tmp_path: Path) -> None:
    # Four patches with distinct, well-separated median stds. A relative percentile gate
    # keeps the low-std patches and rejects only the scene's worst tail -- no absolute
    # value is frozen; the resolved threshold is derived from this scene's distribution.
    gaussians = tmp_path / "gaussians.ply"
    sparse = tmp_path / "sparse.ply"
    write_plane_ply(gaussians, n=8)
    write_plane_ply(sparse, n=8)
    evidence = build_sparse_support_evidence(gaussians, sparse, support_k=3, radius_multiplier=2.0)
    evidence["photometric_std"] = np.zeros(64, dtype=np.float32)
    evidence["photometric_view_count"] = np.full(64, 4, dtype=np.int16)
    rows = np.asarray([0, 1, 2, 3, 20, 21, 22, 23, 40, 41, 42, 43, 60, 61, 62, 63])
    pids = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    stds = {0: 0.02, 1: 0.05, 2: 0.10, 3: 0.40}
    for p, s in stds.items():
        evidence["photometric_std"][rows[pids == p]] = s
    cache = tmp_path / "evidence.npz"
    save_sparse_support_evidence(cache, evidence)

    # percentile=75 => cap at the 75th percentile of {0.02,0.05,0.10,0.40} = 0.175,
    # so only the 0.40 patch is rejected; nothing else is.
    aggregate = aggregate_patch_evidence(
        cache, rows, pids,
        min_supported_fraction=0.5,
        max_photometric_std_percentile=75.0,
        min_photometric_views=2,
    )
    assert aggregate["observationally_supported"].tolist() == [True, True, True, False]
    assert aggregate["reject_reason"][3] == "inconsistent_photometry"
    assert np.isclose(float(aggregate["photometric_std_threshold"]), 0.175, atol=1e-6)
    assert np.isclose(float(aggregate["photometric_std_percentile"]), 75.0)

    # The same relative percentile on a scene whose stds are all 10x larger keeps the
    # same *fraction*, proving it adapts per scene instead of using a fixed number.
    evidence["photometric_std"][:] = 0.0
    for p, s in stds.items():
        evidence["photometric_std"][rows[pids == p]] = s * 10.0
    save_sparse_support_evidence(cache, evidence)
    scaled = aggregate_patch_evidence(
        cache, rows, pids,
        min_supported_fraction=0.5,
        max_photometric_std_percentile=75.0,
        min_photometric_views=2,
    )
    assert scaled["observationally_supported"].tolist() == [True, True, True, False]
    assert np.isclose(float(scaled["photometric_std_threshold"]), 1.75, atol=1e-6)


def test_rigid_deformation_rotates_about_pivot() -> None:
    from manifold_gs.edit_metrics import rigid_deformation

    quarter_turn = np.asarray([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    pivot = np.asarray([1.0, 0.0, 0.0])
    deform = rigid_deformation(quarter_turn, np.zeros(3), pivot=pivot)
    moved = deform(np.asarray([[1.0, 0, 0], [2.0, 0, 0]]))
    assert np.allclose(moved[0], pivot)          # pivot is fixed
    assert np.allclose(moved[1], [1.0, 1.0, 0.0])  # (2,0) rotates to (1,1)


def test_certified_binding_beats_radius_binding_on_leakage() -> None:
    from manifold_gs.edit_metrics import (
        certified_patch_binding,
        edit_propagation_metrics,
        propagate_edit,
        radius_binding,
        rigid_deformation,
    )

    xs = [0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 5.0, 6.0]
    points = np.column_stack([xs, np.zeros(8), np.zeros(8)])
    patch_ids = np.asarray([0, 0, 0, 1, 1, 1, -1, -1])
    edit_region = patch_ids == 0
    residual_mask = patch_ids < 0
    lift = rigid_deformation(np.eye(3), np.asarray([0.0, 0.0, 1.0]))
    target = propagate_edit(points, lift, edit_region)

    certified = propagate_edit(points, lift, certified_patch_binding(patch_ids, [0]))
    certified_metrics = edit_propagation_metrics(
        points, certified, target, edit_region, residual_mask=residual_mask
    )
    assert certified_metrics["target_shift_mean"] == 1.0
    assert certified_metrics["edit_error_max"] == 0.0
    assert certified_metrics["boundary_leakage_max"] == 0.0
    assert certified_metrics["leaked_point_fraction"] == 0.0
    assert certified_metrics["residual_contamination_max"] == 0.0

    nearest = propagate_edit(points, lift, radius_binding(points, points[edit_region], 0.15))
    nearest_metrics = edit_propagation_metrics(
        points, nearest, target, edit_region, residual_mask=residual_mask
    )
    assert nearest_metrics["edit_error_max"] == 0.0            # selected region still correct
    assert np.isclose(nearest_metrics["boundary_leakage_max"], 1.0)  # leaks the x=0.3 neighbour
    assert np.isclose(nearest_metrics["leaked_point_fraction"], 0.2)
    assert nearest_metrics["residual_contamination_max"] == 0.0  # far residuals untouched


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
