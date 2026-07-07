"""Observation-derived support certificates for Gaussian surface patches."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .ply_io import read_vertex_ply


EVIDENCE_SCHEMA = "manifoldgs.observation_evidence.v1"


@dataclass(frozen=True)
class ColmapCamera:
    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    rotation: np.ndarray
    translation: np.ndarray


def _qvec2rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.asarray([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float64)


def read_colmap_training_cameras(scene: str | Path) -> list[ColmapCamera]:
    """Read PINHOLE/SIMPLE_PINHOLE cameras from COLMAP text export."""
    sparse = Path(scene) / "sparse" / "0"
    intrinsics: dict[int, tuple[int, int, float, float, float, float]] = {}
    for line in (sparse / "cameras.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        camera_id, model = int(fields[0]), fields[1]
        width, height = int(fields[2]), int(fields[3])
        params = list(map(float, fields[4:]))
        if model == "PINHOLE":
            fx, fy, cx, cy = params[:4]
        elif model == "SIMPLE_PINHOLE":
            fx, cx, cy = params[:3]
            fy = fx
        else:
            raise ValueError(f"unsupported COLMAP camera model for evidence: {model}")
        intrinsics[camera_id] = (width, height, fx, fy, cx, cy)

    test_file = sparse / "test.txt"
    test_names = set(test_file.read_text(encoding="utf-8").split()) if test_file.is_file() else set()
    cameras: list[ColmapCamera] = []
    lines = (sparse / "images.txt").read_text(encoding="utf-8").splitlines()
    data_lines = [line for line in lines if line and not line.startswith("#")]
    for line in data_lines[::2]:
        fields = line.split()
        q = np.asarray(list(map(float, fields[1:5])))
        t = np.asarray(list(map(float, fields[5:8])))
        camera_id, name = int(fields[8]), fields[9]
        if name in test_names:
            continue
        width, height, fx, fy, cx, cy = intrinsics[camera_id]
        cameras.append(ColmapCamera(
            name, width, height, fx, fy, cx, cy, _qvec2rotmat(q), t,
        ))
    if not cameras:
        raise ValueError(f"no training cameras found in {scene}")
    return cameras


def compute_camera_evidence(
    xyz: np.ndarray,
    support_radius: np.ndarray,
    cameras: list[ColmapCamera],
) -> dict[str, np.ndarray]:
    """Compute frustum view count, maximum parallax, and projected footprint.

    View count is geometric frustum support, not an occlusion test. Free-space and
    photometric checks are separate certificate components and must not be
    inferred from this field.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    centers: list[np.ndarray] = []
    visible_rows: list[np.ndarray] = []
    footprints: list[np.ndarray] = []
    for camera in cameras:
        camera_xyz = xyz @ camera.rotation.T + camera.translation
        z = camera_xyz[:, 2]
        u = camera.fx * camera_xyz[:, 0] / np.maximum(z, 1e-12) + camera.cx
        v = camera.fy * camera_xyz[:, 1] / np.maximum(z, 1e-12) + camera.cy
        visible = (z > 0) & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
        footprint = np.where(
            visible, support_radius * np.sqrt(camera.fx * camera.fy) / np.maximum(z, 1e-12), 0.0,
        )
        centers.append(-camera.rotation.T @ camera.translation)
        visible_rows.append(visible)
        footprints.append(footprint)
    visibility = np.stack(visible_rows, axis=0)
    footprint = np.stack(footprints, axis=0)
    view_count = visibility.sum(axis=0)
    mean_footprint = footprint.sum(axis=0) / np.maximum(view_count, 1)

    max_parallax = np.zeros(xyz.shape[0], dtype=np.float64)
    centers_array = np.stack(centers)
    rays = xyz[None, :, :] - centers_array[:, None, :]
    rays /= np.maximum(np.linalg.norm(rays, axis=2, keepdims=True), 1e-12)
    for left in range(len(cameras)):
        for right in range(left + 1, len(cameras)):
            jointly_visible = visibility[left] & visibility[right]
            cosine = np.sum(rays[left] * rays[right], axis=1).clip(-1.0, 1.0)
            angles = np.degrees(np.arccos(cosine))
            max_parallax[jointly_visible] = np.maximum(
                max_parallax[jointly_visible], angles[jointly_visible],
            )
    return {
        "training_view_count": view_count.astype(np.int16),
        "max_parallax_deg": max_parallax.astype(np.float32),
        "mean_projection_radius_px": mean_footprint.astype(np.float32),
        "camera_support_kind": np.asarray("frustum_no_occlusion"),
        "training_camera_count": np.asarray(len(cameras), dtype=np.int16),
    }


def _project_to_camera(
    xyz: np.ndarray, camera: ColmapCamera
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return pixel ``u``, ``v``, camera-space depth ``z`` and a frustum mask."""
    camera_xyz = xyz @ camera.rotation.T + camera.translation
    z = camera_xyz[:, 2]
    u = camera.fx * camera_xyz[:, 0] / np.maximum(z, 1e-12) + camera.cx
    v = camera.fy * camera_xyz[:, 1] / np.maximum(z, 1e-12) + camera.cy
    visible = (z > 0) & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
    return u, v, z, visible


def _first_hit_mask(
    camera: ColmapCamera,
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    visible: np.ndarray,
    *,
    pixel_bin: float,
    relative_depth_margin: float,
    min_depth_margin: float,
) -> np.ndarray:
    """Per-point first-hit mask: bucket into pixel bins and keep the nearest depths."""
    mask = np.zeros(z.shape[0], dtype=bool)
    if not visible.any():
        return mask
    rows = np.flatnonzero(visible)
    bin_u = np.floor(u[rows] / pixel_bin).astype(np.int64)
    bin_v = np.floor(v[rows] / pixel_bin).astype(np.int64)
    columns = int(np.ceil(camera.width / pixel_bin)) + 1
    bin_key = bin_v * columns + bin_u
    _, inverse = np.unique(bin_key, return_inverse=True)
    first_hit_depth = np.full(inverse.max() + 1, np.inf)
    np.minimum.at(first_hit_depth, inverse, z[rows])
    nearest = first_hit_depth[inverse]
    margin = np.maximum(relative_depth_margin * nearest, min_depth_margin)
    mask[rows] = z[rows] <= nearest + margin
    return mask


def compute_visibility_evidence(
    xyz: np.ndarray,
    cameras: list[ColmapCamera],
    *,
    pixel_bin: float = 4.0,
    relative_depth_margin: float = 0.02,
    min_depth_margin: float = 1e-4,
) -> dict[str, np.ndarray]:
    """Occlusion-aware first-hit visibility using the point cloud as its own occluder.

    For each camera the frustum-visible points are bucketed into integer pixel bins
    of size ``pixel_bin``; the nearest depth inside a bin approximates the first-hit
    surface seen through those pixels. A point is then classified per camera as
    ``first_hit`` when its depth is within ``margin`` of the bin's nearest depth, or
    ``occluded`` when it sits farther behind. ``margin`` scales with depth as
    ``max(relative_depth_margin * first_hit_depth, min_depth_margin)``.

    This is strictly stronger than the frustum count in :func:`compute_camera_evidence`
    because a point behind a nearer surface no longer earns a view. It is still a
    geometric proxy — it uses the reconstructed points, not rendered depth or
    photometry — so it is only the occlusion component of the full certificate and
    must not be read as free-space or photometric support.
    """
    if pixel_bin <= 0:
        raise ValueError("pixel_bin must be positive")
    xyz = np.asarray(xyz, dtype=np.float64)
    n = xyz.shape[0]
    first_hit_count = np.zeros(n, dtype=np.int32)
    occluded_count = np.zeros(n, dtype=np.int32)
    for camera in cameras:
        u, v, z, visible = _project_to_camera(xyz, camera)
        if not visible.any():
            continue
        is_first_hit = _first_hit_mask(
            camera, u, v, z, visible,
            pixel_bin=pixel_bin, relative_depth_margin=relative_depth_margin,
            min_depth_margin=min_depth_margin,
        )
        first_hit_count[is_first_hit] += 1
        occluded_count[visible & ~is_first_hit] += 1
    return {
        "first_hit_view_count": first_hit_count.astype(np.int16),
        "occluded_view_count": occluded_count.astype(np.int16),
        "visibility_support_kind": np.asarray("first_hit_occlusion"),
        "visibility_pixel_bin": np.asarray(pixel_bin, dtype=np.float32),
    }


def compute_photometric_evidence(
    xyz: np.ndarray,
    cameras: list[ColmapCamera],
    images: list[np.ndarray],
    *,
    pixel_bin: float = 4.0,
    relative_depth_margin: float = 0.02,
    min_depth_margin: float = 1e-4,
) -> dict[str, np.ndarray]:
    """Multi-view colour agreement at each point's first-hit projections.

    For every camera in which a point is first-hit visible, its projected pixel colour
    is sampled from the matching image; the per-point RGB variance is accumulated over
    those views with Welford's algorithm. A point whose observed colour disagrees
    across views (high ``photometric_std``) is geometrically placed but not supported
    by consistent appearance — the floater failure mode sparse support alone can miss.

    ``images`` must align with ``cameras`` (same order) and hold ``(H, W, C)`` arrays
    scaled to ``[0, 1]``. Points seen in fewer than two first-hit views get an infinite
    ``photometric_std`` and are left for the view-count threshold to reject. This is a
    CPU proxy that samples the nearest pixel; it is the photometric component of the
    certificate, not free-space or Fisher evidence.
    """
    if len(images) != len(cameras):
        raise ValueError("images must align one-to-one with cameras")
    xyz = np.asarray(xyz, dtype=np.float64)
    n = xyz.shape[0]
    count = np.zeros(n, dtype=np.int32)
    mean = np.zeros((n, 3), dtype=np.float64)
    m2 = np.zeros((n, 3), dtype=np.float64)
    for camera, image in zip(cameras, images):
        image = np.asarray(image, dtype=np.float64)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("each image must be an (H, W, C>=3) array")
        if image.shape[0] != camera.height or image.shape[1] != camera.width:
            raise ValueError("image shape does not match its camera resolution")
        u, v, z, visible = _project_to_camera(xyz, camera)
        first_hit = _first_hit_mask(
            camera, u, v, z, visible,
            pixel_bin=pixel_bin, relative_depth_margin=relative_depth_margin,
            min_depth_margin=min_depth_margin,
        )
        rows = np.flatnonzero(first_hit)
        if rows.size == 0:
            continue
        px = np.clip(np.floor(u[rows]).astype(np.int64), 0, camera.width - 1)
        py = np.clip(np.floor(v[rows]).astype(np.int64), 0, camera.height - 1)
        colours = image[py, px, :3]
        count[rows] += 1
        delta = colours - mean[rows]
        mean[rows] += delta / count[rows][:, None]
        m2[rows] += delta * (colours - mean[rows])

    seen = count >= 2
    variance = np.full((n, 3), np.inf)
    variance[seen] = m2[seen] / (count[seen] - 1)[:, None]
    photometric_std = np.full(n, np.inf)
    photometric_std[seen] = np.sqrt(np.mean(variance[seen], axis=1))
    mean_colour = np.where(count[:, None] >= 1, mean, 0.0)
    return {
        "photometric_view_count": count.astype(np.int16),
        "photometric_std": photometric_std.astype(np.float32),
        "photometric_mean_color": mean_colour.astype(np.float32),
        "photometric_support_kind": np.asarray("first_hit_pixel_sample"),
    }


def _xyz(path: str | Path) -> np.ndarray:
    rows = read_vertex_ply(path).data
    missing = [name for name in ("x", "y", "z") if name not in rows.dtype.names]
    if missing:
        raise ValueError(f"PLY is missing coordinate properties: {missing}")
    return np.stack([rows["x"], rows["y"], rows["z"]], axis=1).astype(np.float64)


def build_sparse_support_evidence(
    gaussian_ply: str | Path,
    sparse_ply: str | Path,
    *,
    source_indices: np.ndarray | None = None,
    support_k: int = 3,
    radius_multiplier: float = 2.0,
    scene: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Measure whether each Gaussian is locally supported by COLMAP sparse points.

    The adaptive radius is the distance from each sparse point to its nearest
    non-self sparse neighbour. This avoids a scene-unit threshold and keeps the
    cache deterministic. It is intentionally only the sparse-geometry component
    of the full observation certificate.
    """
    gaussian_xyz = _xyz(gaussian_ply)
    sparse_xyz = _xyz(sparse_ply)
    if sparse_xyz.shape[0] < 2:
        raise ValueError("at least two COLMAP sparse points are required")
    if support_k < 1:
        raise ValueError("support_k must be positive")
    if radius_multiplier <= 0:
        raise ValueError("radius_multiplier must be positive")

    sparse_tree = cKDTree(sparse_xyz)
    local_spacing = sparse_tree.query(sparse_xyz, k=2)[0][:, 1]
    nearest_distance, nearest_index = sparse_tree.query(gaussian_xyz, k=1)
    support_radius = radius_multiplier * local_spacing[nearest_index]
    supported = nearest_distance <= support_radius
    counts = sparse_tree.query_ball_point(gaussian_xyz, r=support_radius, return_length=True)
    supported &= counts >= support_k

    if source_indices is None:
        source_indices = np.arange(gaussian_xyz.shape[0], dtype=np.int64)
    source_indices = np.asarray(source_indices, dtype=np.int64).reshape(-1)
    if source_indices.size != gaussian_xyz.shape[0]:
        raise ValueError("source_indices must contain one ID per Gaussian")
    evidence = {
        "schema": np.asarray(EVIDENCE_SCHEMA),
        "source_indices": source_indices,
        "sparse_distance": nearest_distance.astype(np.float32),
        "sparse_support_radius": support_radius.astype(np.float32),
        "sparse_support_count": np.asarray(counts, dtype=np.int32),
        "sparse_supported": supported.astype(bool),
        "support_k": np.asarray(support_k, dtype=np.int32),
        "radius_multiplier": np.asarray(radius_multiplier, dtype=np.float32),
    }
    if scene is not None:
        cameras = read_colmap_training_cameras(scene)
        evidence.update(compute_camera_evidence(gaussian_xyz, support_radius, cameras))
        evidence.update(compute_visibility_evidence(gaussian_xyz, cameras))
    return evidence


def save_sparse_support_evidence(path: str | Path, evidence: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **evidence)


def aggregate_patch_evidence(
    evidence_npz: str | Path,
    mesh_source_indices: np.ndarray,
    patch_ids: np.ndarray,
    *,
    min_supported_fraction: float = 0.5,
    min_training_views: int = 0,
    min_parallax_deg: float = 0.0,
    min_first_hit_views: int = 0,
    max_photometric_std: float = float("inf"),
    max_photometric_std_percentile: float | None = None,
    min_photometric_views: int = 0,
) -> dict[str, np.ndarray]:
    """Aggregate cached point evidence into explicit patch accept/reject reasons.

    ``max_photometric_std_percentile`` expresses the photometric-consistency gate as a
    per-scene *relative* percentile of the finite per-patch median std (e.g. ``90``
    rejects the worst 10% of this scene's patches) instead of an absolute value frozen
    from one scene. When both it and ``max_photometric_std`` are given, the stricter
    (smaller) cap wins. Patches whose median std is +inf (points seen in <2 views) are
    always rejected and never enter the percentile.
    """
    if not 0 <= min_supported_fraction <= 1:
        raise ValueError("min_supported_fraction must lie in [0, 1]")
    if max_photometric_std_percentile is not None and not 0 < max_photometric_std_percentile <= 100:
        raise ValueError("max_photometric_std_percentile must lie in (0, 100]")
    cache = np.load(evidence_npz)
    schema = str(np.asarray(cache["schema"]).item())
    if schema != EVIDENCE_SCHEMA:
        raise ValueError(f"unsupported observation evidence schema: {schema}")
    ids = np.asarray(cache["source_indices"], dtype=np.int64).reshape(-1)
    row_by_source = {int(source_id): row for row, source_id in enumerate(ids)}
    mesh_source_indices = np.asarray(mesh_source_indices, dtype=np.int64).reshape(-1)
    patch_ids = np.asarray(patch_ids, dtype=np.int32).reshape(-1)
    missing = sorted(set(map(int, mesh_source_indices)) - set(row_by_source))
    if missing:
        raise ValueError(f"evidence cache is missing {len(missing)} mesh source IDs")
    rows = np.asarray([row_by_source[int(x)] for x in mesh_source_indices])
    supported = np.asarray(cache["sparse_supported"], dtype=bool)[rows]
    distance = np.asarray(cache["sparse_distance"], dtype=np.float64)[rows]
    radius = np.asarray(cache["sparse_support_radius"], dtype=np.float64)[rows]

    unique_patches = np.unique(patch_ids)
    fractions = np.asarray([supported[patch_ids == p].mean() for p in unique_patches])
    normalized_distance = np.asarray([
        np.median(distance[patch_ids == p] / np.maximum(radius[patch_ids == p], 1e-12))
        for p in unique_patches
    ])
    accepted = fractions >= min_supported_fraction
    reasons = np.where(accepted, "accepted", "insufficient_sparse_support").astype("<U40")
    result = {
        "patch_ids": unique_patches.astype(np.int32),
        "sparse_supported_fraction": fractions.astype(np.float32),
        "median_normalized_sparse_distance": normalized_distance.astype(np.float32),
        "observationally_supported": accepted,
        "reject_reason": reasons,
    }
    has_camera_evidence = "training_view_count" in cache and "max_parallax_deg" in cache
    has_visibility_evidence = "first_hit_view_count" in cache
    if min_training_views > 0 or min_parallax_deg > 0:
        if not has_camera_evidence:
            raise ValueError("camera thresholds require camera fields in the evidence cache")
    has_photometric_evidence = "photometric_std" in cache and "photometric_view_count" in cache
    if min_first_hit_views > 0 and not has_visibility_evidence:
        raise ValueError("first-hit threshold requires first_hit_view_count in the evidence cache")
    if (
        np.isfinite(max_photometric_std)
        or max_photometric_std_percentile is not None
        or min_photometric_views > 0
    ) and not has_photometric_evidence:
        raise ValueError("photometric thresholds require photometric fields in the evidence cache")
    if has_camera_evidence:
        view_count = np.asarray(cache["training_view_count"], dtype=np.int32)[rows]
        parallax = np.asarray(cache["max_parallax_deg"], dtype=np.float64)[rows]
        footprint = np.asarray(cache["mean_projection_radius_px"], dtype=np.float64)[rows]
        patch_views = np.asarray([np.median(view_count[patch_ids == p]) for p in unique_patches])
        patch_parallax = np.asarray([np.median(parallax[patch_ids == p]) for p in unique_patches])
        patch_footprint = np.asarray([np.median(footprint[patch_ids == p]) for p in unique_patches])
        view_valid = patch_views >= min_training_views
        parallax_valid = patch_parallax >= min_parallax_deg
        reasons[(result["observationally_supported"]) & ~view_valid] = "insufficient_training_views"
        reasons[(result["observationally_supported"]) & view_valid & ~parallax_valid] = "insufficient_parallax"
        result["observationally_supported"] &= view_valid & parallax_valid
        result.update({
            "median_training_view_count": patch_views.astype(np.float32),
            "median_max_parallax_deg": patch_parallax.astype(np.float32),
            "median_projection_radius_px": patch_footprint.astype(np.float32),
        })
    if has_visibility_evidence:
        first_hit = np.asarray(cache["first_hit_view_count"], dtype=np.int32)[rows]
        patch_first_hit = np.asarray([np.median(first_hit[patch_ids == p]) for p in unique_patches])
        first_hit_valid = patch_first_hit >= min_first_hit_views
        reasons[result["observationally_supported"] & ~first_hit_valid] = "insufficient_first_hit_visibility"
        result["observationally_supported"] &= first_hit_valid
        result["median_first_hit_view_count"] = patch_first_hit.astype(np.float32)
    if has_photometric_evidence:
        photometric_std = np.asarray(cache["photometric_std"], dtype=np.float64)[rows]
        photometric_views = np.asarray(cache["photometric_view_count"], dtype=np.int32)[rows]
        # A patch is only as photometrically consistent as its median point; +inf std
        # (points seen in <2 views) propagates through the median as "unsupported".
        patch_std = np.asarray([np.median(photometric_std[patch_ids == p]) for p in unique_patches])
        patch_photo_views = np.asarray([np.median(photometric_views[patch_ids == p]) for p in unique_patches])
        photo_view_valid = patch_photo_views >= min_photometric_views
        effective_std_cap = max_photometric_std
        if max_photometric_std_percentile is not None:
            finite_std = patch_std[np.isfinite(patch_std)]
            # No finite patch => nothing to keep; cap stays at the absolute value.
            if finite_std.size:
                percentile_cap = float(np.percentile(finite_std, max_photometric_std_percentile))
                effective_std_cap = min(effective_std_cap, percentile_cap)
            result["photometric_std_percentile"] = np.float32(max_photometric_std_percentile)
        result["photometric_std_threshold"] = np.float32(effective_std_cap)
        photo_std_valid = patch_std <= effective_std_cap
        supported_now = result["observationally_supported"]
        reasons[supported_now & ~photo_view_valid] = "insufficient_photometric_views"
        reasons[supported_now & photo_view_valid & ~photo_std_valid] = "inconsistent_photometry"
        result["observationally_supported"] &= photo_view_valid & photo_std_valid
        result["median_photometric_std"] = patch_std.astype(np.float32)
        result["median_photometric_view_count"] = patch_photo_views.astype(np.float32)
    return result
