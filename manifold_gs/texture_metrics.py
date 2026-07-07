"""Per-patch texture baking round-trip metrics (P0.5).

Scores how much appearance survives baking observed surface colour into a finite
per-patch texture and sampling it back:

- ``baking_roundtrip_metrics``: bake a patch's colours into a tangent-plane texture
  grid, sample them back at the same points, and report reprojection error / PSNR
  and texel fill;
- ``seam_error_metrics``: bake each patch independently and measure colour
  disagreement between adjacent patches at their shared boundary — the seam a hybrid
  asset shows when neighbouring charts are textured separately.

Pure numpy/scipy, deterministic, CPU-only. The observed colour is meant to be the
multi-view ``photometric_mean_color`` from :mod:`manifold_gs.observation_evidence`.
This is a nearest-texel proxy, not a full UV atlas or a renderer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class BakedTexture:
    texture: np.ndarray      # (R, R, C)
    mask: np.ndarray         # (R, R) bool: which texels received colour
    origin: np.ndarray       # (2,) uv lower corner
    span: np.ndarray         # (2,) uv extent
    resolution: int


def tangent_plane_uv(points: np.ndarray) -> np.ndarray:
    """Project points onto their best-fit tangent plane and return 2D uv coords."""
    points = np.asarray(points, dtype=np.float64)
    centered = points - points.mean(axis=0)
    # Right singular vectors order the directions by variance; the first two span
    # the tangent plane of a locally flat patch.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def bake_patch_texture(uv: np.ndarray, colors: np.ndarray, resolution: int) -> BakedTexture:
    """Average colours into an ``resolution x resolution`` tangent-plane texture."""
    if resolution < 1:
        raise ValueError("resolution must be positive")
    uv = np.asarray(uv, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    channels = colors.shape[1]
    origin = uv.min(axis=0)
    span = np.maximum(uv.max(axis=0) - origin, 1e-12)
    ij = np.clip(np.floor((uv - origin) / span * resolution).astype(np.int64), 0, resolution - 1)
    flat = ij[:, 0] * resolution + ij[:, 1]
    texture = np.zeros((resolution * resolution, channels))
    count = np.zeros(resolution * resolution)
    np.add.at(texture, flat, colors)
    np.add.at(count, flat, 1.0)
    filled = count > 0
    texture[filled] /= count[filled][:, None]
    return BakedTexture(
        texture.reshape(resolution, resolution, channels),
        filled.reshape(resolution, resolution),
        origin, span, resolution,
    )


def sample_patch_texture(baked: BakedTexture, uv: np.ndarray) -> np.ndarray:
    """Nearest-texel lookup of ``uv`` in a baked texture."""
    uv = np.asarray(uv, dtype=np.float64)
    ij = np.clip(
        np.floor((uv - baked.origin) / baked.span * baked.resolution).astype(np.int64),
        0, baked.resolution - 1,
    )
    return baked.texture[ij[:, 0], ij[:, 1]]


def _psnr(reference: np.ndarray, other: np.ndarray) -> float:
    mse = float(np.mean((reference - other) ** 2))
    if mse <= 1e-24:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def baking_roundtrip_metrics(
    points: np.ndarray,
    colors: np.ndarray,
    resolution: int,
) -> dict[str, float | int]:
    """Bake a patch's colours, sample them back, and report the round-trip error."""
    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    if points.shape[0] != colors.shape[0]:
        raise ValueError("points and colors must have one row per surface sample")
    uv = tangent_plane_uv(points)
    baked = bake_patch_texture(uv, colors, resolution)
    reconstructed = sample_patch_texture(baked, uv)
    error = np.linalg.norm(reconstructed - colors, axis=1)
    return {
        "samples": int(points.shape[0]),
        "resolution": int(resolution),
        "reprojection_error_mean": float(np.mean(error)),
        "reprojection_error_max": float(np.max(error)),
        "reprojection_psnr": _psnr(colors, reconstructed),
        "texel_fill_fraction": float(np.mean(baked.mask)),
    }


def seam_error_metrics(
    points: np.ndarray,
    patch_ids: np.ndarray,
    colors: np.ndarray,
    resolution: int,
    *,
    boundary_radius: float,
) -> dict[str, float | int]:
    """Colour disagreement between adjacent patches at their shared boundary.

    Each patch is baked independently and its points reconstructed from its own
    texture. For every cross-patch pair of points within ``boundary_radius``, the
    seam error is the difference between the two reconstructions at (nearly) the same
    location — what a viewer sees as a visible chart seam.
    """
    if boundary_radius <= 0:
        raise ValueError("boundary_radius must be positive")
    points = np.asarray(points, dtype=np.float64)
    patch_ids = np.asarray(patch_ids).reshape(-1)
    colors = np.asarray(colors, dtype=np.float64)

    reconstructed = np.zeros_like(colors)
    for patch in np.unique(patch_ids):
        rows = np.flatnonzero(patch_ids == patch)
        uv = tangent_plane_uv(points[rows])
        baked = bake_patch_texture(uv, colors[rows], resolution)
        reconstructed[rows] = sample_patch_texture(baked, uv)

    tree = cKDTree(points)
    pairs = tree.query_pairs(r=boundary_radius, output_type="ndarray")
    if pairs.shape[0]:
        cross = patch_ids[pairs[:, 0]] != patch_ids[pairs[:, 1]]
        pairs = pairs[cross]
    if pairs.shape[0] == 0:
        return {
            "boundary_pairs": 0,
            "seam_error_mean": 0.0,
            "seam_error_max": 0.0,
            "seam_psnr": float("inf"),
        }
    diff = reconstructed[pairs[:, 0]] - reconstructed[pairs[:, 1]]
    seam = np.linalg.norm(diff, axis=1)
    # Raw ceiling: the disagreement between the *input* colours of the same
    # cross-patch neighbour pairs, with no baking at all. If the baked seam is no
    # worse than this, the seam is genuine colour variance across the boundary, not a
    # per-patch charting/quantization artifact -- and a shared atlas will not fix it.
    raw = np.linalg.norm(colors[pairs[:, 0]] - colors[pairs[:, 1]], axis=1)
    return {
        "boundary_pairs": int(pairs.shape[0]),
        "seam_error_mean": float(np.mean(seam)),
        "seam_error_max": float(np.max(seam)),
        "seam_psnr": _psnr(reconstructed[pairs[:, 0]], reconstructed[pairs[:, 1]]),
        "raw_seam_error_mean": float(np.mean(raw)),
        "raw_seam_psnr": _psnr(colors[pairs[:, 0]], colors[pairs[:, 1]]),
        "baking_excess_error_mean": float(np.mean(seam) - np.mean(raw)),
    }
