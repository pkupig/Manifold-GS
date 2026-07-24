#!/usr/bin/env python3
"""DTU-official-mask collision coverage for a Gaussian-frame candidate mesh.

This is the P0.4 recall counterpart to ``evaluate_collision_candidate.py``.  It
reproduces DTUeval's two asymmetric filters before measuring coverage:

* candidate samples are mapped from the Gaussian frame to DTU millimetres and
  retained only inside the official ``ObsMask{scan}_10.mat`` volume (with the
  official 60 mm padding);
* reference STL samples are retained only on the positive side of
  ``Plane{scan}.mat``.

The subsequent distances remain in the Gaussian frame, so tolerance fractions
are comparable to the existing asset benchmark.  The report deliberately keeps
this official-mask metric separate from the full-STL precision report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manifold_gs.collision_metrics import sample_mesh_surface
from manifold_gs.mesh_io import read_triangle_mesh_ply


def transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return (np.c_[points, np.ones(len(points))] @ matrix.T)[:, :3]


def observation_mask(points_mm: np.ndarray, mat_path: Path, patch_size: float = 60.0) -> np.ndarray:
    """Apply the exact ObsMask/BB/Res indexing used by official DTUeval."""
    doc = loadmat(mat_path)
    obs = np.asarray(doc['ObsMask'], dtype=bool)
    bb = np.asarray(doc['BB'], dtype=np.float64)
    res = np.asarray(doc['Res'], dtype=np.float64).reshape(-1)
    inbound = np.all((points_mm >= bb[:1] - patch_size) & (points_mm < bb[1:] + 2 * patch_size), axis=1)
    result = np.zeros(len(points_mm), dtype=bool)
    idx = np.flatnonzero(inbound)
    grid = np.around((points_mm[idx] - bb[:1]) / res).astype(np.int64)
    grid_inbound = np.all((grid >= 0) & (grid < np.asarray(obs.shape)[None]), axis=1)
    valid = idx[grid_inbound]
    g = grid[grid_inbound]
    result[valid] = obs[g[:, 0], g[:, 1], g[:, 2]]
    return result


def plane_mask(points_mm: np.ndarray, plane_path: Path) -> np.ndarray:
    plane = np.asarray(loadmat(plane_path)['P'], dtype=np.float64).reshape(4)
    return (np.c_[points_mm, np.ones(len(points_mm))] * plane[None]).sum(axis=1) > 0


def evaluate(candidate: Path, gt: Path, cameras: Path, obs_mask: Path, plane: Path, *, samples: int, seed: int, tolerance_fraction: float) -> dict:
    vertices, faces = read_triangle_mesh_ply(str(candidate))
    cand, cand_normals, cand_area = sample_mesh_surface(vertices, faces, samples, seed=seed)
    scale = np.asarray(np.load(cameras)['scale_mat_0'], dtype=np.float64)
    observed = observation_mask(transform(cand, scale), obs_mask)
    cand, cand_normals = cand[observed], cand_normals[observed]

    data = np.load(gt)
    ref, normals = np.asarray(data['xyz'], dtype=np.float64), np.asarray(data['normals'], dtype=np.float64)
    ref_mm = transform(ref, scale)
    above = plane_mask(ref_mm, plane)
    ref, normals = ref[above], normals[above]
    bbox = float(np.linalg.norm(np.ptp(ref, axis=0)))
    tol = tolerance_fraction * max(bbox, 1e-12)
    if len(cand) == 0 or len(ref) == 0:
        raise ValueError('official mask removed all candidate or reference samples')
    ctree, rtree = cKDTree(cand), cKDTree(ref)
    ref_to_cand, _ = ctree.query(ref, k=1)
    cand_to_ref, match = rtree.query(cand, k=1)
    supported = cand_to_ref <= tol
    dots = np.abs(np.sum(cand_normals[supported] * normals[match[supported]], axis=1))
    sweep = []
    for frac in (0.005, 0.01, 0.02, 0.03, 0.05, 0.08):
        t = frac * bbox
        sweep.append({'tolerance': t, 'coverage': float(np.mean(ref_to_cand <= t)), 'false_surface_fraction': float(np.mean(cand_to_ref > t))})
    return {
        'protocol': 'DTU official ObsMask(candidate) + Plane(reference), then Gaussian-frame distance',
        'candidate': str(candidate.resolve()), 'gt': str(gt.resolve()), 'cameras': str(cameras.resolve()),
        'obs_mask': str(obs_mask.resolve()), 'plane': str(plane.resolve()),
        'samples_before_obs_mask': int(samples), 'candidate_samples_in_obs_mask': int(len(cand)),
        'reference_samples_above_plane': int(len(ref)), 'candidate_area_unmasked': float(cand_area),
        'bbox_diagonal': bbox, 'tolerance_fraction': tolerance_fraction, 'tolerance': tol,
        'coverage': {'coverage': float(np.mean(ref_to_cand <= tol)), 'false_surface_fraction': float(np.mean(~supported)),
                     'candidate_to_reference_p95': float(np.quantile(cand_to_ref, .95)),
                     'reference_to_candidate_p95': float(np.quantile(ref_to_cand, .95)),
                     'hausdorff': float(max(cand_to_ref.max(), ref_to_cand.max())),
                     'supported_normal_median_deg': float(np.median(np.degrees(np.arccos(np.clip(dots, 0, 1))))) if len(dots) else float('nan')},
        'coverage_sweep': {'candidate_area': float(cand_area), 'sweep': sweep},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate', required=True); ap.add_argument('--gt', required=True); ap.add_argument('--cameras', required=True)
    ap.add_argument('--obs-mask', required=True); ap.add_argument('--plane', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--samples', type=int, default=100000); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--tolerance-fraction', type=float, default=.01)
    a = ap.parse_args()
    report = evaluate(Path(a.candidate), Path(a.gt), Path(a.cameras), Path(a.obs_mask), Path(a.plane), samples=a.samples, seed=a.seed, tolerance_fraction=a.tolerance_fraction)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
if __name__ == '__main__': main()
