#!/usr/bin/env python3
"""Run the frozen restricted-fisher/v1 certificate on one 3DGS bundle.

The runner intentionally does not optimize any checkpoint parameter.  For each
accepted patch it replaces only the selected Gaussian centers by +/- normal
translations, renders its cached first-hit training cameras, and restores the
checkpoint centers immediately after each image.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GS = ROOT / "third_party" / "gaussian-splatting"
sys.path.insert(0, str(GS)); sys.path.insert(0, str(ROOT))
import torch
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from manifold_gs.restricted_fisher import PROTOCOL_VERSION, classify_patch, finite_difference_fisher, patch_normals, scene_threshold


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--bundle', required=True); p.add_argument('--evidence', required=True); p.add_argument('--out', required=True)
    p.add_argument('--patches', type=int, nargs='*'); p.add_argument('--max-patches', type=int, default=None)
    p.add_argument('--epsilon-fraction', type=float, default=.005); p.add_argument('--pixels-per-view', type=int, default=4096); p.add_argument('--seed', type=int, default=0)
    known, rest = p.parse_known_args()
    # Reuse the upstream parser/config loader for -s/-m, pipeline and iteration.
    sys.argv = [sys.argv[0], *rest]
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser, sentinel=True); pipe = PipelineParams(parser)
    parser.add_argument('--iteration', type=int, default=7000); parser.add_argument('--quiet', action='store_true')
    args = get_combined_args(parser)
    return known, args, model, pipe


def patch_sources(bundle: Path, evidence: Path, selected: set[int] | None):
    mapping = np.load(bundle / 'asset_mapping.npz')
    source = np.asarray(mapping['attached_source_indices'], np.int64)
    patches = np.asarray(mapping['attached_patch_ids'], np.int64)
    # The manifest stores the supported-patch *count* for presentation; the
    # mapping carries the auditable rejected ID list used by the exporter.
    rejected = set(map(int, np.asarray(mapping['observation_rejected_patch_ids'], np.int64).tolist()))
    accepted = set(map(int, np.unique(patches).tolist())) - rejected
    wanted = sorted((accepted if selected is None else accepted & selected))
    cache = np.load(evidence)
    cache_source = np.asarray(cache['source_indices'], np.int64)
    cache_bits = np.asarray(cache['first_hit_view_bits'], np.uint64)
    names = [str(x) for x in np.asarray(cache['first_hit_view_names']).tolist()]
    bits_by_source = dict(zip(map(int, cache_source), cache_bits))
    result = {}
    for patch in wanted:
        ids = source[patches == patch]
        bits = np.uint64(0)
        for idx in ids: bits |= bits_by_source.get(int(idx), np.uint64(0))
        result[patch] = (np.unique(ids), bits, names)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_shifted(camera, gaussians, pipeline, background, original, rows, delta):
    shifted = original.clone()
    shifted[rows] += delta
    gaussians._xyz = shifted
    with torch.no_grad():
        image = render(camera, gaussians, pipeline, background)['render'].permute(1, 2, 0)
    gaussians._xyz = original
    if camera.alpha_mask is not None:
        image = image * camera.alpha_mask.to(image.device).permute(1, 2, 0)
    return image.detach().cpu().numpy()


def main() -> None:
    known, args, model, pipeline = parse()
    if known.epsilon_fraction != .005 or known.pixels_per_view != 4096 or known.seed != 0:
        raise ValueError('restricted-fisher/v1 parameters are frozen: epsilon=.005, pixels=4096, seed=0')
    bundle, evidence = Path(known.bundle), Path(known.evidence)
    patches = patch_sources(bundle, evidence, set(known.patches) if known.patches else None)
    if known.max_patches is not None: patches = dict(list(patches.items())[:known.max_patches])
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(model.extract(args), gaussians, load_iteration=args.iteration, shuffle=False)
    cameras = {c.image_name: c for c in scene.getTrainCameras()}
    background = torch.zeros(3, dtype=torch.float32, device='cuda')
    original = gaussians._xyz.detach()
    bbox = float(torch.linalg.vector_norm(original.max(0).values - original.min(0).values).item())
    epsilon = known.epsilon_fraction * bbox
    xyz = original.cpu().numpy()
    rng = np.random.default_rng(known.seed); records=[]
    for patch, (source_ids, bits, names) in patches.items():
        rows = torch.as_tensor(source_ids, dtype=torch.long, device=original.device)
        normal_map = patch_normals(xyz[source_ids], np.full(len(source_ids), patch))
        if patch not in normal_map:
            records.append({'patch_id': patch, 'fisher': None, 'views': 0, 'pixels': 0, 'status': 'numerically_unresolved'}); continue
        delta = torch.as_tensor(normal_map[patch] * epsilon, dtype=original.dtype, device=original.device)
        values=[]; pixels=0; views=0
        for bit, name in enumerate(names):
            if not (int(bits) >> bit) & 1 or name not in cameras: continue
            try:
                plus = render_shifted(cameras[name], gaussians, pipeline.extract(args), background, original, rows, delta)
                minus = render_shifted(cameras[name], gaussians, pipeline.extract(args), background, original, rows, -delta)
                # Perturbation support is the deterministic raster footprint.  It is
                # intersected with alpha and uniformly subsampled with frozen seed.
                footprint = np.isfinite(plus).all(-1) & np.isfinite(minus).all(-1) & (np.abs(plus-minus).sum(-1) > 0)
                choices = np.flatnonzero(footprint)
                if len(choices) > known.pixels_per_view: choices = rng.choice(choices, known.pixels_per_view, replace=False)
                valid = np.zeros(footprint.size, bool); valid[choices] = True; valid = valid.reshape(footprint.shape)
                value, count = finite_difference_fisher(plus, minus, epsilon, valid)
                if count: values.append((value, count)); pixels += count; views += 1
            except Exception as exc:
                records.append({'patch_id': patch, 'fisher': None, 'views': views, 'pixels': pixels, 'status': 'numerically_unresolved', 'error': str(exc)}); break
        else:
            fisher = float(np.average([v for v,c in values], weights=[c for v,c in values])) if values else None
            records.append({'patch_id': patch, 'fisher': fisher, 'views': views, 'pixels': pixels})
    threshold = scene_threshold(records)
    for rec in records:
        if 'status' not in rec:
            fisher = float('nan') if rec['fisher'] is None else rec['fisher']
            rec['status'] = classify_patch(fisher, rec['views'], rec['pixels'], threshold)
    checkpoint = Path(args.model_path) / 'point_cloud' / f'iteration_{args.iteration}' / 'point_cloud.ply'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'checkpoint point cloud not found: {checkpoint}')
    out = {'protocol_version': PROTOCOL_VERSION, 'checkpoint': {'iteration': args.iteration, 'path': str(checkpoint), 'sha256': sha256_file(checkpoint)}, 'epsilon_fraction': known.epsilon_fraction, 'epsilon': epsilon, 'pixels_per_view': known.pixels_per_view, 'seed': known.seed, 'bbox_diagonal': bbox, 'fisher_p10_threshold': threshold, 'records': records}
    Path(known.out).parent.mkdir(parents=True, exist_ok=True); Path(known.out).write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'out': known.out, 'patches': len(records), 'threshold': threshold, 'status': {s: sum(r['status']==s for r in records) for s in sorted({r['status'] for r in records})}}, indent=2))
if __name__ == '__main__': main()
