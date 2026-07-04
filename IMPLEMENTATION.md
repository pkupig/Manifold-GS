# Implementation Log

## Repository layout

Reference code is kept under `third_party/`:

- `third_party/gaussian-splatting`: official 3DGS code.
- `third_party/SuGaR`: SuGaR mesh extraction/refinement baseline.
- `third_party/octree-TS`: the original octree/triangle-splatting project to mine and migrate from.

Current reference commits:

- 3DGS: `54c035f7834b564019656c3e3fcc3646292f727d`
- SuGaR: `7c10c4ae4a267dece512f5c7f40ed212a0a2ab44`
- octree-TS: `b09754a0f9e36e7e846d4852809e9b620f8fea68`

Note: the 3DGS recursive clone failed for Inria GitLab submodules `SIBR_viewers` and `simple-knn`. `SIBR_viewers` is only the viewer. `simple-knn` is needed for full 3DGS training, so the training setup step must either fetch a working mirror or install a compatible package. The offline diagnostics here do not depend on those CUDA submodules.

The first local implementation layer is independent of CUDA extensions:

- `manifold_gs/ply_io.py`: minimal PLY reader/writer for 3DGS checkpoint files.
- `manifold_gs/diagnostics.py`: covariance eigensystem, dimensionality labels, and mesh-ready oriented point export.
- `manifold_gs/graph_diagnostics.py`: kNN local manifold scores.
- `manifold_gs/patch_mesh.py`: conservative patch graph and chart triangulation.
- `manifold_gs/losses.py`: renderer-agnostic PyTorch prototypes for geometry-conservative losses.
- `manifold_gs/torch_geometry.py`: differentiable covariance eigen-geometry from 3DGS scales and quaternions.
- `manifold_gs/training_hooks.py`: `ManifoldLossController` for periodic graph refresh and training loss computation.
- `patches/gaussian_splatting_train_mcgs.diff`: minimal official-3DGS training-loop patch.
- `patches/gaussian_splatting_mcgs_headless.diff`: current full diff including the training hook and a headless-safe `network_gui` import fix.
- `scripts/check_or_apply_train_patch.sh`: verifies or applies the training patch.
- `scripts/analyze_gaussians.py`: CLI wrapper.
- `scripts/extract_poisson_mesh.py`: Poisson mesh fallback from oriented surface splats.
- `scripts/extract_patch_mesh.py`: patch-chart mesh extraction from Gaussian covariance geometry.
- `scripts/export_asset_bundle.py`: export grouped OBJ/PLY patches, attached and residual
  full-attribute Gaussian layers, a conservative collision candidate, source mappings,
  and a machine-readable asset manifest.
- `scripts/preflight_3dgs.py`: checks torch CUDA, 3DGS extensions, and PLY dependency.
- `scripts/compare_diagnostics.py`: compares multiple diagnostic `summary.json` files and prints metric deltas.

## First runnable target

Given a trained 3DGS or SuGaR checkpoint:

```bash
cd E-Manifold-GS
python scripts/analyze_gaussians.py \
  --ply path/to/point_cloud/iteration_30000/point_cloud.ply \
  --out experiments/example_diagnostics
```

Outputs:

- `summary.json`: counts and ratios of surface/curve/volume-like splats.
- `diagnostics.npz`: arrays for eigenvalues, normals, mass, labels, and masks.
- `graph_diagnostics.npz`: local rank-2, normal variation, area variation, and curvature-scale scores.
- `surface_oriented_points.ply`: filtered oriented point cloud suitable for Poisson/Screened Poisson reconstruction.

Compare multiple diagnostic runs:

```bash
python scripts/compare_diagnostics.py \
  --summary vanilla=experiments/synthetic_vanilla_2k_diag/summary.json \
  --summary mcgs=experiments/synthetic_mcgs_2k_diag/summary.json \
  --baseline vanilla
```

Poisson fallback:

```bash
python scripts/extract_poisson_mesh.py \
  --points experiments/example_diagnostics/surface_oriented_points.ply \
  --mesh experiments/example_diagnostics/poisson_mesh.ply \
  --depth 9
```

Patch-chart mesh:

```bash
python scripts/extract_patch_mesh.py \
  --ply path/to/point_cloud/iteration_30000/point_cloud.ply \
  --mesh experiments/example_diagnostics/patch_mesh.ply
```

Hybrid asset bundle from the projected pipeline:

```bash
python scripts/export_asset_bundle.py \
  --gaussians experiments/scene/manifold/projected_gaussians.ply \
  --mesh experiments/scene/manifold/patch_mesh.ply \
  --meta experiments/scene/manifold/patch_mesh_meta.npz \
  --source-map experiments/scene/manifold/projected_manifold.npz \
  --out experiments/scene/hybrid_asset
```

The explicit source map is required when projection filtered or reordered the original
Gaussian rows. The collision output is a conservative candidate, not a production physics
asset. The bundle intentionally retains uncertified Gaussians as a residual layer.

## Why this is the right first step

This tests the key hypothesis before touching the renderer:

> Bad sparse-view 3DGS geometry should correlate with invalid covariance spectra, non-surface dimensionality, floaters, and weak manifold structure.

If this diagnostic signal is meaningful, we can add graph scores and then training losses.

## Next engineering steps

1. Apply and test the 3DGS training patch:

```bash
cd E-Manifold-GS
bash scripts/check_or_apply_train_patch.sh check
bash scripts/check_or_apply_train_patch.sh apply
```

Then run 3DGS with weak manifold losses after dependencies are installed:

```bash
python third_party/gaussian-splatting/train.py \
  -s path/to/scene \
  -m output/mcgs_scene \
  --mcgs_warmup 2000 \
  --mcgs_bootstrap_thin \
  --mcgs_lambda_thin 0.02 \
  --mcgs_lambda_area 0.005 \
  --mcgs_lambda_curv 0.005 \
  --mcgs_lambda_rank2 0.002 \
  --mcgs_lambda_normal 0.002
```

Current local environment notes:

- `diff_gaussian_rasterization` and `simple_knn` were built successfully with:

```bash
CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 \
python -m pip install --no-build-isolation \
  third_party/gaussian-splatting/submodules/diff-gaussian-rasterization \
  third_party/gaussian-splatting/submodules/simple-knn
```

- CUDA 12.8 required small include fixes:
  - `cuda_rasterizer/rasterizer_impl.h`: add `<cstdint>`.
  - `simple_knn.cu`: add `<cfloat>`.
- `plyfile==1.0.3` is used to avoid upgrading numpy beyond scipy's supported range.
- In the managed sandbox, GPU access is blocked and PyTorch reports no NVIDIA driver. Running GPU commands with escalated execution exposes the RTX 4060 Laptop GPU correctly.
- A 50-iteration synthetic 3DGS smoke run completed successfully with `--mcgs_bootstrap_thin`.

2. Add mesh extraction adapters:
   - SuGaR coarse extraction fallback;

3. Add training integration:
   - free-space pruning.
   - wire `manifold_gs.losses` into a 3DGS/SuGaR-compatible training loop.

4. Add sparse-view benchmark scripts:
   - train baseline 3DGS/SuGaR;
   - train manifold-conservative variant;
   - compare rendering, geometry, floater, and asset metrics.
- `scripts/generate_analytic_scene.py`: generate train/held-out RGB and geometry
  GT from one plane, sphere, or torus definition.
- `scripts/evaluate_geometry_gt.py`: evaluate checkpoint centers/tangent normals
  against analytic GT, including normalized kernel-varifold distance.
- `scripts/analyze_fundamental_compatibility.py`: compare center-support
  fundamental forms with covariance-normal geometry and report approximate
  Gauss-Codazzi residuals.
