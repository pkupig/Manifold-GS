# Experiment Plan

## Current runnable pieces

### 1. Offline geometry diagnostics

Input:

```text
point_cloud/iteration_*/point_cloud.ply
```

Run:

```bash
python scripts/analyze_gaussians.py \
  --ply path/to/point_cloud.ply \
  --out experiments/scene_diag
```

This gives:

- surface/curve/volume primitive counts;
- covariance thinness;
- local rank-2 score;
- normal variation;
- log area-mass variation;
- curvature-scale score;
- oriented surface point cloud.

Purpose:

> Check whether bad geometry regions correlate with manifold-conservation violations.

### 2. Poisson mesh fallback

Run:

```bash
python scripts/extract_poisson_mesh.py \
  --points experiments/scene_diag/surface_oriented_points.ply \
  --mesh experiments/scene_diag/poisson_mesh.ply \
  --depth 9
```

Purpose:

> Keep an engineering-safe mesh extraction path so the project does not depend entirely on a new patch extraction algorithm.

### 3. Patch-chart mesh prototype

Run:

```bash
python scripts/extract_patch_mesh.py \
  --ply path/to/point_cloud.ply \
  --mesh experiments/scene_diag/patch_mesh.ply
```

Purpose:

> Test whether covariance-derived tangent structure can produce conservative local manifold patches without forcing a watertight global surface.

## Next implementation target

The next useful step is to integrate `manifold_gs.losses` into a trainable 3DGS fork.

Recommended order:

1. Use official 3DGS or SuGaR's embedded 3DGS fork as the training base.
2. Add a periodic graph refresh every `K` iterations:
   - compute activated scales and rotations;
   - classify surface-like splats;
   - build kNN edges on high-confidence surface splats;
   - cache `edges` and `neighbor_indices`.
3. Add losses with low initial weights:
   - `L_thin`;
   - `L_area`;
   - `L_curv`;
   - `L_rank2`;
   - `L_normal`.
4. Anneal weights after appearance stabilizes:
   - no manifold loss before iteration 1k-2k;
   - weak loss during densification;
   - stronger loss after densification slows down.
5. Export both:
   - normal 3DGS checkpoint for rendering metrics;
   - diagnostics/patch mesh for geometry and asset metrics.

## Suggested first benchmark

The previous synthetic COLMAP scene below is a software smoke test only. Its
images are not renderings of its COLMAP point surface, so it has no valid
geometry GT and must not appear in a paper comparison.

The first scientific benchmark is the analytic suite specified in
`THEORY-VALIDATION.md`: plane, sphere, torus, two close sheets, and a
surface-plus-curve scene. Each dataset must derive RGB, mask, depth, normals,
cameras, COLMAP points, dense surface samples, and reference mesh from one
shared scene definition.

Use a small sparse-view subset where full runs are cheap:

- DTU scan with 3, 6, 9 views; or
- a small COLMAP scene already available locally; or
- a synthetic object scene if cameras/mesh are easy.

For a pure pipeline smoke test only, generate the tiny synthetic scene:

```bash
python scripts/make_synthetic_colmap_scene.py \
  --out experiments/synthetic_colmap
```

Before training, check GPU/extension readiness:

```bash
python scripts/preflight_3dgs.py
```

Then run a short training job:

```bash
python third_party/gaussian-splatting/train.py \
  -s experiments/synthetic_colmap \
  -m experiments/synthetic_mcgs_run \
  --iterations 200 \
  --save_iterations 200 \
  --test_iterations 200 \
  --disable_viewer \
  --mcgs_warmup 20 \
  --mcgs_refresh_interval 20 \
  --mcgs_bootstrap_thin \
  --mcgs_lambda_thin 0.02 \
  --mcgs_lambda_area 0.005 \
  --mcgs_lambda_curv 0.005 \
  --mcgs_lambda_rank2 0.002 \
  --mcgs_lambda_normal 0.002
```

Validated smoke command on the synthetic scene:

```bash
python third_party/gaussian-splatting/train.py \
  -s experiments/synthetic_colmap \
  -m experiments/synthetic_mcgs_bootstrap_smoke \
  --iterations 50 \
  --save_iterations 50 \
  --test_iterations 50 \
  --disable_viewer \
  --mcgs_warmup 5 \
  --mcgs_refresh_interval 5 \
  --mcgs_bootstrap_thin \
  --mcgs_lambda_thin 0.02 \
  --mcgs_lambda_area 0.005 \
  --mcgs_lambda_curv 0.005 \
  --mcgs_lambda_rank2 0.002 \
  --mcgs_lambda_normal 0.002
```

Observed result:

- training completed on RTX 4060 Laptop GPU;
- `MCGS` loss was nonzero and decreased from about `0.0036` to `0.0015`;
- diagnostic `thinness_median` improved from about `0.371` in the no-bootstrap smoke run to about `0.299` after 50 bootstrap iterations.

Compare:

- baseline 3DGS;
- baseline SuGaR mesh extraction from 3DGS;
- 3DGS + manifold-conservative losses;
- 3DGS + manifold losses + Poisson fallback;
- 3DGS + manifold losses + patch mesh.

## Expected win condition

The initial go/no-go thresholds are defined before experiments in
`THEORY-VALIDATION.md`. In particular, the full method must beat both vanilla
and a matched thin-only baseline on GT geometry/varifold metrics while losing no
more than 0.3 dB held-out PSNR on average. Qualitative mesh cleanliness alone is
not a win condition.

The first version does not need to beat SuGaR everywhere. It should show:

- fewer floaters under sparse views;
- lower duplicate-layer tendency;
- better local rank-2 manifold score;
- comparable rendering metrics;
- mesh/patch output that is more editable or cleaner in ambiguous regions.

## Practical caution

Do not hard-lock all Gaussians into surfaces. Keep the dimensional split:

- surface primitives for ordinary surfaces;
- curve primitives for thin structures;
- limited volume primitives for residual appearance.

This is important for rendering quality. The geometry losses should shape high-confidence surface regions, not destroy the radiance model.
