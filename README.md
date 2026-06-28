# Manifold-Conservative Gaussian Splatting

## One-line thesis

3D Gaussian Splatting should not be treated as an unconstrained volumetric radiance fitting problem. A well-behaved splat scene should be interpretable as a discretized surface measure induced by one or more manifolds, with a small set of curve-like or volume-like primitives reserved for genuinely non-surface structures.

## Working title

**Manifold-Conservative Gaussian Splatting: Projecting Radiance Gaussians onto Geometry-Consistent Surface Measures**

Alternative titles:

- **Varifold Gaussian Splatting**
- **Topology-Guided Triangle Splatting**
- **Geometry-Conservative Hybrid Splatting**
- **Field-Aligned Surface Splatting with Manifold Projection**

## Core idea

Given a 3DGS representation

```math
G = \{(\mu_i, \Sigma_i, \alpha_i, c_i)\}_{i=1}^N,
```

the covariance eigensystem already contains a local geometric hypothesis:

```math
\Sigma_i = R_i \operatorname{diag}(\lambda_{i1}, \lambda_{i2}, \lambda_{i3})R_i^T.
```

If a splat represents a local surface element, then

```math
\lambda_{i1}, \lambda_{i2} \gg \lambda_{i3}.
```

The two large-eigenvalue directions span a tangent plane, and the smallest-eigenvalue direction gives a local normal. Therefore, a thin 3D Gaussian can be reinterpreted as a probabilistic surface element rather than a free 3D blob.

The research question is:

> When is a set of radiance Gaussians equivalent, or close, to a 2D manifold with an attached appearance field?

This differs from adding normal/depth regularization to 3DGS. The goal is to define a projection from unconstrained splats to a manifold-induced geometric measure.

## Current evidence boundary

The current prototype has verified the engineering path, not the full research claim yet.

Verified:

- official 3DGS can run with the ManifoldGS training hook;
- covariance thinness/rank-style losses are differentiable and nonzero;
- a 50-iteration synthetic smoke run completed;
- bootstrap thinness moved the spectrum in the expected direction:
  - `thinness_median`: about `0.371 -> 0.299`;
  - `r23_median`: about `0.903 -> 0.716`.

Not yet proven:

- default thresholds still classify the smoke checkpoint as volume-like, so a meaningful surface patch graph has not emerged in the short run;
- no real-scene sparse-view comparison has been run yet;
- no claim against SuGaR, SolidGS, GOF, or MeshSplat is justified yet.

The old synthetic scene is now explicitly classified as smoke-only: its RGB
images and sinusoidal COLMAP points do not share a rendering model. The next
paper experiment must use a single analytic geometry source for RGB, depth,
normals, masks, surface samples, and mesh GT. Covariance-spectrum improvements
alone are only thin-Gaussian regularization, not evidence for ManifoldGS.

See `THEORY-VALIDATION.md` for the target propositions, falsifiable hypotheses,
GT definition, metrics, baselines, and pre-registered engineering thresholds.
See `THEORY-PROOF-SKETCH.md` for the proof boundary and
`GEOMETRY-PIPELINE-STATUS.md` for the now-runnable offline projection loop.
The current positioning is audited in
`THEORY-NOVELTY-REAUDIT-2026-06-27.md`; broad first-manifold/varifold/topology
claims are explicitly rejected.
The proposed oral-level differentiator and its current diagnostics are specified
in `FUNDAMENTAL-COMPATIBILITY.md`.

## Why this may still be novel

The current GS geometry line mostly falls into several families:

- **2DGS**: makes each primitive locally surface-like by using 2D disks.
- **SuGaR / 2D-SuGaR**: aligns Gaussians with surfaces and extracts/refines meshes.
- **PGSR / GausSurf / SolidGS / MeshSplat**: improves surface quality using depth, normal, MVS, or mesh priors.
- **Octree-GS**: uses octree organization mainly for scale, LOD, efficiency, and consistency.

The proposed angle is different:

> 2DGS makes individual splats local surface elements; this project asks whether the whole splat set satisfies global manifold, area-measure, normal-integrability, and topology-consistency conditions.

## Main contributions to target

1. **Manifold interpretation of Gaussian splats**
   - Convert covariance eigensystems into tangent planes, normals, area weights, and dimensionality labels.

2. **Geometry-conservative losses**
   - Area-measure conservation.
   - Normal-field integrability.
   - Curvature-scale consistency.
   - Local 2D manifold neighborhood validity.
   - Free-space and duplicate-layer suppression.

3. **Hybrid dimensional primitives**
   - Surface-like thin Gaussians for ordinary surfaces.
   - Curve-like Gaussians for wires, branches, rails, and thin structures.
   - Sparse volume-like Gaussians for fuzzy, translucent, or hard-to-surface effects.

4. **Asset-oriented output**
   - A manifold patch graph or mesh backbone.
   - Appearance splats attached to surface charts.
   - Optional exported mesh, UV/charts, collision geometry, and residual appearance layer.

## Practical positioning

Do not compete only on PSNR/SSIM/LPIPS. The stronger target is:

- sparse-view reconstruction with fewer floaters and duplicate layers;
- geometry-stable splatting;
- editable and engine-usable assets;
- better thin structures and high-curvature areas;
- explainable primitive dimensionality: curve/surface/volume.

## Current implementation status

See:

- `IMPLEMENTATION.md` for repository layout and runnable commands.
- `EXPERIMENTS.md` for the first benchmark path.
- `TODO-RUNS.md` for the current handoff commands.
- `THEORY-VALIDATION.md` for the oral-paper theory and validation contract.
- `NOVELTY_AUDIT_2026-06-27.md` for the novelty boundary against existing GS surface/mesh work.

## Key references

- 3D Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- 2D Gaussian Splatting: https://arxiv.org/abs/2403.17888
- SuGaR: https://arxiv.org/abs/2311.12775
- Octree-GS: https://arxiv.org/abs/2403.17898
- PGSR: https://arxiv.org/abs/2406.06521
- SolidGS: https://arxiv.org/abs/2412.15400
- GausSurf: https://arxiv.org/abs/2411.19454
- MeshSplat: https://arxiv.org/abs/2508.17811
