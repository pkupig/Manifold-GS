# Geometry Pipeline Status

## Closed offline path

```text
3DGS PLY
  -> explicit q (existing geom_mass or kNN area quadrature)
  -> weighted local PCA
  -> quadratic MLS support/tangent projection
  -> confidence without deleting uncertain mass
  -> bounded-normal-variation chart graph
  -> conservative patch triangulation
  -> projected full-attribute Gaussian PLY + patch mesh
  -> reload into official 3DGS training
```

Command:

```bash
python scripts/project_manifold.py \
  --ply path/to/point_cloud.ply \
  --out experiments/scene/manifold \
  --gt path/to/gt/surface.npz
```

Continue training:

```bash
python third_party/gaussian-splatting/train.py \
  -s path/to/scene -m path/to/new_model \
  --mcgs_initial_ply experiments/scene/manifold/projected_gaussians.ply \
  --disable_viewer
```

## Sphere calibration

Input: vanilla 3DGS after 300 iterations, 3 views, 192 opaque primitives.

| metric | before | quadratic projection | change |
|---|---:|---:|---:|
| Chamfer-L1 | 0.070409 | 0.066895 | -4.99% |
| median normal error | 20.480 deg | 4.397 deg | -78.53% |
| normalized varifold | 0.151061 | 0.080272 | -46.86% |
| strict surface primitives | 9 | 165 | +156 |
| non-manifold mesh edges | n/a | 0 | n/a |

Output: 10 charts, 146 triangles, reload/save smoke completed.

The varifold values above use the corrected characteristic RBF kernel on
tangent projectors. Earlier values based on the linear `(n_i dot n_j)^2`
tangent kernel should be treated as legacy diagnostics, not compared directly.

## Important failure boundary

The projected representation immediately renders at roughly 26.2-26.3 dB,
versus 28.68 dB before projection. It violates the 0.3 dB appearance guardrail.
Hard offline projection is therefore suitable for geometry diagnosis and asset
export, but not yet for repeated in-training updates.

The training version needs a proximal update:

```math
theta_next = argmin_theta L_rgb(theta)
  + beta ||mu-P_MLS(mu_old)||_q^2
  + gamma d_Grassmann(P(theta),P_MLS)^2,
```

with annealed `beta/gamma`, appearance recovery, and tangent-aligned split.

An aggressively early-densified isotropic checkpoint is also a required
counterexample: mass is conserved, but local tangent evidence is absent and
quadratic MLS cannot identify the true sphere. The pipeline must report low
confidence rather than claim manifold recovery.

## Still open before paper training

- replace hard projection with differentiable/proximal support and tangent loss;
- split surface primitives only in the estimated tangent plane;
- separate semantic deletion from conservative representation merge;
- add visibility/free-space evidence for duplicate sheets and floaters;
- evaluate topology on plane/sphere/torus meshes, not only point support;
- preserve held-out rendering within the pre-registered guardrail.
