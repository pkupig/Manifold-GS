# Fundamental-form Compatibility for ManifoldGS

## Why this replaces normal smoothness

Smooth neighboring normals are not enough to prove that independently learned
centers, tangent planes, and curvatures come from one surface. The proposed
compatibility object compares two independent geometric descriptions:

1. **support geometry** fitted from Gaussian centers;
2. **predicted extrinsic geometry** obtained from covariance normals and their
   spatial derivatives.

For a local chart `phi(u,v)`, the support yields

```math
I = J_phi^T J_phi,
II^support_ab = <partial_ab phi,n_support>.
```

The covariance normal field yields

```math
II^cov_ab = -<partial_a n_cov,partial_b phi>.
```

If both descriptions represent one regular surface, they should agree and the
second form should be symmetric. Their corresponding shape operators should
satisfy Gauss-Codazzi compatibility.

## Implemented offline diagnostic

`manifold_gs/fundamental_compatibility.py` computes:

- covariance-normal versus support-normal angle;
- relative support/covariance shape-operator discrepancy;
- antisymmetric second-form residual;
- scale-normalized Gauss residual;
- local normal-field curl;
- local normal-coordinate Codazzi residual.

Run:

```bash
python scripts/analyze_fundamental_compatibility.py \
  --ply path/to/point_cloud.ply \
  --out experiments/scene/fundamental
```

The Codazzi implementation uses locally fitted normal coordinates and neglects
finite-neighborhood connection terms. It is a diagnostic approximation, not
yet a coordinate-invariant discrete theorem.

## Analytic falsification tests

Exact sphere and torus normal fields produce low residuals. Shuffling normals
while keeping support fixed increases all principal compatibility quantities.
Random sign flips do not change the result because the representation is
unoriented.

For the analytic sphere checkpoint:

| median metric | vanilla | projected | reduction |
|---|---:|---:|---:|
| normal angle | 21.837 deg | 0.955 deg | 95.6% |
| shape mismatch | 0.383 | 0.113 | 70.5% |
| symmetry residual | 0.364 | 0.029 | 92.0% |
| Gauss residual | 0.300 | 0.112 | 62.7% |
| normal curl | 1.410 | 0.024 | 98.3% |
| Codazzi residual | 0.503 | 0.057 | 88.7% |

This does not prove reconstruction correctness: the projected covariance field
was constructed from the same MLS support, so low post-projection residual is
partly expected. The useful experiment is whether a training loss lowers these
metrics on held-out geometry without hard projection.

## Proximal training design

Use alternating target refresh and differentiable optimization.

### Geometry refresh every K iterations

1. Select high-opacity support but retain uncertain mass separately.
2. Fit quadratic MLS charts from detached centers and explicit `q_i`.
3. Cache target centers, tangent projectors, local coordinates, confidence, and
   weighted least-squares derivative matrices.
4. Do not overwrite trainable Gaussians.

### Differentiable proximal terms

```math
L_support = sum_i q_i c_i ||mu_i-mu_hat_i||^2/r_i^2,
```

```math
L_tangent = sum_i q_i c_i ||P_i-P_hat_i||_F^2.
```

With cached derivative matrices `D_i`, normal derivatives remain differentiable
between refreshes:

```math
partial n_i = D_i [n_j]_{j in N(i)}.
```

This enables:

```math
L_shape = sum_i q_i c_i ||S^cov_i-S^support_i||_F^2 r_i^2,
```

```math
L_sym = sum_i q_i ||II^cov_i-(II^cov_i)^T||^2,
```

```math
L_Gauss = sum_i q_i (K(I_i)-det((I_i)^{-1}II^cov_i))^2,
```

and a discrete covariant Codazzi term after connection coefficients are cached
or consistently estimated.

### Optimization schedule

- appearance warm-up;
- weak support/tangent proximal terms;
- enable symmetry/Gauss terms only on confident charts;
- enable Codazzi after its discrete estimator passes coordinate-change tests;
- tangent-aligned conservative split;
- short appearance recovery after each graph refresh;
- reject an update when held-out rendering or free-space metrics regress.

The training hook implements an independent schedule for second-order terms:

```bash
--mcgs_compatibility_start 100 --mcgs_compatibility_ramp 100
```

Before `start`, only first-order/proximal terms act. Between `start` and
`start+ramp`, symmetry and Gauss weights increase linearly. This avoids asking
derivatives of an unstable covariance-normal field to satisfy a second-order
constraint.

Current implementation status:

- `L_support`, `L_tangent`: enabled in training;
- `L_shape`: enabled in training via `--mcgs_lambda_shape`;
- `L_sym`: enabled in training;
- `L_Gauss`: enabled but still considered experimental;
- discrete Codazzi: still offline diagnostic only, not a headline training term yet.

The local least-squares operators now use radius-normalized coordinates, so
their numerical rank threshold is invariant to scene scale and densification
radius. Offline diagnostics additionally report:

- covariance/support absolute normal alignment (`normal_alignment_abs`);
- scaled linear and quadratic Gram minimum eigenvalues;
- normalized kNN boundary gap;
- normalized PCA normal eigengap.

Training exposes optional `--mcgs_compatibility_alignment_floor` and
`--mcgs_compatibility_gram_floor` gates and an optional
`--mcgs_compatibility_max_cache_drift` adaptive-refresh trigger. All default to
zero, preserving prior behavior until their thresholds are calibrated. Cache
drift is logged relative to neighborhood radius after removing common
translation. Shape-only training now builds the compatibility cache
independently of whether symmetry or Gauss is enabled.

### Cache-drift mechanism check

A 300-step analytic-sphere comparison used identical losses and changed only
cache refresh behavior. Fixed refresh used `interval=100`; adaptive refresh
kept normalized relative center drift below `0.01`.

| metric | fixed cache | drift-adaptive |
|---|---:|---:|
| graph refreshes | 3 | 31 |
| maximum recorded cache drift | 0.10924 | 0.00996 |
| all-opaque Chamfer | 0.069729 | 0.069726 |
| normal median | **9.49 deg** | 10.03 deg |
| normalized kernel varifold | 0.096276 | **0.096048** |
| shape mismatch median | 0.23274 | **0.22991** |
| symmetry median | 0.21058 | **0.18650** |
| Codazzi median | 0.42354 | **0.41432** |

Adaptive refresh improves the compatibility mechanism, especially symmetry,
with about 27% wall-time overhead in this small run. It does not materially
improve GT geometry and slightly worsens the normal median. This supports the
theoretical decomposition: estimator freshness controls realizability defect,
not the separate data-identifiability term. This is one short mechanism run,
not a benchmark claim or a frozen `0.01` threshold.

## 300-step mechanism calibration

The following is one analytic sphere, three training views, seed 0, without
densification. It is a mechanism check, not a benchmark claim.

| method | train PSNR | Chamfer-L1 | normal median | varifold | symmetry |
|---|---:|---:|---:|---:|---:|
| vanilla + explicit q | 28.683 | 0.07039 | 20.59 deg | 0.11714 | 0.27034 |
| tangent only | 28.709 | 0.07029 | 10.10 deg | 0.10029 | 0.21058 |
| tangent + fixed symmetry | 28.714 | 0.07034 | 11.15 deg | 0.09905 | 0.17923 |
| tangent + delayed symmetry | 28.691 | **0.06975** | **10.06 deg** | **0.09814** | **0.17490** |

Delayed symmetry starts at iteration 100 and ramps for 100 iterations. Relative
to tangent-only, it improves every listed geometry quantity while changing
train PSNR by `-0.018 dB`. This supports the narrower claim that second-form
symmetry controls information not supplied by tangent alignment alone.

It does **not** yet establish:

- multi-seed significance;
- improvement with densification and topology changes;
- a useful Gauss or Codazzi training loss;
- superiority to GeoSplat, SuGaR, 2DGS, or sparse-view baselines.

The determinant-only Gauss loss remains poorly conditioned in the current
calibration. It is experimental and should not be enabled in headline runs.

## Densification calibration and certified asset layer

A 7k pilot exposed that confidence-gated symmetry could lower its training
objective by reducing certified chart coverage. The corrected implementation:

- constructs derivative operators on actual trainable centers, not projected
  proxy centers;
- keeps all opaque candidates in compatibility with a confidence weight floor;
- reports certified mass coverage as a guardrail;
- uses a weak support proximal term to tie centers and covariance geometry to
  the same local surface.

On the 1500-step sphere densification calibration, the corrected full method
versus tangent-only gives:

| certified/asset metric | tangent | full |
|---|---:|---:|
| certified mass coverage | 61.0% | 69.3% |
| certified point Chamfer | 0.07165 | 0.05438 |
| certified normal | 9.95 deg | 9.17 deg |
| certified varifold | 0.08878 | 0.08666 |
| symmetry residual | 0.17586 | 0.11182 |
| sampled patch-mesh Chamfer | 0.15229 | 0.13278 |
| sampled patch-mesh normal | 27.77 deg | 23.69 deg |
| nonmanifold edges | 0 | 0 |

The patch mesh still has 30 connected components and 464 boundary edges. It is
a certified editable surface backbone, not yet a watertight final mesh. Global
Poisson reconstruction produced large unsupported surfaces and is rejected as
a headline extraction path.

The corrected 7k seed-0 pilot remains effective after full densification. The
following values use the final registered evaluator:

| metric | vanilla | tangent | manifold full |
|---|---:|---:|---:|
| certified mass coverage | 53.4% | 52.9% | **61.9%** |
| certified Chamfer | 0.2326 | 0.2368 | **0.1075** |
| certified normal | 45.43 deg | 28.52 deg | **18.31 deg** |
| robust certified varifold | 0.0947 | 0.0910 | **0.0777** |
| symmetry residual | 0.3211 | 0.2583 | **0.2228** |
| sampled patch-mesh Chamfer | 0.2835 | 0.3097 | **0.1169** |
| sampled patch-mesh normal | 46.45 deg | 42.78 deg | **31.37 deg** |
| held-out PSNR | 20.856 | 20.940 | **21.100** |
| held-out SSIM | **0.540** | 0.522 | 0.514 |

This is one seed and cannot support a paper claim. Against vanilla, the full
method passes the registered single-run Chamfer, normal, and mesh targets;
robust varifold improves by 17.9%, below the registered 20% target, and SSIM
drops by 0.0264, beyond the 0.01 guardrail. Single-run threshold checks are not
the final statistical decision.

## Three-seed registered result

The complete `sphere_sparse_v2` experiment has now run seeds 0, 1, and 2. Its
registered paired-CI decision is `INCONCLUSIVE`: no primary metric has a 95%
paired t interval wholly beyond its improvement threshold, and neither rendering
guardrail has an interval wholly beyond its failure threshold.

| three-seed mean | vanilla | tangent | manifold full |
|---|---:|---:|---:|
| certified mass coverage | 54.95% | 50.71% | **63.44%** |
| certified Chamfer | 0.24090 | 0.22158 | **0.15211** |
| certified normal | 42.53 deg | 32.45 deg | **27.47 deg** |
| robust certified varifold | 0.09124 | 0.08821 | **0.08047** |
| sampled patch-mesh Chamfer | 0.28287 | 0.26534 | **0.16346** |
| sampled patch-mesh normal | 44.89 deg | 41.78 deg | **36.08 deg** |
| held-out PSNR | **21.4959** | 21.4666 | 21.3542 |
| held-out SSIM | **0.5539** | 0.5514 | 0.5395 |

The means support a geometry/asset improvement direction, including a 37.1%
point-Chamfer and 42.0% mesh-Chamfer reduction against vanilla. They do not yet
constitute a statistical paper claim. Seed 2 also shows that normal accuracy is
the least stable quantity: full reaches 42.67 degrees versus 42.94 for tangent,
despite retaining clear support and mesh-Chamfer gains. The next decisive test is
more seeds plus plane/torus scenes, not post-hoc sphere threshold tuning.

## Required ablations

- first-order normal supervision (`tangent` in historical manifests), an internal
  ablation rather than a GeoSplat reproduction or proxy;
- `q` conservation only;
- support+tangent proximal projection;
- plus second-form symmetry;
- plus support/normal shape-operator match;
- plus Gauss residual;
- plus Codazzi residual.

The internal contribution exists only if the compatibility terms improve
geometry beyond first-order normal supervision. A comparative paper claim
additionally requires running the authors' GeoSplat implementation (or a clearly
documented faithful reimplementation) under the same data and evaluator.
