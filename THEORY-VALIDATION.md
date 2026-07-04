# Theory and Validation Contract

## 0. Paper target

This is the main research line. Contacting faculty is a downstream use of a
credible result, not the optimization target of the project.

The target claim is not:

> Thin Gaussians look more surface-like.

The target claim is:

> Under sparse observations, a Gaussian scene can be projected onto a
> refinement-stable discrete geometric measure whose support and tangent field
> converge toward a manifold, while retaining a separate appearance model.

## 1. Mathematical object

For a compact piecewise-smooth surface `M`, define its unoriented varifold

```math
V_M(\phi)=\int_M \phi(x,T_xM)\,dA(x),
```

where `T_xM` is the tangent 2-plane. A surface-like Gaussian provides a center
`mu_i` and tangent plane `P_i` from its two largest covariance eigenvectors.
The proposed discrete object is

```math
V_G(\phi)=\sum_i q_i\phi(\mu_i,P_i), \qquad q_i>0.
```

`q_i` must be a geometric quadrature mass. It should not be identified blindly
with 3DGS opacity. Opacity participates in view-dependent compositing and is not
an additive area measure. The current prototype quantity
`alpha_i sqrt(lambda_i1 lambda_i2)` is therefore only a diagnostic surrogate.

The implementation should introduce either:

1. an explicit positive geometric mass `q_i`; or
2. normalized local partition-of-unity weights whose sum estimates patch area.

Appearance parameters remain separate from `q_i`.

## 2. Target theoretical result

### Proposition A: consistency of the Gaussian-to-varifold map

Target statement to prove, not a result already established by the code:

Let `M` be a compact `C^2` surface with bounded curvature. Let Gaussian centers
have fill distance `h`, tangent-plane error `O(h)`, and positive quadrature
weights approximate surface area with first-order accuracy. Then for bounded
Lipschitz test functions on position and the Grassmannian,

```math
d_BL(V_G,V_M) <= C_M (h + epsilon_T + epsilon_q).
```

With covariance mollification of tangent radius `sigma` and normal thickness
`eta`, the corresponding spatial measure has an additional
`O(sigma + eta)` approximation term.

This result gives us an actual convergence target. It does not claim that an
arbitrary radiance Gaussian mixture has a unique underlying manifold; the
rendering inverse problem is many-to-one without geometric assumptions.

### Proposition B: refinement conservation

For a split operation replacing parent `i` by children `C(i)`, require

```math
sum_{j in C(i)} q_j = q_i
```

and require child barycenter/tangent moments to approximate the parent up to the
local refinement order. Then the discrete measure changes by at most the local
cell diameter in bounded-Lipschitz distance. Merge is the reverse operation.

This is the conservation law that the octree can enforce. Pairwise smoothing of
`alpha * tangent_area` is not sufficient to establish it.

### Proposition C: topology recovery is conditional

If reach/separation is bounded below, sampling is dense enough, and tangent
errors are bounded, a radius graph or restricted-complex construction can
recover the topology of `M`. This must be presented as a conditional result.
Sparse-view occlusion can violate the assumptions, in which case the method
must report uncertain/open patches instead of inventing watertight topology.

## 3. Falsifiable hypotheses

### H1: measure consistency

At matched primitive count, ManifoldGS reduces kernel-varifold discrepancy to
analytic GT relative to vanilla 3DGS and thin-only 3DGS.

### H2: refinement invariance

After forced split/merge at fixed rendered appearance, geometric mass, first
moment, and varifold discrepancy change less than in an unconstrained baseline.

### H3: geometry from sparse views

At 3/6/9 input views, ManifoldGS improves GT surface distance and normal error
without a material held-out rendering regression.

H3 contains two logically separate hypotheses and must be audited as such:

- **H3a (realizability):** tangent/value and shape residuals reduce distance to
  the locally realizable field set;
- **H3b (identifiability):** the available image/visibility evidence selects the
  GT member within that realizable set.

Bonnet compatibility addresses H3a only. For a visible depth graph,
`THEORY-STABILITY.md` proves that an `H1` rendered-depth error controls position,
tangent, and kernel-varifold error. RGB-only control instead requires a positive
smallest singular value of the multi-view image Jacobian after appearance
gauges are removed; this is an assumption to test, not a consequence of low
PSNR loss.

### H4: topology and asset utility

On closed and boundary surfaces, extracted patches recover the correct number
of components and Euler characteristic more often, and produce fewer
non-manifold edges and duplicate sheets.

### H5: mixed-dimensional necessity

On scenes containing a surface plus a thin curve, surface/curve/volume typing
improves both curve recall and surface precision over an all-2D representation.

Failure of H1 or H2 invalidates the word "conservative". Failure of H3 means the
work is a representation diagnostic rather than a reconstruction method.

## 4. Ground truth

The old `scripts/make_synthetic_colmap_scene.py` dataset is not geometry GT. Its
RGB images and sinusoidal COLMAP points are generated independently. It is valid
only for import/training smoke tests.

The first valid benchmark must render RGB, mask, depth, and normals from the
same analytic surface and camera model. It must also export dense GT surface
samples, tangent planes, area weights, and a reference mesh.

Required analytic scenes:

| Scene | Purpose | Known GT |
|---|---|---|
| plane patch | zero curvature, boundary | depth, normal, area, one component |
| sphere | smooth closed manifold | SDF, normals, area `4 pi r^2`, Euler `2` |
| torus | nonzero genus | SDF, normals, area, Euler `0` |
| two close sheets | duplicate-layer stress | separation and two components |
| surface + curve | mixed dimension | labels and independent measures |

Real benchmarks come after analytic validation: DTU for geometry, then a
sparse-view benchmark with held-out views. Real data has scan/mesh GT but no
perfect latent manifold, so it cannot replace the analytic tests.

### Identifiability audit

The analytic generator already exports train/held-out RGB, depth, normal, and
mask from the same renderer. Use these channels in a diagnostic ladder:

1. RGB only;
2. RGB plus train-view mask/free-space;
3. RGB plus exact train-view depth and depth gradient (oracle);
4. each of 2 and 3 with tangent+shape compatibility.

The oracle rung is not a fair sparse-RGB method and must never be reported as
one. It diagnoses whether failure comes from the compatibility estimator or
from missing scene evidence. If exact visible depth still fails on observed
regions, the representation/optimizer is at fault. If depth succeeds while RGB
fails, H3b is the bottleneck and no compatibility weight can repair it alone.

Each rung must also report the union of train-camera visible GT area and the
model mass assigned to that visible union. Proposition 8 in
`THEORY-STABILITY.md` leaves an unavoidable residual term proportional to
unmatched/unseen mass. View count is not a substitute for this coverage: three
well-spread views and three nearly redundant views have different guarantees.

On `plane_torus_sparse_v2_shape`, within-scene correlations already reject
held-out rendering quality as a reliable geometry proxy. For torus, Spearman
correlation of PSNR with varifold is `-0.067` and with normal error is `0.25`;
for plane, PSNR with varifold is `+0.633` (the wrong direction for an error
metric). Correlations pooled across scenes are confounded by scene difficulty
and must not be used as identifiability evidence.

The train-camera first-hit coverage audit separates the two scenes further.
Across three seeds, plane has `100%` GT mass coverage because it is a single
visible patch. Torus has only about `49--54%` coverage at depth tolerances of
`0.5--1%` of the bounding-box diagonal. Thus torus has a large unavoidable H3b
null region under three views, while plane's failure must come from RGB
Jacobian conditioning, appearance gauge, optimization, or representation error
rather than missing first-hit coverage.

The seed-0, 300-step plane identifiability ladder (no densification) gives:

| method | certified Chamfer | normal median | varifold | held-out PSNR |
|---|---:|---:|---:|---:|
| RGB | 0.03091 | 63.65 deg | 0.17021 | 28.233 |
| RGB + point-depth oracle | **0.02785** | 63.71 deg | 0.16908 | 28.149 |
| RGB + compatibility | 0.03089 | **23.33 deg** | **0.11478** | **28.298** |
| RGB + oracle + compatibility | 0.02768 | 24.65 deg | 0.11731 | 28.216 |

This is a mechanism result, not a statistical claim. Pointwise depth improves
support position but not tangent error, exactly as Proposition 6 predicts when
only the `L2` part and not the depth-gradient part is supplied. Compatibility
provides the large tangent/varifold gain before densification. The poor 7k plane
result must therefore be localized to later optimization, densification, graph
changes, or loss scheduling rather than attributed solely to RGB
non-identifiability.

The follow-up 1500-step plane run includes densification and resolves that
ambiguity:

| method | certified Chamfer | normal median | varifold | mesh Chamfer | PSNR |
|---|---:|---:|---:|---:|---:|
| RGB | 0.05946 | 64.91 deg | 0.19335 | 0.07109 | 28.017 |
| RGB + compatibility | 0.05011 | 55.18 deg | 0.15167 | 0.03593 | 27.662 |
| RGB + adaptive compatibility | 0.04537 | 55.95 deg | 0.14614 | 0.02872 | 30.264 |
| oracle depth + adaptive compatibility | **0.01885** | **8.84 deg** | **0.06056** | **0.00726** | **30.863** |

Adaptive refresh caps normalized cache drift at `0.00996` instead of `0.5623`
and improves support/varifold/mesh metrics, but does not rescue normal accuracy
under RGB alone. After densification, the RGB-only shape training term rises
from about `0.21` to `1.3--1.6`; with oracle depth it remains around `0.5--0.7`.
The combined result supports the stability decomposition: compatibility is
useful but not sufficient after refinement adds geometric degrees of freedom;
a data-coercive support anchor is required. Exact depth remains an oracle and
does not establish a sparse-RGB method.

A controlled depth-noise sweep at the same 1500-step setting gives:

| relative depth noise | certified Chamfer | normal median | varifold | certified mass |
|---:|---:|---:|---:|---:|
| exact | **0.01885** | **8.84 deg** | **0.06056** | 77.0% |
| 0.5% | 0.02864 | 17.21 deg | 0.07522 | 76.0% |
| 1% | 0.02645 | 21.20 deg | 0.08452 | 62.6% |
| 2% | 0.03096 | 36.17 deg | 0.11355 | 41.4% |
| 5% | 0.04685 | 54.38 deg | 0.14113 | 25.4% |
| RGB + adaptive compatibility | 0.04537 | 55.95 deg | 0.14614 | 30.7% |

Normal, varifold, and certified coverage degrade consistently with noise. The
small Chamfer inversion between `0.5%` and `1%` is a single-seed optimization
effect, not evidence that noise helps. Up to `2%` relative iid depth noise still
provides substantial geometry gains over RGB-only adaptive compatibility; at
`5%` the method is effectively back at the RGB-only geometry level. These are
synthetic iid perturbations. Real monocular depth has structured scale, edge,
and occlusion errors and requires a separate robustness audit.

The structured-error audit confirms the predicted null space:

| depth error | Chamfer | normal median | varifold | shape mismatch | mesh Chamfer |
|---|---:|---:|---:|---:|---:|
| 2% scale | 0.04950 | 15.32 deg | 0.08997 | 0.41555 | 0.04169 |
| 2% bias | 0.04656 | 14.69 deg | 0.08926 | 0.40406 | 0.04144 |
| 2% low-frequency warp | 0.03347 | 26.63 deg | 0.10309 | 0.42783 | 0.01799 |
| 30% dropout (invalid old sampler) | 0.05452 | 55.41 deg | 0.15372 | 0.55033 | 0.04440 |
| combined (invalid old sampler) | 0.05824 | 56.16 deg | 0.14841 | 0.54746 | 0.04215 |

Scale and bias preserve a coherent plane and therefore retain good normal and
compatibility scores, while their GT Chamfer and mesh support become slightly
worse than RGB-only adaptive compatibility. This is direct empirical evidence
that compatibility cannot reject a consistently displaced realizable surface.
Low-frequency error remains partially useful; random missing depth is much more
damaging because it collapses the data-anchored coverage. A realistic method
must calibrate affine depth gauge and explicitly manage missing/uncertain support
before applying compatibility.

The original dropout and combined rows are retained for audit history but are
not valid missing-data conclusions: bilinear sampling mixed zero-valued missing
pixels into positive depths, creating artificially shallow targets. The sampler
now divides interpolated depth by interpolated validity mass; a mask-normalized
rerun supersedes those rows.

The corrected rerun gives:

| corrected depth condition | Chamfer | normal median | varifold | certified mass | mesh Chamfer |
|---|---:|---:|---:|---:|---:|
| 30% dropout, mask-normalized | **0.01806** | 11.98 deg | 0.06652 | 72.7% | 0.00892 |
| combined, mask-normalized | 0.03456 | 28.63 deg | 0.10205 | 54.0% | 0.01895 |
| combined + affine calibration | 0.03424 | **28.20 deg** | **0.09781** | **57.0%** | **0.01358** |

Random missing depth alone is therefore not the failure mechanism when validity
is normalized correctly and three views provide redundant coverage. The
combined structured residual remains harder but still substantially outperforms
RGB-only adaptive compatibility. Affine calibration removes its coherent gauge
component and improves varifold, coverage, mesh support, PSNR, and SSIM, while
the low-frequency/iid residual sets the remaining error floor.

Per-view robust affine calibration repairs the coherent gauge. For 2% scale,
Chamfer/normal/varifold change from `0.04950 / 15.32 deg / 0.08997` to
`0.01700 / 10.14 deg / 0.05863`; for 2% bias they change from
`0.04656 / 14.69 deg / 0.08926` to `0.02038 / 11.00 deg / 0.06550`.
Both return close to the exact-depth reference without using GT surface samples.
Calibration does not address low-frequency or missing-depth errors.

## 5. Metrics

### Measure metrics

- kernel-varifold distance using position and tangent-plane kernels;
- relative total geometric mass error;
- first-moment/barycenter error;
- split/merge drift in all three quantities.

### Geometry metrics

- accuracy and completeness against GT surface;
- symmetric Chamfer-L1;
- normal angular error and normal consistency;
- F-score at thresholds normalized by scene bounding-box diagonal;
- free-space violations and duplicate-sheet rate.

### Topology/asset metrics

- connected components, boundary loops, Euler characteristic, and genus when
  the extracted result is closed;
- non-manifold edge ratio, self-intersection count, isolated component count;
- mesh triangle count at matched error and editable patch count.

### Rendering metrics

- PSNR, SSIM, and LPIPS on held-out views only;
- report training-view metrics separately and never use them as geometry proof.

Covariance ratios (`r23`, thinness) are mechanism diagnostics, not success
metrics. A degenerate set of flat splats can score perfectly on thinness while
being far from the GT surface.

## 6. Initial acceptance criteria

These are go/no-go engineering thresholds, not numbers to put in a paper before
running statistics.

Across at least three seeds and 3/6/9-view settings:

- kernel-varifold distance: at least 20% lower than vanilla and thin-only;
- symmetric Chamfer-L1: at least 15% lower than vanilla on analytic scenes;
- normal median error: at least 20% lower than vanilla;
- held-out PSNR regression: no worse than 0.3 dB on average;
- relative mass drift after split/merge: below 1%;
- topology: exact component/Euler recovery on at least 80% of analytic runs;
- non-manifold edge ratio: below 1% for mesh-output runs.

If these fail, report confidence intervals and effect sizes before tuning. Do not
move thresholds after seeing only the preferred method's result.

## 7. Required baselines and ablations

Baselines:

- vanilla 3DGS;
- 2DGS;
- SuGaR or the strongest reproducible mesh/surface baseline;
- thin-only 3DGS with the same covariance pressure as ours.

Ablations:

- no explicit geometric mass;
- no split/merge conservation;
- no tangent integrability;
- no topology filtering;
- no mixed-dimensional typing;
- no curvature-driven refinement.

The thin-only baseline is mandatory. Without it, improvements cannot be
attributed to the manifold formulation.
