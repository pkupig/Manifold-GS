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

