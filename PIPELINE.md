# Pipeline Design

## Goal

Build a sparse-view geometry-stable splatting system that treats Gaussians as samples of a manifold-induced measure rather than as unconstrained radiance particles.

The intended representation is not "mesh only" or "splat only":

```text
input images + cameras
  -> initial Gaussians / depth-normal priors
  -> covariance-to-geometry interpretation
  -> manifold projection and topology-aware filtering
  -> hybrid surface/curve/volume primitive optimization
  -> editable asset: mesh/charts + attached appearance splats
```

## Stage 0: Inputs and initialization

### Inputs

- Calibrated sparse-view images.
- Camera poses from COLMAP or equivalent.
- Optional depth/normal priors from MVS, DUSt3R, MASt3R, VGGT, or monocular normal/depth models.

### Baseline initialization options

Option A: initialize from ordinary 3DGS.

- Train a short 3DGS warm-up.
- Extract Gaussian centers, covariances, opacity, and color/SH.
- Use covariance spectra to classify local geometry.

Option B: initialize from depth/normal priors.

- Backproject depth maps into point clouds.
- Fuse local normals.
- Initialize thin Gaussians oriented by predicted normals.
- This is preferred for sparse-view settings.

### Output of this stage

For each primitive:

```math
g_i = (\mu_i, \Sigma_i, \alpha_i, c_i)
```

and its eigendecomposition:

```math
\Sigma_i = R_i \operatorname{diag}(\lambda_{i1}, \lambda_{i2}, \lambda_{i3})R_i^T,
\quad \lambda_{i1} \ge \lambda_{i2} \ge \lambda_{i3}.
```

## Stage 1: Covariance-to-geometry interpretation

### Dimensionality classification

Use eigenvalue ratios to classify each primitive:

```math
r_{12} = \lambda_{i2}/\lambda_{i1}, \quad
r_{23} = \lambda_{i3}/\lambda_{i2}.
```

Suggested labels:

- **surface-like**: `r12` not too small, `r23` small.
- **curve-like**: `r12` small, `r23` not too meaningful.
- **volume-like**: all eigenvalues comparable.

Practical thresholds:

```text
surface: r12 > 0.25 and r23 < 0.08
curve:   r12 < 0.15 and r23 < 0.5
volume:  r23 > 0.2
```

These should be annealed rather than fixed in early training.

### Surface primitive interpretation

For a surface-like primitive:

- tangent frame:

```math
t_{i1}=R_i[:,1], \quad t_{i2}=R_i[:,2]
```

- normal:

```math
n_i = R_i[:,3]
```

- tangent footprint:

```math
A_i \propto \sqrt{\lambda_{i1}\lambda_{i2}}
```

- normal thickness:

```math
h_i = \sqrt{\lambda_{i3}}
```

### Key design decision

Do not merely force every primitive to be 2D. Instead, estimate which local structures are surface-like, curve-like, or volume-like. This preserves the ability to model fences, wires, leaves, hair-like structures, translucency, and view-dependent residuals.

## Stage 2: Build a local manifold graph

### Neighbor graph

Construct a kNN or radius graph over primitive centers. For each pair `(i, j)`, compute:

- Euclidean distance `||mu_i - mu_j||`.
- Normal compatibility `|n_i dot n_j|`.
- Tangent projection residual:

```math
d^\perp_{ij} = |n_i^T(\mu_j - \mu_i)|.
```

- Appearance compatibility.
- Multi-view co-visibility.

### Edge score

For surface-like primitives:

```math
s_{ij}
= w_d \frac{\|\mu_i-\mu_j\|^2}{\sigma_i^2}
+ w_n (1-|n_i^T n_j|)
+ w_p \frac{(d^\perp_{ij})^2}{h_i^2 + h_j^2}
+ w_v (1-\operatorname{covis}_{ij}).
```

Keep edges below a threshold and connected to enough nearby views.

### Local manifold validity

For each primitive, its accepted neighbors should form a local 2D neighborhood:

- local PCA of neighbor centers should have rank 2;
- neighbor graph should not split into many disconnected fans;
- signed distances to the tangent plane should be small;
- no strong evidence of two opposite normal layers in the same small cell.

### Why this matters

2DGS gives local disks but not global connectivity. This stage creates the missing object:

```text
splats -> local adjacency -> manifold patch graph
```

## Stage 3: Manifold-conservative losses

### 3.1 Photometric rendering loss

Use the existing differentiable splatting renderer:

```math
L_{rgb} = \sum_{p} \rho(I_p - \hat I_p).
```

This remains necessary but should no longer be the only source of geometry.

### 3.2 Depth and normal consistency

If depth/normal priors are available:

```math
L_d = \sum_p \rho(D_p - \hat D_p)
```

```math
L_n = \sum_p (1 - \langle N_p, \hat N_p\rangle).
```

Use confidence weighting, because monocular priors can be wrong.

### 3.3 Area-measure conservation

Interpret each surface-like Gaussian as carrying local area measure:

```math
m_i = \alpha_i \sqrt{\lambda_{i1}\lambda_{i2}}.
```

Regularize sudden changes in local area density:

```math
L_{area} =
\sum_{(i,j)\in E}
\rho\left(
\log m_i - \log m_j
\right).
```

Also penalize excessive normal thickness:

```math
L_{thin}
= \sum_i
\max\left(0,
\frac{\lambda_{i3}}{\lambda_{i1}+\lambda_{i2}}-\tau_h
\right)^2.
```

This prevents a surface primitive from becoming a volumetric opacity blob to explain sparse-view ambiguity.

### 3.4 Normal-field integrability

If a set of normals comes from a real surface, it should be locally integrable. In a local tangent chart, approximate:

```math
\partial_v n_u - \partial_u n_v \approx 0.
```

Implementation-friendly version:

1. Fit a local height function over tangent coordinates around each primitive.
2. Compare the predicted normal from the fitted height field with Gaussian normals.
3. Penalize inconsistency.

Simpler graph version:

```math
L_{int}
= \sum_{i}\sum_{j,k\in N(i)}
\rho\left(
((n_j-n_i)^T t_{ik}) -
((n_k-n_i)^T t_{ij})
\right).
```

This does not have to be perfect differential geometry at first. The key is to penalize arbitrary non-integrable normal fields that look plausible only from training views.

### 3.5 Curvature-scale consistency

Large splats should only exist on low-curvature regions. Estimate local curvature by normal variation:

```math
\kappa_{ij} =
\frac{\arccos(|n_i^T n_j|)}{\|\mu_i-\mu_j\|+\epsilon}.
```

Let tangent radius be:

```math
r_i = \sqrt{\lambda_{i1}+\lambda_{i2}}.
```

Regularize:

```math
L_{curv}
= \sum_{(i,j)\in E}
\max(0, \kappa_{ij} r_i - \tau_\kappa)^2.
```

Interpretation:

- flat wall: large splats allowed;
- edge/corner/high curvature: splats must shrink or split.

This gives a principled octree/refinement signal.

### 3.6 Local 2D manifold loss

For each neighborhood, compute covariance of neighbor centers:

```math
C_i = \sum_{j\in N(i)} w_{ij}(\mu_j-\bar\mu_i)(\mu_j-\bar\mu_i)^T.
```

Let eigenvalues be `eta1 >= eta2 >= eta3`. A surface neighborhood should satisfy:

```math
\eta_1, \eta_2 \gg \eta_3.
```

Loss:

```math
L_{rank2}
= \sum_i
\frac{\eta_{i3}}{\eta_{i1}+\eta_{i2}+\epsilon}.
```

This catches local volumetric clusters and duplicate layers.

### 3.7 Free-space and visibility consistency

For each camera ray with observed surface depth, penalize opaque primitives in free space before the surface:

```math
L_{free}
= \sum_{\text{ray } r}
\sum_{i \in \text{before depth}}
\alpha_i T_i.
```

This is essential in sparse view. Without it, RGB loss may keep floaters.

## Stage 4: Topology-aware projection

### Objective

Project the current splat set onto a manifold-like patch graph.

This is not full mesh extraction yet. It is a graph/charts step that says which splats belong together and whether they form a locally valid surface.

### Patch extraction

1. Start from surface-like primitives.
2. Build graph using the compatibility score from Stage 2.
3. Extract connected components.
4. Split components at:
   - high normal discontinuity;
   - high curvature;
   - low co-visibility;
   - conflicting signed distances;
   - suspected duplicate layers.

### Patch chart fitting

For each patch:

- choose local frame by PCA or robust normal averaging;
- project centers to 2D chart coordinates;
- fit a local height field or local triangulation;
- detect boundaries where neighborhood support is one-sided;
- mark singular/high-curvature nodes.

### Optional varifold projection

Represent the Gaussian set as an oriented or unoriented surface measure:

```math
V_G = \sum_i a_i A_i \delta_{(\mu_i, n_i)}.
```

Compare it to a candidate mesh/surface measure:

```math
V_M = \int_M \delta_{(x,n(x))} dA.
```

Use a kernel varifold distance:

```math
D(V_G,V_M)
= \|V_G - V_M\|_K^2.
```

This is a mathematically clean way to state "the Gaussians should correspond to a manifold surface." It can be introduced as a theoretical view first, then approximated by graph losses in implementation.

## Stage 5: Adaptive refinement with octree

### Role of octree

The octree should not be just a rendering LOD structure. It should be a geometry-refinement hierarchy.

Each cell stores:

- primitives inside the cell;
- photometric residual statistics;
- depth/normal residual statistics;
- local rank-2 score;
- curvature-scale score;
- free-space violation;
- topology uncertainty.

### Split criteria

Split a cell if:

- high photometric residual but geometry is stable: add appearance capacity;
- high normal/depth residual: refine surface geometry;
- high curvature-scale violation: split large splats;
- mixed dimensionality: separate curve/surface/volume primitives;
- topology ambiguity: postpone merge and keep multiple hypotheses.

### Merge/prune criteria

Prune or merge if:

- opacity is low and free-space violation is high;
- primitive is redundant with same patch and same normal;
- primitive creates duplicate layer unsupported by views;
- volume-like primitive persists in a region with strong surface evidence.

### Refinement actions

- Surface region: split into smaller thin Gaussians aligned to local tangent frame.
- Curve region: split along principal curve direction.
- Volume region: keep only if appearance improves validation views and free-space does not reject it.

## Stage 6: Hybrid primitive optimization

### Surface primitive

Thin Gaussian:

```math
\lambda_1,\lambda_2 \gg \lambda_3.
```

Parameters:

- center constrained to local chart or manifold patch;
- normal from patch or covariance;
- tangent axes regularized by local frame field;
- opacity linked to area measure.

### Curve primitive

Line-like Gaussian:

```math
\lambda_1 \gg \lambda_2,\lambda_3.
```

Useful for:

- wires;
- rails;
- plant stems;
- mesh-like thin structures;
- silhouettes.

Regularization:

- centerline smoothness;
- tangent direction consistency;
- avoid becoming a surface sheet.

### Volume primitive

Near-isotropic or full 3D Gaussian:

```math
\lambda_1 \approx \lambda_2 \approx \lambda_3.
```

Only allowed when:

- depth/normal priors are uncertain;
- appearance is genuinely fuzzy/translucent;
- validation rendering improves;
- no free-space conflict.

Volume budget should be small.

## Stage 7: Editable asset extraction

### Surface backbone

From manifold patches:

- triangulate patch centers in chart coordinates;
- optionally simplify;
- preserve boundaries and high-curvature feature lines;
- export mesh.

### Appearance layer

Attach splats to surface:

```math
\mu_i = \phi_p(u_i, v_i) + \epsilon_i n_i.
```

For strict surface asset mode, set `epsilon_i = 0` or very small.

Store:

- chart id;
- barycentric coordinate or `(u,v)`;
- tangent covariance;
- color/SH/material residual.

### Engine/DCC value

The output can support:

- collision proxy from mesh;
- editing geometry separately from appearance;
- LOD through patch hierarchy;
- baking splat appearance to texture;
- preserving residual splats for view-dependent detail.

## Minimal implementation plan

### Phase 1: Geometry diagnostics on existing 3DGS

No renderer changes needed.

Implement:

- covariance eigenspectrum analysis;
- primitive dimensionality labels;
- local rank-2 score;
- normal consistency;
- free-space violation if depth available.

Expected result:

- visualizations showing which Gaussians are surface/curve/volume;
- evidence that floaters and bad regions violate manifold scores.

### Phase 2: Manifold regularization during training

Add:

- thinness loss;
- area-measure smoothness;
- normal consistency;
- curvature-scale loss;
- local rank-2 loss.

Compare:

- 3DGS baseline;
- 2DGS if available;
- proposed regularized thin 3DGS.

### Phase 3: Patch graph and projection

Add:

- neighbor graph;
- patch extraction;
- duplicate-layer rejection;
- chart fitting;
- optional surface mesh extraction.

This is the first phase that clearly differs from 2DGS.

### Phase 4: Hybrid dimensional primitives

Add:

- curve primitive mode;
- volume primitive budget;
- classification annealing;
- octree split/merge.

### Phase 5: Asset export

Add:

- mesh/charts;
- attached splats;
- texture baking prototype;
- Unreal/Blender import experiment if desired.

## Evaluation

### Rendering metrics

- PSNR.
- SSIM.
- LPIPS.

These are necessary but not sufficient.

### Geometry metrics

- Chamfer distance against ground-truth mesh or DTU scan.
- Normal consistency.
- Depth error on held-out views.
- F-score at standard thresholds.
- Number of floaters.
- Duplicate-layer score.
- Local rank-2 violation score.

### Asset metrics

- mesh watertightness or patch manifoldness;
- number of non-manifold edges;
- collision usability;
- triangle count vs visual quality;
- texture/splat baking quality;
- edit stability under moving vertices or removing objects.

### Ablations

- no area conservation;
- no integrability;
- no rank-2 neighborhood loss;
- no free-space;
- no curve/volume primitives;
- no octree refinement.

## Main technical risks

1. **Over-constraining hurts view synthesis**
   - Mitigation: keep residual volume primitives and anneal geometry losses.

2. **Monocular depth/normal priors are wrong**
   - Mitigation: confidence weighting and multi-view consistency checks.

3. **Patch graph mistakes cause wrong topology**
   - Mitigation: delay hard decisions; keep patch graph soft during training.

4. **Varifold formulation may be too heavy**
   - Mitigation: use it as theory; implement graph approximations first.

5. **Asset export may look worse than splat rendering**
   - Mitigation: export hybrid assets, not pure mesh-only assets.

