# Novelty and Feasibility Audit

## Claim to avoid

Do not claim:

> We combine octree, triangle splatting, and mesh splatting.

This is too easy to position as an engineering mixture. It is close to existing work on 2DGS, SuGaR, mesh splatting, and octree Gaussian representations.

## Stronger claim

Claim:

> We reinterpret Gaussian splats as samples of a manifold-induced geometric measure and derive optimization constraints that make the splat set globally geometry-consistent, not merely locally surface-like.

This creates a cleaner distinction:

- 2DGS: local disk primitive.
- SuGaR: surface-aligned Gaussians and mesh extraction/refinement.
- SolidGS/GausSurf/MeshSplat: stronger priors and better surface reconstruction.
- Ours: manifold-conservative projection of a Gaussian mixture using area, normal, integrability, curvature-scale, and topology consistency.

## Relation to 2DGS

2DGS already observes that 3D ellipsoids are geometrically inconsistent and replaces them with 2D oriented disks.

The proposed project should not compete by saying "our Gaussians are also thin."

The distinction should be:

```text
2DGS:
  each primitive is locally surface-like.

Ours:
  the whole primitive set should be close to a valid manifold-induced measure.
```

Concrete differences:

- explicit local adjacency graph;
- manifold neighborhood validity;
- area-measure conservation;
- normal-field integrability;
- curvature-scale consistency;
- duplicate-layer and free-space rejection;
- hybrid dimensionality rather than all-primitives-are-2D;
- asset extraction as an optimization target, not an afterthought.

## Relation to mesh-splatting work

MeshSplat, SuGaR, MaGS-like methods already use mesh/surface structure with splats.

The project must avoid being "mesh with splats attached."

Better positioning:

> We start from Gaussian covariance geometry and ask which Gaussian mixtures admit a manifold interpretation. The mesh or patch graph is the result of a projection, not just a template to attach Gaussians.

## Relation to octree work

Octree-GS and similar work use hierarchical structure for rendering efficiency, LOD, or scale handling.

The project's octree should be tied to geometry:

- split when curvature-scale consistency is violated;
- split when local manifold rank is unclear;
- split when dimensionality is mixed;
- prune when free-space or duplicate-layer evidence rejects a primitive;
- merge when area/normal/appearance are redundant.

This makes the octree a geometry-refinement mechanism.

## Mathematical anchor: surface measure

A smooth surface `M` induces an area measure:

```math
\mu_M(B) = \int_{M \cap B} dA.
```

A splat set can be interpreted as a discrete approximation:

```math
\mu_G = \sum_i a_i A_i \delta_{\mu_i},
\quad
A_i \propto \sqrt{\lambda_{i1}\lambda_{i2}}.
```

With normals:

```math
V_G = \sum_i a_i A_i \delta_{(\mu_i,n_i)}.
```

This resembles a discrete varifold. A valid surface representation should be close to some manifold-induced varifold:

```math
V_M = \int_M \delta_{(x,n(x))}dA.
```

This gives a principled answer to the question:

> What is the mapping from 3DGS to a manifold?

Answer:

> covariance eigensystem -> tangent-normal decomposition -> weighted surface measure -> varifold/manifold projection.

## Candidate conservation laws

### 1. Area-measure conservation

Opacity and covariance footprint should not vary arbitrarily. For surface-like primitives:

```math
m_i = \alpha_i \sqrt{\lambda_{i1}\lambda_{i2}}.
```

Neighboring primitives on the same patch should have smooth local area density unless there is a boundary, occlusion, or material discontinuity.

### 2. Normal integrability

Normals should locally correspond to a surface height field or implicit function:

```math
n(x) \approx \frac{\nabla f(x)}{\|\nabla f(x)\|}.
```

Non-integrable normal fields indicate that splats are fitting images without forming a coherent surface.

### 3. Curvature-scale consistency

Primitive size must match local curvature. A large splat cannot represent a high-curvature region without geometric error.

```math
\kappa_i r_i < \tau.
```

### 4. Local dimensionality conservation

Surface regions should remain rank-2, curve regions rank-1, and volume regions rank-3. A region should not freely change dimension just to satisfy RGB loss.

### 5. Free-space conservation

Observed empty space along camera rays should remain empty. This is crucial for sparse-view reconstruction.

## Why this can produce assets

Current GS is hard to use as an asset because:

- no stable topology;
- no collision surface;
- no UV/chart structure;
- hard to edit semantically;
- floaters and duplicate layers are common;
- geometry and appearance are entangled.

The proposed representation can produce:

- a patch graph or mesh backbone;
- splats attached to charts;
- curve primitives for wires/thin structures;
- sparse volume primitives for residual effects;
- a hierarchy for LOD and editing.

This is useful for:

- Blender import;
- game-engine static assets;
- collision and navigation proxies;
- texture baking;
- object deletion or local geometry edits.

## Feasibility by stages

### Stage A: Low risk

Analyze existing 3DGS checkpoints.

Deliverables:

- eigenvalue-ratio histograms;
- surface/curve/volume classification visualization;
- local rank-2 violation heatmap;
- floater detection using free-space;
- correlation between bad geometry and conservation-law violations.

Why feasible:

- requires only reading Gaussian parameters and camera/depth data;
- no renderer modification.

### Stage B: Medium risk

Add differentiable manifold-conservative losses.

Deliverables:

- regularized 3DGS training;
- sparse-view comparison against baseline 3DGS and 2DGS;
- geometry metrics and held-out rendering metrics.

Why feasible:

- losses are graph/local-neighborhood operations;
- can be implemented in PyTorch;
- does not require full topology solving.

### Stage C: Medium-high risk

Build patch graph and topology-aware projection.

Deliverables:

- connected surface patches;
- duplicate-layer splitting;
- chart fitting;
- preliminary mesh extraction;
- patch-attached splats.

Why feasible:

- graph construction can start as non-differentiable periodic update;
- chart fitting can be local and robust rather than globally optimal.

### Stage D: High risk

Full hybrid dimensional primitives and octree refinement.

Deliverables:

- surface/curve/volume primitive modes;
- split/merge schedule;
- adaptive octree refinement;
- asset export.

Why risky:

- discrete mode switches can destabilize optimization;
- refinement affects renderer performance;
- evaluation must show more than visual quality.

## Minimal paper path

### Paper v1: Geometry diagnostics + regularization

Main result:

> Manifold-conservative losses reduce floaters, duplicate layers, and geometry errors in sparse-view GS.

This is the fastest route to a workshop or early paper.

### Paper v2: Patch graph + asset output

Main result:

> Gaussian splats can be projected to manifold patches and exported as editable hybrid assets.

This is stronger and more differentiated.

### Paper v3: Theory-heavy varifold splatting

Main result:

> A Gaussian splat scene admits a principled interpretation as a discrete varifold, enabling manifold projection.

This is most novel mathematically, but requires careful writing and experiments.

## Recommended immediate experiments

1. Train or obtain a few 3DGS scenes under sparse-view settings.
2. Compute eigenvalue spectra and classify splats into surface/curve/volume.
3. Visualize smallest-eigenvector normals.
4. Build kNN graph and compute:
   - rank-2 neighborhood score;
   - normal compatibility;
   - area density variation;
   - curvature-scale violation.
5. Compare bad visual regions with high violation scores.
6. Add `L_thin + L_area + L_curv + L_rank2`.
7. Evaluate on DTU/Tanks/LLFF or any available scene with approximate geometry.

## Go/no-go criteria

Continue if:

- conservation scores correlate with geometry artifacts;
- regularization improves geometry without destroying rendering;
- patch extraction produces coherent surface components;
- asset extraction is visibly more usable than pure GS.

Pivot if:

- all gains can be matched by 2DGS plus depth/normal loss;
- graph losses are too unstable;
- extracted surfaces are not meaningfully editable;
- hybrid primitives complicate training without measurable benefit.

