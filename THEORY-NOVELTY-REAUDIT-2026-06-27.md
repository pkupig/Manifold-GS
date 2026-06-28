# Theory and Novelty Re-audit, 2026-06-27

## Executive verdict

The broad claim "we are the first to introduce manifold/topology/varifold ideas
into Gaussian Splatting" is false as of June 2026.

The project still has a plausible gap, but only after narrowing the claim to:

> A refinement-conservative geometric measure for radiance splats, with
> explicit mass independent of opacity and compatibility-certified projection
> from Gaussian tangent/curvature fields to editable surface patches.

This exact combination was not found in the audited literature. That is a
plausible novelty statement, not proof of worldwide priority.

## 1. Theory correctness

### Correct and usable

1. `V_G = sum_i q_i delta_(mu_i,P_i)` is a valid finite varifold whenever
   `q_i >= 0` and `P_i` belongs to the Grassmannian. Its support need not
   already be a manifold.
2. The forward bounded-Lipschitz estimate in `THEORY-PROOF-SKETCH.md` follows
   from the stated cell partition, quadrature, support, and tangent errors. It
   is correct but elementary, not a deep new theorem by itself.
3. Conservative split exactly preserves mass and first moment when child
   masses sum correctly and offsets have zero mass-weighted mean. It also
   preserves tangent-projector moment when children inherit the parent plane.
4. Rejecting an unconditional 3DGS-manifold bijection is correct: rendering and
   the Gaussian-to-varifold map are both non-identifiable.

### Correct only after qualification

1. `q_i = pi r_k^2/k` is an area estimator under locally homogeneous 2D
   sampling, not an automatic proof of cell quadrature under boundaries,
   varying density, noise, or duplicate layers.
2. General merge preserves mass and first moment, but preserves the complete
   tangent-projector moment only for agreeing child tangent planes.
3. The old `(n_i dot n_j)^2` tangent kernel matched only a finite projector
   moment. It has been replaced by an RBF on tangent projectors. Kernel
   discrepancy still does not determine topology.
4. Quadratic MLS empirically improves analytic geometry, but no convergence
   rate has been proved for our exact weighted estimator.

### Not proved

- training converges to a rectifiable or integral varifold;
- the patch graph is homeomorphic to the latent surface;
- sparse images uniquely identify that surface;
- low kernel-varifold discrepancy implies correct topology;
- current losses enforce all assumptions of an Allard/NSW-style theorem.

## 2. Direct novelty conflicts

| Work | What it already covers | Consequence |
|---|---|---|
| 2DGS | intrinsic planar surface splats | thin/rank-2 primitives are not new |
| SuGaR and mesh GS | surface alignment, mesh extraction, editable/deformable assets | GS-to-mesh is not new |
| Topology-GS (2024) | persistent homology, topology-guided interpolation/loss | "first topology-aware GS" is unavailable |
| GeoSplat (2025/26) | manifold/varifold estimation, curvature, rank-2 regularization, tangent/curvature-aware densification, sparse-view gains | most broad geometry framing overlaps |
| TagSplat (2025) | Gaussian connectivity and topology-preserving ADC for dynamic meshes | topology graph plus ADC is not new broadly |
| Topo-GS (2026) | covariance-to-tangent alignment and distinct 1D/2D constraints | mixed dimension/tangent alignment are not new broadly |
| Geometry-Grounded GS (2026) | rigorous Gaussian-rendering equivalence to stochastic solids | "first rigorous geometric interpretation" is unavailable |
| Flagfolds | covariance/PCA as weighted nested subspaces across dimensions | covariance-to-stratified geometry has mathematical prior art |

## 3. Plausibly distinct remainder

### A. Explicit additive mass independent of opacity

No audited GS work was found whose central representation carries a separate
surface quadrature mass `q_i`, proves its ADC split/merge behavior, and evaluates
refinement-invariant geometric measure convergence.

Risk: GeoSplat discusses area invariance and discrete varifold geometry. We must
compare exact formulas before claiming this distinction.

### B. Fundamental-form compatibility

GeoSplat estimates tangents and principal curvatures. A more specific question
is whether independently learned support, tangent, and curvature fields are
jointly realizable by one surface. For a chart `phi`, estimate

```math
I = J_phi^T J_phi,
II_ab = - <partial_a n, partial_b phi>.
```

Then penalize discrete Gauss-Codazzi residuals:

```math
K(I) - det(I^{-1}II) = 0,
nabla_a II_bc - nabla_b II_ac = 0.
```

This is a stronger replacement for informal "normal integrability". No GS
paper using this exact compatibility formulation was found in targeted search.

Risk: if both forms are computed from the same explicit embedding, the
identities become nearly tautological. The useful residual must compare
independently predicted covariance/curvature geometry against chart geometry.

### C. Certified reject option for asset projection

Report where reach/separation, sampling, or tangent confidence is insufficient,
and export open patches plus residual curve/volume splats rather than inventing
watertight topology. This needs calibrated coverage guarantees to be more than
a system contribution.

### D. Stratified radiance measure

Surface, curve, and volume measures can have dimension-specific masses and
refinement laws. This is distinct in scene reconstruction but adjacent to
flagfolds and Topo-GS. Keep it as an extension unless thin structures become the
central benchmark.

## 4. Revised oral-level thesis

> Existing geometry-aware splatting estimates local normals and curvature, but
> does not guarantee that adaptive splats represent a refinement-invariant
> geometric measure or that independently learned support and curvature fields
> are realizable by one surface. We introduce an explicit conserved quadrature
> measure and compatibility-aware proximal projection that produces
> confidence-certified editable patches.

Required contributions:

1. explicit `q_i` with exact ADC conservation and a nontrivial stability result;
2. discrete Gauss-Codazzi compatibility between charts and Gaussian geometry;
3. proximal optimization preserving rendering while reducing geometry errors;
4. asset extraction with calibrated reject regions;
5. direct comparison with GeoSplat, Geometry-Grounded GS, 2DGS, and SuGaR.

Without item 2 or an equally deep replacement, the project is likely an
incremental combination of varifold estimation, MLS projection, and GS meshing.
Mass conservation alone is probably insufficient for an oral paper.

## 5. Claims allowed today

- "We formulate a refinement-conservative discrete geometric measure for GS."
- "We prove a conditional forward consistency bound."
- "Our prototype exactly preserves geometric mass through clone/split."
- "Our offline projection improves analytic geometry and exports reloadable
  surface-aligned Gaussians and open patches."

## 6. Claims not allowed today

- "Manifold concepts have not previously been used in GS."
- "We are the first topology-aware or varifold-based GS method."
- "3DGS is equivalent to a manifold."
- "Our training converges to a manifold."
- "Our topology is guaranteed correct."
- "The current method is already an oral-level novelty."

## 7. Primary references checked

- GeoSplat: https://arxiv.org/abs/2509.05075
- Geometry-Grounded GS: https://arxiv.org/abs/2601.17835
- Topology-GS: https://arxiv.org/abs/2412.16619
- TagSplat: https://arxiv.org/abs/2512.01329
- Topo-GS: https://arxiv.org/abs/2605.17011
- 2DGS: https://arxiv.org/abs/2403.17888
- Point-cloud varifold approximation: https://arxiv.org/abs/1609.03625
- Neural Varifolds: https://arxiv.org/abs/2407.04844

