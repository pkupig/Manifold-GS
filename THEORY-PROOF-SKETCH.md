# ManifoldGS Theory: Proof Targets and Non-Claims

## 1. There is no unconditional equivalence

Let a radiance Gaussian be `(mu, Sigma, alpha, c)`. The map that retains only
its center, tangent eigenspace, and geometric mass,

```math
F(G)=V_G=\sum_i q_i\delta_{(\mu_i,P_i)},
```

is not injective: it discards color, opacity, normal sign, normal thickness,
and part of the covariance spectrum. It is not surjective onto smooth
manifolds either, because a finite atomic measure is not a smooth surface
measure. Moreover, image formation is many-to-one. Therefore the paper must
not claim an identity or bijection between arbitrary 3DGS and manifolds.

The defensible mathematical object is an **approximate projection/retraction**:

```text
radiance Gaussians --F--> discrete varifold --R_h--> reconstructed patches
```

under explicit sampling and regularity assumptions.

## 2. Forward consistency theorem

Let `M` be a compact `C^2` embedded surface with area measure `A` and tangent
projector `P(x)`. Let `{C_i}` be a measurable partition of `M`. For each cell,
choose a Gaussian center `mu_i`, tangent projector `P_i`, and mass `q_i`.
Assume

```math
sup_{x in C_i} ||x-mu_i|| <= h,
sup_{x in C_i} ||P(x)-P_i||_F <= epsilon_T,
sum_i |q_i-A(C_i)| <= epsilon_q.
```

Define

```math
V_M(phi)=int_M phi(x,P(x)) dA(x),
V_G(phi)=sum_i q_i phi(mu_i,P_i).
```

For every test function with `||phi||_infinity <= 1` and Lipschitz constant at
most one under

```math
d((x,P),(y,Q))=||x-y||+||P-Q||_F,
```

we have

```math
|V_G(phi)-V_M(phi)|
<= A(M)(h+epsilon_T)+epsilon_q.
```

### Proof

Insert the cell-area quadrature and use the triangle inequality:

```math
|sum_i q_i phi(mu_i,P_i)-sum_i int_{C_i} phi(x,P(x))dA|
<= sum_i |q_i-A(C_i)| ||phi||_infinity
 + sum_i int_{C_i}|phi(mu_i,P_i)-phi(x,P(x))|dA.
```

The first term is at most `epsilon_q`; the second is at most
`A(M)(h+epsilon_T)`. Taking the supremum proves the bounded-Lipschitz bound.

This elementary theorem is already enough to justify the covariance-to-measure
map. The hard research question is making training produce its assumptions.

## 3. Gaussian mollification corollary

If each atom is spatially mollified by a centered Gaussian with expected
displacement at most `rho_i`, coupling the atom and its mollification gives

```math
d_BL(V_G, V_G convolved K) <= sum_i q_i rho_i.
```

Thus tangent radius and normal thickness must vanish with sampling scale for a
continuous Gaussian density to converge to the surface measure. Merely making
the smallest eigenvalue vanish is insufficient if tangent radii stay large
across high curvature.

## 4. Exact refinement theorem

Replace parent atom `(q,mu,P)` by children `(q_j,mu_j,P_j)`. If

```math
sum_j q_j=q,
sum_j q_j mu_j=q mu,
```

then zeroth and first spatial moments are exactly preserved. If additionally

```math
sum_j q_j P_j=qP,
```

the tangent-projector moment is preserved. For a 1-Lipschitz test function,

```math
|sum_j q_j phi(mu_j,P_j)-q phi(mu,P)|
<= sum_j q_j (||mu_j-mu||+||P_j-P||_F).
```

The implemented conservative split enforces all three equalities when children
inherit the parent tangent plane. Its recentered child offsets make first-moment
conservation deterministic rather than true only in expectation.

The current merge operator preserves mass and first moment in general. It
preserves the tangent-projector moment only when merged child tangent planes
agree (including the exact inverse of our split). A general varying-tangent
merge must either retain a higher-rank tangent mixture or accept a quantified
projector-moment residual.

Opacity pruning is not a conservative split or merge. The implemented
representation-pruning option transports each removed mass to its nearest
retained sample. This preserves total mass exactly and gives the first-moment
bound

```math
\left\|\Delta \sum_i q_i\mu_i\right\|
\leq \sum_{i\in removed} q_i\|\mu_i-\mu_{a(i)}\|.
```

It does not preserve the first moment exactly. Exact local zeroth/first-moment
pruning requires nonnegative barycentric transport to retained neighborhoods;
that operator remains future work.

Conservation alone also does not make `q_i` a consistent quadrature after many
topology changes. The certified asset layer therefore applies

```math
q_i'=(1-eta)q_i+eta A_i\frac{\sum_{j\in C}q_j}{\sum_{j\in C}A_j},
\qquad i\in C,
```

where `C` is the certified layer and `A_i` is a winsorized squared kNN radius.
This exactly preserves certified and residual zeroth masses separately. The
winsorization prevents isolated samples from claiming unbounded surface area.
It is not yet a proof of asymptotically optimal quadrature.

## 5. A conditional reverse theorem

Weak varifold convergence alone does not imply that every finite `V_G` is near
a unique manifold, and kernel MMD alone does not preserve topology. A reverse
statement needs assumptions such as:

- support lies in a shrinking tubular neighborhood of a compact surface;
- the target has reach at least `tau > 0`;
- sampling fill distance `h << tau` and no arbitrarily large holes;
- local masses have upper/lower 2-density bounds;
- tangent errors and regularized first variation are bounded;
- duplicate sheets are separated or rejected by visibility evidence.

Under these assumptions, established varifold compactness/rectifiability and
manifold reconstruction results can be used in two separate steps:

1. discrete measures converge to a rectifiable varifold;
2. positive reach plus sufficiently dense sampling permits recovery of homology
   by an appropriate offset/complex.

This is not a single new theorem and should not be presented as one. Our useful
new theorem can instead connect measurable Gaussian quantities to sufficient
conditions for an existing reconstruction theorem.

### 5.1 Quantitative compatibility-to-GT target

The missing link can be isolated as two assumptions rather than hidden inside
an unconditional reconstruction claim. Let `Z` denote the compatible
surface-varifold class and let `D_h(V_G)` collect shape, symmetry, Gauss, and
Codazzi defects. Target an estimator-specific error bound

```math
dist(V_G,Z) <= C_B(D_h(V_G)+epsilon_h),
```

where `epsilon_h` explicitly contains graph fill distance, MLS bias, normal
noise, and mass-quadrature error. Separately, require the restricted rendering
operator to have a stable inverse on `Z`:

```math
d_V(U,V_*) <= C_R d_image(R(U),R(V_*)),  U in Z.
```

If rendering is locally `L_R`-Lipschitz, then for a nearest compatible field
`U=Pi_Z(V_G)` the triangle inequality yields

```math
d_V(V_G,V_*)
<= C_R d_image(R(V_G),R(V_*))
 + (1+C_R L_R)C_B(D_h(V_G)+epsilon_h).
```

This decomposition is the defensible theoretical explanation for combining a
data term with compatibility. Classical Bonnet only identifies the zero set
`Z`; it does not provide `C_B`. Sparse-view image formation is not stably
injective without visibility/scene assumptions, so it does not automatically
provide `C_R` either. Proving or empirically certifying these two constants on
a restricted analytic scene class is now the central reverse-theory target.

The bound is absolute. It cannot imply a registered relative gain such as 20%
without both an upper bound for the candidate and a nonzero lower bound for the
baseline. Percentage PASS thresholds therefore remain experimental criteria,
not theorem conclusions.

`THEORY-STABILITY.md` proves the corresponding linearized graph estimate and
lists the residuals that must be measured separately.

## 6. Projection theorem target

Let noisy centers be within `delta` of `M`, with fill distance `h`, reach
`tau`, and bounded sampling ratio. A local weighted PCA/MLS projection with
radius `r` should target a bound of the form

```math
dist_H(M_h,M) <= C(delta + r^2/tau + h^2/r),
angle(P_h,P_M) <= C(delta/r + r/tau + h/r).
```

The exact constants and exponents depend on the selected MLS estimator and
sampling model. These expressions are currently design targets, not proven
claims. We should either derive them for our estimator or cite a theorem whose
algorithm and assumptions we implement exactly.

## 7. What each loss must establish

| Needed theorem assumption | Observable/operation |
|---|---|
| small fill distance | coverage loss and completeness |
| correct area quadrature | explicit `q_i`, partition-of-unity calibration |
| tangent convergence | local PCA/covariance agreement |
| bounded first variation | normal/curvature regularization weighted by `q_i` |
| tubular support | MLS projection, depth/free-space evidence |
| positive separation | duplicate-sheet and visibility test |
| stable refinement | conservative clone/split/merge |

This table is the bridge from the proof to the pipeline. A loss that does not
control one of these quantities is not part of the core theoretical claim.

The implemented `q_i = pi r_k^2/k` is a kNN area estimator. It is approximately
calibrated for locally homogeneous random 2D samples, but it is not by itself a
proof of the cell-quadrature assumption above. Boundaries, nonuniform density,
noise, and duplicate samples require bias correction or an explicit
partition-of-unity construction.

The implemented kernel-varifold diagnostic uses Gaussian RBF kernels on both
position and tangent projectors. The earlier `(n_i dot n_j)^2` tangent kernel
matched only a finite projector moment and was not sufficient to identify an
arbitrary tangent distribution.

## 8. Counterexamples required in the paper

1. **Flat but misplaced sheet:** perfect thinness and normals, wrong support.
2. **Two close sheets:** weak appearance fit and local rank-2 can both pass,
   while topology is wrong.
3. **Sparse islands:** low accuracy but poor completeness; normalized varifold
   can hide missing total mass if weights are renormalized.
4. **Oscillating normals:** centers lie on a surface but tangent field has
   unbounded first variation.
5. **Opacity-area ambiguity:** identical rendering can redistribute alpha and
   footprint while changing `alpha*area`, showing why `q_i` must be separate.

## 9. Literature anchors

- Buet, Leonardi, Masnou, *A varifold approach to surface approximation*:
  discrete point-cloud varifolds, regularized first variation, and convergence:
  https://arxiv.org/abs/1609.03625
- Buet and Rumpf, *Mean curvature motion of point cloud varifolds*: local
  covariance tangent estimation and consistency under regular sampling:
  https://arxiv.org/abs/2010.09419
- Niyogi, Smale, Weinberger, *Finding the Homology of Submanifolds with High
  Confidence from Random Samples*: topology recovery under condition/reach and
  sampling assumptions: https://doi.org/10.1007/s00454-008-9053-2
- Levin, *The Approximation Power of Moving Least-Squares*: approximation order
  for quasi-uniform scattered samples:
  https://doi.org/10.1090/S0025-5718-98-00974-0

The next theory pass must match our estimator to exact theorem hypotheses and
constants; these papers support the framework but do not automatically prove
our training algorithm.
