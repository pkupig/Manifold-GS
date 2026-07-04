# Quantitative Stability: Compatibility, Realizability, and Ground Truth

This note isolates the theorem that the project can prove locally from the
inverse problem that must remain an explicit assumption. It is the quantitative
bridge missing from the qualitative Bonnet argument in `THEORY-BONNET.md`.

## 1. Local graph model

Let `Omega` be a bounded Lipschitz domain and represent a surface as a graph

```math
X_f(u)=(u_1,u_2,f(u)), \qquad f\in H^2(\Omega).
```

Write `m in H^1(Omega;R^2)` for the tangential part of an independently learned
normal field, using the sign convention for which a realizable graph satisfies
`m=grad f` to first order. Define

```math
r_0=m-\nabla f, \qquad r_1=\nabla m-\nabla^2 f=\nabla r_0.
```

`r_0` is the linearized tangent/value mismatch and `r_1` is the linearized
shape-operator mismatch. Let

```math
Z={ (g,\nabla g):g\in H^2(\Omega) }
```

be the set of locally realizable graph fields.

## 2. Exact linearized error bound

> **Proposition 1 (compatibility coercivity).** For the product norm
>
> ```math
> ||(a,b)||_X^2=||a||_{H^2}^2+||b||_{H^1}^2,
> ```
>
> the distance from `(f,m)` to `Z` satisfies
>
> ```math
> dist_X((f,m),Z) <= ||m-\nabla f||_{H^1}
>                  = (||r_0||_{L^2}^2+||r_1||_{L^2}^2)^{1/2}.
> ```

**Proof.** Choose the admissible comparison field `(f,grad f) in Z`. Its graph
component agrees exactly with `f`; the remaining product-norm difference is
`||m-grad f||_{H^1}`. Taking the infimum over `Z` proves the inequality. `square`

This elementary estimate is useful because it names the minimum coercive loss:
both a value/tangent term and a derivative/shape term are required. It also
matches the implementation better than referring to Gauss-Codazzi alone.

If only `r_1` is available, Poincare gives the weaker statement

```math
||r_0-(r_0)_Omega||_{H^1} <= C_Omega ||r_1||_{L^2}.
```

Thus shape match controls compatibility only modulo a constant normal-tilt
mode. A tangent anchor, boundary condition, or zero-mean gauge is necessary.

## 3. Why symmetry, Gauss, and Codazzi are not coercive alone

At a flat reference graph, symmetry controls the antisymmetric part of
`grad m` (the scalar curl of `m`). Its kernel contains every gradient field,
including gradients of arbitrarily wrong surfaces. Linearized Gauss has no
first-order coercivity at zero curvature because the determinant is quadratic.
Codazzi constrains derivatives of the second form but leaves constant compatible
curvature modes and all exactly realizable wrong graphs.

Concrete null modes are:

1. `m=grad g` for any wrong graph `g`: all exact compatibility identities hold;
2. `m=grad f+c`: derivative-only shape/symmetry/Codazzi terms miss the constant
   tilt `c`;
3. affine `f`: curvature-based terms vanish although position and tilt can both
   be wrong.
4. a coherently scale- or bias-distorted depth graph: it remains a valid smooth
   surface and can satisfy all compatibility identities while being displaced
   from GT;
5. a smooth low-frequency depth warp: compatibility may faithfully realize the
   wrong curved surface supplied by the biased data anchor.

Therefore Gauss-Codazzi terms are discriminants for non-integrability, not data
terms and not standalone distances to ground truth. In the local graph regime,
full shape match already contains the derivative information that symmetry and
Codazzi inspect selectively; those terms may improve conditioning or robustness,
but cannot replace the tangent/value anchor.

## 3.1 Nonlinear bounded-slope extension

The same coercivity survives beyond the first-order normal approximation on a
certified graph chart. For a slope `p in R^2`, define

```math
nu(p)=(-p,1)/sqrt(1+||p||^2).
```

Let `q=grad f` be the support slope and let `p(u)` be the slope encoded by the
independent covariance normal. Write `S_support(q,grad q)` for the graph shape
operator and `S_normal(q,p,grad p)` for the Weingarten operator obtained by
differentiating `nu(p)` along the support chart `X_f`.

> **Proposition 2 (nonlinear chart coercivity).** Assume almost everywhere
>
> ```math
> ||p||,||q|| <= M, \quad ||nabla q|| <= K,
> \quad nu(p) dot nu(q) >= gamma > 0.
> ```
>
> Then there is `C=C(M,K,gamma,Omega)` such that
>
> ```math
> dist_X((f,p),Z)
> <= C(
>   ||P(p)-P(q)||_{L^2}
>   +||S_normal(q,p,nabla p)-S_support(q,nabla q)||_{L^2}
> ).
> ```
>
> Here `P(p)=I-nu(p)nu(p)^T`; the positive-dot condition fixes the local
> unoriented-normal sign.

**Proof.** On the compact slope ball, `p -> nu(p)` is a smooth embedding
whose inverse is `p=-nu_T/nu_3`. The projector residual therefore controls
`||p-q||` after the sign is fixed. With `J_q=D X_f`, differentiation gives

```math
II_normal=-(D nu(p) nabla p)^T J_q.
```

For fixed `(p,q)`, this is a linear map of `nabla p`. At `p=q`,
`-D nu(q)^T J_q=I/sqrt(1+||q||^2)`. The angle condition and bounded slopes keep
this map uniformly invertible. Indeed, with `w_p=sqrt(1+||p||^2)`, direct
differentiation gives

```math
C(p,q):=-D nu(p)^T J_q
=I/w_p+p(q-p)^T/w_p^3,
```

and the matrix determinant lemma gives

```math
det C(p,q)=(1+p dot q)/w_p^4
          =(nu(p) dot nu(q)) w_q/w_p^3.
```

Hence `det C` is bounded away from zero by a constant depending only on
`M,gamma`; its operator norm is bounded on the same compact set, so its smallest
singular value is uniformly positive. Subtracting the support second form,
multiplying by the uniformly bounded graph metric, and using the Lipschitz
dependence of `C(p,q)` on `p` gives

```math
||nabla p-nabla q||
<= C_1||S_normal-S_support||+C_2 K||p-q||.
```

Combining the value and derivative estimates proves the claim by choosing the
comparison field `(f,grad f) in Z`. `square`

The assumptions are operational, not cosmetic. The estimate degenerates when
the chart becomes vertical, covariance and support normals approach orthogonal,
curvature is unbounded, or one kNN neighborhood mixes multiple sheets. Those
cases must be rejected or re-charted rather than assigned a large compatibility
weight.

## 4. Realizability-to-GT decomposition

Let `V(f,m,q)` be the discrete oriented measure represented by positions,
tangents, and geometric masses, and let `Pi_Z V` replace the independent normal
field by the graph normal while retaining the support and masses. For any metric
`d_V`, the triangle inequality gives the exact decomposition

```math
d_V(V,V_*) <= d_V(V,Pi_Z V) + d_V(Pi_Z V,V_*).
```

Under bounded slope, bounded mass, and a Lipschitz tangent kernel, Propositions 1
and 2 together with the forward quadrature estimate imply a bound of the form

```math
d_V(V,Pi_Z V)
<= C_T (||r_0||_{L^2}+||r_1||_{L^2}+epsilon_h+epsilon_q).
```

This is the part controlled by compatibility. The second term compares two
already realizable surfaces and cannot be bounded by compatibility: it requires
stable scene evidence.

### 4.1 Direct bound for the implemented kernel varifold

The evaluator uses the unit-mass kernel

```math
k((x,P),(y,Q))=
exp(-||x-y||^2/(2 sigma^2)-||P-Q||_F^2/(2 tau^2)),
```

where, for unoriented codimension-one planes,
`||P-Q||_F^2=2(1-(n dot m)^2)` and `tau` is `tangent_sigma` in the code.

> **Proposition 3 (coupling-to-kernel bound).** Let `mu` and `nu` be unit-mass
> discrete varifolds and let `pi` be any coupling between them. Their evaluator
> MMD satisfies
>
> ```math
> MMD_k(mu,nu)
> <= [int (||x-y||^2/sigma^2+||P-Q||_F^2/tau^2) d pi]^{1/2}.
> ```

**Proof.** Let `Phi` be the RKHS feature map of `k`. Coupling the two feature
means, Jensen's inequality gives

```math
||int(Phi(z)-Phi(z'))d pi||^2
<= int ||Phi(z)-Phi(z')||^2 d pi
= int 2(1-k(z,z'))d pi.
```

Apply `1-exp(-a/2)<=a/2` with
`a=||x-y||^2/sigma^2+||P-Q||_F^2/tau^2`. `square`

This proposition applies exactly to `normalized_kernel_varifold_distance`,
apart from deterministic evaluator subsampling. A cell-to-sample assignment,
conservative split/merge map, or pruning transport ledger supplies a concrete
coupling `pi`. It follows that the reported varifold error can be bounded by a
scaled RMS spatial transport plus RMS tangent transport. Mass conservation is
needed to retain a coupling without creating or deleting probability mass;
correct quadrature is still needed for the target marginal to approximate area.

The converse does not hold with a useful universal constant: a small
fixed-bandwidth MMD can hide fine geometric or topological errors. Proposition 3
therefore supplies a sufficient route to a low evaluator score, not an
identifiability theorem.

### 4.2 Visible depth gives a constructive identifiability bound

Consider one camera and a surface that is a single visible depth graph over an
image domain `Omega`. Write its world-space parameterization as

```math
X_z(xi)=o+z(xi)r(xi),
```

where `r(xi)` is the known camera ray. Assume bounded depth and slope and a
uniformly nondegenerate surface Jacobian. For two depth graphs `z,z_*`, direct
differentiation gives

```math
X_z-X_z*=(z-z_*)r,

D X_z-D X_z* = D(z-z_*) tensor r +(z-z_*)D r.
```

The tangent-projector map is Lipschitz on matrices whose two nonzero singular
values are bounded away from zero. Therefore:

> **Proposition 6 (depth-to-varifold stability).** Under the bounds above,
> there is a camera/chart constant `C_D` such that
>
> ```math
> ||X_z-X_z*||_{L^2}+||P_z-P_z*||_{L^2}
> <= C_D||z-z_*||_{H^1}.
> ```
>
> If corresponding image cells carry matching normalized masses, Proposition 3
> consequently gives
>
> ```math
> MMD_k(V_z,V_z*)
> <= C_D'(sigma,tau)||z-z_*||_{H^1}.
> ```

**Proof.** The position estimate follows from bounded rays. The displayed
Jacobian identity controls the `L2` difference of the two tangent bases by the
`H1` depth error. Uniform Jacobian nondegeneracy makes the map from a basis to
its orthogonal projector locally Lipschitz. Apply Proposition 3 to the coupling
between corresponding image cells. `square`

For unequal quadrature masses an additional mass-discrepancy term is required.
For occlusions or multiple ray intersections, the common graph coupling ceases
to exist; the image must be split into visibility-consistent charts.

The proposition explains why pointwise depth alone is not the exact theoretical
target: tangent/varifold control requires depth gradients as well. In practice a
multiscale depth loss, rendered-normal loss derived from depth, or an explicit
`H1` depth term supplies this information.

### 4.3 When RGB alone can be locally identifying

Let `F:z -> I` be the differentiable renderer restricted to a fixed visible
graph, with appearance either known or separately gauge-fixed. Assume `F` is
continuously differentiable near `z_*`, its derivative is locally Lipschitz with
constant `L`, and on the admissible deformation space

```math
||D F[z_*] e||_Y >= beta ||e||_{H^1}, \qquad beta>0.
```

> **Proposition 7 (local photometric coercivity).** If
> `||z-z_*||_{H^1}<=beta/L`, then
>
> ```math
> ||z-z_*||_{H^1} <= (2/beta)||F(z)-F(z_*)||_Y.
> ```

**Proof.** Taylor's theorem and derivative Lipschitzness give

```math
||F(z)-F(z_*)||_Y
>= beta||e||_{H^1}-(L/2)||e||_{H^1}^2
>= (beta/2)||e||_{H^1}.
```

Rearrange. `square`

In a finite discretization, `beta` is the smallest singular value of the
multi-view image Jacobian after removing appearance and rigid-motion gauges.
Textureless regions, trainable view-dependent color, occlusion changes, and
unseen surfaces make `beta` zero or extremely small. More views help only if
their stacked Jacobian removes those null directions. Low training RGB loss is
therefore not evidence that this assumption holds.

### 4.4 Visibility coverage is an unavoidable global term

Depth and photometric coercivity apply only on charts that are visible as the
first ray intersection. Let unit-mass varifolds decompose into matched visible
mass and residual mass,

```math
mu=(1-eta)mu_vis+eta mu_res,
nu=(1-eta)nu_vis+eta nu_res.
```

Because the evaluator kernel has `k(z,z)=1`, every kernel mean embedding has
norm at most one. The triangle inequality therefore gives:

> **Proposition 8 (visible-mass decomposition).**
>
> ```math
> MMD_k(mu,nu)
> <= (1-eta)MMD_k(mu_vis,nu_vis)+2 eta.
> ```

**Proof.** Expand the two kernel means using the displayed mixture and apply
the triangle inequality. The residual mean-embedding difference is at most
two. `square`

For unequal visible fractions, the unmatched-mass contribution is bounded by
the corresponding total-variation mismatch and the same unit feature norm.
Consequently, no visible-depth or RGB theorem can give a strong global bound
when camera-visible GT area is small.

A foreground mask certifies only that rays outside the silhouette should remain
empty. A depth map additionally certifies free space before the first hit. Both
still permit unsupported geometry behind the first hit; eliminating duplicate
back sheets requires another view, a scene prior, or an explicit residual-mass
penalty. Required diagnostics are therefore GT visible-area coverage on analytic
scenes and certified model-mass coverage per camera union, not view count alone.

Assume, on a restricted admissible scene and camera class, that image formation
is stably identifying on realizable fields:

```math
d_V(U,V_*) <= C_R d_image(R(U),R(V_*)), \qquad U\in Z.
```

If `R` is locally `L_R`-Lipschitz, the projection argument yields

```math
d_V(V,V_*)
<= C_R d_image(R(V),R(V_*))
 +(1+C_RL_R)C_T
   (||r_0||_{L^2}+||r_1||_{L^2}+epsilon_h+epsilon_q).
```

This is a conditional stability theorem, not an unconditional sparse-view
reconstruction theorem. Stable identifiability is false without restrictions:
unseen sheets, textureless regions, occlusions, and radiance/geometry ambiguity
produce different realizable surfaces with the same images.

## 5. Consequences for the method

The theorem separates four quantities that training and evaluation must not
collapse into one score:

| Bound term | Required mechanism | Observable |
|---|---|---|
| image/data residual | photometric plus visibility evidence | held-out image, mask, depth, free-space |
| `r_0` | tangent/value anchor | covariance-vs-support normal angle |
| `r_1` | shape match | support-vs-normal shape discrepancy |
| `epsilon_h+epsilon_q` | sampling and quadrature control | fill distance, chart coverage, mass error |

The current MLS confidence measures support planarity and sampling anisotropy,
but does not itself enforce the nonlinear theorem's angle margin
`|n_cov dot n_support|>=gamma`. That margin must be logged and used as a chart
certificate (or the chart must be reoriented/rejected) before interpreting a
small shape residual through Proposition 2.

Symmetry, Gauss, and Codazzi should be reported as mechanism diagnostics. They
become headline losses only if an ablation shows that they reduce a failure mode
left by the coercive `r_0+r_1` pair.

The current normalized kernel-varifold score is not directly optimized. A fixed
20% relative improvement cannot follow from the bound without numerical upper
constants for the candidate and a positive lower bound for the baseline. It is
an empirical success criterion, not a theorem threshold.

## 5.1 Discrete least-squares perturbation

The continuum residual is useful only if the cached kNN operators estimate it
consistently. The relevant certificate is the conditioning of the scaled local
design matrix, not neighbor count alone.

Let `u_j` lie in a chart ball of radius `r`, set `xi_j=u_j/r`, and let normalized
nonnegative weights form a scaled design matrix `A`. Assume

```math
lambda_min(A^T A) >= lambda > 0.
```

> **Proposition 4 (fixed-coordinate WLS error).** For a weighted linear fit to
> samples of `v in C^2`, with response perturbations bounded by `eta_v`,
>
> ```math
> ||D_hat v-Dv(0)||
> <= C(lambda,weights)(r||D^2v||_infinity+eta_v/r).
> ```
>
> For a weighted quadratic fit to `f in C^3`, with height perturbations bounded
> by `eta_f`,
>
> ```math
> ||H_hat f-D^2f(0)||
> <= C(lambda,weights)(r||D^3f||_infinity+eta_f/r^2).
> ```

**Proof.** Taylor-expand the responses in scaled coordinates. The omitted term
is `O(r^2||D^2v||)` for the linear model and `O(r^3||D^3f||)` for the quadratic
model. The weighted pseudoinverse has norm at most `lambda^{-1/2}`. Recovering a
first derivative divides the fitted scaled coefficient by `r`; recovering a
second derivative divides by `r^2`. The response-noise terms scale in the same
way, which gives the two inequalities. `square`

Applied to the implementation, the compatibility discretization error has the
schematic form

```math
epsilon_h
<= C[
 r||D^2 n|| + eta_n/r
 +r||D^3 f|| + eta_f/r^2
 +epsilon_frame+epsilon_mass
].
```

This result deliberately assumes fixed, correct chart coordinates. Gaussian
center noise also perturbs the kNN graph, frame, and design matrix
(errors-in-variables); bounding those effects requires a reach/separation margin
and a neighbor-stability argument. Until that extension is proved, local Gram
conditioning, radius, covariance/support angle, and neighbor-set stability must
be treated as independent chart diagnostics.

The bound predicts a bias-noise tradeoff rather than a universally optimal
`k`: shrinking `r` reduces Taylor bias but amplifies normal noise as `1/r` and
height noise as `1/r^2`. A fixed `k=20` across changing densification levels
does not keep these terms fixed.

### 5.2 Neighbor and frame stability

Let `d_k(i)` and `d_{k+1}(i)` be the distances from sample `i` to the last
included and first excluded neighbors. Suppose every center is perturbed by at
most `delta`.

> **Proposition 5 (kNN margin certificate).** If
>
> ```math
> d_{k+1}(i)-d_k(i)>4 delta,
> ```
>
> then the unordered kNN set of `i` is unchanged by the perturbation, and its
> radius changes by at most `2 delta`.

**Proof.** Every pairwise distance changes by at most `2 delta`. An included
distance can therefore increase by at most `2 delta`, while an excluded one can
decrease by at most `2 delta`. The strict `4 delta` gap prevents their order
from crossing. The radius statement follows from the same distance bound.
`square`

For a fixed neighbor set, let `C_i` be the weighted local covariance and let
`g_i=lambda_2(C_i)-lambda_1(C_i)` be its normal eigengap. Perturbing centers,
weights, and radius changes the covariance by

```math
||Delta C_i|| <= C(r delta+delta^2+r^2 epsilon_w),
```

where `epsilon_w` is the local normalized-weight perturbation. Davis-Kahan then
gives

```math
sin angle(n_hat_i,n_i) <= ||Delta C_i||/g_i.
```

Finally, if the scaled WLS Gram matrix changes from `G_i` to `G_i+Delta G_i`,
Weyl's inequality gives

```math
lambda_min(G_i+Delta G_i)
>= lambda_min(G_i)-||Delta G_i||.
```

These statements produce four distinct certificates: normalized kNN gap,
normalized PCA eigengap, covariance/support normal alignment, and scaled Gram
minimum eigenvalue. Planarity alone certifies none of the other three.

## 6. Next proof targets

1. Make the covariance and Gram perturbation constants in Section 5.2 explicit
   for the implementation's Gaussian weights and mass normalization.
2. Bound the implemented kNN/MLS residual by its continuum counterpart plus an
   explicit `epsilon_h` under quasi-uniform sampling and positive reach.
3. Establish a restricted data coercivity result first for analytic textured
   graphs with known cameras and visibility, then test where it fails.
4. Convert explicit mass and tangent errors to the exact Gaussian-kernel MMD
   used by evaluation after accounting for deterministic subsampling and
   quadrature bias. Proposition 3 already gives the pre-subsampling coupling
   bound with exact bandwidth dependence.

Only targets 1, 2, and 4 are geometry/estimator theory. Target 3 is an inverse
rendering theorem and must not be silently attributed to Bonnet compatibility.
