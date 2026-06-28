# Bonnet Compatibility for Gaussian Splatting

This note states the geometric-consistency constraint at the center of the
fundamental-form line, proves it is a non-trivial discriminant in the 3DGS
parameterization, positions it against the closest prior work with a priority
check, and specifies the single attribution experiment that decides whether it
is an oral-level contribution.

It is the theory companion to `manifold_gs/fundamental_compatibility.py`.

---

## 1. Object

A 3DGS scene exposes two geometric fields over the same primitives:

- a **position field** `P = {mu_i}` (Gaussian centers);
- an **orientation/normal field** `N = {n_i}` (third column of the rotation
  `R_i`, i.e. the smallest-covariance-eigenvector direction).

In vanilla 3DGS these are **independent optimization variables**, both driven
only by the photometric loss. A smooth surface forbids this: the unit normal is
the Gauss map of the embedding, `n = nu(x)`, fully determined by position.

From each field we estimate a shape operator (second fundamental form expressed
in a common tangent frame):

- `A_support(i)` from a weighted quadratic **Monge patch** fit of neighbor
  centers `{mu_j}` (depends only on `P`); code: `support_shape`, `support_second`.
- `A_pred(i)` from the **Weingarten map** `II_pred = -dn . dphi` of the learned
  normal field `{n_j}` (depends on `N`); code: `predicted_shape`,
  `predicted_second`.

The compatibility residual is

```math
rho(i) = ||A_pred - A_support||             (shape match)
       + ||A_pred - A_pred^T||              (symmetry / integrability)
       + |det A_pred - det A_support| r^2   (Gauss equation)
       + ||Codazzi(A_pred)|| r^2            (Codazzi-Mainardi)
```

mapping to `shape_relative`, `symmetry_residual`, `gauss_residual_scaled`,
`codazzi_residual_scaled` in `summarize_compatibility`.

---

## 2. Triviality boundary

> **Lemma 1 (triviality).** If the normal field is a deterministic function of
> the position field, `n_i = nu(mu_i; nbr(P))` equal to the unit normal of the
> fitted position surface (e.g. PCA / Monge normal), then
> `A_pred(i) = A_support(i) + O(h)` and `rho(i) = O(h)`, pure fit-truncation
> error of mesh size `h`.

*Proof.* Both operators are then computed from the same height-function germ;
`A_pred` uses the gradient of the same fitted normal, which is the derivative of
the same `C^2` surface, so the two shape operators agree to fit order. The
Gauss-map of a `C^2` surface is `C^1`, so its discrete derivative reproduces the
extrinsic curvature up to `O(h)`. ∎

Lemma 1 is exactly the auditor's worry: **if both forms come from one embedding,
the identities are near-tautological.** It also names the precise condition:
triviality holds iff `N` is a function of `P`. In vanilla 3DGS `N` is *not* a
function of `P` (the orientation is a free quaternion), so the lemma does not
apply.

---

## 3. Non-triviality

### 3.1 Separation vs the standard loss suite (proved)

> **Theorem 1.** Let the loss suite be photometric `L_photo`, thinness
> `L_thin`, and pairwise normal smoothness `L_nsm = mean(1 - |n_i.n_j|)`. When
> `N` is an independent field, for every `eps > 0` there is a configuration with
> `L_photo, L_thin, L_nsm <= eps` and `rho >= c_0 > 0`.

*Construction (rotating normals on flat support).*
Sample centers on the plane `z = 0`, so the support is exactly flat:
`A_support = 0`, support normal `= e_z`. Assign a curl-carrying normal field

```math
n_i = normalize( e_z + delta * kappa * (-y_i, x_i, 0) ).
```

- `L_nsm`: `1 - |n_i.n_j| = O(delta^2 kappa^2 spacing^2) -> 0` as spacing -> 0.
- `L_thin`: each splat is made genuinely flat (lambda_3 -> 0), orientation-free.
- `L_photo`: a thin, tangentially near-isotropic splat is first-order invariant
  to small normal tilt, so against a textured-plane GT a configuration with
  `L_photo <= eps` exists (the orientation gauge slack of thin splats).
- But `II_pred = -dn.dphi = delta*kappa [[0,-1],[1,0]]`, purely antisymmetric, so
  `symmetry_residual = ||II_pred - II_pred^T|| ~ 2 sqrt(2) delta kappa` and
  `shape_relative ~ delta kappa / scale` since `A_support = 0`.

Fix `delta*kappa` while spacing -> 0: the three standard losses vanish while
`rho >= c_0`. ∎

This is the formal version of counterexample #4 ("oscillating normals") in
`THEORY-PROOF-SKETCH.md`. The rejected configuration is precisely the
sparse-view duplicate-sheet / floater family: a normal field that looks locally
smooth and renders well but is realizable by no surface.

`L_nsm` is what most surface-GS methods actually deploy, so Theorem 1 covers the
realistic baseline, not a strawman.

### 3.2 Separation vs first-order normal supervision (qualitative, stated honestly)

Some methods instead **supervise** the free normal toward the position normal,
`n_i -> nu(mu_i)` (e.g. local-mean / PCA normal of neighbor centers). This is the
trivial regime of Lemma 1 imposed as a target. We do **not** claim strict
logical independence from *perfect* supervision: if `n_i = nu(mu_i)` exactly at
all samples, compatibility holds up to `O(h)`. The genuine differences are:

1. **Curvature-aware integrability vs curvature-blind smoothness.**
   `L_nsm` and value-supervision penalize *all* normal variation, fighting the
   legitimate curvature of a curved surface. The symmetry/Codazzi terms penalize
   *only* the non-integrable (curl) part of `dn`, allowing true curvature. On any
   non-flat surface these are different constraints, not rescalings.

2. **No injected target / larger correct fixed-point set.** Supervision pins `n`
   to `nu(mu)`, the noise-sensitive local-PCA normal — the exact estimator
   GeoSplat reports as unreliable. Its fixed point is wrong wherever `P` is wrong.
   Compatibility injects no external target; its fixed-point set is the whole
   Bonnet-realizable family of `(P, N)` pairs, and it lets photometric evidence
   on *both* fields move the pair toward joint consistency.

3. **Second-order sensitivity to the supervision residual.** Soft supervision
   sees only the value residual `e_i = n_i - nu(mu_i)`; `II_pred` sees its
   gradient `de`. A small-amplitude residual with non-zero curl passes value
   supervision while raising `symmetry_residual`.

The honest positioning: compatibility is **not** a strictly stronger superset of
normal supervision, but a *different second-order, target-free* constraint with a
larger correct fixed-point set. Section 5 must demonstrate this empirically; the
theory alone does not settle it.

### 3.3 Meaning (Bonnet)

> **Bonnet (fundamental theorem of surface theory).** On a simply-connected
> domain, a first form `I` (SPD) and a second form `II` (symmetric) satisfying
> the Gauss and Codazzi-Mainardi equations determine an immersion realizing them,
> unique up to a rigid motion of `R^3`.

Hence `rho -> 0` with `I` from `A_support`'s metric and `II = A_pred` symmetric
and Gauss/Codazzi-consistent **certifies that the learned `(P, N)` pair is, in
each chart, the data of an actual smooth surface, pinned up to rigid motion.**
This upgrades "the normals look consistent" to "the fields are Bonnet-realizable
by one surface." Reference: do Carmo, *Differential Geometry of Curves and
Surfaces*, Ch. 4.

---

## 4. Priority check (mid-2026)

Targeted literature search; single-pass, not exhaustive. Confidence marked.

| # | Question | Verdict | Evidence |
|---|---|---|---|
| Q1 | Any GS / differentiable-rendering work using Gauss-Codazzi / Bonnet compatibility as a loss? | **Open** (med-high) | No GS/DR hit. Neural-SDF works compute fundamental forms from one implicit field (trivial regime). GeoSplat and QGS explicitly do not use Gauss/Codazzi. |
| Q2 | Position-vs-orientation decoupling framed as identifiability / gauge freedom needing a compatibility constraint? | **Partially occupied** (med) | The render-invariant ambiguity is widely noted (G3Splat); gauge freedom appears in GS *uncertainty* and *feature-learning* contexts. The standard fix is to **supervise normals to the position normal** (collapses to the trivial regime). The second-order compatibility framing is not found. |
| Q3 | GeoSplat exact overlap. | **Nearest neighbor, distinct core** (high) | GeoSplat uses shape operator, principal curvatures, a varifold approximate second fundamental form (WSFF), local-manifold priors, curvature-aware densification — **all derived from positions**; normals are `u_d1 x u_d2`. It never compares a position-derived form against an independent-normal-derived form and uses no Gauss/Codazzi. |
| Q4 | Varifold / point-cloud discriminant comparing an independent normal field's 2nd-order variation against position curvature. | **Mostly open** (med) | Buet-Rumpf and varifold methods estimate curvature from the points/varifold; the varifold object supports independent orientation, but using the position-vs-normal discrepancy as a discriminant is not an established named tool. |

**Overall.** The specific framing — *impose Bonnet / Gauss-Codazzi compatibility
between a position-derived and an independent-normal-derived fundamental form in
3DGS* — appears unclaimed as of mid-2026, inside a crowded neighborhood
(GeoSplat second-order, QGS curved primitives, normal supervision, gauge/ambiguity
awareness). The gap is real but narrow. The decisive prior fact is Q2: **first-order
normal supervision already exists**, so novelty rests entirely on the second-order,
target-free distinction of Section 3.2, demonstrated by Section 5.

Sources: GeoSplat arXiv:2509.05075; Quadratic Gaussian Splatting arXiv:2411.16392;
G3Splat arXiv:2512.17547 / OpenReview ZfNeovqQkn; Normal-GS arXiv:2410.20593;
Buet & Rumpf arXiv:2010.09419; do Carmo Ch. 4.

---

## 5. The attribution experiment (make-or-break)

Non-trivial and unclaimed are necessary, not sufficient. The constraint earns an
oral only if it **fixes a failure mode that the strongest deployed alternatives
leave**, with the credit isolated.

Setup: analytic sphere + torus + two-close-sheets, 3/6/9 views, >= 3 seeds, the
existing held-out PSNR guardrail.

Ladder (each adds one term, matched primitive count):

1. photometric only;
2. + thinness;
3. + pairwise normal smoothness `L_nsm`;
4. + **first-order normal supervision** `n -> nu(mu)` (the real competitor, the
   Q2 baseline) — this rung is mandatory, not optional;
5. + **Bonnet compatibility** (symmetry + Gauss + Codazzi).

Primary readouts attributable to the last two rungs:
- duplicate-sheet rate and free-space-violation count on two-close-sheets;
- normal **integrability** residual and `symmetry_residual` on the torus
  (curved surface, where curvature-aware vs curvature-blind diverge);
- GT Chamfer-L1 / normal-median with no held-out PSNR regression.

Pass condition for the claim: rung 5 beats rung 4 on the duplicate-sheet /
integrability axis with non-overlapping seed CIs, **and** rung 4 alone does not
close the gap. If rung 4 already closes it, the contribution collapses to
"a principled normal-supervision variant" — report that honestly and fall back to
the conserved-measure thesis in `THEORY-NOVELTY-REAUDIT-2026-06-27.md`.

---

## 6. One-sentence thesis (if Section 5 passes)

> 3DGS decouples position and orientation, so a scene can render perfectly while
> its centers and normals belong to no common surface; we impose the fundamental
> theorem of surfaces — Bonnet compatibility of the position-derived and
> normal-derived fundamental forms — as the missing second-order constraint, and
> show it removes duplicate sheets and non-integrable normal fields that
> photometric, thinness, normal-smoothness, and normal-supervision baselines
> leave behind.
