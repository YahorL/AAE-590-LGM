# Equivariant Filter on SO(3) with Gyro Bias — Derivation

## Context

You want to estimate attitude `R ∈ SO(3)` from a gyroscope (with constant
bias `b ∈ ℝ³`) plus body-frame direction measurements from a magnetometer
and an accelerometer. The deliverable here is a self-contained mathematical
derivation tying the two papers:

- **EqF** (van Goor, Hamel, Mahony, IEEE TAC 2023) — the general recipe
  on a homogeneous space.
- **ABC-EqF** (Fornasier et al.) — specialises EqF to attitude with bias
  and calibration. We use the **n = 0** case (no online extrinsic calibration),
  i.e. state = `(R, b)`, symmetry group **G = SO(3) ⋉ ℝ³**.

The derivation answers: *"why isn't this just an IEKF, what symmetry do I
choose, what are A, B, C, and what are the EqF update equations?"*

Pure math — no code in this document.

---

## 1. Why the IEKF fails when bias is added

The Invariant EKF on bare SO(3) is exact because the attitude error
`E = R̂ R⁻¹` has *group-affine* dynamics: `Ė` depends only on `E` and the
input, **not** on the true state `R`. Linearisation around `E = I` is
then global.

The moment you add a constant bias state `b` and write the joint
state `(R, b) ∈ SO(3) × ℝ³`, the trick breaks:

```
Ṙ = R (Ω - b)×
ḃ = 0
```

`SO(3) × ℝ³` is not a Lie group whose left/right translation matches the
dynamics — the cross-coupling `R · b×` has no compatible group structure.
Barrau & Bonnabel's "Imperfect-IEKF" tacks the bias on as an extra
linear state, sacrificing the group-affine property and the global
linearisation guarantees.

**EqF fix:** keep `(R, b)` on the *manifold* `M = SO(3) × ℝ³`, but choose
a richer Lie group `G` that *acts transitively* on `M`. Run the observer
on `G`. The error dynamics on `G` are again exactly linearisable (or at
least third-order in the equivariant case), and bias couples in
naturally through the group structure.

---

## 2. System

State manifold (homogeneous space, **not** a Lie group):
```
ξ = (R, b) ∈ M = SO(3) × ℝ³
```

Kinematics with measured gyro `Ω ∈ ℝ³`:
```
Ṙ = R (Ω - b)×
ḃ = 0
```
Compact: `f_u(ξ) = (R(Ω - b)×, 0)`, input `u = Ω ∈ 𝕃 = ℝ³`.

Output: two body-frame direction measurements (gravity from accel,
magnetic field from mag), with known global references `d₁, d₂ ∈ ℝ³`:
```
y = h(ξ) = (R^T d₁, R^T d₂) ∈ ℝ⁶ =: 𝒩
```

---

## 3. The symmetry group G = SO(3) ⋉ ℝ³

Take elements
```
X = (A, a),    A ∈ SO(3),    a ∈ ℝ³
```
with the **semidirect-product** group law (this is the key choice):
```
(A₁, a₁) · (A₂, a₂) = (A₁ A₂,  a₁ + A₁ a₂)
identity:               (I, 0)
inverse:                (A, a)⁻¹ = (A^T,  -A^T a)
```

Structurally `G ≅ SE(3)` — embed
```
X ↦ [ A   a ]
    [ 0   1 ]
```
so all group ops are 4×4 matrix multiplication. The Lie algebra is
`g = so(3) × ℝ³ ≅ ℝ⁶` with element
```
λ = (ω×, β) ↦ [ ω×   β ]
              [ 0    0 ]
```
Its `vee` is `λ^∨ = (ω, β) ∈ ℝ⁶`.

> Although `G ≅ SE(3)`, the **action on M is not the SE(3) rigid-motion
> action** — the second component `a` does not translate a position,
> it *shifts the bias estimate*. That is the whole point.

---

## 4. State action φ : G × M → M (right action)

```
φ(X, ξ) = ( R · A,   A^T (b - a) )
```

Verify it is a right action: `φ(X₂, φ(X₁, ξ)) = φ(X₁ X₂, ξ)`. ✓
Verify transitivity: given two states `(R, b), (R', b') ∈ M`, pick
`A = R^T R'`, `a = b - A b' = b - R^T R' b'`; then `φ(X, (R,b)) = (R',b')`. ✓

Geometric reading:
- The `A` factor right-multiplies the rotation (body-frame attitude shift).
- The `a` factor subtracts a body-frame angular-velocity-like quantity
  from the bias and then re-expresses it in the new body frame.

---

## 5. Input action ψ : G × 𝕃 → 𝕃

The input is the gyro reading `Ω`. Define
```
ψ(X, Ω) = A^T (Ω - a)
```

Equivariance check (`Dφ_X · f_u = f_{ψ(X,u)} ∘ φ_X`):

```
Dφ_X · f_Ω(R, b)  = (R(Ω - b)× A, 0)
f_{ψ(X,Ω)}(φ_X(ξ)) = ( RA · (A^T(Ω - a) - A^T(b - a))×, 0 )
                   = ( RA · (A^T(Ω - b))×, 0 )
                   = ( R (Ω - b)× A, 0 )      using (A^T v)× = A^T v× A
```
Equal. ✓ The biased attitude system is **equivariant** under (φ, ψ).

---

## 6. Equivariant lift Λ : M × 𝕃 → g

The lift takes a state and an input and produces a Lie-algebra element
that, when applied at the origin via `Dφ`, reproduces `f_u`.
Required condition (eq. 11 of the EqF paper):
```
D_X|_{id} φ_ξ(X) · Λ(ξ, u)  =  f_u(ξ).
```

Compute `D_X|_{id} φ_{(R,b)}` on a tangent `(ω×, β) ∈ g`. With
`exp(t(ω×, β)) = (exp(tω×), tβ + O(t²))` and using `(A,a)·(B,β)=(AB, a+Aβ)`:
```
φ_{(R,b)}(exp(tω×), tβ) = ( R · exp(tω×),  exp(-tω×) · (b - tβ) )
d/dt|_{t=0}             = ( R ω×,         -ω× b - β )
```

So we need `Λ((R, b), Ω) = (ω̃×, β)` with
```
R ω̃× = R (Ω - b)×        ⇒   ω̃ = Ω - b
-ω̃× b - β = 0            ⇒   β = -(Ω - b)× b = -Ω × b      (b × b = 0)
```

**Lift:**
```
Λ((R, b), Ω) = ( (Ω - b)×,  -Ω × b )         (eq. 7 of ABC-EqF, n=0)
```

Equivariance of the lift (eq. 13 of EqF paper),
`Λ(φ(X, ξ), ψ(X, u)) = Ad_{X⁻¹} Λ(ξ, u)`, holds by construction.

---

## 7. Output action ρ : G × 𝒩 → 𝒩

For body-frame direction measurements with **calibrated** sensors (n=0):
```
ρ(X, y) = (A^T y₁,  A^T y₂)
```
Equivariance:
```
ρ(X, h(ξ)) = (A^T R^T d₁, A^T R^T d₂) = ((RA)^T d₁, (RA)^T d₂) = h(φ_X(ξ))  ✓
```

---

## 8. Lifted observer state and dynamics

Choose the **fixed origin** `ξ₀ = (I, 0) ∈ M`. Observer lives on G:
```
X̂ = (Â, â) ∈ G,    X̂(0) = id = (I, 0).
```

Recovered state estimate:
```
ξ̂ = φ(X̂, ξ₀) = ( Â,  -Â^T â )    ⇒    R̂ = Â,   b̂ = -Â^T â,   â = -Â b̂.
```

Lifted ODE (eq. 17 of the EqF paper):
```
dX̂/dt = X̂ · Λ(ξ̂, Ω) + (correction term)
```

In 4×4 matrix form with `Ω̃ := Ω - b̂`:
```
dX̂/dt = X̂ · [ Ω̃×    -Ω × b̂ ]   +  Δ̂
              [ 0     0       ]
```

The `correction term` Δ̂ comes from the EqF gain (Riccati × innovation),
mapped from `g` back into a tangent vector at X̂ via left translation —
see §10.

---

## 9. Global error and local coordinates

Right-invariant **global error** on M (eq. 18 of EqF paper):
```
e = φ(X̂⁻¹, ξ) = ( R Â^T,   Â b + â ) = ( R R̂^T,   R̂ (b - b̂) )
```

Note the second component is the bias error **rotated into the world
frame**. When `R = R̂` and `b = b̂`, `e = (I, 0) = ξ₀`. The filter's job
is to drive `e → ξ₀`.

Local coordinates `ε ∈ ℝ⁶` via the exponential chart at the origin:
```
ε = ϑ(e) = ( log(e_R)^∨,   e_b )  ∈ ℝ³ × ℝ³,    ϑ(ξ₀) = 0.
```
The first 3 are exponential coordinates of the attitude error; the last
3 are the (already-linear) bias-error coordinates.

---

## 10. Linearised matrices A_t⁰ and C_t⁰

ABC-EqF gives them in closed form (eq. 14a–b, n=0). Derivation outline:
linearise `dε/dt` and `y - ŷ` around `ε = 0`.

### State matrix (6×6)

Define the **origin input** (input transported to the lifted state's
viewpoint):
```
Ω₀ := ψ(X̂⁻¹, Ω) = Â (Ω - b̂) = R̂ (Ω - b̂)        (world-frame corrected gyro)
```

Then
```
       ┌                ┐
A_t⁰ = │  0      -I     │      (top-left and top-right blocks act on  ε_R)
       │  0      Ω₀×    │      (bottom row acts on  ε_b)
       └                ┘
```

Reading: a perturbation in attitude does **not** change the bias error
(bottom-left zero — bias is constant). A perturbation in bias error
feeds straight into the attitude-error rate with a `-I` (bias error
*is* an angular-rate error). The bottom-right `Ω₀×` reflects the rotation
of the world-frame bias error under the corrected angular velocity.

### Output matrix (6×6, two direction measurements)

Output in local coordinates: `ŷ + C_t⁰ ε + O(|ε|²)` with
```
       ┌                  ┐
C_t⁰ = │  d₁×      0      │
       │  d₂×      0      │
       └                  ┘
```

The zeros in the right block say the magnetometer / accelerometer don't
see the bias directly. Information about `b` enters only through the
state coupling in `A_t⁰`.

> **Improved output linearisation (EqF★).** Lemma V.3 of the EqF paper
> replaces `C_t⁰` with a symmetrised matrix
> ```
> C_t★ = ½ ( D_{E|id} ρ(E, y) + D_{E|id} ρ(E, ŷ) ) Ad_{X̂⁻¹}
> ```
> which gives `O(|ε|³)` linearisation error instead of `O(|ε|²)`. For two
> direction measurements:
> `C_t★ = ½ [(y_i + ŷ_i)× | 0]_{i=1,2}` rotated into the right frame.
> Use this in place of `C_t⁰` to get the EqF★ variant — strictly better
> performance, identical computational cost.

---

## 11. EqF equations (continuous-time)

Gain matrices: state covariance `Σ ∈ S₊(6)`, process noise `M_c`,
measurement noise `N` (set as you would for an EKF).

### Propagation
```
dX̂/dt = X̂ · Λ(ξ̂, Ω)                    ← matrix ODE on G
dΣ/dt = A_t⁰ Σ + Σ A_t⁰ᵀ + M_c          ← Riccati
```

### Update on a measurement y = (y₁, y₂)
```
S = C_t⁰ Σ C_t⁰ᵀ + N
K = Σ C_t⁰ᵀ S⁻¹                                       ∈ ℝ^{6×6}

innovation in 𝒩  :  ν = ρ(X̂⁻¹, y) - h(ξ₀) = (Â y₁ - d₁, Â y₂ - d₂)
                                  (re-expressed at the origin)

Δ⁰ = K · ν                                           ∈ ℝ⁶  (in chart coords)
Δ  = D_E ϑ⁻¹(0) · Δ⁰  ∈ g                            (lift to algebra)

X̂ ← exp_G(Δ) · X̂                                    ← left-multiplicative reset
Σ ← (I - K C_t⁰) Σ
```

`exp_G` is the matrix exponential of the 4×4 embedding of g (closed-form,
same shape as `SE(3)` exp).

### Discrete-time tip (ABC-EqF §IV-B)

Between gyro samples, integrate `X̂` with one step of a Lie-group
integrator (e.g. `X̂_{k+1} = X̂_k · exp_G(Λ(ξ̂_k, Ω_k) ΔT)`). For Σ, ABC-EqF
derives a closed-form **state-transition matrix** Φ(t+ΔT, t):

```
       ┌  I    Φ₁₂    ┐
Φ   =  │  0    Φ₂₂    │
       └              ┘

Φ₁₂ ≈ -ΔT ( I + (ΔT/2) Ω₀× + (ΔT²/6) Ω₀× Ω₀× )
Φ₂₂ ≈   I + ΔT  Ω₀× + (ΔT²/2) Ω₀× Ω₀×
```

Then `Σ_{k+1} = Φ Σ_k Φᵀ + M_d` with discrete process noise `M_d`
obtained by integrating `Φ M_c Φᵀ` over `[t, t+ΔT]`. This is faster than
ode45 of the continuous Riccati and exact-to-3rd-order in ΔT — the
runtime advantage in Table II of ABC-EqF.

---

## 12. Recipe summary (Algorithm 1 of EqF, instantiated)

1. **Symmetry group.** `G = SO(3) ⋉ ℝ³`, identified with SE(3) for
   matrix arithmetic.
2. **State action.** `φ((A,a), (R,b)) = (RA, A^T(b - a))`.
3. **Input action.** `ψ((A,a), Ω) = A^T(Ω - a)`. Equivariance verified.
4. **Lift.** `Λ((R,b), Ω) = ((Ω - b)×, -Ω × b)`.
5. **Output equivariance.** `ρ((A,a), y) = (A^T y₁, A^T y₂)`. Verified.
6. **Origin and chart.** `ξ₀ = (I, 0)`, `ϑ = (log(e_R)^∨, e_b)`.
7. **A, C.** As in §10.
8. **EqF equations.** As in §11. Use `C_t★` instead of `C_t⁰` for the
   star variant (recommended; same cost, lower linearisation error).

What makes this design "special" vs. an Imperfect-IEKF:
- Bias enters through the **group structure**, not as a tacked-on linear
  state. The bias error term `-I` in `A_t⁰` falls out exactly from the
  lift derivation (it isn't postulated).
- The Riccati is propagated in the right coordinates: error is global
  (`e = φ(X̂⁻¹, ξ)` defined on all of M, not just a neighbourhood of R̂),
  and the linearisation error is intrinsic to the chart, not to the
  embedding.
- Swapping in `C_t★` cuts the dominant linearisation error term.

---

## Pointers if you want to go further

- **Add accelerometer/mag extrinsic calibration** → upgrade the symmetry
  group to `(SO(3) ⋉ so(3)) × SO(3)ⁿ` per ABC-EqF eq. 4. Each calibration
  state `Cᵢ ∈ SO(3)` adds a 3-dim block to `ε`, and the lift gets the
  third component `C^T(Ω - b)× C` from eq. 7.
- **GNSS velocity / position aiding** → the homogeneous-space machinery
  generalises; the output action picks up an additive translation piece.
- **Reference implementation** — the ABC-EqF authors publish one at
  `github.com/aau-cns/ABC-EqF`. Use it to sanity-check your A, C, and Φ
  derivations against their code.
