"""Training symmetry: the closed forms `MATH_TRAINING_SYMMETRY.md` §3 states, checked.

WHY THIS EXISTS
---------------
`MATH_TRAINING_SYMMETRY.md` is a question register for analytic work on how training moves
Pythia's weights. It states four closed forms as its starting point. CLAUDE.md asks for a sympy
check of each, saying what the check does not prove.

WHAT IT SHOWS
-------------
1. **Gradient flow conserves the QK imbalance.** If the loss depends on (W_Q, W_K) only through
   M = W_Qᵀ W_K (W_Q, W_K of size k × d, rows = head dims), then under gradient flow
   Ψ = W_Q W_Qᵀ − W_K W_Kᵀ (k × k) has dΨ/dt = 0, for any G = ∂L/∂M. The same algebra gives
   the OV law (W_V W_Vᵀ − W_Oᵀ W_O for V = W_O W_V).
2. **Weight decay λ on both factors gives dΨ/dt = −2λΨ**, so Ψ(t) = e^{−2λt} Ψ(0).
3. **Weight decay on W only (not on a LayerNorm gain γ_i that multiplies column i of W)**:
   the rescaling symmetry (γ_i, W[:, i]) → (c γ_i, W[:, i] / c) gives, under flow with decay on
   W alone, d/dt (γ_i² − ‖W[:, i]‖²) = 2λ ‖W[:, i]‖² (Pythia decays matrices, not gains).
4. **RoPE shrinks the QK symmetry group.** Score qᵀ R(θ) k is invariant under q → A q,
   k → A⁻ᵀ k for every θ iff Aᵀ commutes with R(θ). On one rotary pair the commutant is
   {a I + b J}: a complex scalar, 2 parameters instead of 4.
5. **The random-init symmetric fraction.** For M = A Bᵀ with A, B independent d × k with iid
   standard Gaussian entries, E tr(M²) = d k and E ‖M‖_F² = d² k, so the ratio of expectations
   of the symmetric energy fraction (‖M‖_F² + tr M²) / (2 ‖M‖_F²) is (1 + 1/d) / 2.
   At d = 1024 that is 0.500488.

WHAT IT DOES NOT PROVE
----------------------
- 1–3 are continuous-time gradient flow. Finite steps break 1 at O(η²) per step (Kunin et al.,
  2012.04728); Adam's diagonal preconditioner breaks it at first order. Neither is checked here.
- 1 assumes the loss sees (W_Q, W_K) only through M. Pythia's QK biases transform with the
  factors (b_Q → A b_Q, b_K → A⁻ᵀ b_K), and are left out here.
- 4 checks one rotary pair at one generic angle; distinct frequencies across pairs (so no
  mixing between pairs) is argued, not checked.
- 5 is a ratio of expectations, not the expectation of the ratio. Concentration at d = 1024
  makes them close; the gap is not bounded here. Small (d, k) are checked exactly by Wick.
"""
import itertools

import sympy as sp


def check_conservation(k=2, d=3):
    WQ = sp.Matrix(k, d, lambda i, j: sp.Symbol(f"q{i}{j}"))
    WK = sp.Matrix(k, d, lambda i, j: sp.Symbol(f"k{i}{j}"))
    G = sp.Matrix(d, d, lambda i, j: sp.Symbol(f"g{i}{j}"))  # dL/dM, M = WQ^T WK
    lam = sp.Symbol("lam")
    dWQ = -(WK * G.T)  # dL/dWQ = WK G^T  (k x d)
    dWK = -(WQ * G)    # dL/dWK = WQ G
    dPsi = dWQ * WQ.T + WQ * dWQ.T - dWK * WK.T - WK * dWK.T
    assert sp.simplify(dPsi) == sp.zeros(k, k), "1 fails"
    Psi = WQ * WQ.T - WK * WK.T
    dWQd, dWKd = dWQ - lam * WQ, dWK - lam * WK
    dPsi_d = dWQd * WQ.T + WQ * dWQd.T - dWKd * WK.T - WK * dWKd.T
    assert sp.simplify(dPsi_d + 2 * lam * Psi) == sp.zeros(k, k), "2 fails"
    print("1, 2: dPsi/dt = 0 under flow; = -2 lam Psi with decay on both  OK")


def check_gain(d=3):
    g, lam = sp.symbols("g lam")
    w = sp.Matrix(d, 1, lambda i, j: sp.Symbol(f"w{i}"))
    # loss depends on (g, w) only through u = g * w; gradient G_u arbitrary
    Gu = sp.Matrix(d, 1, lambda i, j: sp.Symbol(f"G{i}"))
    dg = -(Gu.T * w)[0]
    dw = -g * Gu - lam * w
    lhs = 2 * g * dg - 2 * (w.T * dw)[0]
    assert sp.simplify(lhs - 2 * lam * (w.T * w)[0]) == 0, "3 fails"
    print("3: d/dt (g^2 - |w|^2) = 2 lam |w|^2 with decay on w only  OK")


def check_rope_commutant():
    a, b, c, e, th = sp.symbols("a b c e th")
    A = sp.Matrix([[a, b], [c, e]])
    R = sp.Matrix([[sp.cos(th), -sp.sin(th)], [sp.sin(th), sp.cos(th)]])
    eqs = list(A * R - R * A)
    sol = sp.solve([sp.simplify(x.subs(th, sp.Rational(7, 10))) for x in eqs], [c, e], dict=True)
    assert sol == [{c: -b, e: a}], sol
    print("4: commutant of a rotary pair's rotation is {a I + b J}  OK")


def wick_expect(poly, gens):
    """E of a polynomial in iid N(0,1) variables: moments 0, 1, 0, 3."""
    mom = {0: 1, 1: 0, 2: 1, 3: 0, 4: 3}
    total = 0
    for monom, coeff in sp.Poly(poly, *gens).terms():
        term = coeff
        for p in monom:
            term *= mom[p]
            if term == 0:
                break
        total += term
    return sp.nsimplify(total)


def check_sym_fraction():
    for d, k in [(2, 1), (3, 2), (4, 2)]:
        A = sp.Matrix(d, k, lambda i, j: sp.Symbol(f"a{i}{j}"))
        B = sp.Matrix(d, k, lambda i, j: sp.Symbol(f"b{i}{j}"))
        gens = list(A) + list(B)
        M = A * B.T
        tr2 = sp.expand((M * M).trace())
        fro = sp.expand(sum(x**2 for x in M))
        assert wick_expect(tr2, gens) == d * k and wick_expect(fro, gens) == d * d * k, (d, k)
    print("5: E tr(M^2) = d k, E |M|_F^2 = d^2 k (Wick, d,k in {(2,1),(3,2),(4,2)})  OK;"
          f" (1 + 1/1024)/2 = {(1 + 1 / 1024) / 2:.6f}")


if __name__ == "__main__":
    check_conservation()
    check_gain()
    check_rope_commutant()
    check_sym_fraction()
