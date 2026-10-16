"""B2a's reduction: what a head's attention reads of a token, the rung D = 0, and the circle case
(`docs/ROADMAP.md` §7, B2a; `trackB_circuit/design-B2a.md` "The math").

WHY THIS EXISTS
---------------
B2a asks for the smallest subspace U of the residual (in the head's LN1 frame) whose reduced
attention reproduces a head's real attention. Before any search, three facts bound the answer and
fix what its rungs mean. A head's logit, with rotary R(Δ) on q and k and Pythia's QKV biases, is

    l_ij = (W_Q y_i + b_q)^T R(i − j) (W_K y_j + b_k)        (the 1/sqrt(64) scale aside)

on y = LN1(x). The reduced logit replaces y by U U^T y in both slots, keeping the biases.

WHAT IT SHOWS
-------------
1. **Ceiling.** l_ij depends on y only through W_Q y and W_K y, so any component of y in the null
   space of [W_Q; W_K] is invisible, at every offset (R acts after the projections). Projecting
   onto the span of the rows of W_Q and W_K is exact: D ≤ rank[W_Q; W_K] ≤ 2·64 = 128 always.
   The question B2a can answer is "how far below 128", not "how far below 1024".
2. **D = 0 is a positional head.** With U = 0 the logit is b_q^T R(Δ) b_k: a function of the
   offset only. In one rotary plane at angle Δ·θ it is |b_q||b_k| cos(Δθ − φ), so a peak at
   Δ = 1 (a previous-token pattern) needs no token content at all. B2a's ladder starts at 0.
3. **The circle case.** If the symmetric part of the content form is α U U^T with U of dimension
   2 and the biases are 0, the content logit is α r_i r_j cos(θ_i − θ_j) with (r, θ) the polar
   coordinates of U^T y. At equal r it is the kernel of `2512.01868` §3.3 (the Kuramoto form on
   the circle, β = α r²). The antisymmetric part of W_Q^T W_K (the part rotary feeds) has no
   such form: it gives sin(θ_i − θ_j), a signed preference, not a symmetric attraction.

WHAT IT DOES NOT PROVE
----------------------
- Anything about D below the ceiling on real tokens; that is the measurement.
- That U U^T y has equal norms across tokens. Real projected tokens do not, so the circle kernel
  of fact 3 holds with a token-dependent β_i = α r_i r_j; B2a reads attention, not the kernel.
- Anything about the value side: whether V maps U into U (needed for the reduced *dynamics* to
  close, `2305.05465` Thms 4.2, 5.2) is a separate weights-only row in the design.
"""

import sympy as sp

# --- 1. ceiling: a null-space component of [W_Q; W_K] leaves every logit unchanged -------------
d, h = 5, 2                                  # residual 5, head 2 (so rank [W_Q; W_K] ≤ 4 < 5)
# exact integer weights (a fixed seed), so the null space is exact; tokens, biases, angle symbolic
_rng = sp.randMatrix(2 * h, d, min=-5, max=5, seed=20261015)
WQ, WK = _rng[:h, :], _rng[h:, :]
bq = sp.Matrix(h, 1, sp.symbols("bq0:2", real=True))
bk = sp.Matrix(h, 1, sp.symbols("bk0:2", real=True))
th = sp.symbols("theta", real=True)
R = sp.Matrix([[sp.cos(th), -sp.sin(th)], [sp.sin(th), sp.cos(th)]])   # one rotary plane, any Δ
yi = sp.Matrix(d, 1, sp.symbols("yi0:5", real=True))
yj = sp.Matrix(d, 1, sp.symbols("yj0:5", real=True))

stack = WQ.col_join(WK)
null = stack.nullspace()
assert len(null) == 1                        # generic: rank 4, a 1-dim null space
n = null[0]
a, b = sp.symbols("a b", real=True)


def logit(u, v):
    return ((WQ * u + bq).T * R * (WK * v + bk))[0]


base = logit(yi, yj)
moved = logit(yi + a * n, yj + b * n)
assert sp.simplify(sp.expand(moved - base)) == 0

# --- 2. D = 0: the logit is the bias form, offset only; one plane gives |bq||bk| cos(Δθ − φ) ----
zero = sp.zeros(d, 1)
l0 = sp.simplify(logit(zero, zero))
assert not (l0.free_symbols & (set(yi) | set(yj)))
rq, rk, phq, phk = sp.symbols("r_q r_k phi_q phi_k", positive=True)
bq_p = sp.Matrix([rq * sp.cos(phq), rq * sp.sin(phq)])
bk_p = sp.Matrix([rk * sp.cos(phk), rk * sp.sin(phk)])
l0_p = sp.simplify((bq_p.T * R * bk_p)[0])
assert sp.simplify(l0_p - rq * rk * sp.cos(th + phk - phq)) == 0

# --- 3. circle: symmetric rank-2 form α U U^T on U^T y = r (cos θ, sin θ) -----------------------
al, ri, rj, ti, tj = sp.symbols("alpha r_i r_j theta_i theta_j", real=True)
ui = sp.Matrix([ri * sp.cos(ti), ri * sp.sin(ti)])
uj = sp.Matrix([rj * sp.cos(tj), rj * sp.sin(tj)])
sym = sp.simplify((ui.T * (al * sp.eye(2)) * uj)[0])
assert sp.simplify(sym - al * ri * rj * sp.cos(ti - tj)) == 0
J = sp.Matrix([[0, -1], [1, 0]])             # the antisymmetric rank-2 form
anti = sp.simplify((ui.T * (al * J) * uj)[0])
assert sp.simplify(anti - al * ri * rj * sp.sin(ti - tj)) == 0

print("b2a_subspace_ceiling: 3 checks pass (ceiling = row span of [W_Q; W_K] at every offset; "
      "D = 0 is the offset-only bias form, cos(Δθ − φ) in one plane; symmetric rank 2 is the "
      "circle kernel α r_i r_j cos(θ_i − θ_j), antisymmetric gives sin)")
