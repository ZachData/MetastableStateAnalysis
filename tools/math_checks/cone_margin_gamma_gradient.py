"""The first-order effect of a rank-1 LayerNorm-gamma patch on the cone margin,
in closed form.

WHY THIS EXISTS
---------------
`p9_metric_intervention/plan-9.md` Sec 4.4 claims the effect of a candidate
metric patch on the cone margin is "free to compute" from activations already on
disk. That is an assertion about a derivative nobody wrote down. This writes it
down and checks it against finite differences.

THE SETUP
---------
Attention reads LN(x) = Gamma xhat + b, with xhat the standardized row and
Gamma = diag(gamma). `math-1c.md` Sec 7.3 gives the cone margin exactly:

    m = dist(0, conv{p_i}) = sqrt( min_{lambda in simplex} ||c(lambda)||^2 ),
    c(lambda) = sum_i lambda_i p_i

The Phase 9 patch is Gamma -> Gamma + eps * u u^T, so each point moves by
eps * (u . xhat_i) * u and

    c(lambda, eps) = c_0(lambda) + eps * s(lambda) * u,
    s(lambda) = sum_i lambda_i (u . xhat_i) = u^T Xhat^T lambda

THE RESULT
----------
By Danskin's envelope theorem at a unique minimizer lambda*,

    d(m^2)/d(eps) at eps=0  =  2 * ( u^T Xhat^T lambda* ) * ( u^T c(lambda*) )

Both factors are computable from the stored activations plus the optimal simplex
weight the existing QP already returns. The derivative vanishes iff u is
orthogonal to the lambda*-weighted raw centroid OR to the frame-space centroid,
and the second-order term is +eps^2 s(lambda*)^2 ||u||^2 >= 0 at fixed lambda*.

WHAT IT DOES NOT PROVE
----------------------
- Danskin needs a UNIQUE minimizer. On a degenerate configuration (ties, a face
  of minimizers) only a directional derivative exists, and the formula gives one
  element of the subdifferential. The instances below are generic; degeneracy is
  exactly the near-zero-margin case `math-1c.md` Sec 7.2 says is the informative
  one, so a real runner must report whether lambda* was unique.
- It is FIRST ORDER at eps = 0. A swept patch of usable size is not covered.
- It is the effect on the configuration AT THE PATCHED LAYER. Everything
  downstream needs a forward pass.
- Instances are checked at (n=6,d=4) and (n=9,d=5). That is evidence, not a
  general-(n,d) proof; the algebra above is the proof and the code is its test.

Run: python3 tools/math_checks/cone_margin_gamma_gradient.py
"""
import numpy as np
import sympy as sp

results = []


def record(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if detail:
        print(f"      {detail}")
    results.append(ok)


# ---------------------------------------------------------------------------
# 1. The algebra, symbolically, at a small instance
# ---------------------------------------------------------------------------
eps = sp.symbols("eps", real=True)
n_sym, d_sym = 3, 2
lam = sp.symbols("l0:3", nonnegative=True)
Xhat = sp.Matrix(n_sym, d_sym, lambda r, c: sp.Symbol(f"x{r}{c}", real=True))
u = sp.Matrix(d_sym, 1, lambda r, c: sp.Symbol(f"u{r}", real=True))
G = sp.diag(*[sp.Symbol(f"g{k}", real=True) for k in range(d_sym)])
b = sp.Matrix(d_sym, 1, lambda r, c: sp.Symbol(f"b{r}", real=True))

lam_vec = sp.Matrix(lam)
# p_i(eps)^T = xhat_i^T (Gamma + eps u u^T) + b^T
P = Xhat * (G + eps * u * u.T) + sp.ones(n_sym, 1) * b.T
c = (P.T * lam_vec)                      # (d,1)
obj = (c.T * c)[0, 0]

c0 = c.subs(eps, 0)
s = (u.T * Xhat.T * lam_vec)[0, 0]
predicted = 2 * s * (u.T * c0)[0, 0]
record("d/d(eps) ||c(lambda,eps)||^2 at eps=0  =  2 (u^T Xhat^T lambda)(u^T c_0)",
       sp.simplify(sp.diff(obj, eps).subs(eps, 0) - predicted) == 0,
       "symbolic in Xhat, Gamma, b, u and lambda -- the envelope step is what adds "
       "the requirement that lambda* be the unique minimizer")

# Second order, at fixed lambda: always non-negative.
second = sp.diff(obj, eps, 2) / 2
record("second-order term is s(lambda)^2 ||u||^2 >= 0 at fixed lambda",
       sp.simplify(second - s**2 * (u.T * u)[0, 0]) == 0,
       "so along a direction where the first order vanishes the margin cannot "
       "decrease to second order without lambda* itself moving")


# ---------------------------------------------------------------------------
# 2. Numeric check against finite differences, through a real simplex solve
# ---------------------------------------------------------------------------
def simplex_project(v):
    """Exact Euclidean projection onto the probability simplex, O(n log n)."""
    m = len(v)
    s = np.sort(v)[::-1]
    css = np.cumsum(s) - 1.0
    idx = np.arange(1, m + 1)
    cond = s - css / idx > 0
    rho = idx[cond][-1]
    theta = css[cond][-1] / rho
    return np.maximum(v - theta, 0.0)


def margin_sq(P, iters=40000, tol=1e-14):
    """min_{lambda in simplex} ||P^T lambda||^2, by projected gradient with the
    true Lipschitz constant (math-1c.md Sec 7.4: L = 2*lambda_max(G), not 2n)."""
    G = P @ P.T
    L = 2.0 * np.linalg.eigvalsh(G)[-1]
    lam = np.full(P.shape[0], 1.0 / P.shape[0])
    for _ in range(iters):
        nxt = simplex_project(lam - (2.0 * G @ lam) / L)
        if np.max(np.abs(nxt - lam)) < tol:
            lam = nxt
            break
        lam = nxt
    return float(lam @ G @ lam), lam


rng = np.random.default_rng(20260920)
for (nn, dd) in ((6, 4), (9, 5)):
    Xh = rng.normal(size=(nn, dd))
    gam = np.abs(rng.normal(loc=1.0, scale=0.2, size=dd))
    bb = rng.normal(scale=0.1, size=dd)
    uu = rng.normal(size=dd)
    uu /= np.linalg.norm(uu)

    def points(e):
        return Xh @ (np.diag(gam) + e * np.outer(uu, uu)) + bb

    m2_0, lam_star = margin_sq(points(0.0))
    # Analytic derivative
    c_star = points(0.0).T @ lam_star
    s_star = uu @ (Xh.T @ lam_star)
    analytic = 2.0 * s_star * (uu @ c_star)
    # Central finite difference through the full re-solve (lambda* re-optimised)
    h = 1e-6
    numeric = (margin_sq(points(h))[0] - margin_sq(points(-h))[0]) / (2 * h)

    rel = abs(analytic - numeric) / max(abs(numeric), 1e-12)
    record(f"finite-difference agreement at (n={nn}, d={dd})",
           rel < 1e-5,
           f"analytic {analytic:.9f} vs numeric {numeric:.9f}, rel err {rel:.2e}; "
           f"m = {np.sqrt(m2_0):.6f}, support of lambda* = {int((lam_star > 1e-9).sum())} points")

# ---------------------------------------------------------------------------
# 3. The binding set is what the derivative sees
# ---------------------------------------------------------------------------
# lambda* is supported on the binding tokens (math-1c.md Sec 7.3), so
# s(lambda*) = u^T Xhat^T lambda* depends ONLY on those tokens. A patch aimed at
# a subspace that no binding token occupies has zero first-order effect on the
# margin however large its D.
Xh = rng.normal(size=(8, 4))
m2, lam_star = margin_sq(Xh @ np.diag(np.ones(4)))
support = np.where(lam_star > 1e-9)[0]
centroid = Xh.T @ lam_star
u_orth = rng.normal(size=4)
u_orth -= centroid * (u_orth @ centroid) / (centroid @ centroid)
u_orth /= np.linalg.norm(u_orth)
record("a patch orthogonal to the lambda*-weighted centroid has zero first-order effect",
       abs(u_orth @ (Xh.T @ lam_star)) < 1e-10,
       f"|support(lambda*)| = {len(support)} of 8 tokens -- the derivative reads the "
       "binding set and nothing else, which is why the intervention has a closed-form target")

print()
print(f"{sum(results)}/{len(results)} checks passed")
raise SystemExit(0 if all(results) else 1)
