"""Checks the closed-form claims in p10_cluster_function/design-10.md "OV1" (the OV cut).

C1  A head adds o_i = sum_j P_ij W_O (W_V y_j + b_V), y_j = gamma * xhat_j + beta. With rows of P
    summing to 1 this is sum_j P_ij K xhat_j + c, K = W_O W_V diag(gamma), c = W_O (W_V beta + b_V).
    Symbolic, n = 2 tokens, d = 2, k = 1: a full proof (an identity in the symbols, given sum_j P_ij
    = 1). Does NOT cover the rotary part (Q and K only; V has none in GPT-NeoX).
C2  On the plane xhat lives in (Pi = I - 11^T/d), xhat^T K xhat = xhat^T S xhat with S = sym(Pi K Pi):
    symbolic, d = 3 (a full proof for d = 3; the general case is the same two lines of algebra).
C3  S has at most k positive and at most k negative eigenvalues when K = W_O W_V diag(gamma) has
    k = d_head columns in W_O: the form vanishes on ker(W_V diag(gamma) Pi), of dimension >= d - k,
    and a positive (negative) eigenspace of dimension > k would meet it. The argument is stated
    here; the check is numerical (300 random draws, d = 24, k = 4, gamma of both signs) and does
    NOT prove it. It is what lets each cut head be written back into a 64-wide head.
C4  The write-back: for S' = U diag(lam) U^T (U orthonormal columns, lam nonzero),
    W_V' = |lam|^(1/2) U^T diag(gamma)^-1, b_V' = -W_V' beta, W_O' = U sign(lam) |lam|^(1/2)
    gives W_O' (W_V' (gamma * x + beta) + b_V') = S' x for every x. Symbolic, d = 2, two pairs
    of either sign: a full proof for that size. Does NOT check float32 round-off (the runner
    reads the written weights back against S', design-10.md "OV1" first checks).
C5  S = S_+ - S_-, both PSD, S_+ S_- = 0, ||S_+||^2 + ||S_-||^2 = ||S||^2 (numerical, 50 draws).
    So att and rep split S's energy between them, and a control scaled to ||S_+|| matches att.

Run: python3 tools/math_checks/ov_cut_ov1.py
"""
import numpy as np
import sympy as sp

results = []


def record(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"      {detail}")
    results.append(bool(ok))


# C1 ------------------------------------------------------------------------------------------
d, k = 2, 1
WO = sp.Matrix(d, k, sp.symbols("o0:2"))
WV = sp.Matrix(k, d, sp.symbols("v0:2"))
bV = sp.Matrix(k, 1, sp.symbols("bv0:1"))
g = sp.symbols("g0:2")
be = sp.Matrix(d, 1, sp.symbols("be0:2"))
X = [sp.Matrix(d, 1, sp.symbols(f"x{j}_0:2")) for j in range(2)]
p = sp.symbols("p")
P = [p, 1 - p]                                  # one row of P, summing to 1
G = sp.diag(*g)
o = sum((P[j] * WO * (WV * (G * X[j] + be) + bV) for j in range(2)), sp.zeros(d, 1))
K = WO * WV * G
c = WO * (WV * be + bV)
o2 = sum((P[j] * K * X[j] for j in range(2)), sp.zeros(d, 1)) + c
record("C1 o_i = sum_j P_ij K xhat_j + c", sp.simplify(o - o2) == sp.zeros(d, 1))

# C2 ------------------------------------------------------------------------------------------
d = 3
Ksym = sp.Matrix(d, d, sp.symbols("k0:9"))
Pi = sp.eye(d) - sp.ones(d, d) / d
S = (Pi * Ksym * Pi + (Pi * Ksym * Pi).T) / 2
a, b = sp.symbols("a b")
xh = sp.Matrix([a, b, -a - b])                  # a generic vector in the plane 1^T x = 0
record("C2 xhat^T K xhat = xhat^T sym(Pi K Pi) xhat on the plane",
       sp.expand((xh.T * Ksym * xh)[0] - (xh.T * S * xh)[0]) == 0)

# C3 ------------------------------------------------------------------------------------------
rng = np.random.default_rng(0)
d, k = 24, 4
Pi = np.eye(d) - 1.0 / d
worst = 0
for _ in range(300):
    WO, WV = rng.normal(size=(d, k)), rng.normal(size=(k, d))
    gam = rng.normal(size=d)                    # both signs: the bound does not need gamma > 0
    Kn = WO @ WV @ np.diag(gam)
    Sn = Pi @ Kn @ Pi
    Sn = (Sn + Sn.T) / 2
    lam = np.linalg.eigvalsh(Sn)
    tol = 1e-9 * np.abs(lam).max()
    worst = max(worst, (lam > tol).sum(), (lam < -tol).sum())
record("C3 n_+, n_- <= k (300 draws)", worst <= k, f"max count {worst} > k = {k}")

# C4 ------------------------------------------------------------------------------------------
d = 2
th = sp.symbols("th", real=True)
U = sp.Matrix([[sp.cos(th), -sp.sin(th)], [sp.sin(th), sp.cos(th)]])
l1, l2 = sp.symbols("l1 l2", positive=True)
g = sp.symbols("g0:2", positive=True)
G = sp.diag(*g)
be = sp.Matrix(d, 1, sp.symbols("be0:2"))
x = sp.Matrix(d, 1, sp.symbols("x0:2"))
ok = True
for s1, s2 in ((1, 1), (1, -1), (-1, -1)):
    lam = [s1 * l1, s2 * l2]
    root = sp.diag(sp.sqrt(l1), sp.sqrt(l2))
    sgn = sp.diag(s1, s2)
    WVp = root * U.T * G.inv()
    bVp = -WVp * be
    WOp = U * sgn * root
    lhs = WOp * (WVp * (G * x + be) + bVp)
    rhs = U * sp.diag(*lam) * U.T * x
    ok &= sp.simplify(lhs - rhs) == sp.zeros(d, 1)
record("C4 write-back gives S' x exactly (both signs)", ok)

# C5 ------------------------------------------------------------------------------------------
ok = True
for _ in range(50):
    A = rng.normal(size=(12, 12))
    Sn = (A + A.T) / 2
    lam, V = np.linalg.eigh(Sn)
    Sp = (V * np.where(lam > 0, lam, 0)) @ V.T
    Sm = -(V * np.where(lam < 0, lam, 0)) @ V.T
    ok &= np.allclose(Sp - Sm, Sn) and np.allclose(Sp @ Sm, 0, atol=1e-10)
    ok &= np.linalg.eigvalsh(Sp).min() > -1e-10 and np.linalg.eigvalsh(Sm).min() > -1e-10
    ok &= np.isclose(np.sum(Sp ** 2) + np.sum(Sm ** 2), np.sum(Sn ** 2))
record("C5 S = S+ - S-, orthogonal, energies add (50 draws)", ok)

print(f"\n{sum(results)}/{len(results)} passed")
raise SystemExit(0 if all(results) else 1)
