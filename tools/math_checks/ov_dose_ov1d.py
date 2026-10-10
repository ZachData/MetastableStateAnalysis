"""Checks the closed-form claims in p10_cluster_function/design-10.md "OV1d" (a dose curve at matched
centred distance).

Notation as ov_cut_ov1.py: K = W_O W_V diag(gamma), Pi = I - 11^T/d, S = sym(Pi K Pi) = S_+ - S_-.

D1  The w family's form on the plane is t S_+: sym(Pi (t S_+) Pi) = t S_+ for every t (S_+ lives in
    the plane), so w+t's attraction is S_+ scaled, attractive for t > 0 and repulsive for t < 0.
    Numerical, 50 draws x 5 values of t: checks the algebra. Does NOT say the stream moves
    linearly in t (it does not need to; the rule measures dc per arm).
D2  The z hook with column 0 of P zeroed in Delta's term gives token i the head output
    sum_j P_ij K xhat_j + sum_{j>=1} P_ij (K + Delta) xhat_j - sum_{j>=1} P_ij K xhat_j + c, i.e. key 0
    through K and keys j >= 1 through K + Delta; for i = 0 (causal: only key 0) it is base's. Symbolic,
    n = 3, d = 2, causal P: a full proof for that size. Does NOT check the float32 hook on the real
    model (the runner's first check does, with column 0 kept).
D3  dc is exactly 0 for a common shift of every kept row (row-centring removes it): symbolic, n = 3,
    d = 2. Does NOT make dc blind to a shift scaled per token (e.g. by each token's P_i0): that moves
    centred rows and counts in dc, as the rule says.
D4  c2a's frame (unit rows first, then the mean direction out) is NOT invariant to a common shift:
    a numerical counterexample where a shift changes the frame's cosine distances. This is why dz is
    stored beside and dc, not dz, is the matched measure only by the user's pick. Does NOT measure how
    much a common shift moves c2a's partition on Pythia's rows.

Run: python3 tools/math_checks/ov_dose_ov1d.py
"""
import numpy as np
import sympy as sp

results = []


def record(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"      {detail}")
    results.append(bool(ok))


# D1 ------------------------------------------------------------------------------------------
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(50):
    d, k = 24, 4
    W_O, W_V = rng.normal(size=(d, k)), rng.normal(size=(k, d))
    g = rng.choice([-1, 1], size=d) * (0.3 + rng.random(d))
    K = W_O @ W_V @ np.diag(g)
    Pi = np.eye(d) - 1.0 / d
    M = Pi @ K @ Pi
    lam, U = np.linalg.eigh((M + M.T) / 2)
    Sp = U[:, lam > 1e-10] @ np.diag(lam[lam > 1e-10]) @ U[:, lam > 1e-10].T
    for t in (-3.0, -0.5, 0.0, 0.25, 1.0):
        F = Pi @ (t * Sp) @ Pi
        worst = max(worst, np.abs((F + F.T) / 2 - t * Sp).max())
record("D1 sym(Pi t S+ Pi) = t S+ (50 draws x 5 t)", worst < 1e-10, f"worst {worst:.1e}")

# D2 ------------------------------------------------------------------------------------------
n, d = 3, 2
p = sp.symbols("p10 p11 p20 p21 p22")
P = sp.Matrix([[1, 0, 0], [p[0], p[1], 0], [p[2], p[3], p[4]]])     # causal; row 0 attends only to key 0
X = sp.Matrix(n, d, sp.symbols("x0:6"))
Kb = sp.Matrix(d, d, sp.symbols("k0:4"))
Dl = sp.Matrix(d, d, sp.symbols("e0:4"))
c = sp.Matrix(1, d, sp.symbols("c0:2"))
ones = sp.ones(n, 1)
P1 = P.copy()
P1[:, 0] = sp.zeros(n, 1)
hook = P * X * Kb.T + ones * c + P1 * X * Dl.T                   # base head + the hook's term
want = P[:, 0] * X[0, :] * Kb.T + P1 * X * (Kb + Dl).T + ones * c
row0 = sp.simplify(hook[0, :] - (X[0, :] * Kb.T + c))
record("D2 z: key 0 through K, keys j >= 1 through K + Delta; token 0 at base (symbolic, n = 3, d = 2)",
       sp.simplify(hook - want) == sp.zeros(n, d) and row0 == sp.zeros(1, d))

# D3 ------------------------------------------------------------------------------------------
H = sp.Matrix(3, 2, sp.symbols("h0:6"))
v = sp.Matrix(1, 2, sp.symbols("v0:2"))
C = sp.eye(3) - sp.ones(3, 3) / 3
record("D3 row-centring removes a common shift exactly (symbolic, n = 3, d = 2)",
       sp.simplify(C * (H + sp.ones(3, 1) * v) - C * H) == sp.zeros(3, 2))

# D4 ------------------------------------------------------------------------------------------


def frame_cos(Y):
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    m = Y.mean(0)
    zh = m / np.linalg.norm(m)
    Z = Y - np.outer(Y @ zh, zh)
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    return 1 - Z @ Z.T


rng = np.random.default_rng(1)
Y = rng.normal(size=(12, 6)) * (0.5 + rng.random((12, 1)))      # rows of unequal norm
shift = 3.0 * rng.normal(size=6)
gap = np.abs(frame_cos(Y + shift) - frame_cos(Y)).max()
record("D4 c2a's frame is not invariant to a common shift (counterexample)", gap > 1e-2, f"gap {gap:.1e}")

print(f"\n{sum(results)}/{len(results)} passed")
raise SystemExit(0 if all(results) else 1)
