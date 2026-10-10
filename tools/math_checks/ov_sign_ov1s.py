"""Checks the closed-form claims in p10_cluster_function/design-10.md "OV1s" (the sign at a fixed subspace).

Notation as ov_cut_ov1.py: K = W_O W_V diag(gamma), Pi = I - 11^T/d, S = sym(Pi K Pi) = S_+ - S_-.

M1  The forms on the plane: sym(Pi K' Pi) is S_+ for K' = K + S_- (norep), -S_- for K' = K - S_+
    (noatt), -S_+ for K' = -S_+ (neg). Holds because S_+- live in the plane (Pi S_+- Pi = S_+-), so
    sym(Pi (K + S_-) Pi) = S + S_- = S_+. Numerical, 50 draws (d = 24, k = 4): checks the algebra,
    does NOT say anything about the antisymmetric part K - S, which norep and noatt keep.
M2  K + S_- does not fit a k-wide head: its rank exceeds k in every draw (numerical, 50 draws; the
    bound is 2k, since K and S_- both live in the 2k-dim span of Pi W_O, Pi diag(gamma) W_V^T and
    W_O). So norep and noatt are added by a hook, not written in the weights. Does NOT show the
    rank is 2k in Pythia's heads.
M3  The hook's term is linear: a head with K' = K + Delta adds sum_j P_ij K' xhat_j + c = (the base
    head's output) + sum_j P_ij Delta xhat_j. Symbolic, n = 2, d = 2: a full proof for that size
    (given sum_j P_ij = 1, as C1 of ov_cut_ov1.py). Does NOT check the float32 hook against the
    weights (the runner's first check does, design-10.md "OV1s").

Run: python3 tools/math_checks/ov_sign_ov1s.py
"""
import numpy as np
import sympy as sp

results = []


def record(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"      {detail}")
    results.append(bool(ok))


def draw(rng, d=24, k=4):
    W_O, W_V = rng.normal(size=(d, k)), rng.normal(size=(k, d))
    g = rng.choice([-1, 1], size=d) * (0.3 + rng.random(d))
    K = W_O @ W_V @ np.diag(g)
    Pi = np.eye(d) - 1.0 / d
    M = Pi @ K @ Pi
    S = (M + M.T) / 2
    lam, U = np.linalg.eigh(S)
    Sp = U[:, lam > 1e-10] @ np.diag(lam[lam > 1e-10]) @ U[:, lam > 1e-10].T
    Sm = -U[:, lam < -1e-10] @ np.diag(lam[lam < -1e-10]) @ U[:, lam < -1e-10].T
    return K, Pi, Sp, Sm, k


def form(Pi, Kp):
    M = Pi @ Kp @ Pi
    return (M + M.T) / 2


# M1 ------------------------------------------------------------------------------------------
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(50):
    K, Pi, Sp, Sm, _k = draw(rng)
    for Kp, want in ((K + Sm, Sp), (K - Sp, -Sm), (-Sp, -Sp)):
        worst = max(worst, np.abs(form(Pi, Kp) - want).max())
record("M1 forms on the plane: norep S+, noatt -S-, neg -S+ (50 draws)", worst < 1e-10, f"worst {worst:.1e}")

# M2 ------------------------------------------------------------------------------------------
rng = np.random.default_rng(1)
ranks = []
for _ in range(50):
    K, Pi, Sp, Sm, k = draw(rng)
    ranks.append((np.linalg.matrix_rank(K + Sm, tol=1e-9), np.linalg.matrix_rank(K - Sp, tol=1e-9), k))
record("M2 rank(K + S-) and rank(K - S+) exceed k, at most 2k (50 draws)",
       all(k < a <= 2 * k and k < b <= 2 * k for a, b, k in ranks), f"{ranks[:5]}")

# M3 ------------------------------------------------------------------------------------------
n, d = 2, 2
P = sp.Matrix(n, n, sp.symbols("p0:4"))
X = sp.Matrix(n, d, sp.symbols("x0:4"))                   # rows xhat_j
Kb = sp.Matrix(d, d, sp.symbols("k0:4"))
Dl = sp.Matrix(d, d, sp.symbols("e0:4"))
c = sp.Matrix(1, d, sp.symbols("c0:2"))
ones = sp.ones(n, 1)
lhs = P * X * (Kb + Dl).T + ones * c
rhs = (P * X * Kb.T + ones * c) + P * X * Dl.T
record("M3 the hook's term is linear in Delta (symbolic, n = d = 2)", sp.simplify(lhs - rhs) == sp.zeros(n, d))

print(f"\n{sum(results)}/{len(results)} passed")
raise SystemExit(0 if all(results) else 1)
