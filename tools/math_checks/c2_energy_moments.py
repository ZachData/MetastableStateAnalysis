"""Checks the closed-form claim C2's draft makes about E_1 and the cloud's mean (draft §4.1; the
referee read of C2 unit 5, point M1).

E_1(X) = (1 / 2n^2) sum_ij exp(<x_i, x_j>) for unit rows x_i, and m = (1/n) sum_i x_i.

M1  sum_ij <x_i, x_j> = n^2 |m|^2 (sympy, symbolic rows, n = 3, d = 2).
M2  E_1 = 1/2 + |m|^2 / 2 + sum_{k>=2} (1 / (2 n^2 k!)) sum_ij <x_i, x_j>^k exactly, so the
    first-order term of E_1 is half the squared mean norm (sympy series of exp; numerical on
    random unit clouds to 40 terms).
M3  On the token-centred cloud (each row minus the mean, then re-normalised), the mean is not
    zero in general: re-normalising after centring leaves a mean whose norm is set by how the
    rows' centred lengths differ (numerical: nonzero on a random cloud, zero when every centred
    row has the same length).

NOT proved: that a drop in E_1 between two layers of a real model is driven by |m|^2 (the higher
terms are not small: |<x_i, x_j>| <= 1 and the diagonal terms are 1); anything about which term
moves at the break (that needs the series on the records); that V_c is a function of the
effective rank (E5 is a different statistic of the same Gram matrix).

Run: python3 tools/math_checks/c2_energy_moments.py
"""
import math

import numpy as np
import sympy as sp

results = []


def record(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if detail:
        print("      " + detail)
    results.append(ok)


# M1: the sum of all inner products is n^2 |m|^2.
n, d = 3, 2
X = sp.Matrix(n, d, lambda i, j: sp.Symbol(f"x{i}{j}", real=True))
gram_sum = sum((X.row(i) * X.row(j).T)[0] for i in range(n) for j in range(n))
m = sp.Matrix([[sum(X[i, j] for i in range(n)) / n for j in range(d)]])
record("M1 sum_ij <x_i,x_j> = n^2 |m|^2", sp.simplify(gram_sum - n**2 * (m * m.T)[0]) == 0)

# M2: the exponential's series, term by term.
t = sp.Symbol("t", real=True)
series = sp.series(sp.exp(t), t, 0, 6).removeO()
record("M2 exp(t) = 1 + t + sum_{k>=2} t^k / k! (to order 5)",
       sp.simplify(series - (1 + t + sum(t**k / sp.factorial(k) for k in range(2, 6)))) == 0)

rng = np.random.default_rng(0)
worst = 0.0
for _ in range(20):
    nn, dd = int(rng.integers(5, 60)), int(rng.integers(3, 40))
    U = rng.standard_normal((nn, dd)) + rng.standard_normal(dd) * rng.uniform(0, 3)
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    G = U @ U.T
    e1 = np.exp(G).sum() / (2 * nn**2)
    mean = U.mean(axis=0)
    rest = sum((G**k).sum() / (2 * nn**2 * math.factorial(k)) for k in range(2, 40))
    worst = max(worst, abs(e1 - (0.5 + mean @ mean / 2 + rest)))
record("M2 E_1 = 1/2 + |m|^2/2 + higher terms (20 random clouds)", worst < 1e-12,
       f"largest gap {worst:.1e}")

# M3: the centred, re-normalised cloud keeps a mean.
U = rng.standard_normal((40, 16)) + 2.0
C = U - U.mean(axis=0)
C /= np.linalg.norm(C, axis=1, keepdims=True)
m_c = np.linalg.norm(C.mean(axis=0))
V = rng.standard_normal((40, 16))
V -= V.mean(axis=0)
V = np.vstack([V, -V])                      # every centred row paired with its negative
V /= np.linalg.norm(V, axis=1, keepdims=True)
m_sym = np.linalg.norm(V.mean(axis=0))
record("M3 centring then re-normalising leaves a mean (nonzero in general, zero by symmetry)",
       m_c > 1e-3 and m_sym < 1e-12, f"|m_c| {m_c:.3f} on a random cloud; {m_sym:.1e} when symmetric")

print(f"\n{sum(results)} of {len(results)} passed")
raise SystemExit(0 if all(results) else 1)
