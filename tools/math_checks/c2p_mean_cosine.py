"""Checks the closed form in trackC_writeup/paper/main.tex §2.1 (C2p, the readings audit, 2026-10-30).

M1  For n unit vectors x_i with mean m = (1/n) sum_i x_i, the mean cosine over the n(n - 1) ordered
    pairs of distinct tokens is (n ||m||^2 - 1) / (n - 1) (sympy, from sum_ij <x_i, x_j> = n^2 ||m||^2
    and the n diagonal terms equal to 1), and it agrees with a direct computation on random unit
    clouds (numerical).

NOT proved: that this is the statistic Ethayarajh (2019) report, which averages over words drawn
from different contexts of a corpus, not over one prompt's tokens; the identity holds for whatever
set of unit vectors the mean is taken over.

Run: python3 tools/math_checks/c2p_mean_cosine.py
"""
import sys

import numpy as np
import sympy as sp

results = []


def record(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if detail:
        print("      " + detail)
    results.append(ok)


n, M2 = sp.symbols("n M2", positive=True)  # M2 = ||m||^2
total = n**2 * M2  # sum over all i, j of <x_i, x_j>
off_diagonal_mean = (total - n) / (n * (n - 1))
record("M1 symbolic", sp.simplify(off_diagonal_mean - (n * M2 - 1) / (n - 1)) == 0)

rng = np.random.default_rng(0)
worst = 0.0
for size, dim in [(2, 3), (7, 5), (512, 64), (50, 1024)]:
    x = rng.standard_normal((size, dim)) + rng.standard_normal(dim)  # a shared direction, so ||m|| > 0
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    g = x @ x.T
    direct = (g.sum() - np.trace(g)) / (size * (size - 1))
    m2 = float(np.sum(x.mean(axis=0) ** 2))
    worst = max(worst, abs(direct - (size * m2 - 1) / (size - 1)))
record("M1 numerical", worst < 1e-12, f"largest gap {worst:.1e}")

sys.exit(0 if all(results) else 1)
