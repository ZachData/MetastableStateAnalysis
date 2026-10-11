"""Checks the closed-form claims in p10_cluster_function/design-10.md "F13" (the centre scan).

Strong Renyi centre: token j with dist(x_j, x_i) > delta for every i < j. Ordinary: dist > delta to
every earlier accepted token (greedy). p_x = mu(B_delta(x)); a_x = the other points of a finite cloud
within delta of x.

M1  Lemma C.1 per point: a point at arrival k is strong with probability (1 - p)^(k-1), so the
    infinite-sequence count is int 1/p dmu and the first n arrivals give (1 - (1 - p)^n)/p.
    Symbolic (geometric series). Does NOT check the paper's i.i.d. assumption on any real cloud.
M2  On S^1 with the uniform law, mu(B_delta) = delta/pi, so E[#strong] = pi/delta (symbolic), and a
    simulation at n = 4000 lands within 3 % of it at delta = 0.3. The d = 2 case only.
M3  The order null's mean is exact: over uniform random orders of a fixed cloud, E[#strong] =
    sum_x 1/(a_x + 1) (x is strong iff it arrives first among itself and its a_x neighbours). This
    equals Lemma C.1 on the cloud's empirical measure (int 1/mu_n(B) dmu_n = sum_x (1/n) n/(a_x+1)).
    With one point held first (position 0): points within delta of it are never strong, the rest
    give 1/(a'_x + 1), a'_x counting neighbours other than it. Exact enumeration of every order
    (n = 7, 3 clouds, both forms, rationals). Does NOT cover the index statistic I or coverage,
    which the producer simulates.
M4  Strong is a subset of ordinary, and the ordinary centres leave no token farther than delta from
    every centre at the layer they were accepted on (so coverage there is 1 by construction, which
    is why F15 follows the centres to later layers). Numerical, 200 random clouds.

Run: python3 tools/math_checks/renyi_f13.py
"""
import itertools
from fractions import Fraction

import numpy as np
import sympy as sp

results = []


def record(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"      {detail}")
    results.append(bool(ok))


def strong(D, order, delta):
    out = []
    for t, j in enumerate(order):
        if all(D[j, i] > delta for i in order[:t]):
            out.append(j)
    return out


def ordinary(D, order, delta):
    out = []
    for j in order:
        if all(D[j, i] > delta for i in out):
            out.append(j)
    return out


# M1 ------------------------------------------------------------------------------------------
p, k, n = sp.symbols("p k n", positive=True)
q = 1 - p
finite = sp.summation(q ** (k - 1), (k, 1, n))
record("M1 sum_{k<=n} (1-p)^(k-1) = (1 - (1-p)^n)/p",
       sp.simplify(finite - (1 - q ** n) / p) == 0)
record("M1 and -> 1/p as n -> oo for 0 < p < 1",
       sp.limit(((1 - q ** n) / p).subs(p, sp.Rational(1, 3)), n, sp.oo) == 3)

# M2 ------------------------------------------------------------------------------------------
dl = sp.symbols("delta", positive=True)
mu_ball = (2 * dl) / (2 * sp.pi)
record("M2 uniform S^1: 1/mu(B_delta) = pi/delta", sp.simplify(1 / mu_ball - sp.pi / dl) == 0)
rng = np.random.default_rng(0)
delta, counts = 0.3, []
for _ in range(300):
    th = rng.uniform(0, 2 * np.pi, 1000)
    G = np.abs(np.angle(np.exp(1j * (th[:, None] - th[None, :]))))
    G[np.triu_indices(1000)] = np.inf
    counts.append(int((G.min(1) > delta).sum()))
cnt = float(np.mean(counts))
record("M2 mean simulated strong count on S^1 (300 runs, n 1000, delta 0.3) within 3 % of pi/delta",
       abs(cnt - np.pi / delta) / (np.pi / delta) < 0.03, f"{cnt:.2f} against {np.pi / delta:.2f}")

# M3 ------------------------------------------------------------------------------------------
ok_free, ok_first = True, True
for seed in range(3):
    r = np.random.default_rng(10 + seed)
    X = r.uniform(0, 1, 7)
    D = np.abs(X[:, None] - X[None, :])
    delta = 0.15
    a = (D <= delta).sum(1) - 1
    want = sum(Fraction(1, int(ai) + 1) for ai in a)
    perms = list(itertools.permutations(range(7)))
    got = Fraction(sum(len(strong(D, list(o), delta)) for o in perms), len(perms))
    ok_free &= got == want
    # position 0 held first, the rest permuted
    near0 = D[0] <= delta
    want0 = 1 + sum(Fraction(1, int((D[x, 1:] <= delta).sum() - 1) + 1) for x in range(1, 7) if not near0[x])
    perms0 = [[0] + list(o) for o in itertools.permutations(range(1, 7))]
    got0 = Fraction(sum(len(strong(D, o, delta)) for o in perms0), len(perms0))
    ok_first &= got0 == want0
record("M3 E_orders[#strong] = sum_x 1/(a_x+1), every order enumerated (n 7, 3 clouds)", ok_free)
record("M3 with position 0 held first: 1 + sum over x outside B(x_0) of 1/(a'_x+1)", ok_first)

# M4 ------------------------------------------------------------------------------------------
sub, cover = True, True
for s in range(200):
    r = np.random.default_rng(100 + s)
    U = r.normal(size=(60, 5))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    D = np.arccos(np.clip(U @ U.T, -1, 1))
    delta = r.uniform(0.3, 1.5)
    order = list(range(60))
    S, O = strong(D, order, delta), ordinary(D, order, delta)
    sub &= set(S) <= set(O)
    rest = [j for j in order if j not in O]
    cover &= all(min(D[j, i] for i in O) <= delta for j in rest)
record("M4 strong is a subset of ordinary (200 clouds)", sub)
record("M4 every non-centre lies within delta of an ordinary centre (200 clouds)", cover)

print(f"\n{sum(results)}/{len(results)} passed")
raise SystemExit(0 if all(results) else 1)
