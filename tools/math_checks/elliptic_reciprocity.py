"""Checks MATH_SPECTRAL_OT.md sec 3's reciprocity parameter tau boundary cases.

    tau = sum_ij V_ij V_ji / sum_ij V_ij^2

The doc asserts tau=1 for symmetric V, tau=0 for Ginibre (generic, no
constraint), tau=-1 for antisymmetric V. Only the two exact endpoints are
algebraic facts checkable in closed form; "tau=0 for Ginibre" is a statement
about a random ensemble's expectation, not an identity, and is out of scope
here (that one needs simulation, which the doc already says is still the
right check to run).

Checked symbolically for a generic n x n matrix (n=4): both endpoints proven
for ANY n by the same argument, so n=4 is illustrative, not load-bearing.

Run: python3 tools/math_checks/elliptic_reciprocity.py
"""
import sympy as sp

results = []


def record(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"      {detail}")
    results.append(ok)


n = 4


def tau(V):
    num = sum(V[i, j] * V[j, i] for i in range(n) for j in range(n))
    den = sum(V[i, j] ** 2 for i in range(n) for j in range(n))
    return sp.simplify(num / den)


# Symmetric case: build a generic symmetric matrix and check tau = 1.
sym_entries = {}
Vs = sp.zeros(n, n)
for i in range(n):
    for j in range(i, n):
        s = sp.Symbol(f"s{i}{j}")
        Vs[i, j] = s
        Vs[j, i] = s
tau_sym = tau(Vs)
record("sec 3: tau = 1 for a generic symmetric V", tau_sym == 1,
       f"got {tau_sym}")

# Antisymmetric case: generic antisymmetric matrix, check tau = -1.
Va = sp.zeros(n, n)
for i in range(n):
    for j in range(i + 1, n):
        a = sp.Symbol(f"a{i}{j}")
        Va[i, j] = a
        Va[j, i] = -a
tau_anti = tau(Va)
record("sec 3: tau = -1 for a generic antisymmetric V", tau_anti == -1,
       f"got {tau_anti}")

# Cross-check against the doc's own S/A machinery: V = S + A with S symmetric,
# A antisymmetric, tau should interpolate and hit +/-1 only at the pure cases
# (a basic sanity check that tau is genuinely reading the S/A mix, not some
# other property) -- verified at a single generic mixed instance that tau is
# strictly between -1 and 1 when both S and A are present with comparable
# scale, using concrete rational entries so the inequality is decidable.
Vm = sp.Matrix([[2, 3, 1, 0], [1, 2, 0, 1], [4, 1, 2, 2], [0, 3, 1, 2]])
tau_mixed = tau(Vm)
record(f"sec 3: a generic mixed V has tau strictly in (-1, 1) (got {tau_mixed})",
       bool(sp.Rational(-1) < tau_mixed) and bool(tau_mixed < sp.Rational(1)))

print()
n_pass = sum(results)
print(f"{n_pass}/{len(results)} checks passed")
if n_pass != len(results):
    raise SystemExit(1)
