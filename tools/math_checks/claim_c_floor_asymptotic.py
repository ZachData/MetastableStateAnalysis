"""Checks POPPER_PLAN.md sec 6l's floor formula (CLAIM-C's cell-drop gate):

    floor = (2^(n-k) + 1) / (2^n + 1)  ~=  2^-k

and the stated special case "= 2/(2^n+1) exactly when k = n".

Both checked exactly and symbolically -- these are finite-n algebraic facts
and an asymptotic limit, not instance checks.

Run: python3 tools/math_checks/claim_c_floor_asymptotic.py
"""
import sympy as sp

results = []


def record(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"      {detail}")
    results.append(ok)


n, k = sp.symbols("n k", positive=True, integer=True)

floor = (2 ** (n - k) + 1) / (2 ** n + 1)

# Special case k = n.
floor_at_k_eq_n = sp.simplify(floor.subs(k, n))
record("sec 6l: floor at k=n simplifies to 2/(2^n+1)",
       sp.simplify(floor_at_k_eq_n - 2 / (2 ** n + 1)) == 0,
       f"got {floor_at_k_eq_n}")

# The approximation claim itself: floor / 2^-k -> 1 as n -> infinity, i.e.
# floor approaches the nonzero constant 2^-k (NOT 0 -- an earlier version of
# this script wrongly checked floor -> 0 and failed, because for fixed k,
# 2^(n-k)/2^n = 2^-k is a nonzero constant as n -> infinity; that was a bug
# in the check, not in POPPER_PLAN.md's claim).
ratio = floor / (sp.Rational(1, 2) ** k)
ratio_limit = sp.limit(ratio, n, sp.oo)
record("sec 6l: floor / 2^-k -> 1 as n -> infinity (the stated approximation "
       "'floor ~= 2^-k' is exact in this limit)",
       ratio_limit == 1)

# Concrete finite-n spot check (n=10, k=3) against direct decimal evaluation,
# the kind of check the source text describes doing "against the gate's own
# enumeration at every k".
floor_num = floor.subs({n: 10, k: 3})
approx_num = sp.Rational(1, 2) ** 3
record(f"sec 6l: at n=10, k=3: floor = {sp.nsimplify(floor_num)} ~= 2^-3 = {approx_num} "
       f"(relative error {float(abs(floor_num - approx_num) / approx_num):.4%})",
       float(abs(floor_num - approx_num) / approx_num) < 0.01)

print()
n_pass = sum(results)
print(f"{n_pass}/{len(results)} checks passed")
if n_pass != len(results):
    raise SystemExit(1)
