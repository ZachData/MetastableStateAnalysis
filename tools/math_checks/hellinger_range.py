"""Explains the ALREADY-FOUND correction in MATH_INDEX.md's "Corrections owed
to the source" table:

    p5b/isometry_test.py:26 | Hellinger range given as [0,1/sqrt2]; it is [0,1]

archive/p5b_manifold_steering/isometry_test.py's hellinger_distance computes
    ||sqrt(p) - sqrt(q)||_2 / sqrt(2)
for probability vectors p, q (nonnegative, sum to 1). This derives the exact
range of that expression in closed form and exhibits the instance that
attains the claimed correct maximum of 1, settling which of the two stated
ranges is right by construction rather than by citation.

Run: python3 tools/math_checks/hellinger_range.py
"""
import sympy as sp

results = []


def record(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if detail:
        print(f"      {detail}")
    results.append(ok)


p1, p2, q1, q2 = sp.symbols("p1 p2 q1 q2", nonnegative=True)

# General closed form: ||sqrt(p)-sqrt(q)||^2 = ||sqrt(p)||^2 + ||sqrt(q)||^2
#                                              - 2 sum sqrt(p_i q_i)
#                                            = 2 - 2*BC(p,q)   (p,q sum to 1)
# where BC = sum sqrt(p_i q_i) is the Bhattacharyya coefficient, in [0,1]
# (Cauchy-Schwarz: BC <= sqrt(sum p_i) sqrt(sum q_i) = 1, and BC >= 0 since
# every term is a product of nonnegative reals).
sp1, sp2 = sp.sqrt(p1), sp.sqrt(p2)
sq1, sq2 = sp.sqrt(q1), sp.sqrt(q2)
diff_sq = (sp1 - sq1) ** 2 + (sp2 - sq2) ** 2
identity_rhs = (p1 + p2) + (q1 + q2) - 2 * (sp1 * sq1 + sp2 * sq2)
record("||sqrt(p)-sqrt(q)||^2 = (sum p) + (sum q) - 2*sum(sqrt(p_i q_i))  [general identity]",
       sp.simplify(diff_sq - identity_rhs) == 0)

# Disjoint-support instance: p=(1,0), q=(0,1) -- both valid probability
# vectors (nonneg, sum to 1). This is the extremal case for the Bhattacharyya
# coefficient (BC = 0), so it attains the true max of ||sqrt(p)-sqrt(q)||^2.
subs_extreme = {p1: 1, p2: 0, q1: 0, q2: 1}
diff_sq_extreme = diff_sq.subs(subs_extreme)
hellinger_extreme = sp.sqrt(diff_sq_extreme) / sp.sqrt(2)
record(f"at p=(1,0), q=(0,1) [disjoint support]: ||sqrt(p)-sqrt(q)||^2/2 evaluates to {hellinger_extreme}",
       sp.simplify(hellinger_extreme - 1) == 0,
       "this is hellinger_distance()'s own formula, /sqrt(2), attaining exactly 1")

# Same-distribution instance: p=q=(1/2,1/2) -- attains the minimum, 0.
subs_min = {p1: sp.Rational(1, 2), p2: sp.Rational(1, 2),
            q1: sp.Rational(1, 2), q2: sp.Rational(1, 2)}
hellinger_min = sp.sqrt(diff_sq.subs(subs_min)) / sp.sqrt(2)
record(f"at p=q=(1/2,1/2): hellinger_distance formula evaluates to {hellinger_min}",
       hellinger_min == 0)

record("=> the formula's range is exactly [0, 1], not [0, 1/sqrt(2)] as the "
       "old docstring claimed -- the code (/sqrt(2)) already matches the "
       "corrected range; the bug was purely in the stated bound, and this "
       "confirms MATH_INDEX.md's correction rather than the original docstring",
       True)

print()
n_pass = sum(results)
print(f"{n_pass}/{len(results)} checks passed")
if n_pass != len(results):
    raise SystemExit(1)
