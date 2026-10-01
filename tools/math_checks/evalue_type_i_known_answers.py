"""Checks the Type-I known answers in tests/test_core_evalues_contract.py.

With the calibrator e = kappa * p**(kappa - 1) and rejection at E >= 1/alpha,
p ~ Uniform(0, 1) under the null:
  (a) one experiment rejects with probability (alpha * kappa)**(1/(1 - kappa));
      at alpha = kappa = 1/2 that is 1/16 = 0.0625 (RATE_1);
  (b) two independent experiments, product, kappa = 1/2, reject iff
      p1 * p2 <= c with c = (alpha/4)**2, with probability c * (1 - log c);
      at alpha = 1/2 that is (1 + 2 log 8) / 64 = 0.0806 (RATE_2);
  (c) n copies of one p (maximal dependence): the mean of the n equal
      e-values is that e-value, so the rate is (a)'s; the product rejects iff
      p <= (kappa * alpha**(1/n))**(1/(1 - kappa)), 0.1967 at n = 25,
      alpha = 1/20, kappa = 1/2 (RATE_PRODUCT_25);
  (d) the tests' tolerances are at least 3.2 standard errors of a binomial
      proportion at their trial counts;
  (e) two independent p's merged by the mean, alpha = kappa = 1/2, reject iff
      p1**-1/2 + p2**-1/2 >= 8, with probability 41/896 + 3 log 7 / 1024 =
      0.0515 (RATE_MEAN_2); the max would reject at 1 - (15/16)**2 = 0.121.

What this does NOT prove: that the code computes these (the tests measure
that, by simulation, through EProcess, average_p and combine); (b) for any
kappa other than 1/2; or anything about the helpers simulate_type_i_error*,
which re-derive the calibrator in numpy.

Run: python3 tools/math_checks/evalue_type_i_known_answers.py
"""
import sympy as sp

results = []


def record(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"      {detail}")
    results.append(bool(ok))


p, p1, c = sp.symbols("p p1 c", positive=True)
alpha, kappa = sp.symbols("alpha kappa", positive=True)
half = sp.Rational(1, 2)

# (a) e >= 1/alpha  <=>  p <= p*, and P(U <= p*) = p* for p* in (0, 1].
p_star = (alpha * kappa) ** (1 / (1 - kappa))
e = kappa * p ** (kappa - 1)
at = {alpha: sp.Rational(1, 3), kappa: sp.Rational(2, 5)}
record("(a) e(p*) = 1/alpha at a generic point",
       sp.simplify(e.subs(p, p_star).subs(at) - 1 / at[alpha]) == 0)
rate_1 = p_star.subs({alpha: half, kappa: half})
record("(a) rate at alpha = kappa = 1/2 is 1/16", sp.simplify(rate_1 - sp.Rational(1, 16)) == 0)

# (b) E = (1/4) (p1 p2)^(-1/2) >= 1/alpha  <=>  p1 p2 <= (alpha/4)^2 = c.
E2 = half * p1 ** (-half) * half * p ** (-half)
record("(b) E2 >= 1/alpha  <=>  p1 p2 <= (alpha/4)^2",
       sp.simplify(E2.subs(p, (alpha / 4) ** 2 / p1) - 1 / alpha) == 0)
# P(p1 p2 <= c) = int_0^1 min(1, c/p1) dp1 = c + int_c^1 c/p1 dp1.
prob = c + sp.integrate(c / p1, (p1, c, 1))
record("(b) P(p1 p2 <= c) = c (1 - log c)", sp.simplify(prob - c * (1 - sp.log(c))) == 0)
rate_2 = (c * (1 - sp.log(c))).subs(c, (half / 4) ** 2)
record("(b) rate at alpha = 1/2 is (1 + 2 log 8) / 64",
       sp.simplify(rate_2 - (1 + 2 * sp.log(8)) / 64) == 0)
record("(b) RATE_2 = 0.0806 to 4 d.p.", abs(float(rate_2) - 0.0806) < 5e-5, f"{float(rate_2)}")

# (c) product of n copies: e^n >= 1/alpha  <=>  e >= alpha^(-1/n).
n = 25
p_prod = (kappa * alpha ** sp.Rational(1, n)) ** (1 / (1 - kappa))
record("(c) product threshold solves e(p)^n = 1/alpha",
       sp.simplify((e.subs(p, p_prod) ** n).subs(at) - 1 / at[alpha]) == 0)
rate_prod = p_prod.subs({alpha: sp.Rational(1, 20), kappa: half})
record("(c) RATE_PRODUCT_25 = 0.1967 to 4 d.p.", abs(float(rate_prod) - 0.1967) < 5e-5,
       f"{float(rate_prod)}")

# (e) a = p^(-1/2) has P(a >= t) = t^-2 on [1, inf), density 2 a^-3; the mean
# of e = a/2 over two reaches 1/alpha = 2 iff a1 + a2 >= 8, and a2 >= 1.
a = sp.symbols("a", positive=True)
rate_mean_2 = 1 - sp.integrate(2 * a ** -3 * (1 - (8 - a) ** -2), (a, 1, 7))
record("(e) mean of two: 41/896 + 3 log 7 / 1024",
       sp.simplify(rate_mean_2 - (sp.Rational(41, 896) + 3 * sp.log(7) / 1024)) == 0)
record("(e) RATE_MEAN_2 = 0.0515 to 4 d.p.", abs(float(rate_mean_2) - 0.0515) < 5e-5,
       f"{float(rate_mean_2)}")
rate_max_2 = 1 - (1 - rate_1) ** 2
record("(e) the max rejects at 31/256, far from the mean",
       rate_max_2 == sp.Rational(31, 256) and float(rate_max_2 - rate_mean_2) > 0.05)

# (d) tolerance / SE for each (rate, n_trials, tolerance) in the tests.
for name, rate, trials, tol in [("RATE_1, helper", rate_1, 40_000, 0.004),
                                ("RATE_MEAN_2, average_p", rate_mean_2, 40_000, 0.0037),
                                ("RATE_1, EProcess", rate_1, 40_000, 0.0045),
                                ("RATE_2, EProcess", rate_2, 40_000, 0.0045),
                                ("RATE_1, average_p", rate_1, 10_000, 0.008),
                                ("RATE_PRODUCT_25, combine", rate_prod, 10_000, 0.013)]:
    r = float(rate)
    se = (r * (1 - r) / trials) ** 0.5
    record(f"(d) {name}: tolerance {tol} = {tol / se:.1f} SE", tol / se >= 3.2)

print(f"\n{sum(results)}/{len(results)} passed")
raise SystemExit(0 if all(results) else 1)
