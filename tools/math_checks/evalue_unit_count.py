"""Checks the unit-count arithmetic in docs/PHASE_SYNTHESIS.md "Attainable E":

    With the Vovk-Wang calibrator e = kappa * p**(kappa - 1), kappa = 1/2, and
    rejection at E >= 1/alpha = 20:
      (a) a single p reaches E >= 20 iff p <= 1/1600;
      (b) a Monte-Carlo permutation p (floor 1/(B+1)) reaches it iff B >= 1599;
      (c) an exact one-sided sign-flip over n informative units (floor 2**-n)
          reaches it iff n >= 11 -- so 8 prompts cannot and 12 can, with one to
          spare;
      (d) the E_max values the table quotes for B = 500, 2000 (floor 1/2001),
          5000 (floor 1/5001) and n = 8, 12.

What this does NOT prove: that a given gate's floor IS 2**-n or 1/(B+1). That
is each gate's own dry-run record (claims/audits/, claims/calibration/). CLAIM-C's
homogeneity correction raises its floor above 2**-n (0.0661 against 0.0078 at
n = 8), so (c) is a necessary condition for CLAIM-C, not a sufficient one.

Run: python3 tools/math_checks/evalue_unit_count.py
"""
import sympy as sp

results = []


def record(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"      {detail}")
    results.append(ok)


p, kappa = sp.symbols("p kappa", positive=True)
k = sp.Rational(1, 2)
E = kappa * p ** (kappa - 1)
Ek = E.subs(kappa, k)

# (a) threshold p. E is decreasing in p for kappa < 1, so E >= 20 iff p <= p*.
p_star = sp.solve(sp.Eq(Ek, 20), p)
record("(a) E = 20 at p = 1/1600", p_star == [sp.Rational(1, 1600)], f"got {p_star}")
dE = sp.diff(Ek, p)
record("(a) dE/dp = -p^(-3/2)/4 < 0, so E is decreasing in p",
       sp.simplify(dE + p ** sp.Rational(-3, 2) / 4) == 0, f"got {dE}")

# (b) permutation draws.
B = sp.symbols("B", positive=True, integer=True)
Bmin = next(b for b in range(1, 5000) if Ek.subs(p, sp.Rational(1, b + 1)) >= 20)
record("(b) smallest B with E_max >= 20 is 1599", Bmin == 1599, f"got {Bmin}")

# (c) sign-flip units.
nmin = next(n for n in range(1, 64) if Ek.subs(p, sp.Rational(1, 2 ** n)) >= 20)
record("(c) smallest n with 2^-n floor reaching E >= 20 is 11", nmin == 11, f"got {nmin}")

# (d) the quoted E_max values, to 2 decimals.
quoted = {
    "B=500": (sp.Rational(1, 501), 11.19),
    "B=2000": (sp.Rational(1, 2001), 22.37),
    "B=5000": (sp.Rational(1, 5001), 35.36),
    "n=8": (sp.Rational(1, 2 ** 8), 8.00),
    "n=12": (sp.Rational(1, 2 ** 12), 32.00),
    "CLAIM-C n=8 corrected 0.0661": (sp.Rational(661, 10000), 1.94),
    "CLAIM-C n=12 raw 0.000488": (sp.Rational(488162069807176, 10 ** 18), 22.63),
    "CLAIM-B floor 1/20": (sp.Rational(1, 20), 2.24),
    "P-I3 4 controls, 4 sets": (sp.Rational(1, 5 ** 4), 12.50),
    "P-I3 4 controls, 5 sets": (sp.Rational(1, 5 ** 5), 27.95),
}
for name, (pv, want) in quoted.items():
    got = float(Ek.subs(p, pv))
    record(f"(d) E_max {name} = {want}", abs(got - want) < 0.01, f"got {got:.4f}")

print(f"\n{sum(results)}/{len(results)} passed")
raise SystemExit(0 if all(results) else 1)
