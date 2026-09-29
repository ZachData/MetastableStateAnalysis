"""Lemma 5.1's lower bound on `c` at the two β the project has recorded --
`p10_cluster_function/math-10.md` §5.4, `p1d_cluster_ensemble/status-1d.md`
"Attention communities".

WHY THIS EXISTS
---------------
math-10 §5.4 inverts the saturating strong-centre count for `c = δ√β` and
reads the result against "Lemma 5.1 needs c > 1", at β = 0.50 ("scaled")
and β = 4.0 ("unscaled"). Two things change that reading (2026-09-26):

1. `c > 1` is the paper's sufficient condition for β > 1 only. The bound
   itself is `c > c_min(β) = √β · arccos((−1 + √(4β² + 1)) / (2β))`
   (2411.04990, Eq. 2). At β = 0.5 that is 0.809, not 1.
2. The 0.50 is β_raw / 8: the slope of the softmax's own input on unit LN1
   rows (4.00 on the same run) with the model's 1/√d_h divided out a second
   time. β_raw is the coefficient in `softmax(β⟨u_i, u_j⟩)`, the theory's β.

Checked: c_min at 0.5 and 4.0 (and 4.44, the median without the sink on the
Stage 0 run); c_min < 1 for all β > 1 and c_min → 1 as β → ∞ (so "c > 1
suffices" is right); c scales as √β at fixed δ, so the table's two columns
differ by √8; and which of the table's d_eff rows clear c_min at each β.

WHAT IT DOES NOT PROVE
----------------------
- Nothing about whether the i.i.d. spherically symmetric hypothesis behind
  the count holds for tokens (math-10 §5.4 holds it loosely; so does this).
- Nothing about which β is right for a trained multi-head model: it takes
  the two recorded medians as given. The single-β idealisation fits real
  attention at median R² 0.18 (`status-1c.md` finding 2).
- The table's `c` values are math-10's, recomputed here only through the
  √8 ratio, not from the Rényi-centre volume.
"""
import sympy as sp

results = []


def record(msg, ok, note=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + msg + (f"  ({note})" if note else ""))


b = sp.symbols("beta", positive=True)
c_min = sp.sqrt(b) * sp.acos((-1 + sp.sqrt(4 * b**2 + 1)) / (2 * b))

v05 = float(c_min.subs(b, sp.Rational(1, 2)))
v4 = float(c_min.subs(b, 4))
v444 = float(c_min.subs(b, sp.Rational(444, 100)))
record("c_min(0.5) = 0.809", abs(v05 - 0.809) < 5e-4, f"{v05:.4f}")
record("c_min(4.0) = 0.978", abs(v4 - 0.978) < 5e-4, f"{v4:.4f}")
record("c_min(4.44) < 1", v444 < 1, f"{v444:.4f}")

# sympy's `limit` returns 0 here (wrong: the argument is 1 - 1/(2β) + O(β^-2)
# and arccos(1 - ε) ~ √(2ε), so c_min -> 1). Checked at 50 digits instead.
far = [sp.N(c_min.subs(b, sp.Integer(10) ** k), 50) for k in (4, 8, 12)]
record("c_min -> 1 as beta -> oo (1e4, 1e8, 1e12)",
       all(abs(float(v) - 1) < 10 ** -(k // 2 + 1) for v, k in zip(far, (4, 8, 12))),
       ", ".join(f"{float(v):.12f}" for v in far))
grid = [1 + k * 0.25 for k in range(1, 400)]
record("c_min < 1 on beta in (1, 100]", all(float(c_min.subs(b, x)) < 1 for x in grid))

# math-10 §5.4's table: c = delta * sqrt(beta) at fixed delta
table = {2: (0.042, 0.120), 5: (0.411, 1.161), 8: (0.568, 1.608), 22: (0.793, 2.242)}
record("the two columns differ by sqrt(8)",
       all(abs(hi / lo - 8 ** 0.5) < 0.02 * 8 ** 0.5 for lo, hi in table.values()))
at05 = [d for d, (lo, _) in table.items() if lo > v05]
at4 = [d for d, (_, hi) in table.items() if hi > v4]
record("at beta = 0.5 no d_eff row clears c_min", at05 == [], f"margin at d_eff 22: {0.793 - v05:+.3f}")
record("at beta = 4.0 d_eff >= 5 clears c_min", at4 == [5, 8, 22], str(at4))

print(f"\n{sum(results)}/{len(results)} checks passed")
raise SystemExit(0 if all(results) else 1)
