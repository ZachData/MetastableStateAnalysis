"""Checks the closed-form claims in trackC_writeup/design-C6.md (C6 U0, the bands per schedule).

NeoX's rate (`c6_warmup.lr`): p x / W for x <= W, then cosine from p to p/10 at D = 143 000.

B1  Inside both warmups, equal summed rate maps 70m's step t to the run's t' = t sqrt(W_r / W_b)
    and equal rate to t' = t W_r / W_b at a shared peak (sympy); with W_r = 10 W_b, sqrt(10) and 10.
B2  If only the ratio p / W is pinned (as the early weights pin it, `c6_schedule.py`), the
    in-warmup maps do not depend on the peak itself (sympy: t' in terms of p_r / W_r only).
B3  `c6_warmup.cum` equals the integral of `c6_warmup.lr` after the warmup (sympy, closed form),
    and the discrete NeoX sum within 1/t + 0.1 % at the steps the maps use (numerical).
B4  `c6_warmup.band` equals a fine scan of m_h over each reference interval (numerical), the
    instantaneous map's turn at 70m's warmup end included.
B5  Under "step" the window and early set are unit 3b's (1000, 2000, 4000; steps <= 128).

NOT proved: that a date follows any of the three schedules (Adam's step ~ lr per coordinate makes
"summed rate" a stylised dose; "the rate reaches a level" is another); that the run's warmup is
14 300 at Pythia's peak (`c6_schedule.py` reads that off the weights, approximately for the peak);
that 70m's ten seeds span the run's own init-and-order spread (they vary both; the run shares
seed 0's init and first batch, so the union is conservative).

Run: python3 tools/math_checks/c6_bands.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trackA_force.a1a import STEPS  # noqa: E402
from trackC_writeup import c6_warmup as w  # noqa: E402
from trackC_writeup.c6_schedule import cum as neox_cum  # noqa: E402

results = []


def record(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if detail:
        print("      " + detail)
    results.append(ok)


t, tp, p, pr, Wb, Wr, x, u, L, mn = sp.symbols("t t' p p_r W_b W_r x u L m", positive=True)
c_b = sp.integrate(p * x / Wb, (x, 0, t))
c_r = sp.integrate(p * x / Wr, (x, 0, tp))
s1 = sp.solve(sp.Eq(c_r, c_b), tp)
s2 = sp.solve(sp.Eq(p * tp / Wr, p * t / Wb), tp)
record("B1 cumulative: t' = t sqrt(W_r/W_b)", len(s1) == 1 and sp.simplify(s1[0] - t * sp.sqrt(Wr / Wb)) == 0, str(s1))
record("B1 instantaneous: t' = t W_r/W_b", len(s2) == 1 and sp.simplify(s2[0] - t * Wr / Wb) == 0, str(s2))
k = sp.symbols("k", positive=True)                       # k = p_r / W_r, the pinned ratio
s3 = sp.solve(sp.Eq(k * tp ** 2 / 2, c_b), tp)
s4 = sp.solve(sp.Eq(k * tp, p * t / Wb), tp)
record("B2 in-warmup maps depend on p_r / W_r only", len(s3) == 1 and len(s4) == 1
       and not (s3[0].free_symbols | s4[0].free_symbols) & {pr, Wr}, f"{s3}, {s4}")

post = mn + (p - mn) / 2 * (1 + sp.cos(sp.pi * u / L))
closed = mn * u + (p - mn) / 2 * (u + L / sp.pi * sp.sin(sp.pi * u / L))
record("B3 the cosine part's integral is the closed form",
       sp.simplify(sp.integrate(post.subs(u, x), (x, 0, u)) - closed) == 0)
worst = max(abs(w.cum(s, W) - neox_cum(0, s, w.PEAK, W)) / neox_cum(0, s, w.PEAK, W) - 1 / s
            for W in (w.W_BASE, w.W_RUN) for s in (128, 512, 1000, 2000, 4000, 8000, 16000, 32000))
record("B3 continuous cum within 1/t + 0.1 % of NeoX's discrete sum (t(t-1) against t^2 in warmup)",
       worst < 1e-3, f"worst excess over 1/t {worst:.2e}")


def fine_band(g, h):
    ts = np.linspace(w.prev(g), g, 4001)
    ms = [w.m(h, float(t_)) for t_ in ts]
    lo, hi = min(ms), max(ms)
    return {s for s in STEPS[1:] if s > lo and w.prev(s) < hi}


ok = all(w.band([g], h) == frozenset(fine_band(g, h)) for g in STEPS[1:12] for h in w.HYPS)
record("B4 band equals a fine scan (dates 2 … 2000)", ok)
record("B5 step's window and early set are unit 3b's",
       w.window("step") == (1000, 2000, 4000) and w.early("step") == tuple(s for s in STEPS if s <= 128))

print("\nper 70m date, the run's grid steps under each schedule:")
for g in STEPS[1:13]:
    print(f"  {g:>5}: " + "; ".join(f"{h} {sorted(w.band([g], h))}" for h in w.HYPS))
print("windows:", {h: w.window(h) for h in w.HYPS}, " early tops:", {h: w.early(h)[-1] for h in w.HYPS})
sys.exit(0 if all(results) else 1)
