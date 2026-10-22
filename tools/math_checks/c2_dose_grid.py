"""Checks the closed-form claims in trackC_writeup/design-C2.md "Unit 3b: the LR-dose arm" (C2, 99 (a)).

Pythia's schedule inside the warmup (W = 1430 steps, every size): lr(t) = p t / W.

D1  Equal cumulative LR. sum_{s<=t} lr(s) is p t^2 / 2W (continuous), so a size of peak p'
    matches 410m's step t at t' = t sqrt(p / p') (sympy). The discrete sum p t (t + 1) / 2W gives
    the same grid predictions for both sizes at both counts (numerical).
D2  Equal instantaneous LR: t' = t p / p' (sympy).
D3  Both maps stay inside the warmup for every 410m reference date (t <= 1000 < W), so the linear
    form is the one in force (numerical).
D4  `c2_dose.predicted` equals a brute-force scan of the scaled crossing intervals (numerical),
    and the outcome table it implies is printed: per size, count and grid date, the schedules
    consistent with it.

NOT proved: that the break follows either dose (they are stylised: Adam's step is about lr per
coordinate, so displacement ~ cumulative lr is an assumption, and the instantaneous form says the
break waits for an LR level); that 70m's and 160m's seed spread is 410m's (`REF` is 410m's 10
seeds); anything about size (width, depth), which moves with the LR throughout.

Run: python3 tools/math_checks/c2_dose_grid.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trackA_force.a1a import STEPS  # noqa: E402
from trackC_writeup import c2_dose as d  # noqa: E402

results = []


def record(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if detail:
        print("      " + detail)
    results.append(ok)


t, tp, p, pp, W = sp.symbols("t t' p p' W", positive=True)
cum = sp.integrate(p * t / W, (t, 0, t))
sol = sp.solve(sp.Eq(cum.subs(p, pp).subs(t, tp), cum), tp)
record("D1 equal cumulative LR: t' = t sqrt(p/p')", len(sol) == 1 and sp.simplify(sol[0] - t * sp.sqrt(p / pp)) == 0,
       str(sol))
sol2 = sp.solve(sp.Eq(pp * tp / W, p * t / W), tp)
record("D2 equal instantaneous LR: t' = t p/p'", len(sol2) == 1 and sp.simplify(sol2[0] - t * p / pp) == 0, str(sol2))


def grid(x):
    return min(s for s in STEPS if s >= x)


def discrete_pred(ref, p_ref, p_new):
    """Grid steps where the discrete cumulative LR of the new size crosses 410m's at a ref crossing."""
    out = set()
    for g in ref:
        lo, hi = d.prev(g), g
        for x in np.linspace(lo, hi, 2001)[1:]:
            target = p_ref * x * (x + 1)                 # 2W cancels
            y = (-1 + math.sqrt(1 + 4 * target / p_new)) / 2
            out.add(grid(y))
    return frozenset(out)


ok = True
for size, (_, lr) in d.SIZES.items():
    for kind in ("u", "c"):
        a = d.predicted(d.REF[kind], d.factor("cumulative", lr))
        b = discrete_pred(d.REF[kind], d.REF_LR, lr)
        ok &= a == b
        if a != b:
            print(f"      {size} {kind}: continuous {sorted(a)} discrete {sorted(b)}")
record("D1 the discrete cumulative sum gives the same grid predictions", ok)

record("D3 every reference date is inside the warmup", max(max(v) for v in d.REF.values()) < d.WARMUP)

ok = True
for size, (_, lr) in d.SIZES.items():
    for kind in ("u", "c"):
        for h in d.HYPS:
            f = d.factor(h, lr)
            fine = {grid(x) for g in d.REF[kind] for x in np.linspace(f * d.prev(g), f * g, 20001)[1:]}
            ok &= d.predicted(d.REF[kind], f) == frozenset(fine)
record("D4 predicted equals a brute-force scan", ok)

print("\nOutcome table: per size and count, the schedules consistent with each grid date (others: none)")
for size, (_, lr) in d.SIZES.items():
    for kind, name in (("u", "E1 break"), ("c", "centred")):
        rows = []
        for s in STEPS[1:]:
            cons = d.consistent([s], kind, lr)
            if cons:
                rows.append(f"{s}: {'/'.join(h for h in d.HYPS if h in cons)}")
        print(f"  {size:>4} {name:<8} " + "; ".join(rows))

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
