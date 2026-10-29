"""Checks the closed-form claims in trackT_timaeus/design-T2.md (T2: V and E_β on Dodecahedron).

410m inside its warmup (W_p = 1430, peak p_p = 3e-4, 2,097,152 tokens a step): lr_p(t) = p_p t / W_p.
Dodecahedron (card: W_d = 1000, peak p_d = 3.5e-3, linear warmup from 0; tokens a step 524,288 by
the card's micro-batch arithmetic or 4,194,304 by its 8 GPUs and 40B total): lr_d(s) = p_d s / W_d.

T1  Equal cumulative LR (p_p t² / 2W_p = p_d s² / 2W_d) gives s = t √(p_p W_d / (p_d W_p)); equal
    instantaneous LR gives s = t p_p W_d / (p_d W_p) (sympy).
T2  The discrete sums (Σ_{i ≤ t} p i / W) give the same grid predictions as the continuous forms,
    for both counts (numerical).
T3  Every 410m reference date sits inside 410m's warmup, and its mapped Dodecahedron step under
    each warmup-dependent schedule (cumulative, instantaneous, fraction) inside Dodecahedron's, so
    the linear forms are the ones in force (numerical).
T4  `predicted` (interval intersection on the grid) equals a brute-force scan of the mapped
    crossing intervals (numerical).
T5  The plateau lies past every schedule's latest predicted date at either batch, and EARLY
    before every schedule's earliest (numerical).
Then prints the outcome table: per count and Dodecahedron grid date, the schedules consistent with
it, with tokens at each batch as its own column.

NOT proved: that the break follows any schedule (cumulative LR stands in for how far Adam has moved
the weights, about lr per coordinate per update; an assumption); that Dodecahedron's schedule is
the card's (warmup from 0, linear, 1000 steps: card only, no training log read); which batch is
true; that 410m's seed spread is Dodecahedron's (`REF` is 410m's 10 seeds); anything about the
architecture (Llama, RMSNorm, GQA), the data (FineWeb) or the tokenizer, which all differ.

Run: python3 tools/math_checks/t2_grid.py
"""
import math
import sys

import numpy as np
import sympy as sp

P_LR, P_WARMUP, P_TOKENS = 3e-4, 1430, 2_097_152
D_LR, D_WARMUP = 3.5e-3, 1000
BATCHES = {"tok524k": 524_288, "tok4M": 4_194_304}
P_GRID = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000)   # 410m's grid up to 4000
GRID = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 19, 22, 25, 30, 35, 40, 47, 55, 65, 75, 88, 103,
        121, 141, 165, 193, 225, 263, 308, 359, 420, 491, 574, 671, 785, 917, 1072, 1253, 1465,
        1713, 2002, 2340, 2735, 3198, 3738, 4369, 5108, 5970, 6979, 8158, 9536)
REF = {"u": (256, 512), "c": (512, 1000)}      # 410m's 10 seeds: E₁'s break; the centred count
HYPS = ("instantaneous", "cumulative", "fraction", "step", "tok4M", "tok524k")
PLATEAU = (5108, 5970, 6979, 8158)             # past every predicted date; 9536 is the preflight's step
EARLY = (1, 2)

results = []


def record(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if detail:
        print("      " + detail)
    results.append(ok)


def to_dodeca(hyp, t):
    if hyp == "step":
        return t
    if hyp in BATCHES:
        return t * P_TOKENS / BATCHES[hyp]
    if hyp == "fraction":
        return t * D_WARMUP / P_WARMUP
    if hyp == "cumulative":
        return t * math.sqrt(P_LR * D_WARMUP / (D_LR * P_WARMUP))
    if hyp == "instantaneous":
        return t * P_LR * D_WARMUP / (D_LR * P_WARMUP)
    raise ValueError(hyp)


def p_prev(g):
    return P_GRID[P_GRID.index(g) - 1]


def prev(s):
    i = GRID.index(s)
    if i == 0:
        raise ValueError("step 1 has no interval: a half-rise at 1 is unreadable")
    return GRID[i - 1]


def predicted(ref, hyp):
    out = set()
    for g in ref:
        lo, hi = to_dodeca(hyp, p_prev(g)), to_dodeca(hyp, g)
        out |= {s for s in GRID[1:] if s > lo and prev(s) < hi}
    return frozenset(out)


def consistent(dates, kind):
    ds = set(dates)
    return frozenset(h for h in HYPS if ds & predicted(REF[kind], h))


t, s, pp, pd, Wp, Wd = sp.symbols("t s p_p p_d W_p W_d", positive=True)
sol = sp.solve(sp.Eq(pp * t**2 / (2 * Wp), pd * s**2 / (2 * Wd)), s)
record("T1 equal cumulative LR: s = t √(p_p W_d / (p_d W_p))",
       len(sol) == 1 and sp.simplify(sol[0] - t * sp.sqrt(pp * Wd / (pd * Wp))) == 0, str(sol))
sol = sp.solve(sp.Eq(pp * t / Wp, pd * s / Wd), s)
record("T1 equal instantaneous LR: s = t p_p W_d / (p_d W_p)",
       len(sol) == 1 and sp.simplify(sol[0] - t * pp * Wd / (pd * Wp)) == 0, str(sol))


def grid(y):
    return min(x for x in GRID if x >= y)


cd = np.cumsum([0.0] + [D_LR * i / D_WARMUP for i in range(1, D_WARMUP + 1)])
ok = True
for kind in ("u", "c"):
    disc = set()
    for g in REF[kind]:
        for tt in np.linspace(p_prev(g), g, 4001)[1:]:
            target = P_LR * tt * (tt + 1) / (2 * P_WARMUP)
            disc.add(grid(max(1, int(np.searchsorted(cd, target)))))
    a = predicted(REF[kind], "cumulative")
    ok &= a == frozenset(disc)
    if a != frozenset(disc):
        print(f"      {kind}: continuous {sorted(a)} discrete {sorted(disc)}")
record("T2 the discrete sums give the same grid predictions", ok)

top = max(max(v) for v in REF.values())
record("T3 reference dates and their mapped steps are inside the warmups",
       top < P_WARMUP and all(to_dodeca(h, top) <= D_WARMUP for h in ("cumulative", "instantaneous", "fraction")),
       "; ".join(f"{h} {to_dodeca(h, top):.1f}" for h in HYPS))

ok = True
for kind in ("u", "c"):
    for h in HYPS:
        fine = set()
        for g in REF[kind]:
            lo, hi = to_dodeca(h, p_prev(g)), to_dodeca(h, g)
            fine |= {grid(y) for y in np.linspace(lo, hi, 20001)[1:]}
        ok &= predicted(REF[kind], h) == frozenset(fine)
record("T4 predicted equals a brute-force scan", ok)

allp = [x for k in REF for h in HYPS for x in predicted(REF[k], h)]
record("T5 the plateau is past, and EARLY before, every predicted date",
       max(allp) < min(PLATEAU) and max(EARLY) < min(allp), f"predicted dates span {min(allp)}–{max(allp)}")

print("\nCrossing bands (Dodecahedron steps) for 410m's (128, 512] and (256, 1000]:")
for h in HYPS:
    print(f"  {h:<13} E1 ({to_dodeca(h, 128):.1f}, {to_dodeca(h, 512):.1f}]   centred "
          f"({to_dodeca(h, 256):.1f}, {to_dodeca(h, 1000):.1f}]")
print("\nOutcome table: per count, the schedules consistent with each grid date (others: none)")
for kind, name in (("u", "E1 break"), ("c", "centred")):
    rows, last = [], None
    for x in GRID[1:]:
        cons = consistent([x], kind)
        key = "/".join(h for h in HYPS if h in cons)
        if key:
            rows.append(f"{x}: {key}")
    print(f"  {name}:\n    " + "\n    ".join(rows))

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
