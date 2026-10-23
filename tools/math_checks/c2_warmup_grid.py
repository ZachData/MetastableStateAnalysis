"""Checks the closed-form claims in trackC_writeup/design-C2.md "Unit 3c: the warmup arm" (C2, 92 (ii)).

410m inside its warmup (W_p = 1430, peak p_p = 3e-4): lr_p(t) = p_p t / W_p. GPT-2 small
(Mistral; W_g = 4000, p_g = 6e-4): lr_g = p_g ln(n) / ln W_g with n = t − K scheduler iterations.

W1  Equal cumulative LR: ∫₁ⁿ p_g ln x / ln W_g dx = p_g (n ln n − n + 1) / ln W_g (sympy), and
    equal instantaneous LR gives n = W_g^(p_p t / (W_p p_g)) (sympy).
W2  The log form fits every logged LR of the five seeds from step 50 to 4000
    (`trackC_writeup/gpt2_small_lr_log.json`): n = exp(lr ln W_g / p_g) is an integer to 1e-3 and
    K = t − n lies in `c2_warmup.KS`; the tokens per step implied by each seed's epoch count at
    4100 against OpenWebText's 9 035 582 198 GPT-2 tokens is G_TOKENS to 0.5 % (numerical).
W3  The discrete sums (Σ p_p s / W_p; Σ_{x ≤ n} p_g ln x / ln W_g) give the same grid predictions
    as the continuous forms, for both counts (numerical).
W4  Every 410m reference date and its mapped GPT-2 step sit inside the respective warmups, so the
    warmup forms are the ones in force (numerical).
W5  `c2_warmup.predicted` equals a brute-force scan of the mapped crossing intervals (numerical),
    and the outcome table it implies is printed: per count and GPT-2 grid date, the schedules
    consistent with it.

NOT proved: that the break follows any of the five schedules (cumulative LR stands in for how
far Adam has moved the weights, about lr per coordinate per update; an assumption); that GPT-2
small's seed spread is 410m's (`REF` is 410m's 10 seeds); anything about the model, the data,
the tokenizer or the 1/layer attention scaling, which all differ; where in steps 1–50 the
skipped updates fell (the predictions take the union over K = 15–18); OpenWebText's token count
under Mistral's own preprocessing (W2 uses nanoGPT's count, so the batch is checked to 0.5 %).

Run: python3 tools/math_checks/c2_warmup_grid.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trackC_writeup import c2_dose as d  # noqa: E402
from trackC_writeup import c2_warmup as w  # noqa: E402

results = []


def record(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if detail:
        print("      " + detail)
    results.append(ok)


x, n, t, pp, pg, Wp, Wg = sp.symbols("x n t p_p p_g W_p W_g", positive=True)
cum = sp.integrate(pg * sp.log(x) / sp.log(Wg), (x, 1, n))
record("W1 ∫₁ⁿ p ln x / ln W dx = p (n ln n − n + 1) / ln W",
       sp.simplify(cum - pg * (n * sp.log(n) - n + 1) / sp.log(Wg)) == 0, str(sp.simplify(cum)))
sol = sp.solve(sp.Eq(pg * sp.log(n) / sp.log(Wg), pp * t / Wp), n)
record("W1 equal instantaneous LR: n = W_g^(p_p t / (W_p p_g))",
       len(sol) == 1 and sp.simplify(sp.log(sol[0]) - sp.log(Wg ** (pp * t / (Wp * pg)))) == 0, str(sol))

log = json.loads((Path(w.REPO) / "trackC_writeup" / "gpt2_small_lr_log.json").read_text())
ok, ks, toks = True, set(), []
for seed, v in log["seeds"].items():
    for step, lr in v["lr"]:
        if 50 <= step <= w.G_WARMUP:
            m = math.exp(lr * math.log(w.G_WARMUP) / w.G_LR)
            ok &= abs(m - round(m)) < 1e-3
            ks.add(step - round(m))
    toks.append(v["epoch_at_4100"] * 9_035_582_198 / 4100)
record("W2 the log warmup fits every logged LR, 50 ≤ t ≤ 4000", ok and ks <= set(w.KS),
       f"skips K seen: {sorted(ks)}; KS = {w.KS}")
record("W2 tokens per step from the epoch count", all(abs(x_ / w.G_TOKENS - 1) < 5e-3 for x_ in toks),
       f"{sorted({round(x_) for x_ in toks})} against {w.G_TOKENS}")


def g_cum_discrete(nn):
    return w.G_LR / math.log(w.G_WARMUP) * sum(math.log(i) for i in range(1, int(nn) + 1))


def grid(y):
    return min(s for s in w.GRID if s >= y)


def discrete_pred(ref, k):
    """Grid steps where GPT-2's discrete summed LR first reaches 410m's discrete sum at a crossing."""
    out = set()
    cg = np.cumsum([0.0] + [w.G_LR / math.log(w.G_WARMUP) * math.log(i) for i in range(1, 5000)])
    for g in ref:
        for tt in np.linspace(d.prev(g), g, 2001)[1:]:
            target = w.P_LR * tt * (tt + 1) / (2 * w.P_WARMUP)
            nn = int(np.searchsorted(cg, target))       # iterations needed
            out.add(grid(k + nn))
    return frozenset(out)


ok = True
for kind in ("u", "c"):
    a = w.predicted(w.REF[kind], "cumulative")
    b = frozenset().union(*(discrete_pred(w.REF[kind], k) for k in w.KS))
    ok &= a == b
    if a != b:
        print(f"      {kind}: continuous {sorted(a)} discrete {sorted(b)}")
record("W3 the discrete sums give the same grid predictions", ok)

inside = max(max(v) for v in w.REF.values()) < w.P_WARMUP and all(
    w.to_gpt2(h, max(max(v) for v in w.REF.values()), max(w.KS)) - max(w.KS) < w.G_WARMUP
    for h in ("cumulative", "instantaneous", "fraction"))
record("W4 every reference date and its mapped step are inside the warmups", inside)

ok = True
for kind in ("u", "c"):
    for h in w.HYPS:
        fine = set()
        for k in w.KS:
            for g in w.REF[kind]:
                lo, hi = w.to_gpt2(h, d.prev(g), k), w.to_gpt2(h, g, k)
                fine |= {grid(y) for y in np.linspace(lo, hi, 20001)[1:]}
        ok &= w.predicted(w.REF[kind], h) == frozenset(fine)
record("W5 predicted equals a brute-force scan", ok)

print("\nOutcome table: per count, the schedules consistent with each GPT-2 grid date (others: none)")
for kind, name in (("u", "E1 break"), ("c", "centred")):
    rows = []
    for s in w.GRID[1:]:
        cons = w.consistent([s], kind)
        if cons:
            rows.append(f"{s}: {'/'.join(h[:4] for h in w.HYPS if h in cons)}")
    print(f"  {name:<8} " + "; ".join(rows))

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
