"""Checks the base rate the C2p draft quotes in its section 5 (`trackC_writeup/paper/main.tex`, "The order"):
two dates drawn independently and uniformly from the four grid steps 256, 512, 1000 and 2000 are within
one grid step of each other with probability 10/16 = 0.625 (the second fresh read's M3 (i),
`trackC_writeup/referee-C2p.md`; STATE Blocked 109 (a)).

B1  P(|i - j| <= 1) for i, j uniform and independent on {0, 1, 2, 3} is 10/16 = 5/8 (sympy, exact).
B2  The same by enumerating the four Pythia steps themselves, "within one grid step" read as adjacent
    or equal on Pythia's released grid (numerical).
B3  Seven aligned of nine read against Bin(9, 5/8): its mean is 45/8 = 5.625 (sympy).
B4  Anchored at the break (512, the second of the four steps), a date uniform on the four is
    within one grid step of it with probability 3/4 (`/challenge-pr` on #269, finding 4: seven of
    the nine pairs read include the break, so 0.625 understates chance for them).

NOT proved: that the rows' dates are independent or uniform on those four steps (they are not: they
share one ramp, and a row's locator bounds where it can fall); that the four steps are the right
support (two rows date before 256); so 0.625 sets a scale for the count, not a null for it, and the
draft quotes it as that.

Run: python3 tools/math_checks/c2p_alignment_base_rate.py
"""
import itertools
import sys
from pathlib import Path

import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trackA_force.a1a import STEPS  # noqa: E402

results = []


def record(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if detail:
        print("      " + detail)
    results.append(ok)


# B1
n = 4
hits = sum(1 for i in range(n) for j in range(n) if abs(i - j) <= 1)
p = sp.Rational(hits, n * n)
record("B1 P(within one step) on 4 adjacent steps = 10/16", hits == 10 and p == sp.Rational(5, 8), f"{hits}/{n * n} = {p}")

# B2
support = [256, 512, 1000, 2000]
grid = sorted(set(STEPS))
idx = {s: grid.index(s) for s in support}
adjacent = all(idx[b] - idx[a] == 1 for a, b in zip(support, support[1:]))
pairs = sum(1 for a, b in itertools.product(support, support) if abs(idx[a] - idx[b]) <= 1)
record("B2 the four steps are consecutive on Pythia's grid, 10 of 16 pairs within one step",
       adjacent and pairs == 10, f"grid indices {[idx[s] for s in support]}")

# B3
mean = 9 * p
record("B3 Bin(9, 5/8) mean = 45/8 = 5.625", mean == sp.Rational(45, 8), f"{mean} = {float(mean)}")

# B4
anchored = sp.Rational(sum(1 for s in support if abs(idx[s] - idx[512]) <= 1), len(support))
record("B4 within one step of the break at 512 = 3/4", anchored == sp.Rational(3, 4), f"{anchored}")

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
