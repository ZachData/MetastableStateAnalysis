"""A5's draft registration: what the seed-count statistic can attain, and which break location
survives on seed 0 (`trackA_force/design-A5.md`, draft; `docs/ROADMAP.md` §6 A5, §10.9).

WHY THIS EXISTS
---------------
The draft registers "each PolyPythias 410m seed's energy break falls in (128, 512]" and scores
K, the number of the 9 seeds whose break lands there, against a binomial null. Before the user
picks the null's p0 and the break statistic, this fixes (1) the p-value and the e-value
(`core.evalues.calibrate`, e = κ p^(κ−1), κ = 0.5 from `claims/registry.json`) at every K for
each candidate p0, and (2) where each candidate break statistic puts seed 0, the one seed already
seen, on the 18-step grid.

WHAT IT SHOWS
-------------
1. **The binomial tail is exact and attains E ≥ 20 at modest K.** P(K ≥ k) for K ~ Bin(9, p0),
   in exact rationals. At p0 = 2/17 (uniform over the grid's 17 non-zero steps), K = 5 already
   gives E ≈ 12 and K = 6 gives E > 20; at the stricter p0 = 1/3, E ≥ 20 needs all 9 (K = 8 gives 16).
2. **Seed 0's break by three statistics** (violation counts from `data/analysis/colocation_panel.csv`,
   column `p1_energy_nviol_b1`: β = 1, sphere frame, rel_tol 1e-3, the 7 prompts of
   `build_colocation_panel.META_PROMPTS`; copied here because `data/` is untracked):
   - half-rise (first step with V ≥ V(143000)/2): 512, inside (128, 512];
   - first non-zero step: 256, inside;
   - CLAIM-B's change-mass centroid (positive increments, interval midpoints in log10(step + 1)):
     about step 830, **outside** (128, 512], pulled late by the 16000–32000 rise. So the centroid
     cannot carry a window that seed 0 itself sits in.

WHAT IT DOES NOT PROVE
----------------------
- That the binomial null is the right H0. It treats a seed's break as uniform over the chosen
  steps and the seeds as independent. Independence holds (separate runs), but every seed shares
  Pythia's LR schedule (warmup to about step 1430), so a high K cannot separate "a training event"
  from "the schedule": it shows the location is not the seed's, nothing about why.
- That seed 0's numbers hold at β = 3.5 in the LN1 frame (the project's decided β); only β = 1 on
  the sphere is on record. The draft's dry run on seed 0 is what checks that.
- Anything about the seeds themselves: none of their intermediate steps has been read.
"""
import math

import sympy as sp

KAPPA = sp.Rational(1, 2)
N_SEEDS = 9
GRID = (0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 54000, 143000)
WINDOW = (256, 512)  # the break's step on the grid: the interval (128, 256] or (256, 512]

# Seed 0 (standard pythia-410m), violations summed over 7 prompts, at the 18-step grid.
SEED0_V = {0: 0, 2: 0, 4: 0, 8: 0, 16: 0, 32: 0, 64: 0, 128: 0, 256: 19, 512: 56, 1000: 56,
           2000: 61, 4000: 57, 8000: 59, 16000: 64, 32000: 76, 54000: 72, 143000: 70}


def tail(k, p0, n=N_SEEDS):
    return sum(sp.binomial(n, j) * p0**j * (1 - p0) ** (n - j) for j in range(k, n + 1))


def evalue(p):
    return KAPPA * p ** (KAPPA - 1)


def half_rise(v):
    half = sp.Rational(v[GRID[-1]], 2)
    return next(s for s in GRID if v[s] >= half) if v[GRID[-1]] > 0 else None


def first_nonzero(v):
    return next((s for s in GRID if v[s] > 0), None)


def centroid_step(v):
    w, x = [], []
    for a, b in zip(GRID[:-1], GRID[1:]):
        w.append(max(v[b] - v[a], 0))
        x.append((math.log10(a + 1) + math.log10(b + 1)) / 2)
    c = sum(wi * xi for wi, xi in zip(w, x)) / sum(w)
    return 10**c - 1


def main():
    nonzero = len(GRID) - 1
    p0s = {"2/17 (uniform, 17 non-zero steps)": sp.Rational(len(WINDOW), nonzero),
           "1/6 (uniform, 12 steps up to 4000)": sp.Rational(1, 6),
           "1/3 (a 'moves early' null)": sp.Rational(1, 3)}
    ok = True
    for name, p0 in p0s.items():
        print(f"p0 = {name}")
        prev = sp.Integer(0)
        for k in range(N_SEEDS, -1, -1):
            p = tail(k, p0)
            ok &= bool(prev < p <= 1)  # the tail grows as k falls
            prev = p
            print(f"  K >= {k}: p = {float(p):.3g}, E = {float(evalue(p)):.3g}")
        # check 1: exact tail sums to 1 at k = 0
        ok &= tail(0, p0) == 1
    assert ok, "binomial tail not monotone or not normalised"

    h, f, c = half_rise(SEED0_V), first_nonzero(SEED0_V), centroid_step(SEED0_V)
    print(f"seed 0: half-rise {h}, first non-zero {f}, change-mass centroid ~{c:.0f}")
    assert h in WINDOW and f in WINDOW, "seed 0 should sit inside the window"
    assert not (128 < c <= 512), "centroid expected outside (128, 512]"
    print("PASS (3 checks)")


if __name__ == "__main__":
    main()
