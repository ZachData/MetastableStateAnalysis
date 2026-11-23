"""C4′ (d)+(f)'s bands and the A0 fact the design rests on (trackC_writeup/design-C4pdf.md).

Proves: (1) for a flat arm at r·peak, an event F (r = 1) dates at t₁ has its step band
[t₁/ρ, t₁ρ] and its dose band [t₁/(rρ), t₁ρ/r]; they are disjoint iff r < 1/ρ², which holds at
r = ½ and r = ⅒ with ρ = 1.25; (2) on C4's grid the bands for the transient (t₁ = 48) and for
induction (t₁ = 320) hold the grid steps the design lists, and the dose band's upper edge is
below each arm's horizon at D1 (a) (F½ 1000, F⅒ 768) but not at the ROADMAP's (768, 384); (3) A0
(linear ramp to the peak over W = 1430) reaches F's onset dose (48 peak-steps) at t = √(2W·48)
≈ 370.5, where its rate is ≈ 0.259 of the peak; (4) the GPU estimate at 26.5 s an update.

Does NOT prove: that either clock is right, that the transient has an onset on the new arms, or
that ρ = 1.25 is the right spread (it is C4's placed RHO_MIN; C4′ (c) measured one grid step from
one reordering).
"""
import math

import sympy as sp

r, rho, t1 = sp.symbols("r rho t1", positive=True)

# (1) disjoint iff the dose band's lower edge exceeds the step band's upper edge
lo_dose, hi_step = t1 / (r * rho), t1 * rho
cond = sp.simplify(lo_dose / hi_step)            # = 1/(r rho^2)
assert sp.simplify(cond - 1 / (r * rho**2)) == 0
for rv in (sp.Rational(1, 2), sp.Rational(1, 10)):
    assert cond.subs({r: rv, rho: sp.Rational(5, 4)}) > 1

GRID = (0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768,
        1000, 1250, 1500, 2000, 3000, 4000)


def band(t, rv, dose, p=1.25):
    return (t / (rv * p), t * p / rv) if dose else (t / p, t * p)


def on_grid(b):
    return [s for s in GRID if b[0] <= s <= b[1]]


# (2) the grid steps in each band
assert on_grid(band(48, 1, False)) == [48]
assert on_grid(band(48, 0.5, True)) == [80, 96]
assert on_grid(band(48, 0.1, True)) == [384, 512]
assert on_grid(band(320, 1, False)) == [256, 320, 384]
assert on_grid(band(320, 0.5, True)) == [512, 640, 768]
assert band(320, 0.1, True) == (2560.0, 4000.0)
# horizons: D1 (a) passes each dose band's top by a grid step; the ROADMAP's do not
assert band(48, 0.1, True)[1] < 768 and band(48, 0.1, True)[1] > 384      # F⅒: 600
assert band(320, 0.5, True)[1] < 1000 and band(320, 0.5, True)[1] > 768   # F½ induction: 800

# (3) A0's summed rate t²/(2W) (peak-steps) reaches 48 at sqrt(2W·48)
W = 1430
t = sp.symbols("t", positive=True)
cross = sp.solve(sp.Eq(t**2 / (2 * W), 48), t)[0]
assert abs(float(cross) - 370.513) < 1e-3 and abs(float(cross) / W - 0.2591) < 1e-4
assert 320 < float(cross) < 512                                           # between A0's 320 and 512

# (4) hours at 26.5 s an update
h = lambda n: n * 26.5 / 3600
assert abs(h(1000) - 7.36) < 0.01 and abs(h(768) - 5.65) < 0.01 and abs(h(256) - 1.88) < 0.01
assert abs(h(1000 + 768) - 13.0) < 0.05 and abs(h(768 + 384) - 8.48) < 0.01
assert abs(h(1000 + 768 + 512) - 16.8) < 0.05
print("c4pdf_bands: all checks pass")
