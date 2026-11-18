"""C4′ (c)'s item-9 rule and its data budget (trackC_writeup/design-C4pc.md).

Proves: (1) for 0 < a < b, a date x > 0 is nearer a than b in log distance
(|ln x/a| < |ln x/b|) iff x < sqrt(a b); (2) with a = m_cum(t) = t^2 / 2W and b = t (a < b iff
t < 2W), that threshold is t^(3/2) / sqrt(2W); at W = 1430 and t = 768 (A0's induction date) it is
about 398, so on C4's dense grid F at 384 or earlier (C4 read 320) is nearer the summed-rate map
and F at 512 would not be; (2b) C4's label exists only for ρ < (2W / t_A0)^{1/3}, 1.647 at the
centred date 640 (below item 6's margin 1.727) and 1.953 at V's 384 (above V's 1.466); (3) the rows
of shard 0 left after C4's training rows and H4's held-out rows hold 3149 whole updates of 1024
rows, fewer than 4000.

Does NOT prove: that log distance is the right scale for "follows more nearly" (it is the scale
in which both maps are ratios, as C4's ρ is); anything about the dates themselves; that the
shard's last row is whole (it is not: 30e9 is not a multiple of a row's 4098 bytes, so the last
partial row is dropped, as the floor does).
"""
import math

import sympy as sp

x, a, b, t, W = sp.symbols("x a b t W", positive=True)

# (1) nearer a than b in log  <=>  x < sqrt(ab), for a < b. Squared distances differ by
# (ln x - ln a)^2 - (ln x - ln b)^2 = (ln b - ln a)(2 ln x - ln a - ln b), whose sign is
# that of ln x - ln sqrt(ab) when b > a.
lx, la, lb = sp.symbols("lx la lb", real=True)
diff = (lx - la) ** 2 - (lx - lb) ** 2
assert sp.expand(diff - (lb - la) * (2 * lx - la - lb)) == 0

# (2) the threshold with a = t^2/2W, b = t
thr = sp.sqrt((t**2 / (2 * W)) * t)
assert sp.simplify(thr - t ** sp.Rational(3, 2) / sp.sqrt(2 * W)) == 0
val = float(thr.subs({t: 768, W: 1430}))
assert 397 < val < 399 and 384 < val < 512          # grid steps ≤ 384 pass (F read 320); 512 does not
m768 = 768**2 / (2 * 1430)
assert abs(m768 - 206.2) < 0.05
# numeric spot check of (1) on C4's pair: 320 nearer 206.2 than 768
assert abs(math.log(320 / m768)) < abs(math.log(320 / 768))
assert abs(math.log(512 / m768)) > abs(math.log(512 / 768))

# (2b) M2's cap (/challenge-pr on #361, finding 1): `c4_read.separable` needs t rho^3 < 2W, so a
# label exists only for rho < (2W/t)^(1/3): 1.647 at the centred date 640 (below item 6's margin
# 1.727, so the centred label cannot become "cumulative"), 1.953 at V's 384 (above V's 1.466).
rho = sp.symbols("rho", positive=True)
cap = sp.solve(sp.Eq((t * rho) ** 2 / (2 * W), t / rho), rho)[0]
assert sp.simplify(cap - (2 * W / t) ** sp.Rational(1, 3)) == 0
cap640, cap384 = (float(cap.subs({t: tt, W: 1430})) for tt in (640, 384))
margin640 = 640 / math.sqrt(2 * 1430 * 48)        # c4_close.rho_star(640, 48, "cumulative")
margin384 = 384 / math.sqrt(2 * 1430 * 24)
assert abs(cap640 - 1.647) < 1e-3 and abs(margin640 - 1.727) < 1e-3 and cap640 < margin640
assert abs(cap384 - 1.953) < 1e-3 and abs(margin384 - 1.466) < 1e-3 and margin384 < cap384

# (3) shard 0's unused rows
ROW_BYTES = 2049 * 2
rows_in_shard = 30_000_000_000 // ROW_BYTES
used = 4000 * 1024 + 64                              # C4's training rows + H4's held-out rows
left = rows_in_shard - used
assert rows_in_shard == 7_320_644 and left == 3_224_580
assert left // 1024 == 3149 < 4000

print("c4pc_nearer: ok", round(val, 1), left // 1024)
