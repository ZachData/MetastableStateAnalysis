"""C4′ (d)+(f)'s bands and the A0 facts the design rests on (trackC_writeup/design-C4pdf.md;
revised after /challenge-pr on #371, findings 1 and 2).

Proves: (1) with the reference event known to (t⁻, t] and a spread ρ₀, a flat arm at r·peak has a
step band [t⁻/ρ₀, tρ₀] and a dose band [t⁻/(rρ₀), tρ₀/r], disjoint iff r < t⁻/(tρ₀²); (2) on C4's
grid (t⁻ = 32, t = 48; or 48, 64) the transient's bands meet at r = ½ and not at r = ⅒, and on F¹'s
grid D (t⁻ = t − 4) they are disjoint at both rates for t from 48 to 64; induction's (256, 320] is
disjoint at r = ½ by a thin margin (< 0.512); (3) the band values the design quotes, and that 1000
passes every transient dose band (to 800) and F½'s induction band (to 800); (4) grid D has 65
states, ends at 1000 and holds the dose bands' interior states; (5) A0 (a linear ramp over W = 1430)
reaches F's onset dose (48 peak-steps) at t = √(2W·48) ≈ 370.5, at ≈ 0.259 of the peak; (6) the
GPU estimate at 26.5 s an update.

Does NOT prove: that either clock is right, that the new arms have an onset, that ρ₀ = 1.25 is the
right spread (it is C4's placed RHO_MIN), or that F¹ reproduces F (H-d checks that).
"""
import sympy as sp

r, rho, t, tm = sp.symbols("r rho t tm", positive=True)

# (1) disjoint iff dose's lower edge > step's upper edge
ratio = sp.simplify((tm / (r * rho)) / (t * rho))
assert sp.simplify(ratio - tm / (r * t * rho**2)) == 0       # > 1  <=>  r < tm/(t rho^2)


def bands(tm_, t_, r_, p=1.25):
    return (tm_ / p, t_ * p), (tm_ / (r_ * p), t_ * p / r_)


def sep(tm_, t_, p=1.25):
    return tm_ / (t_ * p * p)


# (2) C4's grid against D
assert 0.42 < sep(32, 48) < 0.43 and 0.47 < sep(48, 64) < 0.49          # r = ½ refused on C4's grid
assert sep(32, 48) > 0.1 and sep(48, 64) > 0.1                          # r = ⅒ fine
for t_ in range(48, 65, 4):
    assert sep(t_ - 4, t_) > 0.5                                         # D: r = ½ disjoint
assert 0.5 < sep(256, 320) < 0.513                                       # induction: thin

# (3) values quoted
(s, d) = bands(44, 48, 0.5); assert s == (35.2, 60.0) and abs(d[0] - 70.4) < 1e-9 and d[1] == 120.0
(s, d) = bands(44, 48, 0.1); assert abs(d[0] - 352) < 1e-9 and abs(d[1] - 600) < 1e-9
(s, d) = bands(60, 64, 0.5); assert s == (48.0, 80.0) and d == (96.0, 160.0)
(s, d) = bands(60, 64, 0.1); assert abs(d[0] - 480) < 1e-9 and abs(d[1] - 800) < 1e-9
(s, d) = bands(256, 320, 0.5); assert s == (204.8, 400.0) and abs(d[0] - 409.6) < 1e-9 and d[1] == 800.0
(s, d) = bands(256, 320, 0.1); assert abs(d[0] - 2048) < 1e-9 and abs(d[1] - 4000) < 1e-9
# the widest transient dose band over the gated arms' coarse references: [256, 800] at r = ⅒
assert abs(bands(32, 48, 0.1)[1][0] - 256) < 1e-9 and abs(bands(48, 64, 0.1)[1][1] - 800) < 1e-9
assert 1000 > 800

# (4) grid D
D = sorted(set(range(0, 129, 4)) | set(range(144, 257, 16)) | set(range(288, 993, 32)) | {1000})
assert len(D) == 65 and D[-1] == 1000
assert [x for x in D if 352 <= x <= 600] and [x for x in D if 70.4 <= x <= 120]

# (5) A0 crosses 48 peak-steps
W = 1430
x = sp.symbols("x", positive=True)
cross = float(sp.solve(sp.Eq(x**2 / (2 * W), 48), x)[0])
assert abs(cross - 370.513) < 1e-3 and abs(cross / W - 0.2591) < 1e-4 and 320 < cross < 512

# (6) hours at 26.5 s an update: F¹ 128, F½ 1000, F⅒ 1000, two branches of 256
h = lambda n: n * 26.5 / 3600
assert abs(h(128) - 0.94) < 0.01 and abs(h(2000) - 14.72) < 0.01 and abs(h(512) - 3.77) < 0.01
assert abs(h(128 + 2000 + 512) - 19.4) < 0.05 and abs(h(768 + 384) - 8.48) < 0.01
print("c4pdf_bands: all checks pass")
