"""C4's flat-arm maps (trackC_writeup/design-C4.md "The predictions on the flat arm").

Proves: (1) under Pythia's linear ramp eta(t) = eta_hat * t / W (continuous time), the summed rate
to t <= W is eta_hat t^2 / 2W, and the flat arm (eta_hat from the start) reaches it at
t_F = t^2 / 2W, which is <= W for t <= W; (2) the step band [t/rho, t rho] and the cumulative band
[(t/rho)^2 / 2W, (t rho)^2 / 2W] are disjoint iff t < 2W / rho^3; (3) at W = 1430, rho = 1.25 the
bound is about 1464, above both dates (256, 1000).

Does NOT prove: anything about the discrete schedule's off-by-one (NeoX's rate is 0 at iteration 0:
the discrete sum to checkpoint t is eta_hat t (t - 1) / 2W, as the loop below sums; within 1/t of the continuous one, checked numerically
below at t >= 16); that the dates themselves follow either hypothesis; anything past t = W, where
both arms share the cosine.
"""
import sympy as sp

t, W, eta, rho, s = sp.symbols("t W eta rho s", positive=True)

cum_ramp = sp.integrate(eta * s / W, (s, 0, t))
assert sp.simplify(cum_ramp - eta * t**2 / (2 * W)) == 0
tF = sp.solve(sp.Eq(eta * s, cum_ramp), s)[0]
assert sp.simplify(tF - t**2 / (2 * W)) == 0
assert sp.simplify((t**2 / (2 * W)).subs(t, W) - W / 2) == 0          # t_F <= W on t <= W

# disjoint bands: cumulative upper end < step lower end
cond = sp.simplify((t * rho) ** 2 / (2 * W) - t / rho)
bound = sp.solve(sp.Eq(cond, 0), t)
assert any(sp.simplify(b - 2 * W / rho**3) == 0 for b in bound)

val = float((2 * W / rho**3).subs({W: 1430, rho: sp.Rational(5, 4)}))
assert 1460 < val < 1470 and val > 1000

# discrete off-by-one: sum_{k=0}^{t-1} eta k / W = eta t (t - 1) / 2W  (rate 0 at iteration 0)
for tt in (16, 256, 1000):
    disc = sum(k for k in range(tt)) / 1430
    cont = tt**2 / (2 * 1430)
    assert abs(disc - cont) / cont <= 1 / tt + 1e-12

print("c4_flat_maps: ok", round(val, 1))
