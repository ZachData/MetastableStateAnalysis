"""Checks the closed-form claims in trackN_neurons/design-N1.md "What a pole is under GeLU".

Pythia's MLP neuron i: z_i = (W_in LN(x) + b_in)_i, h_i = GeLU(z_i), the write is h_i w_out,i.
`hidden_act` is "gelu" (every Pythia config read here), which transformers maps to the exact
GeLU(z) = z Phi(z), Phi the standard normal CDF.

G1  GeLU has one minimum, at z* = -0.75179 (h = -0.169971): GeLU' = Phi + z phi changes sign once,
    from - to +, at z* (sympy: GeLU'' = phi (2 - z^2); limit of GeLU' at -inf is 0 from below;
    the root by nsolve). So h is decreasing in z on (-inf, z*) and increasing on (z*, inf).
G2  sign(GeLU(z)) = sign(z) for every z (Phi > 0): a GeLU neuron has no double-negative regime.
    The essay's regime is SwiGLU's (G5).
G3  The strongest negative votes write almost nothing: |GeLU(z)| <= |z| Phi(z) is below 0.0041 at
    z <= -3 and below 1.3e-4 at z <= -4 (and falls with z). So a pole read at the most negative z
    is a closed gate, not a write.
G4  Ranking by h does not rank the negative side by z: for z1 < z2 <= z*, h(z1) > h(z2) (from G1).
    The most negative h are the z nearest z*, a band, not the strongest negative votes.
G5  SwiGLU for comparison: SiLU(g) = g sigma(g) has its minimum -0.27846 at g = -1.27846; the
    product SiLU(g) v is unbounded in both signs, and g < 0, v < 0 gives h > 0 with
    h <= 0.27846 |v| (the essay's "double negative").
G6  The tanh approximation (dropped from design-N1.md's questions on this bound) differs from the exact form by at
    most about 4.7e-4 on a fine grid, and its minimum is within 1e-3 of G1's in place and value.

NOT proved: that a pole read at the most negative z carries a job (2509.24198's ablation
removes the output where z < 0, which is largest at z*, so it bears on the band, G1 and G3;
U1 counts populated readings on 410m); which GeLU Pythia was trained with (G6 bounds the
difference only); anything about LayerNorm, the
parallel residual or the bias, which set z's scale per neuron.

Run: python3 tools/math_checks/n1_gelu_pole.py
"""
import sys

import numpy as np
import sympy as sp
from scipy.special import ndtr

results = []


def record(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if detail:
        print("      " + detail)
    results.append(bool(ok))


z = sp.symbols("z", real=True)
phi = sp.exp(-z**2 / 2) / sp.sqrt(2 * sp.pi)
Phi = (1 + sp.erf(z / sp.sqrt(2))) / 2
gelu = z * Phi

# G1
d1 = sp.simplify(sp.diff(gelu, z))
d2 = sp.simplify(sp.diff(gelu, z, 2))
ok_d1 = sp.simplify(d1 - (Phi + z * phi)) == 0
ok_d2 = sp.simplify(d2 - phi * (2 - z**2)) == 0
lim = sp.limit(d1, z, -sp.oo)
zs = sp.nsolve(d1, z, -0.75)
hs = gelu.subs(z, zs).evalf()
# d1 increases on (-sqrt2, sqrt2) (d2 > 0), decreases on (-inf, -sqrt2) from its limit 0,
# so d1 < 0 on (-inf, -sqrt2]; d1 > 0 for z >= 0; one root, in (-sqrt2, 0).
ok_sign = d1.subs(z, -sp.sqrt(2)).evalf() < 0 and d1.subs(z, 0).evalf() > 0
grid = np.linspace(-12, 6, 180001)

fd1 = lambda t: ndtr(t) + t * np.exp(-t**2 / 2) / np.sqrt(2 * np.pi)  # noqa: E731
s = np.sign(fd1(grid))
changes = np.flatnonzero(np.diff(s) != 0)
record("G1 GeLU' = Phi + z phi, GeLU'' = phi (2 - z^2)", ok_d1 and ok_d2)
record("G1 GeLU' -> 0 at -inf, < 0 at -sqrt2, > 0 at 0: one sign change",
       lim == 0 and ok_sign and len(changes) == 1,
       f"limit {lim}; grid sign changes {len(changes)} at z ~ {grid[changes[0]]:.5f}")
record("G1 minimum at z* = -0.75179, h = -0.16997",
       abs(float(zs) + 0.751791524693564) < 1e-9 and abs(float(hs) + 0.16997120748) < 1e-10,
       f"z* = {float(zs):.12f}, GeLU(z*) = {float(hs):.12f}")

# G2
g = lambda t: t * ndtr(t)  # noqa: E731  (ndtr keeps the tail: no underflow to 0 above z = -37)
gz = g(grid)
record("G2 sign(GeLU(z)) = sign(z) on the grid", np.all(np.sign(gz) == np.sign(grid)))

# G3
vals = {t: abs(g(t)) for t in (-1.0, -2.0, -3.0, -4.0)}
tail3 = np.max(np.abs(g(np.linspace(-40, -3, 100001))))
tail4 = np.max(np.abs(g(np.linspace(-40, -4, 100001))))
record("G3 |GeLU| < 0.0041 for z <= -3, < 1.3e-4 for z <= -4",
       tail3 < 0.0041 and tail4 < 1.3e-4,
       "  ".join(f"|GeLU({t:g})| = {v:.3g}" for t, v in vals.items()))

# G4
neg = grid[grid <= float(zs)]
record("G4 h strictly decreasing in z on z <= z* (ranking by h reverses z there)",
       np.all(np.diff(g(neg)) < 0))

# G5
gg = sp.symbols("g", real=True)
silu = gg / (1 + sp.exp(-gg))
gs = sp.nsolve(sp.diff(silu, gg), gg, -1.3)
ss = silu.subs(gg, gs).evalf()
record("G5 SiLU minimum -0.27846 at g = -1.27846",
       abs(float(gs) + 1.278464542761074) < 1e-9 and abs(float(ss) + 0.278464542761074) < 1e-9,
       f"g* = {float(gs):.12f}, SiLU(g*) = {float(ss):.12f} (equal: SiLU(g*) = g* + 1)")

# G6
gt = lambda t: 0.5 * t * (1 + np.tanh(np.sqrt(2 / np.pi) * (t + 0.044715 * t**3)))  # noqa: E731
fine = np.linspace(-12, 12, 2400001)
diff = np.max(np.abs(gt(fine) - g(fine)))
i = np.argmin(gt(fine))
record("G6 tanh approximation within 5e-4 of exact; its minimum within 1e-3 of G1's",
       diff < 5e-4 and abs(fine[i] - float(zs)) < 1e-3 and abs(gt(fine[i]) - float(hs)) < 1e-3,
       f"max |diff| = {diff:.3g}; tanh min at z = {fine[i]:.5f}, h = {gt(fine[i]):.6f}")

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
