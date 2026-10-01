"""The closed forms `p1d_cluster_ensemble/identity_sim.py` is tested against.

WHAT IT CHECKS
--------------
1. Under the full mask (Geshkovski et al.'s (SA), Q = K = V = I, self
   included), for n points whose pairwise inner products all equal g, the
   field's component gives d<x_i, x_k>/dt = (6.9)'s right-hand side:
       2 e^{bg} (1 - g)((n - 1) g + 1) / (e^b + (n - 1) e^{bg}).
   This is the reduction `p1c_frames/gamma_ode.py` integrates, re-derived
   from the field `identity_sim.velocity` computes.
2. Under the causal mask with n = 2 (2411.04990's (CSA)), the first token
   does not move and
       dg/dt = e^{bg} (1 - g^2) / (e^b + e^{bg}),
   which is exactly half of (6.9) at n = 2. So g_causal(t) = g_(6.9)(t / 2).

WHAT IT DOES NOT PROVE
----------------------
That the integrator is right: `tests/test_phase1d_identity_sim.py` checks the
numerical trajectories against `gamma_ode.integrate_gamma`. It also does not
show that the equal-angle reduction holds for the causal mask at n > 2. It
does not: the mask breaks permutation symmetry, so there is no one-scalar
closed form there, and the tests use Thm 4.1 (x_1 fixed, all tokens -> x_1(0))
instead.

Run: python3 tools/math_checks/identity_sim_closed_form.py
"""
import sympy as sp

b, g = sp.symbols("beta g", real=True)
n = sp.symbols("n", positive=True, integer=True)


def full_mask_rate():
    # <field_i, x_i> and <field_i, x_k> for unit x's with all pairwise <.,.> = g.
    Z = sp.exp(b) + (n - 1) * sp.exp(b * g)
    along_i = (sp.exp(b) + (n - 1) * g * sp.exp(b * g)) / Z
    along_k = (sp.exp(b) * g + sp.exp(b * g) * (1 + (n - 2) * g)) / Z
    # d<x_i, x_k>/dt = <P_i F_i, x_k> + <x_i, P_k F_k> = 2 (along_k - g along_i) by symmetry
    return sp.simplify(2 * (along_k - g * along_i))


def causal_pair_rate():
    # x_2' = P_{x_2}((e^{bg} x_1 + e^b x_2) / Z), Z = e^{bg} + e^b; x_1' = 0.
    Z = sp.exp(b * g) + sp.exp(b)
    return sp.simplify(sp.exp(b * g) * (1 - g ** 2) / Z)


eq69 = 2 * sp.exp(b * g) * (1 - g) * ((n - 1) * g + 1) / (sp.exp(b) + (n - 1) * sp.exp(b * g))

assert sp.simplify(full_mask_rate() - eq69) == 0, "full-mask field does not reduce to (6.9)"
print("1. full mask, equal angles: d g/dt = (6.9)                      OK")
assert sp.simplify(causal_pair_rate() - eq69.subs(n, 2) / 2) == 0, "causal pair is not (6.9)/2"
print("2. causal, n = 2: d g/dt = (6.9) at n = 2, halved               OK")
