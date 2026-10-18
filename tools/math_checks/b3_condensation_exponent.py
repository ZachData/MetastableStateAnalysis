"""B3's routing: what the condensation exponent of a head's attention can and cannot say, and how
2609.28448's d = 2 threshold sits inside the general critical scale (`docs/ROADMAP.md` §7, B3;
`trackB_circuit/design-B3.md` "The math").

WHY THIS EXISTS
---------------
ROADMAP's B3 row asks for "per-head β against the paper's condensation thresholds (does a head
condense when it forms?)". 2609.28448 defines condensation as a limit, lim_{N→∞} Y_A > 0, with
Y_A the mean row inverse participation ratio Σ_j A_ij² (its eq. attention_ipr,
d2_condensation_definition); at finite N it is "only a smooth crossover". So B3 reads the
participation number N_eff,i = 1 / Σ_j A_ij² against the number of keys n_i a row can see
(causal: n_i = i after position 0 is dropped), and its slope α = d log N_eff / d log n. This check
fixes what α is on the cases the design names, before any head is read.

WHAT IT SHOWS
-------------
1. **Diffuse gives α = 1, condensed gives α = 0.** Uniform attention over n keys has N_eff = n;
   attention spread equally over k fixed partners has N_eff = k whatever n is.
2. **Fixed-variance logits are diffuse.** For i.i.d. logits z with variance s² (Gaussian),
   E[e^{2z}] / E[e^z]² = e^{s²}, so N_eff → n e^{−s²} as n grows: α → 1 at any fixed s. The
   general form is the paper's bound (eq. finite_d_no_condensation): logits in [−B, B] give
   N_eff ≥ (n − 1) e^{−4B}, so at a fixed β nothing condenses as n → ∞. A head whose α stays
   below 1 over a range of n is one whose logit gap grows with the context, or one with O(1)
   partners set by position (a previous-token head).
3. **2609.28448's d = 2 threshold is the general one at d = 2.** On the circle the softmax
   window holds N_att ~ N (√2 β)^{−1/2} tokens (its eq. d2_number_attended), O(1) iff
   √2 β = χ N². Bruno, Chen, Lin, Polyanskiy, Rigollet (2605.08505, abstract level) put the
   critical scale for uniform keys on S^{d−1} at β_c ≍ n^{2/(d−1)}; at d = 2 that is n², the
   same exponent. As d grows the exponent goes to 0, consistent with the paper's β = O(1) at
   d = N (numerics only there).

WHAT IT DOES NOT PROVE
----------------------
- That a slope over n ∈ [16, 512] (v1's rows) measures the limit: a finite-n α is the crossover the paper
  says finite N gives. Check 2 measures it: i.i.d. Gaussian logits at a fixed s = 2 give a slope
  well below 1 over n = 16–256, rising only past n ≈ 4000. So α is read against the same step's
  heads and step 0, with the row logit spread beside, not as "condensed" or "diffuse".
- That a trained head's low α is the paper's condensation: the paper's routing is to neighbours
  in representation space (Q = K = I, no position); a previous-token head routes by position.
- Any threshold for a trained head: n^{2/(d−1)} needs the local dimension of the key cloud near
  the query (2605.08505's local behaviour of the distance law), which B3 does not estimate.
"""

import numpy as np
import sympy as sp

n, k, s, B, beta, chi, N, d = sp.symbols("n k s B beta chi N d", positive=True)

# --- 1. uniform: N_eff = n (α = 1); k equal partners: N_eff = k (α = 0) -------------------------
Y_uniform = n * (1 / n) ** 2
Y_k = k * (1 / k) ** 2
assert sp.simplify(1 / Y_uniform - n) == 0 and sp.simplify(1 / Y_k - k) == 0
alpha = lambda Neff: sp.simplify(sp.diff(sp.log(Neff), n) * n)
assert alpha(1 / Y_uniform) == 1 and alpha(1 / Y_k) == 0

# --- 2. Gaussian logits: E e^{2z} / (E e^z)² = e^{s²}, so N_eff ≈ n e^{−s²}, α → 1 ---------------
z = sp.Symbol("z", real=True)
pdf = sp.exp(-z**2 / (2 * s**2)) / sp.sqrt(2 * sp.pi * s**2)
E1 = sp.simplify(sp.integrate(sp.exp(z) * pdf, (z, -sp.oo, sp.oo)))
E2 = sp.simplify(sp.integrate(sp.exp(2 * z) * pdf, (z, -sp.oo, sp.oo)))
assert sp.simplify((E2 / E1**2).rewrite(sp.erf) - sp.exp(s**2)) == 0
assert alpha(n * sp.exp(-s**2)) == 1
g = np.random.default_rng(20261018)


def fit_slope(sn, ns, rows=200):
    Ne = []
    for m in ns:
        w = np.exp(sn * g.standard_normal((rows, m)))
        A = w / w.sum(1, keepdims=True)
        Ne.append(np.median(1 / (A**2).sum(1)))
    return np.polyfit(np.log(ns), np.log(Ne), 1)[0]


small = np.array([16, 64, 256])                          # B3's range of n (rows of v1's prompts)
for sn in (0.5, 1.0):
    assert fit_slope(sn, small) > 0.85                   # diffuse at a modest fixed logit scale
# ... but a large fixed scale is still in the crossover at these n: the slope is well below 1
# and rises with n (the lognormal's e^{s²} sets the n where it settles). So a low α at n ≤ 512
# can be a sharp head with no growing gap: the design reads α beside the row logit spread.
s2_small, s2_large = fit_slope(2.0, small), fit_slope(2.0, np.array([4096, 16384, 65536]), 50)
assert s2_small < 0.8 and s2_large > s2_small + 0.1, (s2_small, s2_large)
# the bound: logits in [−B, B] give max_j A_ij ≤ e^{2B}/(n−1)... so Y ≤ e^{4B}/(n−1)
for Bn in (0.5, 1.5):
    m = 500
    zz = g.uniform(-Bn, Bn, (300, m))
    A = np.exp(zz) / np.exp(zz).sum(1, keepdims=True)
    assert ((1 / (A**2).sum(1)) >= (m - 1) * np.exp(-4 * Bn)).all()

# --- 3. d = 2 window: N_att = N (√2 β)^{−1/2} is O(1) iff √2 β = χ N²; = n^{2/(d−1)} at d = 2 ------
N_att = N / sp.sqrt(sp.sqrt(2) * beta)
assert sp.simplify(N_att.subs(beta, chi * N**2 / sp.sqrt(2)) - 1 / sp.sqrt(chi)) == 0
assert sp.limit(N_att.subs(beta, chi * N / sp.sqrt(2)), N, sp.oo) == sp.oo   # β ∝ N is too slow
assert sp.simplify((2 / (d - 1)).subs(d, 2) - 2) == 0
assert sp.limit(2 / (d - 1), d, sp.oo) == 0

print("b3_condensation_exponent: 3 checks pass (uniform attention has α = 1 and k fixed partners "
      "α = 0; fixed-variance or bounded logits are diffuse, N_eff ≈ n e^{−s²}, α → 1; the d = 2 "
      "window is O(1) iff β ∝ N², which is the n^{2/(d−1)} critical scale at d = 2)")
