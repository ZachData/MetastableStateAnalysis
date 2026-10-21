"""S1's statistic: the mean projector of a pool of head subspaces, what its spectrum says, and why
its rotation, Gaussian and Haar nulls are one law (`trackS_subspace/design-S1.md` "The math").

WHY THIS EXISTS
---------------
S1 reads, for a pool of n heads with orthonormal bases U_h (d × k_h), P̄ = (1/n) Σ U_h U_hᵀ, and
counts its eigenvalues above a simulated null's largest λ₁. `docs/ROADMAP.md` §7a proposed three
nulls (step 0, a norm-matched random draw, and each trained head rotated by its own Haar map).
This check makes exact what the eigenvalues mean, which identity ties them to the projection
kernel of 2601.10266, and the planted floor that sets the gate's doses.

WHAT IT SHOWS
-------------
1. **Range and trace.** P̄ is a convex combination of orthogonal projectors, so 0 ⪯ P̄ ⪯ I: each
   eigenvalue lies in [0, 1], and tr P̄ = (1/n) Σ k_h. An eigenvalue λ along u is the pool's mean
   of uᵀ P_h u, the share of the heads that hold u (exactly k/n when k heads hold u fully and the
   rest are orthogonal to it).
2. **tr P̄² is the mean pairwise kernel.** tr P̄² = (1/n²)(Σ_h k_h + Σ_{h≠h'} tr P_h P_h'), so
   Σ λ² is a pool's mean projection kernel (principal-angle overlap Σ cos² θ) up to the diagonal.
3. **The planted floor.** If u (unit) lies in k of the n subspaces, uᵀ P̄ u ≥ k/n, so λ₁ ≥ k/n by
   Rayleigh. With the 2000-draw Haar maxima of λ₁ at k_h = 16, d = 1024 (0.1402–0.1409 for a
   layer's 16 heads, 0.0311 for all 384; the design's preflight and exploratory draws,
   `trackS_subspace/design-S1.md` "Seen before"), a direction held by 3 of
   16 or by 12 of 384 heads is guaranteed above the threshold (3/16 = 0.1875; 12/384 = 0.03125).
4. **One null law.** P̄ depends on each head only through span(U_h). (a) For Q Haar on O(d) and
   any fixed U, span(Q U) is uniform on the Grassmannian Gr(k, d), so the rotation null is
   independent Haar subspaces of the heads' own dimensions, whatever the trained bases are.
   (b) For A with i.i.d. normal entries, Q A has A's law, so the span of A Bᵀ's top-r left
   singular vectors (and of its invariant subspaces of each eigenvalue type, given their
   dimension) is rotation invariant: Haar. Pythia's init draws Q, K, V and O i.i.d. normal, so
   step 0 is one draw of that law. Checked numerically below on small d (a sanity check of the
   argument, not the proof, which is the invariance itself).

WHAT IT DOES NOT PROVE
----------------------
- That directions above the threshold are used by the model, or are anything beyond the
  residual stream's global anisotropy (the beside check reads activations).
- That in a *trained* pool a planted direction adds exactly one eigenvalue above the threshold:
  it mixes with the directions already shared. The gate (G3) measures the count's increment.
- Anything about weighted projectors (Σ σ² u uᵀ), for which the rotation null keeps each head's
  spectrum and differs from Haar with a Gaussian spectrum. S1 does not read them.
- That the max-based threshold is tight: it controls the false count at every rank at once and
  so undercounts a bulk that is shifted but below the null's top edge.
"""

import numpy as np
import sympy as sp

# --- 1. 0 ⪯ P̄ ⪯ I and tr P̄ = mean k: on a symbolic 3-head pool in d = 3 ---------------------------
t = sp.symbols("t0:3", real=True)
def rot(a):
    return sp.Matrix([[sp.cos(a), -sp.sin(a), 0], [sp.sin(a), sp.cos(a), 0], [0, 0, 1]])
e = [sp.Matrix([1, 0, 0]), sp.Matrix([0, 0, 1])]
Us = [rot(t[0]) * e[0], rot(t[1]) * sp.Matrix.hstack(e[0], e[1]), rot(t[2]) * e[0]]
P = [U * U.T for U in Us]
for Ph in P:
    assert sp.simplify(Ph * Ph - Ph) == sp.zeros(3, 3)          # each P_h is a projector
Pbar = (P[0] + P[1] + P[2]) / 3
assert sp.simplify(Pbar.trace() - sp.Rational(1 + 2 + 1, 3)) == 0
assert sp.simplify((sp.eye(3) - Pbar).trace() - (3 - sp.Rational(4, 3))) == 0
# I - P̄ is the mean of the complements I - P_h, also projectors, so it is PSD as well.

# --- 2. tr P̄² = (1/n²)(Σ k_h + Σ_{h≠h'} tr P_h P_h') ---------------------------------------------
lhs = (Pbar * Pbar).trace()
rhs = sp.Rational(1, 9) * (sum(Ph.trace() for Ph in P)
                           + sum((P[a] * P[b]).trace() for a in range(3) for b in range(3) if a != b))
assert sp.simplify(lhs - rhs) == 0

# --- 3. the planted floor: u in k of n subspaces ⇒ uᵀ P̄ u ≥ k/n ----------------------------------
# uᵀ P_h u = 1 for a head holding u and ∈ [0, 1] otherwise, so the mean is ≥ k/n. Symbolically:
k, n = sp.symbols("k n", positive=True, integer=True)
q = sp.symbols("q", nonnegative=True)          # the mean of uᵀ P_h u over the n − k others, in [0, 1]
mean_quad = (k * 1 + (n - k) * q) / n
assert sp.simplify(mean_quad - k / n - (n - k) * q / n) == 0  # = k/n + a nonnegative term
for kk, nn, edge in ((3, 16, 0.1409), (12, 384, 0.0311)):
    assert kk / nn > edge
assert 2 / 16 < 0.1409 and 11 / 384 < 0.0311                  # one fewer head is not guaranteed

# --- 4. numeric sanity: rotation, Gaussian-product and Haar give one λ₁ law (d = 24) ---------------
rng = np.random.default_rng(20261020)
d, kh, nh, draws = 24, 3, 5, 3000


def l1(bases):
    B = np.concatenate(bases, 1)
    return np.linalg.eigvalsh(B @ B.T / len(bases))[-1]


def haar(m):
    return np.linalg.qr(rng.standard_normal((d, m)))[0]


fixed = [np.linalg.qr(np.eye(d)[:, :kh] + 0.1 * rng.standard_normal((d, kh)))[0] for _ in range(nh)]
x_haar = np.array([l1([haar(kh) for _ in range(nh)]) for _ in range(draws)])
x_rot = np.array([l1([np.linalg.qr(rng.standard_normal((d, d)))[0] @ U for U in fixed])
                  for _ in range(draws)])


def top_left(A, B, r):
    U, _, _ = np.linalg.svd(A @ B.T)
    return U[:, :r]


x_gau = np.array([l1([top_left(rng.standard_normal((d, 6)), rng.standard_normal((d, 6)), kh)
                      for _ in range(nh)]) for _ in range(draws)])
from scipy.stats import ks_2samp  # noqa: E402
assert ks_2samp(x_haar, x_rot).pvalue > 1e-3
assert ks_2samp(x_haar, x_gau).pvalue > 1e-3
assert l1(fixed) > np.quantile(x_haar, 0.99)    # the fixed pool shares; rotating it removes that

print("s1_projector_null: facts 1-3 exact, fact 4 sanity (KS) passed")
