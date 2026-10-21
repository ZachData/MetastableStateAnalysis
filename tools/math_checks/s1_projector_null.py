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
5. **The participation ratio's null and the additive plant** (R1′, Blocked 82 (b)). For P' Haar
   of dim k', E P' commutes with every orthogonal map, so E P' = (k'/d) I, and for independent
   heads E tr P_h P_h' = k_h k_h'/d. Hence E₀ tr P̄² = (1/n²)(Σk + ((Σk)² − Σk²)/d) exactly, and
   PR₀ = (tr P̄)² / E₀ tr P̄² = 879.67 at n = 384, k_h = 16, d = 1024. Adding to head h a direction
   orthogonal to its subspace (P_h → P_h + Q_h, Q_h PSD) cannot lower any tr P_h P_h', since each
   added term tr(Q P'), tr(P Q'), tr(Q Q') is a trace of a product of PSD matrices (= a sum of
   squared norms), so it cannot lower tr P̄².
6. **The additive plant dilutes the carrier's excess** (found by G4, 2026-10-21). With
   χ = tr P̄²/(tr P̄)² − 1/PR₀, adding dimensions raises tr P̄ by a share ε, so to first order
   Δχ = (Δ tr P̄² − Δ E₀ tr P̄²)/(tr P̄)² − 2ε χ₀: the plant's own excess minus 2ε times the
   carrier's excess χ₀. On a Haar carrier χ₀ ≈ 0; on a concentrated one the second term can
   outweigh the first. Singular families keep k_h = 16 at every step, so the data never move ε.

WHAT IT DOES NOT PROVE
----------------------
- That directions above the threshold are used by the model, or are anything beyond the
  residual stream's global anisotropy (the beside check reads activations).
- That in a *trained* pool a planted direction adds exactly one eigenvalue above the threshold:
  it mixes with the directions already shared. The gate (G3) measures the count's increment.
- Anything about weighted projectors (Σ σ² u uᵀ), for which the rotation null keeps each head's
  spectrum and differs from Haar with a Gaussian spectrum. S1 does not read them.
- That PR falls under an additive plant: tr P̄ rises with the added dimensions as well, so PR's
  move is the excess over the null's own rise (≈ (k² − k)/n² per direction on a Haar carrier, an
  expectation, not a bound). G4 measures it.
- That fact 6's first-order dilution is the whole deficit on a real carrier (G4 reads it: it
  predicts −1.08 against −1.08 observed, `qk_query` at k = 12, and 0.24 against 0.26 at k = 32).
- That E₀ PR equals PR₀ (PR₀ is the ratio at the expected kernel; the gap is O(var), G0′ reads it).
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

# --- 5. PR's null value and the additive plant -------------------------------------------------------
nn_, dd_ = sp.symbols("n d", positive=True)
ks_ = sp.symbols("k0:3", positive=True)                 # three heads' dims, symbolic
pair = sum(ks_[a] * ks_[b] / dd_ for a in range(3) for b in range(3) if a != b)
closed = (sum(ks_) + (sum(ks_) ** 2 - sum(x ** 2 for x in ks_)) / dd_)
assert sp.simplify(sum(ks_) + pair - closed) == 0       # diagonal tr P_h² = k_h, plus the pairs
S, S2 = 384 * 16, 384 * 16 ** 2
PR0 = sp.Rational(S) ** 2 / (S + sp.Rational(S ** 2 - S2, 1024))
assert abs(float(PR0) - 879.67) < 0.01
# E tr P P' = k k'/d, numerically (d = 24, k = 3, k' = 5)
proj = lambda U: U @ U.T  # noqa: E731
x = np.mean([np.trace(proj(haar(3)) @ proj(haar(5))) for _ in range(4000)])
assert abs(x - 3 * 5 / d) < 0.02
# additive plant: tr(Q P') = Σ_i ‖P' q_i‖² ≥ 0 and tr(Q Q') = Σ (q_iᵀ q'_j)² ≥ 0, symbolically for rank 1
v = sp.Matrix(sp.symbols("v0:3", real=True))
w = sp.Matrix(sp.symbols("w0:3", real=True))
assert sp.expand((v * v.T * w * w.T).trace() - (v.T * w)[0] ** 2) == 0

# --- 6. first-order dilution of χ under an additive plant ------------------------------------------
T2, T, D2, eps, E0, DE0 = sp.symbols("T2 T D2 epsilon E0 DE0", positive=True)
chi_ = lambda a, t, e: a / t ** 2 - e / t ** 2                     # noqa: E731  χ = (tr2 − E₀)/tr²
dchi = chi_(T2 + D2, T * (1 + eps), E0 + DE0) - chi_(T2, T, E0)
first = sp.series(dchi, eps, 0, 2).removeO()
claim = (D2 - DE0) / T ** 2 - 2 * eps * chi_(T2, T, E0) - 2 * eps * (D2 - DE0) / T ** 2
assert sp.simplify(sp.expand(first - claim)) == 0                  # the last term is second order in the plant

print("s1_projector_null: facts 1-3, 5 and 6 exact, fact 4 and 5's E tr PP' sanity (numeric) passed")
