"""B2's spectra: what the sign count of a head's symmetrised OV or QK can and cannot say, and the
chance level of the principal-angle overlap (`docs/ROADMAP.md` §7, B2; `trackB_circuit/design-B2.md`
"The math").

WHY THIS EXISTS
---------------
B2 reads each head's symmetrised OV, S = ½(V + Vᵀ) with V = W_O W_V of rank ≤ 64 in d = 1024, and
its symmetrised QK form ½(M + Mᵀ), M = W_Qᵀ W_K. ROADMAP's row asks for "dominant negative
eigenpairs". `2609.22990` reports that equal counts of positive and negative eigenvalues of a
head's symmetric QK part follow from the low-rank construction, not from training. This check
makes the claim exact before any column counts signs, and gives the chance level of the
principal-angle column.

WHAT IT SHOWS
-------------
1. **Inertia is fixed by construction.** For any V = A Bᵀ with A, B of size d × r,
   S = ½(A Bᵀ + B Aᵀ) = C J Cᵀ with C = [A B] and J = ½[[0, I], [I, 0]], whose eigenvalues are
   ±½, r of each. By Sylvester's law of inertia, when C has full column rank 2r, S has exactly
   r positive and r negative eigenvalues (and d − 2r zeros), whatever A and B are. So a head
   whose read and write spaces are in general position always has 64 negative eigenvalues of
   its symmetrised OV (and of its symmetrised QK). **Sign counts carry no information; only the
   weights (the |λ| shares, the top eigenvalues) can.**
2. **The exception is degenerate C.** When span(A) and span(B) overlap, the count can move: A = B
   gives S = A Aᵀ ⪰ 0, inertia (r, 0). Departure from (r, r) measures how far a head's read
   and write spaces coincide, not attraction.
3. **The share's null is ½.** If the law of (A, B) is invariant under A → −A (Gaussian init),
   S and −S have the same law, so the attractive share a = Σλ₊ / Σ|λ| and 1 − a are equal in
   law: E[a] = ½. Pythia's step 0 is one such draw per head.
4. **Principal-angle chance.** For U, W orthonormal d × r, Σ_k cos² θ_k = tr(P_U P_W). With U
   Haar and independent of W, E[P_U] = (r/d) I, so E[Σ cos² θ_k] = r²/d (the overlap per
   direction, Σ cos² / r, has mean r/d: 1/64 at r = 16, d = 1024).

WHAT IT DOES NOT PROVE
----------------------
- That a share or a top eigenvalue differing from step 0 is attraction in the dynamics:
  ⟨x, V x⟩ > 0 for x in a subspace says the head's write has a positive component along x, not
  that tokens move together (attention weights it; `trackB_circuit/status-B.md` "B2a on one
  page" found the reduced dynamics reproduce no head).
- That real step-0 heads match the Gaussian law beyond fact 3's symmetry; the gate measures it.
- Anything about V's own (nonsymmetric) eigenvalues, which B2a's ladder and
  `data/analysis/ov_per_head_series.json` read; facts 1–2 are about S only.
"""

import numpy as np
import sympy as sp

# --- 1. S = C J C^T, J = ½[[0, I], [I, 0]] has r eigenvalues +½ and r of −½ --------------------
d, r = 4, 2
A = sp.Matrix(d, r, lambda i, j: sp.Symbol(f"a{i}{j}", real=True))
B = sp.Matrix(d, r, lambda i, j: sp.Symbol(f"b{i}{j}", real=True))
S = (A * B.T + B * A.T) / 2
C = A.row_join(B)
J = sp.Matrix(sp.BlockMatrix([[sp.zeros(r), sp.eye(r)], [sp.eye(r), sp.zeros(r)]])) / 2
assert sp.expand(S - C * J * C.T) == sp.zeros(d)
evJ = J.eigenvals()
assert evJ == {sp.Rational(1, 2): r, -sp.Rational(1, 2): r}

# Sylvester at the real shape, numerically: generic rank-64 factors give inertia (64, 64)
g = np.random.default_rng(20261017)
dn, rn = 1024, 64
An, Bn = g.standard_normal((dn, rn)), g.standard_normal((dn, rn))
ev = np.linalg.eigvalsh((An @ Bn.T + Bn @ An.T) / 2)
tol = np.abs(ev).max() * 1e-9
assert (ev > tol).sum() == rn and (ev < -tol).sum() == rn
# ... and the same when A carries a planted attractive bias (A = B + small noise): still (64, 64)
Ab = Bn + 0.1 * g.standard_normal((dn, rn))
ev = np.linalg.eigvalsh((Ab @ Bn.T + Bn @ Ab.T) / 2)
tol = np.abs(ev).max() * 1e-9
assert (ev > tol).sum() == rn and (ev < -tol).sum() == rn
pos_share = ev[ev > 0].sum() / np.abs(ev).sum()
assert pos_share > 0.9                       # the weights move, the count does not

# --- 2. degenerate C: A = B gives S = A A^T ⪰ 0, inertia (r, 0) --------------------------------
S_deg = (A * A.T + A * A.T) / 2
assert sp.expand(S_deg - A * A.T) == sp.zeros(d)
ev = np.linalg.eigvalsh(An @ An.T)
tol = np.abs(ev).max() * 1e-9
assert (ev > tol).sum() == rn and (ev < -tol).sum() == 0

# --- 3. share null: A → −A maps S → −S, so a ↔ 1 − a ---------------------------------------------
assert sp.expand(((-A) * B.T + B * (-A).T) / 2 + S) == sp.zeros(d)
shares = []
for _ in range(200):
    a_, b_ = g.standard_normal((256, 16)), g.standard_normal((256, 16))
    e = np.linalg.eigvalsh((a_ @ b_.T + b_ @ a_.T) / 2)
    shares.append(e[e > 0].sum() / np.abs(e).sum())
assert abs(np.mean(shares) - 0.5) < 0.01

# --- 4. principal angles: E[tr(P_U P_W)] = tr((r/d) I · P_W) = r²/d -----------------------------
ds, rs = sp.symbols("d r", positive=True)
assert sp.simplify(rs / ds * rs - rs**2 / ds) == 0      # tr((r/d) P_W) with tr P_W = r
Wq, _ = np.linalg.qr(g.standard_normal((dn, 16)))
ov = []
for _ in range(400):
    Uq, _ = np.linalg.qr(g.standard_normal((dn, 16)))
    ov.append(np.linalg.norm(Uq.T @ Wq) ** 2)
assert abs(np.mean(ov) - 16**2 / dn) < 0.02             # 0.25 at r = 16, d = 1024

print("b2_symmetric_inertia: 4 checks pass (sym(A B^T) = C J C^T has inertia (r, r) whenever "
      "[A B] has full column rank, so sign counts are structural and only |λ| weights move; "
      "A = B gives (r, 0); the attractive share's null is ½ under A → −A; Haar principal-angle "
      "overlap tr(P_U P_W) has mean r²/d)")
