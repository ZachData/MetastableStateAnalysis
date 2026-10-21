"""Checks the closed-form claims in trackS_subspace/design-S2.md "The arms" (S2, what rotation does).

Per head, in LN1's folded, centred frame (as OV1, tools/math_checks/ov_cut_ov1.py C1-C2):
B = Pi W_O (d x k), C = W_V diag(gamma) Pi (k x d), K = B C, A = C B (k x k), Z = [B, C^T] (d x 2k).

C1  One core per arm. K = Z E Z^T with E = [[0, I], [0, 0]]; sym(K) = Z J Z^T with J = (E + E^T)/2;
    B G C = Z [[0, G], [0, 0]] Z^T for any k x k G. Symbolic, d = 3, k = 1, and d = 3, k = 2 for
    the G form: a full proof at those sizes (the general case is block multiplication).
C2  Arm (b), rotation removed: with A = Q Lam Q^-1 (diagonalisable, no zero eigenvalue) and
    g(A) = Q diag(Re lam / lam) Q^-1, g(A) is real, and K_b = B g(A) C has the nonzero eigenvalues
    Re lam on the eigenvectors B q (K's own), col(K_b) in col(B) (write space kept), rank =
    #{Re lam != 0}, and K's real eigenvalues unchanged. Symbolic for one 2 x 2 real block
    M [[a, b], [-b, a]] M^-1 (g(A) A = a I, real); numerical for heads (300 draws, d = 48, k = 8).
    Does NOT cover a defective A (the runner refuses a head whose eigenvector matrix has
    condition number above its placed bound).
C3  ||Z M Z^T||_F^2 = tr(Gm M Gm M^T), Gm = Z^T Z: the control's scale is exact through a
    2k x 2k Gram (symbolic, d = 3, k = 1; numerical at d = 48, k = 8).
C4  Arm (a) removes exactly K's antisymmetric part: K - sym(K) = skew(K) and
    ||K||^2 = ||sym K||^2 + ||skew K||^2 (symbolic, d = 3). So the arm's edit size is ||skew K||.
C5  Selected pairs: g = Re lam / lam on a set of eigenvalues closed under conjugation, 1 elsewhere,
    is real, and only those pairs turn real (numerical, 200 draws). This is the S-arm and its
    random-pair control (design "Q3").
C6  NOT a theorem about Pythia, a numerical fact about the reading: the energy E_1 on the sphere
    (identity Gram, beta = 1, as A5's e1) is not monotone along the attention-only particle flow
    with a symmetric V and an untied Q^T K in some draws, and is monotone in every draw with
    Q^T K = V = I (Euler steps, n = 12, d = 6, 200 draws). The gradient-flow results S2 leans on
    tie V to Q^T K; a symmetric V alone is not in them, so S2's Q1 has no theorem predicting
    "monotone" (design "What a reading can mean"). Beside, the baseline: the same draws with V
    unsymmetrised (59 of 200 against 53, added after `/challenge-pr` on #246, finding 7). The
    check does NOT model MLPs, LayerNorm, multiple heads or causal masking.
C7  Arm (b-schur), rotation removed in the residual's metric: B = U R, T = R C U, real Schur
    T = V S V^T, each 2x2 block of S replaced by its symmetric part, K' = U V S' V^T U^T + U N.
    Numerical (200 Gaussian heads, d = 48, k = 8): every nonzero eigenvalue real, write space
    col(B) kept, ||K' - K|| = ||S' - S||, K' inside the head's span Z (so one hook core). The
    medians of ||K' - K|| / ||K|| it prints (Schur-plane against eigenbasis Re lam) are on Gaussian
    heads, NOT Pythia's.
C8  Doubly matched controls: alpha K + N_perp (N_perp random in the span, orthogonal to K, of the
    arm's remaining size) has the arm's edit size and the arm's edited-head norm, through the Gram
    (numerical, 100 draws).

Run: python3 tools/math_checks/s2_ov_arms.py
"""
import numpy as np
import sympy as sp

results = []


def record(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"      {detail}")
    results.append(bool(ok))


def zero(M):
    return sp.simplify(M) == sp.zeros(*M.shape)


# C1 ------------------------------------------------------------------------------------------
d, k = 3, 1
B = sp.Matrix(d, k, sp.symbols("b0:3"))
C = sp.Matrix(k, d, sp.symbols("c0:3"))
Z = B.row_join(C.T)
E = sp.Matrix([[0, 1], [0, 0]])
J = (E + E.T) / 2
K = B * C
record("C1 K = Z E Z^T", zero(K - Z * E * Z.T))
record("C1 sym(K) = Z J Z^T", zero((K + K.T) / 2 - Z * J * Z.T))
k2 = 2
B2 = sp.Matrix(d, k2, sp.symbols("p0:6"))
C2 = sp.Matrix(k2, d, sp.symbols("q0:6"))
G2 = sp.Matrix(k2, k2, sp.symbols("g0:4"))
Z2 = B2.row_join(C2.T)
core = sp.zeros(2 * k2, 2 * k2)
core[:k2, k2:] = G2
record("C1 B G C = Z [[0, G], [0, 0]] Z^T (k = 2)", zero(B2 * G2 * C2 - Z2 * core * Z2.T))

# C2 ------------------------------------------------------------------------------------------
a, b = sp.symbols("a b", real=True)
m = sp.Matrix(2, 2, sp.symbols("m0:4", real=True))
R = sp.Matrix([[a, b], [-b, a]])
A = m * R * m.inv()
# On the block, Q diag(Re lam / lam) Q^-1 = m (a R^-1) m^-1, since R's complex structure commutes.
gA = m * (a * R.inv()) * m.inv()
record("C2 g(A) A = a I on a real 2x2 block (real, no rotation left)", zero(gA * A - a * sp.eye(2)))
record("C2 g(A) commutes with A", zero(gA * A - A * gA))

rng = np.random.default_rng(20261021)


def g_of(A, select=None):
    lam, Q = np.linalg.eig(A)
    f = np.ones_like(lam)
    idx = np.arange(lam.size) if select is None else select
    f[idx] = lam[idx].real / lam[idx]
    return (Q @ np.diag(f) @ np.linalg.inv(Q)), lam


def nz_eigs(M, tol=1e-8):
    ev = np.linalg.eigvals(M)
    return ev[np.abs(ev) > tol * max(np.abs(ev).max(), 1e-300)]


def match(x, y, tol):
    """Each of x within tol of a distinct y (greedy on sorted complex values)."""
    if x.size != y.size:
        return False
    y = list(y)
    for v in x:
        j = int(np.argmin([abs(v - w) for w in y]))
        if abs(v - y[j]) > tol:
            return False
        y.pop(j)
    return True


ok_real = ok_eig = ok_vec = ok_col = ok_rank = ok_realeig = True
dd, kk = 48, 8
for t in range(300):
    Bn = rng.standard_normal((dd, kk))
    Cn = rng.standard_normal((kk, dd))
    An = Cn @ Bn
    g, lam = g_of(An)
    ok_real &= np.abs(g.imag).max() < 1e-9
    g = g.real
    Kb = Bn @ g @ Cn
    ok_eig &= match(nz_eigs(Kb), lam.real[np.abs(lam.real) > 1e-8], 1e-6 * np.abs(lam).max())
    _, Qn = np.linalg.eig(An)
    V = Bn @ Qn
    ok_vec &= np.abs(Kb @ V - V * lam.real[None, :]).max() < 1e-7 * np.abs(lam).max() * np.abs(V).max()
    P_B = Bn @ np.linalg.pinv(Bn)
    ok_col &= np.abs(P_B @ Kb - Kb).max() < 1e-8 * np.abs(Kb).max()
    ok_rank &= np.linalg.matrix_rank(Kb, tol=1e-8 * np.abs(Kb).max()) == int((np.abs(lam.real) > 1e-8).sum())
    realK = lam[np.abs(lam.imag) < 1e-10].real
    ok_realeig &= all(np.min(np.abs(nz_eigs(Kb) - r)) < 1e-6 * np.abs(lam).max() for r in realK)
record("C2 g(A) is real (300 heads)", ok_real)
record("C2 K_b's nonzero eigenvalues are Re lam", ok_eig)
record("C2 K_b's eigenvectors are K's (B q)", ok_vec)
record("C2 col(K_b) in col(B): write space kept", ok_col)
record("C2 rank K_b = #{Re lam != 0}", ok_rank)
record("C2 K's real eigenvalues are kept", ok_realeig)

# C3 ------------------------------------------------------------------------------------------
Mc = sp.Matrix(2, 2, sp.symbols("r0:4"))
lhs = sum(x ** 2 for x in (Z * Mc * Z.T))
Gm = Z.T * Z
record("C3 ||Z M Z^T||^2 = tr(Gm M Gm M^T) (symbolic)", sp.simplify(lhs - (Gm * Mc * Gm * Mc.T).trace()) == 0)
ok = True
for t in range(50):
    Zn = rng.standard_normal((dd, 2 * kk))
    Mn = rng.standard_normal((2 * kk, 2 * kk))
    Gn = Zn.T @ Zn
    ok &= abs(np.linalg.norm(Zn @ Mn @ Zn.T) ** 2 - np.trace(Gn @ Mn @ Gn @ Mn.T)) < 1e-8 * np.linalg.norm(Zn @ Mn @ Zn.T) ** 2
record("C3 the same, numerical (50 draws, d = 48, k = 8)", ok)

# C4 ------------------------------------------------------------------------------------------
Kf = sp.Matrix(3, 3, sp.symbols("k0:9"))
S = (Kf + Kf.T) / 2
W = (Kf - Kf.T) / 2
fro = lambda M: sum(x ** 2 for x in M)
record("C4 K - sym K = skew K", zero(Kf - S - W))
record("C4 ||K||^2 = ||sym K||^2 + ||skew K||^2", sp.expand(fro(Kf) - fro(S) - fro(W)) == 0)

# C5 ------------------------------------------------------------------------------------------
ok_real = ok_sel = True
for t in range(200):
    An = rng.standard_normal((kk, kk))
    lam = np.linalg.eigvals(An)
    cplx = np.flatnonzero(lam.imag > 1e-10)
    if cplx.size == 0:
        continue
    pick = cplx[rng.random(cplx.size) < 0.5]
    sel = []
    for i in pick:                                  # close under conjugation
        sel.append(i)
        sel.append(int(np.argmin(np.abs(lam - np.conj(lam[i])))))
    g, lam0 = g_of(An, np.array(sel, dtype=int) if sel else np.zeros(0, int))
    ok_real &= np.abs(g.imag).max() < 1e-9
    new = np.linalg.eigvals(g.real @ An)
    want = lam0.copy()
    want[sel] = lam0[sel].real
    ok_sel &= match(new, want, 1e-7 * np.abs(lam0).max())
record("C5 selected-pair g is real (200 draws)", ok_real)
record("C5 only the selected pairs turn real", ok_sel)

# C6 ------------------------------------------------------------------------------------------
def flow_energy(Q, Kq, V, x0, beta=1.0, dt=0.02, steps=150):
    x = x0.copy()
    out = []
    for _ in range(steps + 1):
        G = x @ x.T
        out.append(np.exp(beta * G).sum() / (2 * beta * x.shape[0] ** 2))
        s = beta * (x @ Q.T) @ (x @ Kq.T).T
        P = np.exp(s - s.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        v = P @ (x @ V.T)
        v -= (v * x).sum(1, keepdims=True) * x          # tangent to the sphere
        x = x + dt * v
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return np.array(out)


def violations(e, rel_tol=1e-3):
    return int((np.diff(e) < -rel_tol * np.abs(e[:-1])).sum())


n, dm = 12, 6
viol_sym, viol_gen, viol_id = 0, 0, 0
for t in range(200):
    x0 = rng.standard_normal((n, dm))
    x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
    M = rng.standard_normal((dm, dm))
    Vs = (M + M.T) / 2
    Q, Kq = rng.standard_normal((dm, dm)), rng.standard_normal((dm, dm))
    viol_sym += violations(flow_energy(Q, Kq, Vs, x0)) > 0
    viol_gen += violations(flow_energy(Q, Kq, M, x0)) > 0     # the baseline: the same draw, V = M
    I = np.eye(dm)
    viol_id += violations(flow_energy(I, I, I, x0)) > 0
print(f"      C6: untied QK, E_1 non-monotone with V = sym(M) in {viol_sym} of 200, with V = M "
      f"(not symmetric) in {viol_gen}; Q = K = V = I: {viol_id} of 200")
record("C6 a symmetric V alone does not make E_1 monotone (some draw violates)", viol_sym > 0)
record("C6 Q = K = V = I: E_1 monotone in every draw", viol_id == 0)

# C7 ------------------------------------------------------------------------------------------
# Arm (b) in the residual's metric (design "Arm (b): two definitions"): B = U R (QR), the restriction
# T = R C U of K to col(B), K = U T U^T + U N with N = R C (I - U U^T); real Schur T = V S V^T;
# each 2x2 diagonal block of S replaced by its symmetric part (S'); K' = U V S' V^T U^T + U N.
import scipy.linalg as sla

ok_real = ok_col = ok_size = ok_span = True
ratio_schur, ratio_eig = [], []
for t in range(200):
    Bn = rng.standard_normal((dd, kk))
    Cn = rng.standard_normal((kk, dd))
    U, Rr = np.linalg.qr(Bn)
    T = Rr @ Cn @ U
    N = Rr @ Cn @ (np.eye(dd) - U @ U.T)
    S_, Vv = sla.schur(T, output="real")
    S2 = S_.copy()
    i = 0
    while i < kk:
        if i + 1 < kk and abs(S_[i + 1, i]) > 1e-12:
            blk = S_[i:i + 2, i:i + 2]
            S2[i:i + 2, i:i + 2] = (blk + blk.T) / 2
            i += 2
        else:
            i += 1
    Kn = Bn @ Cn
    Ks = U @ Vv @ S2 @ Vv.T @ U.T + U @ N
    ev = nz_eigs(Ks)
    ok_real &= np.abs(ev.imag).max() < 1e-7 * np.abs(ev).max()
    ok_col &= np.abs(U @ U.T @ Ks - Ks).max() < 1e-8 * np.abs(Ks).max()
    ok_size &= abs(np.linalg.norm(Ks - Kn) - np.linalg.norm(S2 - S_)) < 1e-8 * np.linalg.norm(Kn)
    Zn = np.hstack([Bn, Cn.T])                       # K' lies in the head's span: Z M Z^T for some M
    Pz = Zn @ np.linalg.pinv(Zn)
    ok_span &= np.abs(Pz @ Ks @ Pz - Ks).max() < 1e-8 * np.abs(Ks).max()
    ratio_schur.append(np.linalg.norm(Ks - Kn) / np.linalg.norm(Kn))
    g, _ = g_of(Cn @ Bn)
    ratio_eig.append(np.linalg.norm(Bn @ g.real @ Cn - Kn) / np.linalg.norm(Kn))
print(f"      C7: ||K' - K|| / ||K|| on Gaussian heads, median: Schur-plane {np.median(ratio_schur):.2f}, "
      f"eigenbasis Re lam {np.median(ratio_eig):.2f} (random heads, not Pythia's)")
record("C7 Schur-plane de-rotation: every nonzero eigenvalue real", ok_real)
record("C7 Schur-plane de-rotation keeps the write space col(B)", ok_col)
record("C7 its edit size is ||S' - S|| (the blocks' skew parts only)", ok_size)
record("C7 it lies in the head's span Z (one hook core M)", ok_span)

# C8 ------------------------------------------------------------------------------------------
# Doubly matched controls (design "Controls"): for an arm's edit D, alpha = <D, K> / ||K||^2 and a
# random N_perp in the span, orthogonal to K, of norm ||D - alpha K||: the control alpha K + N_perp
# has ||control|| = ||D|| and ||K + control|| = ||K + D|| exactly, both through the Gram.
ok = True
for t in range(100):
    Zn = rng.standard_normal((dd, 2 * kk))
    Gn = Zn.T @ Zn
    ip = lambda M1, M2: float(np.trace(Gn @ M1 @ Gn @ M2.T))   # <Z M1 Z^T, Z M2 Z^T>_F
    Ec = np.zeros((2 * kk, 2 * kk)); Ec[:kk, kk:] = np.eye(kk)
    Dm = rng.standard_normal((2 * kk, 2 * kk))
    alpha = ip(Dm, Ec) / ip(Ec, Ec)
    Rn = rng.standard_normal((2 * kk, 2 * kk))
    Rn = Rn - ip(Rn, Ec) / ip(Ec, Ec) * Ec
    Rn *= np.sqrt(ip(Dm - alpha * Ec, Dm - alpha * Ec) / ip(Rn, Rn))
    ctl = alpha * Ec + Rn
    ok &= abs(ip(ctl, ctl) - ip(Dm, Dm)) < 1e-8 * ip(Dm, Dm)
    ok &= abs(ip(Ec + ctl, Ec + ctl) - ip(Ec + Dm, Ec + Dm)) < 1e-8 * ip(Ec + Dm, Ec + Dm)
record("C8 doubly matched control: same edit size and same edited-head norm (100 draws)", ok)

print(f"\n{sum(results)}/{len(results)} passed")
raise SystemExit(0 if all(results) else 1)
