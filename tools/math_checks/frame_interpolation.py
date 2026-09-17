"""Checks MATH_SPECTRAL_OT.md sec 2.5.1, 2.5.2, 2.5.4 (the M -> M^T frame path).

Two different rigor levels, stated per check:
  * sec 2.5.1's rank bound is checked EXACTLY (integer/rational arithmetic,
    exact rank via sympy's rref).
  * sec 2.5.2/2.5.4's isometry-path properties are checked NUMERICALLY at a
    single concrete (d=4, k=2) Stiefel example with the paper's own polar
    retraction. This is an instance check, not a proof for a general gamma --
    the doc's own table already gives the proof (orthonormality of gamma(t)
    is enough; the polar retraction is one way to build such a gamma). What
    is verified here is that the retraction genuinely stays on the Stiefel
    manifold and that the claimed invariants hold along it, not just at the
    endpoints.

Run: python3 tools/math_checks/frame_interpolation.py
"""
import sympy as sp
from sympy import Rational as Rt

results = []


def record(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"      {detail}")
    results.append(ok)


# --- sec 2.5.1: rank(S) <= 2k where M = A B, A (d,k), B (k,d), d=5, k=2 -----
# Exact integers so rank is unambiguous (no float rank-deficiency ambiguity).
d, k = 5, 2
Amat = sp.Matrix([[1, 0], [2, 1], [0, 3], [1, 1], [4, 2]])          # (d,k)
Bmat = sp.Matrix([[1, 2, 0, 1, 1], [0, 1, 1, 2, 3]])                # (k,d)
assert Amat.shape == (d, k) and Bmat.shape == (k, d)
Mmat = Amat * Bmat
S = (Mmat + Mmat.T) / 2

# S as the stated concatenation product [A, B^T] [B; A^T] / 2
concat_left = Amat.row_join(Bmat.T)      # (d, 2k)
concat_right = Bmat.col_join(Amat.T)     # (2k, d)
S_concat = (concat_left * concat_right) / 2
record("sec 2.5.1: S = (M+M^T)/2 equals [A,B^T][B;A^T]/2 exactly",
       (sp.simplify(S - S_concat)).is_zero_matrix,
       f"residual nonzero: {S - S_concat}")

record(f"sec 2.5.1: rank(S) <= 2k = {2*k} (measured rank {S.rank()}, d={d})",
       S.rank() <= 2 * k,
       f"rank(S) = {S.rank()} exceeds 2k = {2*k}")

record(f"sec 2.5.1: rank(S) is NOT full rank d={d} (a head cannot write its own S)",
       S.rank() < d,
       f"rank(S) = {S.rank()} = d, contradicts the claim")


# --- sec 2.5.2 / 2.5.4: numeric instance on d=4, k=2 ------------------------
import numpy as np

rng = np.random.default_rng(0)


def random_stiefel(dd, kk):
    X = rng.standard_normal((dd, kk))
    Q, _ = np.linalg.qr(X)
    return Q[:, :kk]


dd, kk = 4, 2
U = random_stiefel(dd, kk)
V = random_stiefel(dd, kk)
Sigma = np.diag(sorted(rng.uniform(0.5, 2.0, size=kk), reverse=True))
M = U @ Sigma @ V.T


def polar_frame(t):
    Y = (1 - t) * U + t * V
    # gamma(t) = Y (Y^T Y)^{-1/2}, via eigendecomposition of the SPD Y^T Y.
    YtY = Y.T @ Y
    w, Q = np.linalg.eigh(YtY)
    assert (w > 1e-12).all(), "sigma_min(Y(t)) hit 0 -- refusal condition triggered"
    inv_sqrt = Q @ np.diag(w ** -0.5) @ Q.T
    return Y @ inv_sqrt


tol = 1e-8

g0 = polar_frame(0.0)
record("sec 2.5.2: gamma(0) = U", np.allclose(g0, U, atol=tol))

g1 = polar_frame(1.0)
record("sec 2.5.2: gamma(1) = V", np.allclose(g1, V, atol=tol))

sv_M = np.linalg.svd(M, compute_uv=False)
frob_M = np.linalg.norm(M, "fro")
rank_M = np.linalg.matrix_rank(M, tol=1e-9)

all_sv_ok, all_frob_ok, all_rank_ok, all_orthonormal_ok = True, True, True, True
for t in np.linspace(0, 1, 11):
    g_t = polar_frame(t)
    g_1mt = polar_frame(1 - t)
    Mt = g_t @ Sigma @ g_1mt.T

    if not np.allclose(g_t.T @ g_t, np.eye(kk), atol=1e-7):
        all_orthonormal_ok = False
    sv_t = np.linalg.svd(Mt, compute_uv=False)
    if not np.allclose(sorted(sv_t, reverse=True), sv_M, atol=1e-6):
        all_sv_ok = False
    if not np.isclose(np.linalg.norm(Mt, "fro"), frob_M, atol=1e-6):
        all_frob_ok = False
    if np.linalg.matrix_rank(Mt, tol=1e-9) != rank_M:
        all_rank_ok = False

record("sec 2.5.2: gamma(t) stays on the Stiefel manifold (gamma^T gamma = I) for t in [0,1]",
       all_orthonormal_ok)
record("sec 2.5.2: singular values of M(t) = Sigma for every sampled t",
       all_sv_ok)
record("sec 2.5.2: ||M(t)||_F = ||M||_F for every sampled t",
       all_frob_ok)
record("sec 2.5.2: rank M(t) = k for every sampled t",
       all_rank_ok)

M0 = polar_frame(0.0) @ Sigma @ polar_frame(1.0).T
record("sec 2.5.2: M(0) = M", np.allclose(M0, M, atol=1e-6))

M1 = polar_frame(1.0) @ Sigma @ polar_frame(0.0).T
record("sec 2.5.2: M(1) = M^T", np.allclose(M1, M.T, atol=1e-6))

g_half = polar_frame(0.5)
M_half = g_half @ Sigma @ g_half.T
record("sec 2.5.2: M(1/2) is symmetric",
       np.allclose(M_half, M_half.T, atol=1e-6))
eigvals_half = np.linalg.eigvalsh(M_half)
record("sec 2.5.2: M(1/2) is PSD (all eigenvalues >= 0)",
       bool((eigvals_half >= -1e-8).all()),
       f"eigenvalues: {eigvals_half}")

# --- sec 2.5.4: M_R = U R Sigma V^T, R in O(k) ------------------------------
theta = 0.7
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
assert np.allclose(R @ R.T, np.eye(kk), atol=1e-9)
M_R = U @ R @ Sigma @ V.T
sv_MR_full = np.linalg.svd(M_R, compute_uv=False)  # length dd; only top kk are nonzero
sv_MR = sorted(sv_MR_full, reverse=True)[:kk]
record("sec 2.5.4: sigma(M_R) = Sigma for R in O(k) (top-k singular values; rest are ~0, matching rank k)",
       np.allclose(sv_MR, np.diag(Sigma), atol=1e-6)
       and np.allclose(sorted(sv_MR_full, reverse=True)[kk:], 0.0, atol=1e-6),
       f"top-k: {sv_MR}, expected {np.diag(Sigma)}; tail: {sorted(sv_MR_full, reverse=True)[kk:]}")
record("sec 2.5.4: rank(M_R) = k",
       np.linalg.matrix_rank(M_R, tol=1e-9) == kk)
read_space_unchanged = np.allclose(
    np.linalg.svd(U.T @ np.linalg.svd(M_R, full_matrices=False)[0], compute_uv=False),
    np.ones(kk), atol=1e-6)
record("sec 2.5.4: write space span(U) unchanged by R (U still spans M_R's column space)",
       read_space_unchanged)

print()
n_pass = sum(results)
print(f"{n_pass}/{len(results)} checks passed")
if n_pass != len(results):
    raise SystemExit(1)
