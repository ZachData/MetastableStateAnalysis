"""Checks the closed-form claims in p1e_energy_field/design-1e.md "The math" (M1-M6).

M1  E_beta = (1/2 beta) sum_i phi_beta(x_i), phi_beta(x) = sum_j exp(beta <x, x_j>); so the
    per-token deviations phi_beta(x_i) - mean_i phi_beta(x_i) sum to 0. Symbolic, n = 3, d = 2:
    a full proof (an identity in the symbols). Does NOT say which per-token quantity Pythia's
    dynamics changes: with learned Q, K, V there is no single E_beta it must follow.
M2  On the unit sphere the tangential gradient of log phi_beta at x is beta * P_x^perp m(x),
    m(x) = sum_j softmax_j(beta <x, x_j>) x_j: the mean-shift direction is the field's ascent
    direction. Symbolic in the ambient gradient, n = 3, d = 2; the projection is then linear algebra.
    Does NOT show that a trained attention update points along it (that is U2's measurement) or
    hold for the causal sum's per-token energies beyond replacing the sum's range.
M3  For n <= d + 1 unit vectors, sum_{i != j} exp(beta <x_i, x_j>) >= n(n-1) exp(-beta/(n-1)),
    with equality at the regular simplex: the repulsive (V = -I) energy minimiser. The two steps
    (sum_{i != j} <x_i, x_j> = |sum x|^2 - n >= -n; Jensen on exp) are checked symbolically, and the
    equality case numerically at n = 4, d = 3. Does NOT prove anything about Pythia, whose V is not
    -I and whose mask is causal (math-10.md sec 7.5).
M4  As beta -> 0, m(x) -> the plain mean of the x_j, with first-order correction
    (beta/n) sum_j (<x, x_j> - mean_k <x, x_k>) x_j: at small beta the field only says "towards the
    centroid", so its local part is m_beta - m_0. Series check, n = 3, d = 2. Does NOT bound the
    remainder at beta = 3.5, where the series is not used.
M5  On the unit sphere exp(beta <x, y>) = exp(beta) exp(-(beta/2) |x - y|^2): log of the mean
    pairwise phi_beta is beta plus Wang & Isola's uniformity log E exp(-t |x - y|^2) at t = beta/2.
    Does NOT carry over their claims about what uniformity predicts (contrastive features, not LMs).
M6  Critical points of phi_beta on the sphere lie in span(x_j) OR have m(x) = 0: grad = beta P_x^perp
    m(x) vanishes only where m(x) is parallel to x (so x is in the span) or m(x) = 0. *Corrected after
    /challenge-pr on #156, finding 4:* for x orthogonal to the span all weights are equal and m(x) is
    the plain mean of the x_j, which is 0 in a cloud-centred frame: there the whole subsphere
    orthogonal to the span is critical (the field's floor). Checked: (a) ascent from off-span starts
    ends in the span (maxima); (b) on a centred cloud, a point orthogonal to the span has zero
    gradient; (c) on uncentred rows with nonzero mean the same point does not. Does NOT locate
    saddles (U3's job) or count wells.

Run: python3 tools/math_checks/energy_field_1e.py
"""
import numpy as np
import sympy as sp

results = []


def record(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"      {detail}")
    results.append(bool(ok))


beta = sp.Symbol("beta", positive=True)
n, dim = 3, 2
X = [sp.Matrix([sp.Symbol(f"x{i}_{c}") for c in range(dim)]) for i in range(n)]
y = sp.Matrix([sp.Symbol(f"y_{c}") for c in range(dim)])


def phi(x):
    return sum(sp.exp(beta * x.dot(xj)) for xj in X)


# --- M1 ------------------------------------------------------------------------
E = sum(sp.exp(beta * X[i].dot(X[j])) for i in range(n) for j in range(n)) / (2 * beta)
record("M1 E = (1/2b) sum_i phi(x_i)", sp.simplify(E - sum(phi(X[i]) for i in range(n)) / (2 * beta)) == 0)
dev = [phi(X[i]) - sum(phi(X[k]) for k in range(n)) / n for i in range(n)]
record("M1 deviations sum to 0", sp.simplify(sum(dev)) == 0)

# --- M2 ------------------------------------------------------------------------
grad = sp.Matrix([sp.diff(sp.log(phi(y)), y[c]) for c in range(dim)])
w = [sp.exp(beta * y.dot(xj)) / phi(y) for xj in X]
m = sum((w[j] * X[j] for j in range(n)), sp.zeros(dim, 1))
record("M2 ambient grad log phi = beta m(y)", sp.simplify(grad - beta * m) == sp.zeros(dim, 1))

# --- M3 ------------------------------------------------------------------------
xs = [sp.Matrix([sp.Symbol(f"z{i}_{c}") for c in range(3)]) for i in range(4)]
S = sum(xs, sp.zeros(3, 1))
lhs = sum(xs[i].dot(xs[j]) for i in range(4) for j in range(4) if i != j)
rhs = S.dot(S) - sum(x.dot(x) for x in xs)
record("M3 sum_{i!=j} <x_i,x_j> = |sum x|^2 - sum |x_i|^2", sp.expand(lhs - rhs) == 0)
uu = sp.Symbol("u", real=True)
record("M3 exp convex (Jensen applies)", sp.diff(sp.exp(beta * uu), uu, 2).is_positive)
simplex = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
G = simplex @ simplex.T
b = 3.5
e_simplex = np.exp(b * G[~np.eye(4, dtype=bool)]).sum()
bound = 4 * 3 * np.exp(-b / 3)
rng = np.random.default_rng(0)
worse = []
for _ in range(2000):
    Z = rng.standard_normal((4, 3))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    Gz = Z @ Z.T
    worse.append(np.exp(b * Gz[~np.eye(4, dtype=bool)]).sum() >= e_simplex - 1e-9)
record("M3 simplex attains the bound (n=4, d=3, beta=3.5)", abs(e_simplex - bound) < 1e-9,
       f"{e_simplex} vs {bound}")
record("M3 2000 random configs never beat the simplex", all(worse))

# --- M4 ------------------------------------------------------------------------
m0 = sum(X, sp.zeros(dim, 1)) / n
ip = [y.dot(xj) for xj in X]
first = beta / n * sum(((ip[j] - sum(ip) / n) * X[j] for j in range(n)), sp.zeros(dim, 1))
ser = [sp.series(m[c], beta, 0, 2).removeO() for c in range(dim)]
record("M4 m(y) = mean + (beta/n) sum_j (<y,x_j> - mean) x_j + O(beta^2)",
       all(sp.simplify(ser[c] - (m0[c] + first[c])) == 0 for c in range(dim)))

# --- M5 ------------------------------------------------------------------------
u = sp.Matrix(sp.symbols("u0:3", real=True))
v = sp.Matrix(sp.symbols("v0:3", real=True))
on_sphere = {u[2]: sp.sqrt(1 - u[0] ** 2 - u[1] ** 2), v[2]: sp.sqrt(1 - v[0] ** 2 - v[1] ** 2)}
lhs5 = (beta * u.dot(v)).subs(on_sphere)
rhs5 = (beta - beta / 2 * (u - v).dot(u - v)).subs(on_sphere)
record("M5 beta<x,y> = beta - (beta/2)|x-y|^2 on the sphere", sp.simplify(sp.expand(lhs5 - rhs5)) == 0)

# --- M6 ------------------------------------------------------------------------
d6, n6, b6 = 40, 6, 3.5
A = rng.standard_normal((n6, d6))
A /= np.linalg.norm(A, axis=1, keepdims=True)
Q, _ = np.linalg.qr(A.T)                         # orthonormal basis of span
in_span = []
for _ in range(50):
    x = rng.standard_normal(d6)
    x /= np.linalg.norm(x)
    for _ in range(5000):                        # projected gradient ascent on log phi (mean shift)
        wts = np.exp(b6 * (A @ x - 1))
        mm = wts @ A / wts.sum()
        g = mm - (mm @ x) * x
        if np.linalg.norm(g) < 1e-12:
            break
        x = mm / np.linalg.norm(mm)
    in_span.append(np.linalg.norm(x - Q @ (Q.T @ x)) < 1e-8)
record("M6 (a) maxima reached from off-span starts lie in span(x_j)", all(in_span))
C = A - A.mean(0)                                # cloud-centred rows (the centred arm)
C /= np.linalg.norm(C, axis=1, keepdims=True)
Qc, _ = np.linalg.qr(C.T)
z = rng.standard_normal(d6)
z -= Qc @ (Qc.T @ z)
z /= np.linalg.norm(z)                           # orthogonal to the centred span


def tan_grad(rows, x):
    wts = np.exp(b6 * (rows @ x - 1))
    mm = wts @ rows / wts.sum()
    return mm - (mm @ x) * x


Cm = C - C.mean(0)                                # exact zero mean, then the check point
zc = rng.standard_normal(d6)
Qm, _ = np.linalg.qr(Cm.T)
zc -= Qm @ (Qm.T @ zc)
zc /= np.linalg.norm(zc)
record("M6 (b) centred, zero-mean rows: a point orthogonal to the span is critical",
       np.linalg.norm(tan_grad(Cm, zc)) < 1e-12, f"{np.linalg.norm(tan_grad(Cm, zc))}")
z2 = rng.standard_normal(d6)
z2 -= Q @ (Q.T @ z2)
z2 /= np.linalg.norm(z2)
record("M6 (c) uncentred unit rows: a point orthogonal to the span is not critical",
       np.linalg.norm(tan_grad(A, z2)) > 1e-3, f"{np.linalg.norm(tan_grad(A, z2))}")

print(f"\n{sum(results)}/{len(results)} passed")
raise SystemExit(0 if all(results) else 1)
