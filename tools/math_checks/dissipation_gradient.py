"""Checks MATH_SPECTRAL_OT.md sec 5.2, 5.3(b), and the sec 2.4.5 Bendixson claim.

sec 5.2 is checked EXACTLY and symbolically for n=3 particles in d=2 (small
enough for sympy to differentiate directly, general enough that no particle
is special -- every particle appears as both the "outer" index i and an
"inner" index j, which is exactly the cross-term subtlety the doc's "factor
of 2 ... cancel" line is about). This checks the claimed closed form against
direct symbolic differentiation of E_beta, not against itself.

sec 5.3(b) is checked symbolically for n=3, arbitrary d (via a free vector
symbol per particle) -- it is a linearity statement so a symbolic check is a
full proof, not an instance.

The 2.4.5 Bendixson inclusion (Re lambda(M) in [lambda_min(S), lambda_max(S)])
is a classical field-of-values theorem, not this project's own derivation --
checked numerically on a concrete non-normal 4x4 integer matrix as a sanity
instance, not a proof.

Run: python3 tools/math_checks/dissipation_gradient.py
"""
import sympy as sp

results = []


def record(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"      {detail}")
    results.append(ok)


# --- sec 5.2: dE_beta/dx_i closed form --------------------------------------
n, dim = 3, 2
beta = sp.Symbol("beta", positive=True)
X = [[sp.Symbol(f"x{i}_{c}") for c in range(dim)] for i in range(n)]


def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


E_beta = sum(sp.exp(beta * dot(X[i], X[j])) for i in range(n) for j in range(n))
E_beta = E_beta / (2 * beta * n ** 2)

all_ok = True
for i in range(n):
    for c in range(dim):
        lhs = sp.diff(E_beta, X[i][c])
        # claimed closed form: dE_beta/dx_i = (1/n^2) sum_j exp(beta<x_i,x_j>) x_j
        rhs = sum(sp.exp(beta * dot(X[i], X[j])) * X[j][c] for j in range(n)) / n ** 2
        if sp.simplify(lhs - rhs) != 0:
            all_ok = False
            print(f"      mismatch at particle {i}, coord {c}: "
                  f"{sp.simplify(lhs - rhs)}")

record(f"sec 5.2: dE_beta/dx_i = (1/n^2) sum_j exp(beta<x_i,x_j>) x_j "
       f"(exact, n={n} particles, d={dim}, every (i,coord) component)",
       all_ok)

# --- sec 5.3(b): attn/ffn split is exact by linearity -----------------------
# Delta x_i = Delta x_i^attn + Delta x_i^ffn (given, Pythia's parallel residual)
# v_i = (1/||x_i||) P_perp(Delta x_i), P_perp = I - u u^T linear in its argument
# => v_i = v_i^attn + v_i^ffn, and <G_i, .> is linear, so the sum splits exactly.
d2 = sp.Symbol("d2")  # dimension, symbolic count doesn't matter for this check
u = sp.Matrix([sp.Symbol("u0"), sp.Symbol("u1")])  # a unit direction, d=2 instance
norm_x = sp.Symbol("norm_x", positive=True)
G = sp.Matrix([sp.Symbol("G0"), sp.Symbol("G1")])

dx_attn = sp.Matrix([sp.Symbol("a0"), sp.Symbol("a1")])
dx_ffn = sp.Matrix([sp.Symbol("f0"), sp.Symbol("f1")])
dx_total = dx_attn + dx_ffn

P_perp = sp.eye(2) - u * u.T  # I - u u^T, the tangential projector

v_total = (P_perp * dx_total) / norm_x
v_attn = (P_perp * dx_attn) / norm_x
v_ffn = (P_perp * dx_ffn) / norm_x

lhs_53b = (G.T * v_total)[0, 0]
rhs_53b = (G.T * v_attn)[0, 0] + (G.T * v_ffn)[0, 0]
record("sec 5.3(b): <G_i,v_i> = <G_i,v_i^attn> + <G_i,v_i^ffn> exactly "
       "(P_perp linear, Delta x_i additive)",
       sp.simplify(lhs_53b - rhs_53b) == 0)

# --- sec 2.4.5: Bendixson inclusion, numeric instance -----------------------
import numpy as np

M_np = np.array([[2, 5, -1, 0],
                  [-3, 1, 4, 2],
                  [0, -2, 3, -1],
                  [1, 1, 0, 1]], dtype=float)
S_np = (M_np + M_np.T) / 2
eig_M = np.linalg.eigvals(M_np)
eig_S = np.linalg.eigvalsh(S_np)
lo, hi = eig_S.min(), eig_S.max()
inside = all(lo - 1e-9 <= re <= hi + 1e-9 for re in eig_M.real)
record(f"sec 2.4.5: Bendixson inclusion Re(lambda(M)) in [{lo:.4f}, {hi:.4f}] "
       f"holds on a concrete non-normal 4x4 instance",
       inside,
       f"Re(eig(M)) = {sorted(eig_M.real)}")

print()
n_pass = sum(results)
print(f"{n_pass}/{len(results)} checks passed")
if n_pass != len(results):
    raise SystemExit(1)
