"""The matched-covariance Gaussian draw has the tokens' mean, covariance and
mean squared norm -- `p1d_cluster_ensemble/gaussian_null.py`.

WHY THIS EXISTS
---------------
`gaussian_draw` samples `x = mu + (G @ Zc) / sqrt(n)` row by row, with
`G` an n x n standard normal matrix and `Zc` the centred token rows. The
module docstring claims each row has mean `mu`, covariance
`Zc^T Zc / n` (plug-in, ddof=0), and hence `E|x|^2 = |mu|^2 + tr Sigma =
mean_i |z_i|^2`, which is 1 for unit rows: the draws match the tokens'
scale before they are renormed.

For one row `x = mu + sum_j g_j zc_j / sqrt(n)` with `g_j` iid N(0,1):
E[x] = mu, Cov[x] = sum_j zc_j zc_j^T / n. Checked symbolically at n=3,
d=2 with the rows as free symbols.

WHAT IT DOES NOT PROVE
----------------------
- Nothing about the renormalised draws: projecting to the sphere changes
  the covariance (the projected normal's moments have no closed form here).
  The match is exact before the projection only.
- Nothing about whether the plug-in covariance is the right null (SigClust
  shrinks its eigenvalues for HDLSS data; this does not).
- n=3, d=2 only; the algebra is the same for any n, d, but only this size
  is evaluated.
"""
import sympy as sp

results = []


def record(msg, ok, note=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + msg + (f"  ({note})" if note else ""))


n, d = 3, 2
Z = sp.Matrix(n, d, lambda i, j: sp.Symbol(f"z{i}{j}", real=True))
mu = sp.Matrix([[sum(Z[i, j] for i in range(n)) / n for j in range(d)]])
Zc = Z - sp.ones(n, 1) * mu

# E[g_j] = 0, E[g_j g_k] = delta_jk, so for x = mu + g^T Zc / sqrt(n):
mean_x = mu
cov_x = sp.zeros(d, d)
for j in range(n):
    cov_x += Zc[j, :].T * Zc[j, :] / n
plugin = (Zc.T * Zc) / n
record("Cov[x] equals the plug-in covariance Zc^T Zc / n",
       sp.simplify(cov_x - plugin) == sp.zeros(d, d))

e_norm2 = (mean_x * mean_x.T)[0, 0] + cov_x.trace()
mean_sq = sum((Z[i, :] * Z[i, :].T)[0, 0] for i in range(n)) / n
record("E|x|^2 = |mu|^2 + tr Sigma equals mean_i |z_i|^2",
       sp.simplify(sp.expand(e_norm2 - mean_sq)) == 0,
       "so unit token rows give E|x|^2 = 1 before the renormalisation")

print()
n_pass = sum(results)
print(f"{n_pass}/{len(results)} checks passed")
if n_pass != len(results):
    raise SystemExit(1)
