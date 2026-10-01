import numpy as np, sys
sys.path.insert(0, ".")
import p1d_cluster_ensemble.position_null as pn
from p1d_cluster_ensemble.admit import admit_record
from p1d_cluster_ensemble.gaussian_null import _unit_rows
orig_fit, orig_draw = pn.fit, pn.draw
def scaled_draw(mu, R, rng):
    s = np.linalg.norm(R, axis=1); s = np.maximum(s, 1e-12)
    Rn = R / s[:, None]; Rn = Rn - Rn.mean(0)
    n = R.shape[0]; G = rng.standard_normal((n, n)) / np.sqrt(n)
    N = G @ Rn
    N *= (s / np.maximum(np.linalg.norm(N, axis=1), 1e-12))[:, None]   # each row keeps its own residual norm
    return _unit_rows(mu + N)
n, d, a = 150, 300, 6.0
def run(seed, mode, frame, opening=True):
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n, d)); E = rng.standard_normal((n, d))
    PM = a * np.cumsum(V, 0) / np.arange(1, n + 1)[:, None] if opening else 0 * V
    X = E + PM
    def oracle(Z, p, null):
        U = X / np.linalg.norm(X, axis=1, keepdims=True)
        _, s, Vt = np.linalg.svd(U, full_matrices=False); B = Vt[s > 1e-12 * s[0]].T
        M = (PM / np.linalg.norm(X, axis=1, keepdims=True)) @ B
        mu = M + (Z - M).mean(0); R = Z - mu; R -= R.mean(0)
        return mu, R, {"null": "oracle"}
    pn.fit = (lambda Z, p, nl: oracle(Z, p, nl)) if mode.startswith("oracle") else orig_fit
    pn.draw = scaled_draw if mode.endswith("+scale") else orig_draw
    rec = admit_record(X, frame, 39, seed, null_kind="prefix", positions=np.arange(n))
    return any(r["admitted_excess"] and 0 in r["members"] for r in rec["arms"]["2"]["groups"]), \
           any(r["admitted_excess"] for r in rec["arms"]["2"]["groups"])
for mode, frame in (("oracle+scale", "raw"), ("prefix+scale", "raw"), ("prefix+scale", "centred")):
    res = [run(s, mode, frame) for s in range(10)]
    print(mode, frame, sum(r[0] for r in res), "of 10 admit a group holding position 0")
res = [run(s, "prefix+scale", "centred", opening=False) for s in range(10)]
print("no opening (pure noise), prefix+scale centred:", sum(r[1] for r in res), "of 10 admit anything")
