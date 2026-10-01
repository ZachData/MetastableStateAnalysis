import numpy as np, sys
sys.path.insert(0, ".")
import p1d_cluster_ensemble.position_null as pn
from p1d_cluster_ensemble.admit import admit_record
orig = pn.fit
n, d, a = 150, 300, 6.0
def run(seed, mode, frame):
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n, d)); E = rng.standard_normal((n, d))
    PM = a * np.cumsum(V, 0) / np.arange(1, n + 1)[:, None]; X = E + PM
    def oracle(Z, p, null):
        U = X / np.linalg.norm(X, axis=1, keepdims=True)
        _, s, Vt = np.linalg.svd(U, full_matrices=False); B = Vt[s > 1e-12 * s[0]].T
        if frame == "centred":
            return orig(Z, p, "gaussian")  # oracle only defined for raw here
        M = (PM / np.linalg.norm(X, axis=1, keepdims=True)) @ B
        mu = M + (Z - M).mean(0); R = Z - mu; R -= R.mean(0)
        return mu, R, {"null": "oracle", "r2": 0.0}
    pn.fit = (lambda Z, p, nl: oracle(Z, p, nl)) if mode == "oracle" else orig
    rec = admit_record(X, frame, 39, seed, null_kind="prefix" if mode != "gaussian" else "gaussian",
                       positions=np.arange(n))
    return any(r["admitted_excess"] and 0 in r["members"] for r in rec["arms"]["2"]["groups"])
for mode, frame in (("gaussian", "raw"), ("prefix", "raw"), ("oracle", "raw"), ("prefix", "centred")):
    print(mode, frame, sum(run(s, mode, frame) for s in range(10)), "of 10 seeds admit a group holding position 0")
