import numpy as np, sys
sys.path.insert(0, ".")
import p1d_cluster_ensemble.position_null as pn
from p1d_cluster_ensemble.admit import layer_groups, rank_p
from p1d_cluster_ensemble.gaussian_null import frame_vectors, span_coordinates
n, d, a = 150, 300, 6.0
def admit_pre(Y, frame, B, seed, kind):
    """Fit and draw in the un-normalised space, then apply the frame to each draw."""
    Z = span_coordinates(frame_vectors(Y, frame)[0])
    Ys = span_coordinates(Y)
    mu, R, info = pn.fit(Ys, np.arange(len(Y)), kind)
    _, rows, _ = layer_groups(Z, 2)
    rng = np.random.default_rng(seed); mx = []
    for _ in range(B):
        X = mu + (rng.standard_normal((n, n)) / np.sqrt(n)) @ R
        Xf = span_coordinates(frame_vectors(X, frame)[0])
        mx.append(max((r["excess"] for r in layer_groups(Xf, 2)[1]), default=0.0))
    return [r for r in rows if rank_p(r["excess"], np.array(mx)) <= 0.05], info
for kind in ("prefix", "smooth"):
  for frame in ("raw", "centred"):
    hit = 0; noise = 0; cs = []
    for s in range(10):
        rng = np.random.default_rng(s)
        V = rng.standard_normal((n, d)); E = rng.standard_normal((n, d))
        Y = E + a * np.cumsum(V, 0) / np.arange(1, n + 1)[:, None]
        adm, info = admit_pre(Y, frame, 39, s, kind); cs.append(info.get("c", info.get("h")))
        hit += any(0 in r["members"] for r in adm)
        adm0, _ = admit_pre(E, frame, 39, s, kind); noise += bool(adm0)
    print(f"{kind} {frame}: opening admitted {hit}/10; pure noise admits {noise}/10; fit {np.round(cs, 2).tolist()[:4]}")
