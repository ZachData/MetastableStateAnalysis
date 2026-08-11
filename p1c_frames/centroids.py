"""
p1c_frames/centroids.py — which centroids feed sub-experiment F, and how
P-S1 is adjudicated across checkpoints.

THE PROBLEM AS ORIGINALLY STATED

`design_report` takes cluster centroids. Phase 1 produces three clusterings
per layer (kmeans at a silhouette-selected k, agglomerative at a threshold
sweep, HDBSCAN), they give different centroid counts m, and m enters the
random baseline — so it looked as though the clusterer choice moved the
reference and had to be fixed before F could run at all. That was the
reason F was left unwired in run_1c.

MEASUREMENT, WHICH CHANGES THE ANSWER

The matched-(m, d) baseline does its job. Q_k / Q_k^random for i.i.d.
uniform configurations at d = 256, over m from 4 to 128:

    m         4      8     16     32     64    128
    ratio Q1  1.104  1.048  0.967  0.922  0.951  0.901
    ratio Q2  1.003  1.001  1.021  0.994  0.993  0.986

Flat at 1 across a 32x range in m. And a genuinely sharp configuration
stays low at every m: the regular simplex — a spherical 1-design — gives
ratio Q1 = 0.000 at m = 5, 10, 20 and 40.

So **the ratio is comparable across different centroid counts**, and P-S1
can be adjudicated between checkpoints whose clusterings disagree on m.
The clusterer choice still has to be FIXED for a clean comparison, but it
does not have to be fixed by matching m, which is what made it look hard.

TWO THINGS THE SAME MEASUREMENT SETTLES

1. THE EFFECT-SIZE FLOOR. The random ratio has sd ~ 0.075-0.089 at
   m = 8, 32, 128, so the 2-sigma band is about 0.15. A trained-vs-step-0
   difference in Q_k ratio smaller than that is inside the sampling noise
   of the baseline itself. P-S1 must be adjudicated against this band, not
   against zero, and `random_band` returns it.

2. THE BAND NARROWS SHARPLY WITH k, WHICH IS THE OPPOSITE OF WHAT THE RAW
   RATIOS SUGGEST. Measured 2-sigma bands at d = 256:

       k          1      2      3
       m = 8    0.164  0.015  0.002
       m = 32   0.173  0.013  0.002
       m = 128  0.189  0.015  0.002

   Reading the raw ratios alone would suggest higher k is uninformative —
   the simplex gives 0.000 at k = 1 but 0.977 at k = 2, which looks like
   "no signal". It is not: at m = 20 the k = 2 band is 0.017, so a
   deviation of 0.023 is OUTSIDE it. Both k = 1 and k = 2 register as
   improvements for the simplex under the banded test.

   So higher k is MORE sensitive in relative terms, not less. Both the
   deviation and the noise shrink with k, and the banded test is what
   compares them; a fixed absolute tolerance at any k would be wrong in a
   different direction for each. t_max = 3 below is a cost choice — each
   degree needs its own baseline simulation — not a statement about where
   the power is.

THE CLUSTERER DECISION

Primary: **kmeans at the silhouette-selected k.** Not because it is the
best clusterer, but because it is the only one whose CENTROIDS Phase 1
already persists (`clusters.npz: kmeans_centroids_L{i}`); the other two
persist labels only, so their centroids must be recomputed from
activations and are therefore a second artifact dependency. Since the
ratio is m-comparable, the silhouette-varying k is not the objection it
appeared to be.

Secondary arms, both supported here and both worth running once as a
sensitivity check: agglomerative at the mid threshold (the only method
whose RULE is checkpoint-independent — kmeans selects k by silhouette on
the data, so the selection rule sees the checkpoint) and HDBSCAN (which
has noise tokens, excluded from centroids, and may be absent entirely).

If the three arms disagree about P-S1, that is the result — it means the
design signal is a property of the clustering rather than of the geometry.
"""

from __future__ import annotations

import numpy as np

from .design_test import design_report, random_baseline_Q, adjudicate_p_s1


METHODS = ("kmeans", "agglomerative", "hdbscan")


def centroids_from_labels(X: np.ndarray, labels, drop_noise: bool = True,
                          min_size: int = 1) -> tuple:
    """
    Cluster centroids from per-token labels.

    drop_noise : exclude label -1 (HDBSCAN's noise marker). Including noise
                 tokens as a cluster would put a centroid at the mean of
                 everything HDBSCAN could not assign, which is close to the
                 global centroid and would drag the configuration toward
                 its own common mode — the opposite of what a sharpness
                 test should measure.
    min_size   : drop clusters with fewer than this many tokens. A
                 singleton "centroid" is just a token, and a configuration
                 padded with singletons is measuring the token cloud, not
                 the cluster structure.

    Centroids are NOT normalized here — design_report normalizes, since the
    design question is about directions and centroids of clusters on the
    sphere lie strictly inside it.

    Returns (centroids, info).
    """
    X = np.asarray(X, dtype=np.float64)
    lab = np.asarray(labels).ravel()
    if lab.size != X.shape[0]:
        raise ValueError(f"{lab.size} labels for {X.shape[0]} tokens")

    uniq = [u for u in np.unique(lab) if not (drop_noise and u == -1)]
    cents, sizes, kept = [], [], []
    for u in uniq:
        m = lab == u
        if m.sum() < min_size:
            continue
        cents.append(X[m].mean(axis=0))
        sizes.append(int(m.sum()))
        kept.append(int(u))

    n_noise = int((lab == -1).sum())
    return (np.array(cents) if cents else np.zeros((0, X.shape[1]))), {
        "n_centroids": len(cents),
        "cluster_sizes": sizes,
        "labels_kept": kept,
        "n_noise_tokens": n_noise,
        "noise_fraction": float(n_noise / max(lab.size, 1)),
        "n_dropped_small": len(uniq) - len(cents),
        "min_size": min_size,
    }


def load_centroids(run_dir, layer: int, method: str = "kmeans",
                   activations: np.ndarray = None, **kw) -> tuple:
    """
    Centroids for one layer from a Phase 1 run directory.

    kmeans        : read directly from clusters.npz (`kmeans_centroids_L{i}`)
    agglomerative : recomputed from `agglom_mid_labels_L{i}` + activations
    hdbscan       : recomputed from `hdbscan_labels_L{i}` + activations

    `activations` is required for the latter two and its absence raises
    rather than silently falling back to kmeans — a sensitivity check that
    quietly returns the primary arm is not a sensitivity check.
    """
    from pathlib import Path
    p = Path(run_dir) / "clusters.npz"
    if not p.exists():
        raise FileNotFoundError(f"no clusters.npz under {run_dir}")
    z = np.load(p)

    if method == "kmeans":
        key = f"kmeans_centroids_L{layer}"
        if key not in z.files:
            raise KeyError(f"{p.name} has no {key}; Phase 1 saved no kmeans "
                           f"centroids for this layer")
        C = np.asarray(z[key], dtype=np.float64)
        return C, {"method": "kmeans", "n_centroids": int(C.shape[0]),
                   "source": "persisted"}

    key = {"agglomerative": f"agglom_mid_labels_L{layer}",
           "hdbscan": f"hdbscan_labels_L{layer}"}.get(method)
    if key is None:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if key not in z.files:
        raise KeyError(f"{p.name} has no {key}")
    if activations is None:
        raise ValueError(
            f"method={method!r} persists labels only, so centroids must be "
            f"recomputed and `activations` is required. Refusing to fall "
            f"back to kmeans — a sensitivity arm that silently returns the "
            f"primary arm is not one."
        )
    C, info = centroids_from_labels(activations, z[key], **kw)
    return C, {"method": method, "source": "recomputed", **info}


# ---------------------------------------------------------------------------
# The effect-size floor
# ---------------------------------------------------------------------------

def random_band(m: int, d: int, t_max: int = 3, n_trials: int = 200,
                n_sigma: float = 2.0, seed: int = 0) -> dict:
    """
    The sampling band of the Q_k ratio for i.i.d. uniform configurations at
    this (m, d).

    A trained-vs-step-0 difference smaller than this band is inside the
    noise of the baseline itself. Measured at d = 256: sd ~ 0.075-0.089
    across m = 8, 32, 128, i.e. a 2-sigma band around 0.15 — so P-S1
    requires a ratio difference of roughly that size before it means
    anything, and the original prediction's "low-order Gegenbauer moments
    smaller" carries no threshold at all.
    """
    ratios = np.empty((n_trials, t_max))
    base = random_baseline_Q(m, d, t_max=t_max, n_trials=n_trials, seed=seed)
    rng = np.random.default_rng(seed + 1)
    from .design_test import gegenbauer_moments
    for i in range(n_trials):
        X = rng.normal(size=(m, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        Q = gegenbauer_moments(X @ X.T, t_max=t_max, d=d, check=False)["Q"]
        ratios[i] = Q / np.maximum(base["mean"], 1e-30)
    return {
        "mean": ratios.mean(axis=0), "sd": ratios.std(axis=0),
        "band": float(n_sigma) * ratios.std(axis=0),
        "n_sigma": float(n_sigma), "m": int(m), "d": int(d),
        "n_trials": int(n_trials),
    }


# ---------------------------------------------------------------------------
# The P-S1 protocol
# ---------------------------------------------------------------------------

def run_design_test(centroids: np.ndarray, d: int = None, t_max: int = 3,
                    n_trials: int = 200) -> dict:
    """
    design_report plus the sampling band, which is what makes the numbers
    readable.

    t_max defaults to 3 on cost grounds — every degree needs its own
    baseline simulation — not because higher degrees lack power. The
    module docstring has the measured bands: they narrow from 0.17 at
    k = 1 to 0.002 at k = 3, so higher degrees are more sensitive in
    relative terms and raising t_max is worth the compute if a result
    hinges on it.

    Never read Q_k against a fixed absolute tolerance. Both the deviation
    and the noise shrink with k, at different rates, so a single tolerance
    is wrong in a different direction at every degree.
    """
    rep = design_report(centroids, d=d, t_max=t_max, n_trials=n_trials)
    if "Q_ratio" not in rep:
        return rep
    band = random_band(rep["n_centroids"], rep["d"], t_max=t_max,
                       n_trials=n_trials)
    rep["random_band"] = band["band"]
    rep["outside_band"] = (np.abs(np.asarray(rep["Q_ratio"]) - band["mean"])
                           > band["band"]).tolist()
    return rep


def adjudicate_p_s1_banded(trained: dict, step0: dict) -> dict:
    """
    P-S1 with the effect-size floor applied.

    The registered falsifier is "no difference", which has no threshold.
    This adds one: a degree counts as improved only when the trained-minus-
    step0 ratio difference exceeds the random band at the trained
    configuration's own (m, d). Without it, six degrees of pure sampling
    noise give a coin-flip's worth of "improvements" and a PARTIAL verdict
    on nothing.
    """
    base = adjudicate_p_s1(trained, step0)
    if "Q_ratio" not in trained or "Q_ratio" not in step0:
        return base

    rt = np.asarray(trained["Q_ratio"], dtype=np.float64)
    r0 = np.asarray(step0["Q_ratio"], dtype=np.float64)
    k = min(len(rt), len(r0))
    band = np.asarray(trained.get("random_band", np.zeros(k)), dtype=np.float64)[:k]
    diff = r0[:k] - rt[:k]
    meaningful = diff > band

    base["diff_vs_band"] = {
        "difference": diff.tolist(),
        "band": band.tolist(),
        "meaningful": meaningful.tolist(),
        "n_meaningful": int(meaningful.sum()),
    }
    if meaningful.any():
        base["banded_verdict"] = (
            f"{int(meaningful.sum())}/{k} degrees improve by more than the "
            f"random band. Degrees: "
            f"{[i+1 for i in range(k) if meaningful[i]]}.")
    else:
        base["banded_verdict"] = (
            "NO DEGREE improves by more than the random sampling band. "
            "Whatever the unbanded verdict says, this is not a detection — "
            "the effect is smaller than the baseline's own noise.")
    return base
