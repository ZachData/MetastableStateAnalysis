"""
p1d_cluster_ensemble/scale_spectrum.py — unit 3 of the Blocked 11⁗ programme:
one family on a continuous scale (`design-1d.md` "Unit 3"; the synthetic and
its first check are fixed there, "The synthetic and the first check", before
any run).

The family is the average-linkage merge tree on cosine distance
(`merge_tree.layer_merge_tree`), cut at each of ``GRID_N`` relative heights
``r`` (log-spaced, ``GRID_LO``..``GRID_HI``); the absolute cut is
``delta = r x`` the cloud's median pairwise cosine distance in its frame.
Per scale (`spectrum`), in each **arm** (``ARMS``: clusters of ``>=
SUBSTANTIAL_CLUSTER_SIZE`` tokens, and of ``>= 2``; design "The re-run"):

- **(a) stability**: Hennig's (2007) cluster-wise stability. For each
  cluster C of the arm's size, over ``N_SUBSAMPLES`` subsamples S of
  ``SUBSAMPLE_FRAC`` (without replacement) with ``|C ∩ S| >= 2``, the best
  Jaccard of ``C ∩ S`` against the clusters of the subsample's own tree cut
  at the same absolute ``delta``; the scale's value is the mean over the
  arm's clusters.
- **(b) stability against the Gaussian** (Blocked 15, option 1): its rank p
  (higher tail) and ``z_G`` among ``N_DRAWS`` matched-covariance Gaussian
  draws of the cloud (same frame; `gaussian_null.span_coordinates` +
  `gaussian_draw`, as unit 2), each cut at the same ``r`` times its own
  median with its own subsamples; a draw with no cluster of the arm's size
  scores 0. On real input the design ranks ``z_G`` among unit 2's re-inits;
  that reader is not built here. The count's rank p (the first check's
  failed (b)) is kept beside as ``p_count``.

A **robust plateau** (`robust_plateaus`) is a maximal run of ``>= MIN_RUN``
consecutive grid points with one count ``k >= 2`` and, at every point,
stability ``>= STABLE`` and rank p ``<= ALPHA``.

**The multi-scale synthetic** (`synthetic`; unit 4's row): 3 groups x 3
sub-groups from von Mises–Fisher draws on S^1023 (two planted spreads, set
as mean pairwise cosine distances ``d_f`` within a sub-group and ``d_c``
between sub-groups of one group), 100 uniform background points, a seeded
order, then the opening: `identity_sim`'s β = 0 causal flow to ``OPEN_T``,
then T1 (position 0 dropped). The spreads are exact in expectation: for
independent draws ``E[x·y] = E[x]·E[y]``, so two points of one sub-group
have mean cosine ``ρ_f²`` and two of sibling sub-groups ``ρ_f² ρ_c²``.

**First check** (`first_check`), per arm: in the centred frame, each planted
scale (3 groups; 9 sub-groups) is found by a robust plateau whose ARI to the
planted labels (planted tokens only) is ``>= ARI_BAR`` at every point, and at
most ``MAX_GAUSSIAN_PLATEAUS`` of ``N_GAUSSIAN_CHECKS`` matched-covariance
Gaussian clouds of the synthetic have a robust plateau.

Tier 1: exploratory, unregistered.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .constants import SUBSTANTIAL_CLUSTER_SIZE
from .gaussian_null import frame_vectors, gaussian_draw, span_coordinates
from .identity_sim import integrate_converged
from .merge_tree import labels_at_delta, layer_merge_tree
from .methods import LayerData

FRAMES = ("centred", "raw")

#: The scale grid (design row "grid"; PLACED).
GRID_LO, GRID_HI, GRID_N = 0.01, 1.5, 40
#: Hennig's subsampling (design row "per scale" (a)).
N_SUBSAMPLES = 50
SUBSAMPLE_FRAC = 0.8
#: Gaussian draws per cloud for (b).
N_DRAWS = 50
#: The arms (design "The re-run"): name -> the smallest cluster counted.
ARMS = {"main": SUBSTANTIAL_CLUSTER_SIZE, "size2": 2}
#: Hennig's (2007) "stable" bound; PLACED (not his 0.85 "highly stable").
STABLE = 0.75
ALPHA = 0.05
MIN_RUN = 3
MIN_K = 2

# The synthetic (design "The synthetic and the first check"; PLACED).
DIM = 1024
N_GROUPS = 3
SUB_SIZES = (34, 33, 33)
N_BACKGROUND = 100
D_FINE = 0.15
D_COARSE = 0.40
OPEN_T = 2.0
#: Positions 1..OPEN_K (after T1: rows 0..OPEN_K-1) are what the opening rule reads.
OPEN_K = 4
LADDER_D_FINE = (0.20, 0.25, 0.30)

ARI_BAR = 0.8
N_GAUSSIAN_CHECKS = 50
#: Pass bound (design "The re-run"; PLACED): at most this many Gaussian clouds with a plateau.
MAX_GAUSSIAN_PLATEAUS = 2
#: Seed offsets for independent streams within one seed.
_SUB, _NULL, _GAUSS = 1, 2, 100


def relative_grid() -> np.ndarray:
    return np.geomspace(GRID_LO, GRID_HI, GRID_N)


# ---------------------------------------------------------------------------
# von Mises–Fisher on S^{d-1}
# ---------------------------------------------------------------------------

def mean_cosine(kappa: float, d: int) -> float:
    """
    ``A_d(κ) = I_{d/2}(κ) / I_{d/2-1}(κ)``: the mean cosine of a vMF draw to
    its mean. By the backward recurrence ``R_ν = 1 / (2ν/κ + R_{ν+1})`` from
    ``R = 0`` far above ``ν`` (stable; scipy's ``ive`` underflows to 0/0 at
    order 512 for small κ).
    """
    nu = d / 2.0
    top = nu + 2.0 * kappa + 200.0
    r, v = 0.0, math.floor(top - nu) + nu
    while v >= nu:
        r = 1.0 / (2.0 * v / kappa + r)
        v -= 1.0
    return float(r)


def kappa_for(rho: float, d: int) -> float:
    """The concentration whose mean cosine to the mean direction is ``rho``."""
    from scipy.optimize import brentq
    if not 0.0 < rho < 1.0:
        raise ValueError(f"mean cosine must be in (0, 1), got {rho}")
    # Banerjee et al. (2005)'s approximation, bracketed by a factor 8 each way.
    k0 = rho * (d - rho ** 2) / (1 - rho ** 2)
    return float(brentq(lambda k: mean_cosine(k, d) - rho, k0 / 8, k0 * 8, xtol=1e-10, rtol=1e-12))


def vmf(mu: np.ndarray, kappa: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` draws from vMF(``mu``, ``kappa``) on S^{d-1} (Wood 1994)."""
    mu = np.asarray(mu, dtype=np.float64)
    mu = mu / np.linalg.norm(mu)
    d = mu.size
    b = (d - 1) / (2 * kappa + math.sqrt(4 * kappa ** 2 + (d - 1) ** 2))
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + (d - 1) * math.log(1 - x0 ** 2)
    w = np.empty(n)
    i = 0
    while i < n:
        z = rng.beta((d - 1) / 2.0, (d - 1) / 2.0)
        t = (1 - (1 + b) * z) / (1 - (1 - b) * z)
        if kappa * t + (d - 1) * math.log(1 - x0 * t) - c >= math.log(rng.uniform()):
            w[i] = t
            i += 1
    V = rng.standard_normal((n, d))
    V -= np.outer(V @ mu, mu)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return w[:, None] * mu + np.sqrt(np.clip(1 - w ** 2, 0, None))[:, None] * V


def _uniform_sphere(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    X = rng.standard_normal((n, d))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def synthetic(seed: int = 0, d_fine: float = D_FINE, d_coarse: float = D_COARSE,
              open_t: float = OPEN_T, dim: int = DIM) -> Dict:
    """
    The multi-scale synthetic after the opening and T1.

    Returns ``Y`` (n − 1, dim) in position order with position 0 dropped,
    ``fine`` (sub-group 0..8, background −1), ``coarse`` (group 0..2,
    background −1), ``order`` (the planted row at each kept position), and
    ``info`` (κ, the opening's ODE record).
    """
    if not 0 < d_fine < d_coarse < 1:
        raise ValueError(f"need 0 < d_fine < d_coarse < 1, got {d_fine}, {d_coarse}")
    rng = np.random.default_rng(seed)
    rho_f = math.sqrt(1 - d_fine)
    rho_c = math.sqrt((1 - d_coarse) / (1 - d_fine))
    k_f, k_c = kappa_for(rho_f, dim), kappa_for(rho_c, dim)
    rows, fine, coarse = [], [], []
    centres = _uniform_sphere(N_GROUPS, dim, rng)
    for g in range(N_GROUPS):
        subs = vmf(centres[g], k_c, len(SUB_SIZES), rng)
        for s, size in enumerate(SUB_SIZES):
            rows.append(vmf(subs[s], k_f, size, rng))
            fine += [g * len(SUB_SIZES) + s] * size
            coarse += [g] * size
    rows.append(_uniform_sphere(N_BACKGROUND, dim, rng))
    fine += [-1] * N_BACKGROUND
    coarse += [-1] * N_BACKGROUND
    X = np.vstack(rows)
    order = rng.permutation(X.shape[0])
    X = X[order]
    info = {"seed": int(seed), "d_fine": d_fine, "d_coarse": d_coarse, "dim": dim,
            "rho_fine": rho_f, "rho_coarse": rho_c, "kappa_fine": k_f, "kappa_coarse": k_c,
            "open_t": float(open_t), "n_planted": int(sum(SUB_SIZES) * N_GROUPS),
            "n_background": N_BACKGROUND}
    if open_t > 0:
        S, ode = integrate_converged(X, 0.0, "causal", times=(float(open_t),))
        X = S[-1]
        info["ode"] = ode
    keep = slice(1, None)  # T1
    return {"Y": X[keep], "fine": np.asarray(fine)[order][keep],
            "coarse": np.asarray(coarse)[order][keep], "order": order[keep], "info": info}


def construction_distances(Y: np.ndarray, fine: np.ndarray, frame: str = "centred") -> Dict:
    """Mean pairwise cosine distance within sub-groups, between sibling sub-groups,
    among the first ``OPEN_K`` kept positions, and the median, in ``frame``."""
    Z, _ = frame_vectors(Y, frame)
    D = 1.0 - Z @ Z.T
    subs = [c for c in np.unique(fine) if c >= 0]
    per_group = len(SUB_SIZES)
    iu = np.triu_indices(D.shape[0], 1)
    within = [D[np.ix_(fine == c, fine == c)][np.triu_indices(int((fine == c).sum()), 1)].mean()
              for c in subs]
    between = [D[np.ix_(fine == a, fine == b)].mean() for a in subs for b in subs
               if a < b and a // per_group == b // per_group]
    k = OPEN_K
    return {"frame": frame, "within_sub": float(np.mean(within)),
            "between_sub": float(np.mean(between)),
            "opening": float(D[:k, :k][np.triu_indices(k, 1)].mean()),
            "median": float(np.median(D[iu]))}




# ---------------------------------------------------------------------------
# The spectrum
# ---------------------------------------------------------------------------

def _tree(data: LayerData) -> np.ndarray:
    mt = layer_merge_tree(data, linkage="average")
    if mt["branch"] not in ("plateaus", "two_tokens"):
        raise RuntimeError(f"merge tree refused: {mt['branch']}")
    return mt["_Z"]


def _median_distance(data: LayerData) -> float:
    return float(np.median(data.cos_dist[np.triu_indices(data.n, 1)]))


def substantial_count(labels: np.ndarray, min_size: int = SUBSTANTIAL_CLUSTER_SIZE) -> int:
    _, counts = np.unique(labels, return_counts=True)
    return int(np.sum(counts >= min_size))


def cluster_stability(data: LayerData, labels: Sequence[np.ndarray], deltas: Sequence[float],
                      rng: np.random.Generator, n_sub: int = N_SUBSAMPLES,
                      frac: float = SUBSAMPLE_FRAC, min_size: int = min(ARMS.values())
                      ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Per scale, ``(sizes, stability)`` over every cluster id of that cut: Hennig's
    mean best Jaccard of ``C ∩ S`` over the subsamples S with ``|C ∩ S| >= 2``
    (NaN for a cluster smaller than ``min_size`` or never counted). ``labels`` are
    `labels_at_delta`'s (ids 0..K-1).
    """
    n = data.n
    m = int(round(frac * n))
    sizes = [np.bincount(lab) for lab in labels]
    sums = [np.zeros(sz.size) for sz in sizes]
    hits = [np.zeros(sz.size) for sz in sizes]
    for _ in range(int(n_sub)):
        idx = np.sort(rng.choice(n, size=m, replace=False))
        Zb = _tree(data.subset(idx))
        for g, delta in enumerate(deltas):
            if not np.any(sizes[g] >= min_size):
                continue
            sub = labels_at_delta(Zb, m, delta).astype(np.int64)
            orig = labels[g][idx].astype(np.int64)  # subsample position -> original cluster
            ns = int(sub.max()) + 1
            keys, inter = np.unique(orig * ns + sub, return_counts=True)
            c_of, s_of = keys // ns, keys % ns
            in_c = np.bincount(orig, minlength=sizes[g].size)  # |C ∩ S|
            jac = inter / (in_c[c_of] + np.bincount(sub, minlength=ns)[s_of] - inter)
            best = np.zeros(sizes[g].size)
            np.maximum.at(best, c_of, jac)
            ok = (sizes[g] >= min_size) & (in_c >= 2)
            sums[g][ok] += best[ok]
            hits[g][ok] += 1
    out = []
    for sz, s, h in zip(sizes, sums, hits):
        st = np.full(sz.size, np.nan)
        st[h > 0] = s[h > 0] / h[h > 0]
        out.append((sz, st))
    return out


def arm_stability(per_scale: Sequence[Tuple[np.ndarray, np.ndarray]], min_size: int
                  ) -> List[Optional[float]]:
    """Per scale, the mean stability over clusters of ``>= min_size`` tokens (None if none)."""
    out: List[Optional[float]] = []
    for sz, st in per_scale:
        v = st[(sz >= min_size) & ~np.isnan(st)]
        out.append(float(v.mean()) if v.size else None)
    return out


def hennig_stability(data: LayerData, labels: Sequence[np.ndarray], deltas: Sequence[float],
                     rng: np.random.Generator, n_sub: int = N_SUBSAMPLES,
                     frac: float = SUBSAMPLE_FRAC, min_size: int = SUBSTANTIAL_CLUSTER_SIZE
                     ) -> List[Optional[float]]:
    """Per scale, the mean over clusters of ``>= min_size`` tokens of Hennig's
    cluster-wise stability (None where the cut has none)."""
    return arm_stability(cluster_stability(data, labels, deltas, rng, n_sub, frac, min_size), min_size)


def _z(obs: float, null: np.ndarray) -> Optional[float]:
    sd = float(np.std(null, ddof=1)) if null.size > 1 else 0.0
    return None if sd <= 0 else float((obs - null.mean()) / sd)


def rank_p_higher(obs: float, ref: np.ndarray) -> float:
    """(1 + #{ref >= obs}) / (N + 1)."""
    ref = np.asarray(ref, dtype=np.float64)
    return float((1 + np.sum(ref >= obs)) / (ref.size + 1))


def arm_rows(grid: np.ndarray, deltas: Sequence[float], labels: Sequence[np.ndarray],
             stability: Sequence[Optional[float]], null_k: np.ndarray, null_st: np.ndarray,
             min_size: int) -> List[Dict]:
    """
    One arm's rows. ``p`` / ``z`` is (b): the cut's stability against the draws'
    (``null_st``, NaN = no cluster of the arm's size, scored 0); ``p = 1`` where the
    cut itself has none. ``p_count`` is the substantial count's rank p (higher tail),
    the first check's failed (b), kept beside.
    """
    rows = []
    for g, r in enumerate(grid):
        k_sub = substantial_count(labels[g], min_size)
        ref = np.nan_to_num(null_st[:, g], nan=0.0)
        stab = stability[g]
        rows.append({"r": float(r), "delta": float(deltas[g]), "k": int(np.unique(labels[g]).size),
                     "k_sub": k_sub, "stability": stab,
                     "null_stab_mean": float(ref.mean()),
                     "null_stab_sd": float(ref.std(ddof=1)) if ref.size > 1 else 0.0,
                     "null_n_empty": int(np.isnan(null_st[:, g]).sum()),
                     "z": None if stab is None else _z(stab, ref),
                     "p": 1.0 if stab is None else rank_p_higher(stab, ref),
                     "null_k_mean": float(null_k[:, g].mean()),
                     "p_count": rank_p_higher(k_sub, null_k[:, g])})
    return rows


def spectrum(Y: np.ndarray, frame: str, seed: int, n_sub: int = N_SUBSAMPLES,
             n_draws: int = N_DRAWS) -> Dict:
    """
    One cloud's scale spectrum in ``frame``: ``rows[arm]``, per grid point ``r``
    (`arm_rows`). ``_labels`` (per grid point) and ``_null[arm]`` (the draws'
    counts and stabilities, (n_draws, GRID_N)) are for the caller in-process.
    """
    Z, info = frame_vectors(Y, frame)
    Zs = span_coordinates(Z)
    data = LayerData.from_normed(Zs)
    tree = _tree(data)
    med = _median_distance(data)
    grid = relative_grid()
    deltas = [float(r * med) for r in grid]
    labels = [labels_at_delta(tree, data.n, d) for d in deltas]
    per = cluster_stability(data, labels, deltas, np.random.default_rng([seed, _SUB]), n_sub)
    rng = np.random.default_rng([seed, _NULL])
    null_k = {a: np.zeros((int(n_draws), grid.size), dtype=int) for a in ARMS}
    null_st = {a: np.full((int(n_draws), grid.size), np.nan) for a in ARMS}
    for i in range(int(n_draws)):
        G = LayerData.from_normed(gaussian_draw(Zs, rng))
        Zg, mg = _tree(G), _median_distance(G)
        dg = [float(r * mg) for r in grid]
        lg = [labels_at_delta(Zg, G.n, d) for d in dg]
        pg = cluster_stability(G, lg, dg, np.random.default_rng([seed, _SUB, i]), n_sub)
        for a, size in ARMS.items():
            null_k[a][i] = [substantial_count(lab, size) for lab in lg]
            null_st[a][i] = [np.nan if v is None else v for v in arm_stability(pg, size)]
    rows = {a: arm_rows(grid, deltas, labels, arm_stability(per, size), null_k[a], null_st[a], size)
            for a, size in ARMS.items()}
    return {"frame": frame, "n": int(data.n), "median": med,
            "info": {k: info[k] for k in ("mean_share", "eff_dim")},
            "rows": rows, "_labels": labels, "_null": {a: (null_k[a], null_st[a]) for a in ARMS}}


def robust_plateaus(rows: Sequence[Dict], use_p: bool = True, p_key: str = "p") -> List[Dict]:
    """Maximal runs of ``>= MIN_RUN`` consecutive grid points with one ``k_sub >= MIN_K``,
    stability ``>= STABLE`` and (if ``use_p``) ``rows[p_key] <= ALPHA`` at every point."""
    def ok(row: Dict) -> bool:
        return (row["k_sub"] >= MIN_K and row["stability"] is not None
                and row["stability"] >= STABLE and (not use_p or row[p_key] <= ALPHA))

    out, start = [], None
    for g in range(len(rows) + 1):
        cont = (g < len(rows) and ok(rows[g]) and start is not None
                and rows[g]["k_sub"] == rows[start]["k_sub"])
        if cont:
            continue
        if start is not None and g - start >= MIN_RUN:
            out.append({"start": start, "end": g - 1, "k_sub": rows[start]["k_sub"],
                        "r_lo": rows[start]["r"], "r_hi": rows[g - 1]["r"]})
        start = g if g < len(rows) and ok(rows[g]) else None
    return out


def matched_count(rows: Sequence[Dict], ks: np.ndarray, st: np.ndarray, k: int) -> Dict:
    """Per draw, its largest stability over grid points where it has ``k`` clusters
    of the arm's size; against the cloud's smallest stability over its own points at ``k``."""
    obs = [r["stability"] for r in rows if r["k_sub"] == k and r["stability"] is not None]
    per_draw = [float(np.nanmax(st[i, ks[i] == k])) for i in range(ks.shape[0])
                if np.any((ks[i] == k) & ~np.isnan(st[i]))]
    lo = min(obs) if obs else None
    return {"k": int(k), "cloud_min": lo, "n_draws_with_k": len(per_draw),
            "draw_max": max(per_draw) if per_draw else None,
            "draw_median": float(np.median(per_draw)) if per_draw else None,
            "n_draws_at_or_above": (int(sum(v >= lo for v in per_draw)) if lo is not None else None)}


# ---------------------------------------------------------------------------
# The first check
# ---------------------------------------------------------------------------

def planted_ari(spec: Dict, plateaus: Sequence[Dict], planted: Dict[str, np.ndarray]) -> List[Dict]:
    """Each plateau with its ARI to each planted labelling at every point (planted tokens only)."""
    from sklearn.metrics import adjusted_rand_score
    out = []
    for pl in plateaus:
        row = dict(pl)
        for name, lab in planted.items():
            keep = lab >= 0
            aris = [float(adjusted_rand_score(lab[keep], spec["_labels"][g][keep]))
                    for g in range(pl["start"], pl["end"] + 1)]
            row[f"ari_{name}"] = aris
            row[f"ari_{name}_min"] = min(aris)
        out.append(row)
    return out


def found_scales(plateaus: Sequence[Dict], names: Sequence[str] = ("coarse", "fine")) -> Dict[str, bool]:
    return {nm: any(p[f"ari_{nm}_min"] >= ARI_BAR for p in plateaus) for nm in names}


def _public(spec: Dict, with_null: bool = False) -> Dict:
    out = {k: v for k, v in spec.items() if not k.startswith("_")}
    if with_null:
        out["null"] = {a: {"k_sub": k.tolist(), "stability": np.where(np.isnan(s), None, s).tolist()}
                       for a, (k, s) in spec["_null"].items()}
    return out


def read_synthetic(syn: Dict, frame: str, seed: int) -> Dict:
    """Per arm: robust plateaus with (b) and without, their ARI to the planted
    labels, which scales are found, and the matched-count reading per plateau."""
    spec = spectrum(syn["Y"], frame, seed)
    planted = {"coarse": syn["coarse"], "fine": syn["fine"]}
    arms = {}
    for a in ARMS:
        rows = spec["rows"][a]
        pl = planted_ari(spec, robust_plateaus(rows), planted)
        wo = planted_ari(spec, robust_plateaus(rows, use_p=False), planted)
        ks, st = spec["_null"][a]
        arms[a] = {"plateaus": pl, "plateaus_without_b": wo, "found": found_scales(pl),
                   "found_without_b": found_scales(wo),
                   "matched_count": [matched_count(rows, ks, st, k)
                                     for k in sorted({p["k_sub"] for p in pl + wo})]}
    return {"spectrum": _public(spec, with_null=True), "arms": arms}


def read_gaussian(Y: np.ndarray, frame: str, seed: int, i: int) -> Dict:
    """The i-th matched-covariance Gaussian cloud of ``Y``, read like the synthetic."""
    Z, _ = frame_vectors(Y, frame)
    G = gaussian_draw(span_coordinates(Z), np.random.default_rng([seed, _GAUSS + i]))
    spec = spectrum(G, frame, seed + 1000 * (i + 1))
    return {"draw": i, "spectrum": _public(spec),
            "arms": {a: {"plateaus": robust_plateaus(spec["rows"][a]),
                         "plateaus_without_b": robust_plateaus(spec["rows"][a], use_p=False)}
                     for a in ARMS}}


def _one_thread() -> None:
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def _job(kind: str, *args):
    return kind, args[-1], (read_synthetic if kind == "synthetic" else read_gaussian)(*args[:-1])


def _plateau_str(pls: Sequence[Dict]) -> str:
    return str([(p["k_sub"], round(p["r_lo"], 3), round(p["r_hi"], 3)) for p in pls])


def first_check(seed: int, jobs: int = 1, log=print) -> Dict:
    """
    Design "The re-run": the synthetic (both frames), ``N_GAUSSIAN_CHECKS`` Gaussian
    clouds per frame, and beside it the ``d_f`` ladder and t = 0 (centred). Each cloud
    is one job; every job has its own seeded streams, so ``jobs`` does not change a number.
    """
    t0 = time.time()
    syn = synthetic(seed)
    beside_syn = {f"d_fine={d}": synthetic(seed, d_fine=d) for d in LADDER_D_FINE}
    beside_syn["open_t=0"] = synthetic(seed, open_t=0.0)
    out: Dict = {"synthetic": syn["info"],
                 "construction": {f: construction_distances(syn["Y"], syn["fine"], f) for f in FRAMES},
                 "frames": {f: {"gaussians": [None] * N_GAUSSIAN_CHECKS} for f in FRAMES},
                 "beside": {k: {"construction": construction_distances(s["Y"], s["fine"])}
                            for k, s in beside_syn.items()}}
    tasks = [("synthetic", syn, f, seed, ("main", f)) for f in FRAMES]
    tasks += [("synthetic", s, "centred", seed, ("beside", k)) for k, s in beside_syn.items()]
    tasks += [("gaussian", syn["Y"], f, seed, i, ("gauss", f, i))
              for f in FRAMES for i in range(N_GAUSSIAN_CHECKS)]
    done = 0
    with ProcessPoolExecutor(max_workers=max(1, int(jobs)), initializer=_one_thread) as ex:
        futs = [ex.submit(_job, *t) for t in tasks]
        for fut in as_completed(futs):
            kind, key, rec = fut.result()
            done += 1
            if key[0] == "main":
                out["frames"][key[1]].update(rec)
            elif key[0] == "beside":
                out["beside"][key[1]].update(rec)
            else:
                out["frames"][key[1]]["gaussians"][key[2]] = rec
            msg = "; ".join(f"{a} {_plateau_str(r['plateaus'])}"
                            + (f" found {r['found']}" if "found" in r else "")
                            for a, r in rec["arms"].items())
            log(f"[{done}/{len(tasks)} {time.time() - t0:.0f}s] {' '.join(map(str, key))}: {msg}", flush=True)
    for f in FRAMES:
        fr = out["frames"][f]
        fr["gaussian_with_plateau"] = {a: sum(bool(g["arms"][a]["plateaus"]) for g in fr["gaussians"])
                                       for a in ARMS}
        fr["gaussian_with_plateau_without_b"] = {
            a: sum(bool(g["arms"][a]["plateaus_without_b"]) for g in fr["gaussians"]) for a in ARMS}
    c = out["frames"]["centred"]
    out["pass"] = {a: bool(all(c["arms"][a]["found"].values())
                           and c["gaussian_with_plateau"][a] <= MAX_GAUSSIAN_PLATEAUS) for a in ARMS}
    for f in FRAMES:
        log(f"{f}: found {[(a, out['frames'][f]['arms'][a]['found']) for a in ARMS]}; Gaussian clouds "
            f"with a plateau {out['frames'][f]['gaussian_with_plateau']} of {N_GAUSSIAN_CHECKS} "
            f"(without (b): {out['frames'][f]['gaussian_with_plateau_without_b']})", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _git_head() -> str:
    """HEAD, with ``-dirty`` if a tracked file differs from it."""
    try:
        d = Path(__file__).parent
        head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=d, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                        cwd=d, text=True).strip()
        return head + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def synthetic_cmd(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(prog="scale_spectrum synthetic")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    git = _git_head()
    rec = first_check(a.seed, a.jobs)
    rec["git"] = git
    rec["constants"] = {"grid": [GRID_LO, GRID_HI, GRID_N], "n_subsamples": N_SUBSAMPLES,
                        "subsample_frac": SUBSAMPLE_FRAC, "n_draws": N_DRAWS, "stable": STABLE,
                        "alpha": ALPHA, "min_run": MIN_RUN, "min_k": MIN_K, "arms": ARMS,
                        "ari_bar": ARI_BAR, "n_gaussian_checks": N_GAUSSIAN_CHECKS,
                        "max_gaussian_plateaus": MAX_GAUSSIAN_PLATEAUS}
    (a.out / "first_check.json").write_text(json.dumps(rec, indent=1))
    verdict = ", ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in rec["pass"].items())
    print(f"first check: {verdict} ({rec['seconds']} s; {a.out / 'first_check.json'})")
    return 0 if rec["pass"]["main"] else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmds = {"synthetic": synthetic_cmd}
    if not argv or argv[0] not in cmds:
        print(f"usage: scale_spectrum {{{','.join(cmds)}}} ...", file=sys.stderr)
        return 2
    return cmds[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
