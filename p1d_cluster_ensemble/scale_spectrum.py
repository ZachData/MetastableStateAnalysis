"""
p1d_cluster_ensemble/scale_spectrum.py — unit 3 of the Blocked 11⁗ programme:
one family on a continuous scale (`design-1d.md` "Unit 3"; the synthetic and
its first check are fixed there, "The synthetic and the first check", before
any run).

The family is the average-linkage merge tree on cosine distance
(`merge_tree.layer_merge_tree`), cut at each of ``GRID_N`` relative heights
``r`` (log-spaced, ``GRID_LO``..``GRID_HI``); the absolute cut is
``delta = r x`` the cloud's median pairwise cosine distance in its frame.
Per scale (`spectrum`):

- **(a) stability**: Hennig's (2007) cluster-wise stability. For each
  substantial cluster C (``>= SUBSTANTIAL_CLUSTER_SIZE`` tokens), over
  ``N_SUBSAMPLES`` subsamples of ``SUBSAMPLE_FRAC`` (without replacement),
  the best Jaccard of ``C ∩ S`` against the clusters of the subsample's own
  tree cut at the same absolute ``delta``; the scale's value is the mean over
  its substantial clusters.
- **(b) count against the Gaussian**: the substantial count's ``z_G`` and its
  rank p (higher tail) among ``N_DRAWS`` matched-covariance Gaussian draws of
  the cloud (same frame; `gaussian_null.span_coordinates` + `gaussian_draw`,
  as unit 2), each cut at the same ``r`` times its own median. On real input
  the design ranks ``z_G`` among unit 2's re-inits instead; that reader is
  not built here.

A **robust plateau** (`robust_plateaus`) is a maximal run of ``>= MIN_RUN``
consecutive grid points with one substantial count ``k >= 2`` and, at every
point, stability ``>= STABLE`` and rank p ``<= ALPHA``.

**The multi-scale synthetic** (`synthetic`; unit 4's row): 3 groups x 3
sub-groups from von Mises–Fisher draws on S^1023 (two planted spreads, set
as mean pairwise cosine distances ``d_f`` within a sub-group and ``d_c``
between sub-groups of one group), 100 uniform background points, a seeded
order, then the opening: `identity_sim`'s β = 0 causal flow to ``OPEN_T``,
then T1 (position 0 dropped). The spreads are exact in expectation: for
independent draws ``E[x·y] = E[x]·E[y]``, so two points of one sub-group
have mean cosine ``ρ_f²`` and two of sibling sub-groups ``ρ_f² ρ_c²``.

**First check** (`first_check`): in the centred frame, each planted scale
(3 groups; 9 sub-groups) is found by a robust plateau whose ARI to the
planted labels (planted tokens only) is ``>= ARI_BAR`` at every point, and
none of ``N_GAUSSIAN_CHECKS`` matched-covariance Gaussian clouds of the
synthetic has a robust plateau.

Tier 1: exploratory, unregistered.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
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
N_GAUSSIAN_CHECKS = 5
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


def _best_jaccards(member_lab: np.ndarray, sub_labels: np.ndarray) -> float:
    """Best Jaccard of a set (its members' labels in the subsample's cut) against
    any cluster of that cut (``sub_labels`` over the whole subsample)."""
    ids, inter = np.unique(member_lab, return_counts=True)
    sizes = np.bincount(sub_labels)[ids]
    return float(np.max(inter / (member_lab.size + sizes - inter)))


def hennig_stability(data: LayerData, labels: Sequence[np.ndarray], deltas: Sequence[float],
                     rng: np.random.Generator, n_sub: int = N_SUBSAMPLES,
                     frac: float = SUBSAMPLE_FRAC) -> List[Optional[float]]:
    """Per scale, the mean over substantial clusters of Hennig's cluster-wise
    stability (None where the cut has no substantial cluster)."""
    n = data.n
    m = int(round(frac * n))
    clusters = []
    for lab in labels:
        ids, counts = np.unique(lab, return_counts=True)
        clusters.append([np.flatnonzero(lab == c) for c in ids[counts >= SUBSTANTIAL_CLUSTER_SIZE]])
    sums = [np.zeros(len(cs)) for cs in clusters]
    hits = [np.zeros(len(cs)) for cs in clusters]
    for _ in range(int(n_sub)):
        idx = np.sort(rng.choice(n, size=m, replace=False))
        pos = np.full(n, -1)
        pos[idx] = np.arange(m)
        Zb = _tree(data.subset(idx))
        for g, delta in enumerate(deltas):
            if not clusters[g]:
                continue
            sub = labels_at_delta(Zb, m, delta)
            for j, members in enumerate(clusters[g]):
                p = pos[members]
                p = p[p >= 0]
                if p.size:
                    sums[g][j] += _best_jaccards(sub[p], sub)
                    hits[g][j] += 1
    out: List[Optional[float]] = []
    for s, h in zip(sums, hits):
        ok = h > 0
        out.append(float(np.mean(s[ok] / h[ok])) if ok.any() else None)
    return out


def _z(obs: float, null: np.ndarray) -> Optional[float]:
    sd = float(np.std(null, ddof=1)) if null.size > 1 else 0.0
    return None if sd <= 0 else float((obs - null.mean()) / sd)


def rank_p_higher(obs: float, ref: np.ndarray) -> float:
    """(1 + #{ref >= obs}) / (N + 1)."""
    ref = np.asarray(ref, dtype=np.float64)
    return float((1 + np.sum(ref >= obs)) / (ref.size + 1))


def spectrum(Y: np.ndarray, frame: str, seed: int, n_sub: int = N_SUBSAMPLES,
             n_draws: int = N_DRAWS) -> Dict:
    """
    One cloud's scale spectrum in ``frame``: per grid point ``r``, ``delta``,
    ``k`` (all clusters), ``k_sub``, ``stability``, the draws' count mean / SD,
    ``z`` and ``p``. ``_labels`` (per grid point) is for the caller in-process.
    """
    Z, info = frame_vectors(Y, frame)
    Zs = span_coordinates(Z)
    data = LayerData.from_normed(Zs)
    tree = _tree(data)
    med = _median_distance(data)
    grid = relative_grid()
    deltas = [float(r * med) for r in grid]
    labels = [labels_at_delta(tree, data.n, d) for d in deltas]
    stab = hennig_stability(data, labels, deltas, np.random.default_rng([seed, _SUB]), n_sub)
    rng = np.random.default_rng([seed, _NULL])
    null = np.zeros((int(n_draws), grid.size))
    for i in range(int(n_draws)):
        G = LayerData.from_normed(gaussian_draw(Zs, rng))
        Zg, mg = _tree(G), _median_distance(G)
        null[i] = [substantial_count(labels_at_delta(Zg, G.n, r * mg)) for r in grid]
    rows = []
    for g, r in enumerate(grid):
        k_sub = substantial_count(labels[g])
        rows.append({"r": float(r), "delta": deltas[g], "k": int(np.unique(labels[g]).size),
                     "k_sub": k_sub, "stability": stab[g],
                     "null_mean": float(null[:, g].mean()),
                     "null_sd": float(null[:, g].std(ddof=1)) if n_draws > 1 else 0.0,
                     "z": _z(k_sub, null[:, g]), "p": rank_p_higher(k_sub, null[:, g])})
    return {"frame": frame, "n": int(data.n), "median": med,
            "info": {k: info[k] for k in ("mean_share", "eff_dim")},
            "rows": rows, "_labels": labels}


def robust_plateaus(rows: Sequence[Dict], use_p: bool = True) -> List[Dict]:
    """Maximal runs of ``>= MIN_RUN`` consecutive grid points with one ``k_sub >= MIN_K``,
    stability ``>= STABLE`` and (if ``use_p``) ``p <= ALPHA`` at every point."""
    def ok(row: Dict) -> bool:
        return (row["k_sub"] >= MIN_K and row["stability"] is not None
                and row["stability"] >= STABLE and (not use_p or row["p"] <= ALPHA))

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


def _public(spec: Dict) -> Dict:
    return {k: v for k, v in spec.items() if not k.startswith("_")}


def read_synthetic(syn: Dict, frame: str, seed: int) -> Dict:
    spec = spectrum(syn["Y"], frame, seed)
    planted = {"coarse": syn["coarse"], "fine": syn["fine"]}
    plateaus = planted_ari(spec, robust_plateaus(spec["rows"]), planted)
    without = planted_ari(spec, robust_plateaus(spec["rows"], use_p=False), planted)
    return {"spectrum": _public(spec), "plateaus": plateaus, "plateaus_without_b": without,
            "found": found_scales(plateaus), "found_without_b": found_scales(without)}


def read_gaussians(syn: Dict, frame: str, seed: int) -> List[Dict]:
    Z, _ = frame_vectors(syn["Y"], frame)
    Zs = span_coordinates(Z)
    out = []
    for i in range(N_GAUSSIAN_CHECKS):
        G = gaussian_draw(Zs, np.random.default_rng([seed, _GAUSS + i]))
        spec = spectrum(G, frame, seed + 1000 * (i + 1))
        out.append({"draw": i, "spectrum": _public(spec),
                    "plateaus": robust_plateaus(spec["rows"]),
                    "plateaus_without_b": robust_plateaus(spec["rows"], use_p=False)})
    return out


def first_check(seed: int = 0, log=print) -> Dict:
    t0 = time.time()
    syn = synthetic(seed)
    out: Dict = {"synthetic": syn["info"],
                 "construction": {f: construction_distances(syn["Y"], syn["fine"], f) for f in FRAMES},
                 "frames": {}}
    for frame in FRAMES:
        rec = read_synthetic(syn, frame, seed)
        rec["gaussians"] = read_gaussians(syn, frame, seed)
        out["frames"][frame] = rec
        log(f"{frame}: plateaus {[(p['k_sub'], round(p['r_lo'], 3), round(p['r_hi'], 3)) for p in rec['plateaus']]}"
            f" found {rec['found']}; Gaussian plateaus "
            f"{[len(g['plateaus']) for g in rec['gaussians']]} (without (b): "
            f"{[len(g['plateaus_without_b']) for g in rec['gaussians']]}) [{time.time() - t0:.0f}s]")
    c = out["frames"]["centred"]
    out["pass"] = bool(all(c["found"].values()) and all(not g["plateaus"] for g in c["gaussians"]))
    beside = {}
    for d_f in LADDER_D_FINE:
        s = synthetic(seed, d_fine=d_f)
        beside[f"d_fine={d_f}"] = {"construction": construction_distances(s["Y"], s["fine"]),
                                   **read_synthetic(s, "centred", seed)}
    s = synthetic(seed, open_t=0.0)
    beside["open_t=0"] = {"construction": construction_distances(s["Y"], s["fine"]),
                          **read_synthetic(s, "centred", seed)}
    for k, v in beside.items():
        log(f"beside {k}: plateaus {[(p['k_sub'], round(p['r_lo'], 3), round(p['r_hi'], 3)) for p in v['plateaus']]}"
            f" found {v['found']}")
    out["beside"] = beside
    out["seconds"] = round(time.time() - t0, 1)
    return out


# ---------------------------------------------------------------------------
# Post hoc, after the first check failed (Blocked 15's evidence; not a pass)
# ---------------------------------------------------------------------------

def stability_null(Y: np.ndarray, frame: str, seed: int, n_draws: int = N_DRAWS,
                   n_sub: int = N_SUBSAMPLES) -> Tuple[np.ndarray, np.ndarray]:
    """
    Option 1's null: per matched-covariance draw (the same draws as `spectrum`'s,
    same stream) and grid point, ``k_sub`` and Hennig stability (NaN where the
    draw's cut has no substantial cluster). Shapes (n_draws, GRID_N).
    """
    Z, _ = frame_vectors(Y, frame)
    Zs = span_coordinates(Z)
    grid = relative_grid()
    rng = np.random.default_rng([seed, _NULL])
    ks = np.zeros((int(n_draws), grid.size), dtype=int)
    st = np.full((int(n_draws), grid.size), np.nan)
    for i in range(int(n_draws)):
        G = LayerData.from_normed(gaussian_draw(Zs, rng))
        Zg, mg = _tree(G), _median_distance(G)
        deltas = [float(r * mg) for r in grid]
        labels = [labels_at_delta(Zg, G.n, d) for d in deltas]
        ks[i] = [substantial_count(lab) for lab in labels]
        s = hennig_stability(G, labels, deltas, np.random.default_rng([seed, _SUB, i]), n_sub)
        st[i] = [np.nan if v is None else v for v in s]
    return ks, st


def option1_rows(rows: Sequence[Dict], st: np.ndarray, empty: str) -> List[Dict]:
    """``rows`` with ``p`` replaced by the rank p (higher tail) of the cut's stability
    among the draws' at the same grid point. ``empty``: "zero" scores a draw with no
    substantial cluster 0; "drop" leaves it out (p = 1 if none is left)."""
    out = []
    for g, row in enumerate(rows):
        ref = st[:, g]
        ref = np.nan_to_num(ref, nan=0.0) if empty == "zero" else ref[~np.isnan(ref)]
        p = (1.0 if row["stability"] is None or ref.size == 0
             else rank_p_higher(row["stability"], ref))
        out.append({**row, "p": p})
    return out


def matched_count(rows: Sequence[Dict], ks: np.ndarray, st: np.ndarray, k: int) -> Dict:
    """Per draw, its largest stability over grid points where it has ``k`` substantial
    clusters; against the synthetic's smallest stability over its own points at ``k``."""
    obs = [r["stability"] for r in rows if r["k_sub"] == k and r["stability"] is not None]
    per_draw = [float(np.nanmax(st[i, ks[i] == k])) for i in range(ks.shape[0])
                if np.any((ks[i] == k) & ~np.isnan(st[i]))]
    lo = min(obs) if obs else None
    return {"k": int(k), "synthetic_min": lo, "n_draws_with_k": len(per_draw),
            "draw_max": max(per_draw) if per_draw else None,
            "draw_median": float(np.median(per_draw)) if per_draw else None,
            "n_draws_at_or_above": (int(sum(v >= lo for v in per_draw)) if lo is not None else None)}


def posthoc(seed: int = 0, log=print) -> Dict:
    """
    Blocked 15's numbers, all on seeds already seen: the runs found without (b)
    with their ARI (main synthetic both frames, t = 0, the ladder), and option 1
    on the main synthetic (both empty-draw rules; matched count beside).
    """
    out: Dict = {"seed": int(seed), "without_b": {}, "option1": {}}
    variants = [("t=2", {}, f) for f in FRAMES] + [("t=0", {"open_t": 0.0}, "centred")]
    variants += [(f"d_fine={d}", {"d_fine": d}, "centred") for d in LADDER_D_FINE]
    syns: Dict[str, Dict] = {}
    for name, kw, frame in variants:
        key = f"{name} {frame}"
        syn = syns.setdefault(name, synthetic(seed, **kw))
        rec = read_synthetic(syn, frame, seed)
        out["without_b"][key] = {"plateaus": rec["plateaus_without_b"], "found": rec["found_without_b"]}
        log(f"without (b) {key}: {[(p['k_sub'], round(p['r_lo'], 3), round(p['r_hi'], 3), round(p['ari_coarse_min'], 2), round(p['ari_fine_min'], 2)) for p in rec['plateaus_without_b']]}")
        if name != "t=2":
            continue
        spec = spectrum(syn["Y"], frame, seed)
        ks, st = stability_null(syn["Y"], frame, seed)
        planted = {"coarse": syn["coarse"], "fine": syn["fine"]}
        o1 = {"null_k_sub": ks.tolist(), "null_stability": np.where(np.isnan(st), None, st).tolist()}
        for empty in ("zero", "drop"):
            rows = option1_rows(spec["rows"], st, empty)
            pl = planted_ari(spec, robust_plateaus(rows), planted)
            o1[empty] = {"p": [r["p"] for r in rows], "plateaus": pl, "found": found_scales(pl)}
            log(f"option 1 ({empty}) {frame}: {[(p['k_sub'], round(p['r_lo'], 3), round(p['r_hi'], 3)) for p in pl]} found {found_scales(pl)}")
        o1["matched_count"] = [matched_count(spec["rows"], ks, st, k)
                               for k in sorted({p["k_sub"] for p in out["without_b"][key]["plateaus"]})]
        log(f"matched count {frame}: {o1['matched_count']}")
        out["option1"][frame] = o1
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
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    rec = first_check(a.seed)
    rec["git"] = _git_head()
    rec["constants"] = {"grid": [GRID_LO, GRID_HI, GRID_N], "n_subsamples": N_SUBSAMPLES,
                        "subsample_frac": SUBSAMPLE_FRAC, "n_draws": N_DRAWS, "stable": STABLE,
                        "alpha": ALPHA, "min_run": MIN_RUN, "min_k": MIN_K,
                        "substantial": SUBSTANTIAL_CLUSTER_SIZE, "ari_bar": ARI_BAR,
                        "n_gaussian_checks": N_GAUSSIAN_CHECKS}
    (a.out / "first_check.json").write_text(json.dumps(rec, indent=1))
    print(f"first check: {'PASS' if rec['pass'] else 'FAIL'} ({rec['seconds']} s; {a.out / 'first_check.json'})")
    return 0 if rec["pass"] else 1


def posthoc_cmd(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(prog="scale_spectrum posthoc")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rec = posthoc(a.seed)
    rec["git"] = _git_head()
    rec["seconds"] = round(time.time() - t0, 1)
    (a.out / "posthoc.json").write_text(json.dumps(rec, indent=1))
    print(f"posthoc written ({rec['seconds']} s; {a.out / 'posthoc.json'})")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmds = {"synthetic": synthetic_cmd, "posthoc": posthoc_cmd}
    if not argv or argv[0] not in cmds:
        print(f"usage: scale_spectrum {{{','.join(cmds)}}} ...", file=sys.stderr)
        return 2
    return cmds[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
