"""
p1d_cluster_ensemble/gaussian_null.py — are tokens lumpier than their
covariance alone explains?

The null is SigClust's (Liu, Hayes, Nobel & Marron 2008; `lit-1d.md` §7):
one Gaussian with the tokens' own mean and covariance. Each draw is `n`
points from `N(mu, Sigma)`, put through the same L2 normalisation the
tokens went through, then clustered exactly as the tokens are. A statistic
the tokens beat is structure the covariance does not carry.

Three frames, because a single shared direction and a few rogue
coordinates (Timkey & van Schijndel 2021) dominate the raw cosine
geometry (`status-1d.md` "Matched-covariance Gaussian null"):

- ``raw``: the L2-normed rows 1d clusters now.
- ``centred``: the shared mean direction projected out of every row, then
  renormed.
- ``centred_norogue``: the ``N_ROGUE`` coordinates with the largest
  contribution to the mean cosine similarity (``m_i**2``, `m` the mean
  unit row: Timkey & van Schijndel's measure) zeroed first, then renormed,
  then centred as above.

The Gaussian is fitted *in each frame*, so each frame asks its own
question. Plug-in covariance (``ddof=0``), rank at most ``n - 1``: every
draw lives in the span of the rows, so everything is computed in an
orthonormal basis of that span (all statistics are rotation invariant),
and ``E|x|^2 = |mu|^2 + tr Sigma = mean |y|^2 = 1`` before the
renormalisation (`tests/test_phase1d_gaussian_null.py`).

**Not calibrated at nominal level.** SigClust shrinks the HDLSS
eigenvalues; this uses the plug-in covariance, and ``--calibrate``
measures what that costs. Some statistics are far off nominal level (the
numbers are in `status-1d.md` "Matched-covariance Gaussian null"), so
real results are read against a calibration on the same inputs
(`gaussian_null_report.py` refuses any other), not against the nominal p.

``mt_k`` / ``mt_life`` are 0 when a layer has no robust plateau: an
ordinal encoding ("no plateau" ranks below any plateau), so their tail
counts are meaningful and their z less so. Tier 1: exploratory,
unregistered.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.holdout import add_holdout_args, refuse_held_out

FRAMES = ("raw", "centred", "centred_norogue")

#: PLACED. Timkey & van Schijndel find 1-3 rogue dimensions; 410m's quick
#: look found one dominant coordinate per layer, a different one by depth.
N_ROGUE = 3

#: Which tail is "lumpier than the Gaussian" for each statistic. The two
#: count statistics are descriptive and read in both tails.
LUMPIER = {"ci2": "lower", "nn1": "lower", "hdb_noise": "lower",
           "mt_life": "higher", "hdb_k": None, "mt_k": None}
STATISTICS = tuple(LUMPIER)


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

def _unit_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def mean_direction(Y: np.ndarray) -> np.ndarray:
    m = np.asarray(Y, dtype=np.float64).mean(axis=0)
    return m / max(float(np.linalg.norm(m)), 1e-12)


def rogue_dims(Y: np.ndarray, n_rogue: int = N_ROGUE) -> np.ndarray:
    """Coordinates by contribution ``m_i**2`` to the mean cosine similarity, descending."""
    m = _unit_rows(Y).mean(axis=0)
    return np.argsort(-(m ** 2), kind="stable")[:int(n_rogue)]


def participation_ratio(Y: np.ndarray) -> float:
    """Effective dimensions of the centred cloud: ``(sum l)^2 / sum l^2``."""
    Yc = np.asarray(Y, dtype=np.float64) - np.mean(Y, axis=0)
    ev = np.linalg.svd(Yc, compute_uv=False) ** 2
    s2 = float((ev ** 2).sum())
    return float(ev.sum() ** 2 / s2) if s2 > 0 else 0.0


def frame_vectors(Y: np.ndarray, frame: str, n_rogue: int = N_ROGUE
                  ) -> Tuple[np.ndarray, Dict]:
    """
    Unit rows of ``Y`` in ``frame`` and what the frame removed.

    ``info``: ``mean_share`` (mean of ``(y . m_hat)^2`` over the raw unit
    rows), the rogue coordinates with their ``m_i^2`` contribution share
    and mean-square share, and the effective dimensions of the result.
    """
    if frame not in FRAMES:
        raise ValueError(f"unknown frame {frame!r}; use one of {FRAMES}")
    Y = _unit_rows(Y)
    m = Y.mean(axis=0)
    mh = mean_direction(Y)
    rogue = rogue_dims(Y, n_rogue)
    m2 = float((m ** 2).sum())
    info = {"frame": frame,
            "mean_norm": float(np.linalg.norm(m)),
            "mean_share": float(((Y @ mh) ** 2).mean()),
            "rogue": [int(i) for i in rogue],
            "rogue_cos_share": [float(m[i] ** 2 / m2) if m2 > 0 else 0.0 for i in rogue],
            "rogue_sq_share": [float((Y[:, i] ** 2).mean()) for i in rogue]}
    Z = Y
    if frame == "centred_norogue":
        Z = Z.copy()
        Z[:, rogue] = 0.0
        Z = _unit_rows(Z)
    if frame in ("centred", "centred_norogue"):
        zh = mean_direction(Z)
        Z = _unit_rows(Z - np.outer(Z @ zh, zh))
    info["eff_dim"] = participation_ratio(Z)
    return Z, info


# ---------------------------------------------------------------------------
# The null
# ---------------------------------------------------------------------------

def span_coordinates(Z: np.ndarray) -> np.ndarray:
    """``Z`` in an orthonormal basis of its row span: same Gram matrix, width <= n."""
    Z = np.asarray(Z, dtype=np.float64)
    U, s, _ = np.linalg.svd(Z, full_matrices=False)
    keep = s > s[0] * 1e-12 if s.size else s > 0
    return U[:, keep] * s[keep]


def gaussian_draw(Z: np.ndarray, rng: np.random.Generator, renorm: bool = True
                  ) -> np.ndarray:
    """
    ``n`` points from ``N(mean(Z), cov(Z, ddof=0))``, unit rows if ``renorm``.

    ``x = mu + Zc^T g / sqrt(n)`` with ``g ~ N(0, I_n)`` has exactly that
    covariance and lies in the row span of ``Z``.
    """
    Z = np.asarray(Z, dtype=np.float64)
    n = Z.shape[0]
    mu = Z.mean(axis=0)
    G = rng.standard_normal((n, n)) / np.sqrt(n)
    X = mu + G @ (Z - mu)
    return _unit_rows(X) if renorm else X


# ---------------------------------------------------------------------------
# Lumpiness statistics
# ---------------------------------------------------------------------------

def cluster_index_2(Z: np.ndarray, seed: int = 0) -> float:
    """SigClust's cluster index: 2-means within-SS over total SS (lower = lumpier)."""
    from sklearn.cluster import KMeans
    Z = np.asarray(Z, dtype=np.float64)
    tot = float(((Z - Z.mean(axis=0)) ** 2).sum())
    if tot <= 0 or Z.shape[0] < 3:
        return float("nan")
    km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(Z)
    return float(km.inertia_ / tot)


def lumpiness(Z: np.ndarray, seed: int = 0) -> Dict[str, float]:
    """
    Every statistic of one set of unit rows, by the routes 1d uses.

    ``ci2`` SigClust's 2-means index; ``nn1`` mean cosine distance to the
    nearest other token; ``hdb_k`` / ``hdb_noise`` Phase 1's shipped
    HDBSCAN (``min_cluster_size=2``, precomputed float64 cosine) cluster
    count and noise fraction; ``mt_k`` / ``mt_life`` stage F's top robust
    plateau (average linkage, >= 2 clusters of >= 4 tokens): its substantial
    cluster count and lifetime, 0 when there is none.
    """
    from .merge_tree import layer_merge_tree
    from .methods import LayerData, _fit_hdbscan

    data = LayerData.from_normed(Z)
    D = data.cos_dist.copy()
    np.fill_diagonal(D, np.inf)
    hdb = _fit_hdbscan({"min_cluster_size": 2}, data, seed)
    tree = layer_merge_tree(data)
    top = tree["robust"][0] if tree["robust"] else None
    return {"ci2": cluster_index_2(Z, seed),
            "nn1": float(D.min(axis=1).mean()),
            "hdb_k": float(len(set(hdb.tolist()) - {-1})),
            "hdb_noise": float(np.mean(hdb == -1)),
            "mt_k": float(top["k_substantial"]) if top else 0.0,
            "mt_life": float(top["lifetime"]) if top else 0.0}


def compare(obs: float, null: np.ndarray) -> Dict:
    null = np.asarray(null, dtype=np.float64)
    b = null.size
    return {"obs": float(obs), "null_mean": float(null.mean()),
            "null_sd": float(null.std(ddof=1)) if b > 1 else 0.0,
            "q025": float(np.quantile(null, 0.025)), "q975": float(np.quantile(null, 0.975)),
            "p_lower": float((1 + np.sum(null <= obs)) / (b + 1)),
            "p_upper": float((1 + np.sum(null >= obs)) / (b + 1))}


#: Seed offset for `calibrate`'s pseudo-data, away from the null's own draws.
CALIBRATE_SEED_OFFSET = 10_000


def null_record(Y: np.ndarray, frame: str, n_draws: int, seed: int,
                keep_draws: bool = True, calibrate: bool = False) -> Dict:
    """
    One (layer, frame): the frame's info, the observed statistics, the null's.

    ``calibrate`` replaces the tokens by one draw of their own Gaussian and
    refits the null to that draw: the null is true there, so a calibrated
    statistic puts it in neither tail more often than ``alpha``. The
    plug-in covariance makes this fail in a known direction (the tests'
    small-n case); this measures by how much at real sizes.
    """
    Z, info = frame_vectors(Y, frame)
    Zs = span_coordinates(Z)
    if calibrate:
        pseudo = gaussian_draw(Zs, np.random.default_rng(seed + CALIBRATE_SEED_OFFSET))
        Zs = span_coordinates(pseudo)
        info = {**info, "calibrate": True}
    obs = lumpiness(Zs, seed)
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in STATISTICS}
    for _ in range(int(n_draws)):
        st = lumpiness(gaussian_draw(Zs, rng), seed)
        for k in STATISTICS:
            draws[k].append(st[k])
    out = {"info": info, "n_draws": int(n_draws), "seed": int(seed),
           "stats": {k: compare(obs[k], np.array(draws[k])) for k in STATISTICS}}
    if keep_draws:
        out["draws"] = draws
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _step_prompt(run_dir: Path) -> Tuple[str, str]:
    """``pythia-410m-step143000_wiki_paragraph`` -> ``("step143000", "wiki_paragraph")``."""
    model, _, prompt = run_dir.name.partition("_")
    return model.rsplit("-", 1)[-1], prompt


#: A deduplicated prompt with fewer distinct strings than this is skipped
#: (``repeated_tokens`` has 3), not fitted: a covariance of 3 points is not
#: a null. PLACED.
MIN_TOKENS = 20


def first_occurrences(tokens: Sequence[str]) -> np.ndarray:
    """Positions of each token string's first occurrence, in order."""
    seen, keep = set(), []
    for i, t in enumerate(tokens):
        if t not in seen:
            seen.add(t)
            keep.append(i)
    return np.asarray(keep, dtype=int)


def run_tokens(run_dir: Path) -> List[str]:
    geo = json.loads((Path(run_dir) / "geometry.json").read_text())
    return [str(t) for t in geo.get("tokens") or []]


def _job(args: Tuple) -> Dict:
    run_dir, layer, frame, n_draws, seed, calibrate, dedupe = args
    acts = np.load(Path(run_dir) / "activations.npz")["activations"]
    step, prompt = _step_prompt(Path(run_dir))
    base = {"run_dir": str(run_dir), "step": step, "prompt": prompt,
            "layer": int(layer), "n_tokens": int(acts.shape[1])}
    Y = acts[layer]
    if dedupe:
        tokens = run_tokens(Path(run_dir))
        if len(tokens) != acts.shape[1]:
            raise ValueError(f"{run_dir}: {len(tokens)} token strings for "
                             f"{acts.shape[1]} activation rows; refusing to dedupe")
        keep = first_occurrences(tokens)
        base["n_kept"] = int(keep.size)
        if keep.size < MIN_TOKENS:
            return {**base, "skipped": f"{keep.size} distinct strings < {MIN_TOKENS}",
                    "info": {"frame": frame}}
        Y = Y[keep]
    t0 = time.time()
    rec = null_record(Y, frame, n_draws, seed, calibrate=calibrate)
    rec.update(base)
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def band_of(layer: int) -> str:
    return "L0" if layer == 0 else ("L1-8" if layer <= 8 else ("L9-16" if layer <= 16 else "L17-24"))


def summarise(records: List[Dict], alpha: float = 0.025) -> List[Dict]:
    """Per (step, frame, band, statistic): how many layer-records the null places in each tail."""
    rows: Dict[Tuple, Dict] = {}
    for r in records:
        if "skipped" in r:
            continue
        for k in STATISTICS:
            s = r["stats"][k]
            key = (r["step"], r["info"]["frame"], band_of(r["layer"]), k)
            row = rows.setdefault(key, {"step": key[0], "frame": key[1], "band": key[2],
                                        "stat": k, "n": 0, "below": 0, "above": 0,
                                        "z": []})
            row["n"] += 1
            row["below"] += int(s["p_lower"] <= alpha)
            row["above"] += int(s["p_upper"] <= alpha)
            if s["null_sd"] > 0:
                row["z"].append((s["obs"] - s["null_mean"]) / s["null_sd"])
    out = []
    for row in rows.values():
        z = row.pop("z")
        row["median_z"] = float(np.median(z)) if z else float("nan")
        out.append(row)
    return sorted(out, key=lambda r: (r["stat"], r["step"], FRAMES.index(r["frame"]), r["band"]))


def summary_text(out: Dict) -> str:
    skipped = sum("skipped" in r for r in out["records"])
    lines = [("CALIBRATION (tokens replaced by one Gaussian draw each): " if out.get("calibrate") else "")
             + ("DEDUPED (first occurrence of each token string): " if out.get("dedupe_strings") else "")
             + f"Matched-covariance Gaussian null: {out['n_draws']} draws, seed {out['seed']}, "
             f"{len(out['records']) - skipped} layer-records ({skipped} skipped)",
             "below / above = records with obs in the null's lower / upper 2.5 % tail "
             "(p <= 0.025); lumpier = " + ", ".join(f"{k} {v}" for k, v in LUMPIER.items() if v)]
    stat = None
    for r in out["summary"]:
        if r["stat"] != stat:
            stat = r["stat"]
            lines.append(f"\n[{stat}]  step       frame             band    n  below above  median z")
        lines.append(f"          {r['step']:<10} {r['frame']:<17} {r['band']:<7} {r['n']:>2}  "
                     f"{r['below']:>5} {r['above']:>5}  {r['median_z']:>8.2f}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, nargs="+", required=True,
                    help="Phase 1 prompt directories (each with activations.npz)")
    ap.add_argument("--out", type=Path, required=True, help="JSON to write (summary beside it)")
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="default: every layer")
    ap.add_argument("--frames", nargs="*", default=list(FRAMES), choices=FRAMES)
    ap.add_argument("--n-draws", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibrate", action="store_true",
                    help="run on one Gaussian draw of each layer instead of the tokens")
    ap.add_argument("--dedupe-strings", action="store_true",
                    help="keep only each token string's first occurrence")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    add_holdout_args(ap)
    args = ap.parse_args(argv)

    runs, record = refuse_held_out(list(args.runs), allow=args.allow_holdout,
                                   drop=args.v1_only, context="gaussian_null")
    missing = [r for r in runs if not (r / "activations.npz").exists()]
    if missing:
        print(f"refusing: no activations.npz in {missing}", file=sys.stderr)
        return 1
    if not runs:
        print("no runs", file=sys.stderr)
        return 1
    jobs = []
    for r in runs:
        n_layers = np.load(r / "activations.npz")["activations"].shape[0]
        for L in (args.layers if args.layers else range(n_layers)):
            for f in args.frames:
                jobs.append((str(r), int(L), f, args.n_draws, args.seed, args.calibrate,
                             args.dedupe_strings))

    t0 = time.time()
    if args.workers > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(args.workers) as pool:
            records = pool.map(_job, jobs, chunksize=1)
    else:
        records = [_job(j) for j in jobs]
    out = {"n_draws": args.n_draws, "seed": args.seed, "n_rogue": N_ROGUE,
           "calibrate": bool(args.calibrate), "dedupe_strings": bool(args.dedupe_strings),
           "frames": args.frames, "holdout": record, "inputs": [str(r) for r in runs],
           "seconds": round(time.time() - t0, 1), "records": records}
    out["summary"] = summarise(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    text = summary_text(out)
    args.out.with_suffix(".txt").write_text(text + "\n")
    print(text)
    print(f"  wrote {args.out} ({out['seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
