"""
p1d_cluster_ensemble/admit.py — which HDBSCAN groups beat their covariance?

Build step 1 of `design-1d.md` ("The proposed definition"). At one layer of
one run, a *cluster* is an HDBSCAN group of deduplicated tokens whose
per-group statistic exceeds what the tokens' own matched-covariance
Gaussian (`gaussian_null.py`) produces **anywhere** in a draw:

- **groups** HDBSCAN (``min_cluster_size``, EOM, float64 cosine) on the
  level-set tree, `level_set_hdbscan`: every mutual-reachability edge of one
  weight is merged at once. The shipped call (hdbscan's binary tree) orders
  tied edges by processing order, and ties are the rule, so its groups and
  their stabilities change when unrelated rows are added; it glued a stray
  background row to a planted cap in the tests, and on real layers many of
  its ``min_cluster_size`` 2 groups are never a component of the graph at
  any distance (`status-1d.md` "Admission"). Its labels, its tie artefacts
  (`shipped_check`) and its ARI to ours are written beside every record.
- **statistic** ``excess`` = ``S_C / |C|``, the group's EOM stability
  (sum over members of ``lambda_p - lambda_birth``, ``lambda`` = 1 /
  mutual-reachability distance) over its size. It depends only on the
  group's own branch. hdbscan's ``cluster_persistence_`` is the same number
  divided by the tree's largest ``lambda``, so one tight group anywhere
  scores every other group down (`/challenge-pr` on #121;
  `tests/test_phase1d_admit.py` pins both). Sensitivity arm: ``log_life``
  = ``log(lambda_death / lambda_birth)``.
- **null** each draw is clustered the same way and its *largest* statistic
  kept; a group's ``p = (1 + #{draw max >= obs}) / (B + 1)`` and it is
  admitted at ``p <= alpha``. Under the null, P(any group admitted) <=
  alpha per record whatever the group count (a max statistic,
  single-step).
- **arms** ``min_cluster_size`` 2 (Phase 1's shipped call) and 4
  (`SUBSTANTIAL_CLUSTER_SIZE`), on the same draws.

The frames, the draw, the deduplication and ``--calibrate`` are
`gaussian_null.py`'s, unchanged, and the draws are the same ones it makes
for the same seed. As there, the null is not at nominal level everywhere,
so real counts are read against a calibration on the same inputs
(``report``, which refuses any other) and labels are released per (step,
frame, arm, band) only if that band's calibration, and the same band of the
step-0 control, each admit in at most ``RELEASE_BOUND`` of their records
(`table`); the control's own labels are never released. The bound treats
a band's 56 records (7 prompts x 8 adjacent layers) as if independent,
which they are not.

Label codes in ``labels.json``: ``>= 0`` an admitted group (its
level-set HDBSCAN label); ``-1`` not tested (a later occurrence of a string, dropped by
deduplication); ``-2`` tested and not in an admitted group (HDBSCAN noise,
or a group that did not beat the null); ``-3`` not tested because it sits before
``--min-position`` (the file's ``min_position`` says where). Tier 1: exploratory, unregistered.
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

from .gaussian_null import (CALIBRATE_SEED_OFFSET, MIN_TOKENS, _step_prompt, band_of,
                            first_occurrences, frame_vectors, gaussian_draw,
                            input_fingerprint, run_tokens, span_coordinates)

FRAMES = ("centred", "raw")
STATISTICS = ("excess", "log_life")
#: PLACED (`design-1d.md`): 2 is the shipped call, 4 the sensitivity arm.
MIN_CLUSTER_SIZES = (2, 4)
#: PLACED: per-record error of the max statistic.
ALPHA = 0.05
#: PLACED: a band's labels are released only if its calibration admits in at
#: most this share of records (2 alpha).
RELEASE_BOUND = 2 * ALPHA
#: The untrained checkpoint: no learned content, so it must not admit.
CONTROL_STEP = "step0"
BANDS = ("L1-8", "L9-16", "L17-24")
NOT_TESTED, NOT_ADMITTED, BEFORE_MIN_POSITION = -1, -2, -3


# ---------------------------------------------------------------------------
# Per-group statistics from the condensed tree
# ---------------------------------------------------------------------------

def fit_hdbscan(D: np.ndarray, min_cluster_size: int):
    """Phase 1's call (`methods._fit_hdbscan`), returning the fitted object.

    Needs the `hdbscan` package: sklearn's HDBSCAN does not expose the
    condensed tree, and 1d refuses to substitute another method.
    """
    try:
        import hdbscan as _h
    except ImportError as e:  # pragma: no cover - the deps tier has it
        raise RuntimeError("admit.py needs the `hdbscan` package (condensed tree)") from e
    return _h.HDBSCAN(min_cluster_size=int(min_cluster_size), metric="precomputed",
                      cluster_selection_method="eom").fit(
        np.ascontiguousarray(D, dtype=np.float64))


def mutual_reachability(D: np.ndarray, min_samples: int) -> np.ndarray:
    """hdbscan's: ``max(core_a, core_b, d_ab)``, core = distance to the
    ``min_samples``-th row of the sorted distances, self included at 0
    (checked against hdbscan's single-linkage tree in the tests)."""
    D = np.asarray(D, dtype=np.float64)
    core = np.sort(D, axis=1)[:, int(min_samples)]
    mr = np.maximum(np.maximum(core[:, None], core[None, :]), D)
    np.fill_diagonal(mr, 0.0)
    return mr


def _mst(W: np.ndarray) -> List[Tuple[float, int, int]]:
    """Edges ``(weight, a, b)`` of a minimum spanning tree of dense ``W``, ascending."""
    from scipy.sparse.csgraph import minimum_spanning_tree
    if W.shape[0] < 2:
        return []
    mst = minimum_spanning_tree(W).tocoo()
    edges = sorted(zip(mst.data.tolist(), mst.row.tolist(), mst.col.tolist()))
    if len(edges) != W.shape[0] - 1:
        raise ValueError("a zero or missing edge: the graph is not connected")
    return edges


def _level_tree(mr: np.ndarray) -> Tuple:
    """
    The single-linkage tree of ``mr`` with every edge of one weight merged
    at once: node ``(weight, children, size)``, a point ``(None, (i,), 1)``.

    Mutual-reachability edges tie as a rule (a point's core distance is the
    weight of several of its edges). hdbscan's binary tree orders a tie by
    processing order, which other rows change; merged at once, the nodes are
    the level-set components, which every MST gives.
    """
    edges = _mst(mr)
    parent = list(range(mr.shape[0]))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def kids_at(node: Tuple, w: float) -> Tuple:
        # A node formed at this same weight is merged into, not nested.
        return node[1] if node[0] == w else (node,)

    top: Dict[int, Tuple] = {i: (None, (i,), 1) for i in range(mr.shape[0])}
    for w, a, b in edges:
        a, b = find(a), find(b)
        kids = kids_at(top.pop(a), w) + kids_at(top.pop(b), w)
        parent[b] = a
        top[a] = (w, kids, sum(k[2] for k in kids))
    (node,) = top.values()
    return node


def _points(node: Tuple) -> List[int]:
    out, stack = [], [node]
    while stack:
        k = stack.pop()
        if k[0] is None:
            out.append(k[1][0])
        else:
            stack.extend(k[1])
    return sorted(out)


def _split(node: Tuple, mcs: int) -> Tuple[List[Tuple], int]:
    """A level's pieces of ``min_cluster_size`` or more, and how many points fall out."""
    big = [k for k in node[1] if k[2] >= mcs]
    return big, node[2] - sum(k[2] for k in big)


def level_set_hdbscan(D: np.ndarray, min_cluster_size: int) -> Tuple[np.ndarray, List[Dict]]:
    """
    HDBSCAN (Campello et al. 2013; EOM selection, no single cluster) on the
    tie-merged tree of `_level_tree`: labels, and one row per selected group
    with its birth, death and stability.

    Condensing: at each level, pieces under ``min_cluster_size`` fall out
    (their points leave at that ``lambda`` = 1 / distance); two or more
    pieces of ``min_cluster_size`` or more end the cluster and start new
    ones; one continues it. Stability is the sum over a cluster's points of
    ``lambda_out - lambda_birth``; death is its last level. A group's row
    depends only on its own branch: its internal edges and its cheapest edge
    out (`branch` recomputes it from those alone; the tests check they agree).
    Labels are numbered by each group's smallest member.
    """
    D = np.asarray(D, dtype=np.float64)
    if np.any(D[~np.eye(D.shape[0], dtype=bool)] <= 0):
        raise ValueError("a zero distance between distinct rows (duplicate points?); "
                         "refusing rather than scoring a group infinite")
    mcs = int(min_cluster_size)
    return condense_and_select(_level_tree(mutual_reachability(D, mcs)), mcs)


def condense_and_select(tree: Tuple, min_cluster_size: int) -> Tuple[np.ndarray, List[Dict]]:
    """Condense a ``(weight, children, size)`` tree and select by EOM (see
    `level_set_hdbscan`). Takes any tree, so the tests can hand it hdbscan's
    own binary one and compare."""
    mcs, n = int(min_cluster_size), tree[2]
    clusters = [{"birth": 0.0, "death": 0.0, "stability": 0.0, "children": [],
                 "members": list(range(n))}]
    stack = [(tree, 0)]
    while stack:
        node, cid = stack.pop()
        c = clusters[cid]
        lam = 1.0 / node[0]
        big, fall = _split(node, mcs)
        c["stability"] += fall * (lam - c["birth"])
        c["death"] = lam
        if len(big) == 1:
            stack.append((big[0], cid))
        elif big:
            for k in big:
                c["stability"] += k[2] * (lam - c["birth"])
                c["children"].append(len(clusters))
                stack.append((k, len(clusters)))
                clusters.append({"birth": lam, "death": lam, "stability": 0.0,
                                 "children": [], "members": _points(k)})

    # Excess of mass, as hdbscan: a cluster is kept unless its children's
    # (propagated) stability is strictly larger. Children have larger ids.
    prop, selected = {}, set()
    for cid in range(len(clusters) - 1, 0, -1):
        c = clusters[cid]
        sub = sum(prop[k] for k in c["children"])
        if c["children"] and sub > c["stability"]:
            prop[cid] = sub
            continue
        prop[cid] = c["stability"]
        selected.add(cid)
        desc = list(c["children"])
        while desc:
            k = desc.pop()
            selected.discard(k)
            desc.extend(clusters[k]["children"])
    chosen = sorted((clusters[k] for k in selected), key=lambda c: c["members"][0])
    labels = np.full(n, -1, dtype=int)
    rows = []
    for g, c in enumerate(chosen):
        labels[c["members"]] = g
        size = len(c["members"])
        rows.append({"label": g, "size": size, "members": c["members"],
                     "birth": c["birth"], "death": c["death"], "stability": c["stability"],
                     "excess": c["stability"] / size,
                     "log_life": float(np.log(c["death"] / c["birth"]))})
    return labels, rows


def branch(mr: np.ndarray, members: Sequence[int], min_cluster_size: int
           ) -> Optional[Dict[str, float]]:
    """
    A set's own branch of the hierarchy from its edges alone, or None if it
    is never a branch: ``G`` is a connected component of the
    mutual-reachability graph at some distance only if its internal
    bottleneck is below its cheapest edge out, ``w_out``. It is then born at
    ``1 / w_out`` and condensed as in `level_set_hdbscan`.

    Used to check `level_set_hdbscan`'s rows, and to ask which of the
    shipped call's groups exist only through hdbscan's tie order.
    """
    idx = np.asarray(sorted(members), dtype=int)
    out = np.ones(mr.shape[0], dtype=bool)
    out[idx] = False
    w_out = float(mr[np.ix_(idx, np.flatnonzero(out))].min()) if out.any() else float("inf")
    sub = mr[np.ix_(idx, idx)]
    if idx.size < 2 or max(e[0] for e in _mst(sub)) >= w_out:
        return None
    birth = 1.0 / w_out
    node, stability = _level_tree(sub), 0.0
    while True:
        lam = 1.0 / node[0]
        big, fall = _split(node, int(min_cluster_size))
        stability += fall * (lam - birth)
        if len(big) != 1:
            stability += sum(k[2] for k in big) * (lam - birth)
            return {"birth": birth, "death": lam, "stability": stability}
        node = big[0]


def shipped_check(D: np.ndarray, labels: np.ndarray, min_cluster_size: int) -> Dict:
    """
    The shipped call's groups against the level-set tree: how many are never
    a branch (they exist only through hdbscan's tie order), and their sizes.
    """
    mr = mutual_reachability(D, min_cluster_size)
    groups = sorted(set(np.asarray(labels).tolist()) - {-1})
    artefacts = [g for g in groups
                 if branch(mr, np.flatnonzero(labels == g), min_cluster_size) is None]
    return {"k": len(groups), "tie_artefacts": len(artefacts),
            "artefact_sizes": [int(np.sum(labels == g)) for g in artefacts]}


def layer_groups(Z: np.ndarray, min_cluster_size: int) -> Tuple[np.ndarray, List[Dict], Dict]:
    """
    Level-set HDBSCAN on unit rows ``Z`` by Phase 1's distance route
    (float64 cosine, `LayerData`), and the shipped call on the same
    distances: its labels, `shipped_check`, and its ARI to ours.
    """
    from sklearn.metrics import adjusted_rand_score
    from .methods import LayerData
    D = LayerData.from_normed(Z).cos_dist
    labels, rows = level_set_hdbscan(D, min_cluster_size)
    shipped = fit_hdbscan(D, min_cluster_size).labels_
    check = shipped_check(D, shipped, min_cluster_size)
    check["ari_to_level_set"] = float(adjusted_rand_score(shipped, labels))
    check["labels"] = shipped.astype(int).tolist()
    return labels, rows, check


def rank_p(obs: float, null_max: np.ndarray) -> float:
    """``(1 + #{null >= obs}) / (B + 1)``: the repo's rank p, against draw maxima."""
    null_max = np.asarray(null_max, dtype=np.float64)
    return float((1 + np.sum(null_max >= obs)) / (null_max.size + 1))


# ---------------------------------------------------------------------------
# One record
# ---------------------------------------------------------------------------

def admit_record(Y: np.ndarray, frame: str, n_draws: int, seed: int,
                 min_cluster_sizes: Sequence[int] = MIN_CLUSTER_SIZES,
                 alpha: float = ALPHA, calibrate: bool = False) -> Dict:
    """
    One (layer, frame): every level-set HDBSCAN group, its statistics, p
    and verdict, per ``min_cluster_size`` arm; the shipped call's labels and
    `shipped_check` beside them; and, per draw, the null's maxima and group
    counts (ours and the shipped call's).

    ``calibrate`` replaces the rows by one draw of their own Gaussian, as
    `gaussian_null.null_record` does; the draws are then the same ones
    `gaussian_null` makes for this seed.
    """
    if frame not in FRAMES:
        raise ValueError(f"unknown frame {frame!r}; use one of {FRAMES}")
    Z, info = frame_vectors(Y, frame)
    Zs = span_coordinates(Z)
    if calibrate:
        Zs = span_coordinates(gaussian_draw(Zs, np.random.default_rng(seed + CALIBRATE_SEED_OFFSET)))
        info = {**info, "calibrate": True}
    obs = {m: layer_groups(Zs, m) for m in min_cluster_sizes}
    null = {m: {"k": [], "shipped_k": [], "shipped_tie_artefacts": [],
                **{s: [] for s in STATISTICS}} for m in min_cluster_sizes}
    rng = np.random.default_rng(seed)
    for _ in range(int(n_draws)):
        X = gaussian_draw(Zs, rng)
        for m in min_cluster_sizes:
            _, rows, check = layer_groups(X, m)
            null[m]["k"].append(len(rows))
            null[m]["shipped_k"].append(check["k"])
            null[m]["shipped_tie_artefacts"].append(check["tie_artefacts"])
            for s in STATISTICS:
                # A draw with no group sets no bar: its maximum is 0, which
                # every real group's statistic (>= 0) meets, so it counts
                # against nobody (`rank_p` uses >=). Both statistics are >= 0.
                null[m][s].append(max((r[s] for r in rows), default=0.0))
    arms = {}
    for m in min_cluster_sizes:
        labels, rows, check = obs[m]
        for r in rows:
            for s in STATISTICS:
                r[f"p_{s}"] = rank_p(r[s], null[m][s])
                r[f"admitted_{s}"] = bool(r[f"p_{s}"] <= alpha)
        arms[str(m)] = {"labels": labels.astype(int).tolist(), "groups": rows,
                        "n_groups": len(rows), "shipped": check,
                        "n_admitted": {s: int(sum(r[f"admitted_{s}"] for r in rows))
                                       for s in STATISTICS},
                        "null": null[m],
                        "threshold_q95": {s: float(np.quantile(null[m][s], 1 - alpha))
                                          for s in STATISTICS} if n_draws else {}}
    return {"info": info, "n_draws": int(n_draws), "seed": int(seed),
            "alpha": float(alpha), "arms": arms}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _job(args: Tuple) -> Dict:
    run_dir, layer, frame, n_draws, seed, calibrate, min_position = args
    acts = np.load(Path(run_dir) / "activations.npz")["activations"]
    tokens = run_tokens(Path(run_dir))
    if len(tokens) != acts.shape[1]:
        raise ValueError(f"{run_dir}: {len(tokens)} token strings for "
                         f"{acts.shape[1]} activation rows; refusing to dedupe")
    keep = first_occurrences(tokens)
    keep = keep[keep >= int(min_position)]
    step, prompt = _step_prompt(Path(run_dir))
    base = {"run_dir": str(run_dir), "step": step, "prompt": prompt, "layer": int(layer),
            "n_tokens": int(acts.shape[1]), "n_kept": int(keep.size), "keep": keep.tolist(),
            "min_position": int(min_position)}
    if keep.size < MIN_TOKENS:
        return {**base, "skipped": f"{keep.size} distinct strings < {MIN_TOKENS}",
                "info": {"frame": frame}}
    t0 = time.time()
    rec = admit_record(acts[layer][keep], frame, n_draws, seed, calibrate=calibrate)
    rec.update(base)
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def run(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="admit run", description="Admission on Phase 1 runs.")
    ap.add_argument("--runs", type=Path, nargs="+", required=True,
                    help="Phase 1 prompt directories (activations.npz + geometry.json)")
    ap.add_argument("--out", type=Path, required=True, help="JSON to write")
    ap.add_argument("--layers", type=int, nargs="*", default=list(range(1, 25)),
                    help="default L1-24 (`design-1d.md`: L0 is token identity)")
    ap.add_argument("--frames", nargs="*", default=list(FRAMES), choices=FRAMES)
    ap.add_argument("--n-draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibrate", action="store_true",
                    help="run on one Gaussian draw of each layer instead of the tokens")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--min-position", type=int, default=0,
                    help="diagnostic arm: test only kept tokens at absolute position >= this "
                         "(drops the prompt's opening; `status-1d.md` \"Position\")")
    add_holdout_args(ap)
    args = ap.parse_args(argv)

    runs, holdout = refuse_held_out(list(args.runs), allow=args.allow_holdout,
                                    drop=args.v1_only, context="admit")
    missing = [r for r in runs if not ((r / "activations.npz").exists()
                                       and (r / "geometry.json").exists())]
    if missing or not runs:
        print(f"refusing: no runs, or missing activations.npz / geometry.json in {missing}",
              file=sys.stderr)
        return 1
    jobs = [(str(r), int(L), f, args.n_draws, args.seed, args.calibrate, args.min_position)
            for r in runs for L in args.layers for f in args.frames]

    # Resumable as `gaussian_null`: each record is written as it finishes and
    # reused only if its settings and its run's input files match this call's.
    parts = args.out.with_suffix(".parts")
    parts.mkdir(parents=True, exist_ok=True)
    settings = {"n_draws": args.n_draws, "seed": args.seed, "calibrate": bool(args.calibrate),
                "alpha": ALPHA, "min_cluster_sizes": list(MIN_CLUSTER_SIZES),
                "min_position": args.min_position}
    fingerprints = {str(r): input_fingerprint(r, ("activations.npz", "geometry.json"))
                    for r in runs}

    def part_of(job) -> Path:
        return parts / f"{Path(job[0]).parent.name}__{Path(job[0]).name}__L{job[1]}__{job[2]}.json"

    done, todo = [], []
    for j in jobs:
        p = part_of(j)
        if p.exists():
            rec = json.loads(p.read_text())
            if rec.get("_settings") == {**settings, "input": fingerprints[j[0]]}:
                done.append(rec)
                continue
        todo.append(j)
    print(f"  {len(done)} of {len(jobs)} records already done; running {len(todo)}", flush=True)
    t0 = time.time()

    def keep(job, rec):
        rec["_settings"] = {**settings, "input": fingerprints[job[0]]}
        tmp = part_of(job).with_suffix(".tmp")
        tmp.write_text(json.dumps(rec))
        tmp.replace(part_of(job))
        done.append(rec)
        if len(done) % 50 == 0 or len(done) == len(jobs):
            print(f"  {len(done)}/{len(jobs)} done ({time.time() - t0:.0f} s this call)", flush=True)

    def key(rec):
        return rec["run_dir"], rec["layer"], rec["info"]["frame"]

    if args.workers > 1 and todo:
        from multiprocessing import get_context
        by_key = {(j[0], j[1], j[2]): j for j in todo}
        with get_context("spawn").Pool(args.workers) as pool:
            for rec in pool.imap_unordered(_job, todo, chunksize=1):
                keep(by_key[key(rec)], rec)
    else:
        for job in todo:
            keep(job, _job(job))
    order = {(j[0], j[1], j[2]): i for i, j in enumerate(jobs)}
    records = sorted(done, key=lambda r: order[key(r)])
    for r in records:
        r.pop("_settings", None)
    out = {**settings, "frames": args.frames, "holdout": holdout,
           "inputs": [str(r) for r in runs], "seconds": round(time.time() - t0, 1),
           "records": records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(f"  wrote {args.out} ({out['seconds']} s, {len(records)} records, "
          f"{sum('skipped' in r for r in records)} skipped)")
    return 0


# ---------------------------------------------------------------------------
# Report: counts against the calibration, and the label release
# ---------------------------------------------------------------------------

def load(path: Path) -> Dict:
    d = json.loads(Path(path).read_text())
    d["records"] = [r for r in d["records"] if "skipped" not in r]
    return d


def _cell(recs: List[Dict], arm: str, stat: str) -> Dict:
    groups = [g for r in recs for g in r["arms"][arm]["groups"]]
    adm = [g for g in groups if g[f"admitted_{stat}"]]
    return {"n": len(recs),
            "records_admitting": int(sum(r["arms"][arm]["n_admitted"][stat] > 0 for r in recs)),
            "groups": len(groups), "admitted": len(adm),
            "shipped_groups": int(sum(r["arms"][arm]["shipped"]["k"] for r in recs)),
            "shipped_tie_artefacts": int(sum(r["arms"][arm]["shipped"]["tie_artefacts"]
                                             for r in recs)),
            "admitted_pairs": int(sum(g["size"] == 2 for g in adm)),
            "admitted_median_size": float(np.median([g["size"] for g in adm])) if adm else None}


def table(real: Dict, cal: Dict) -> List[Dict]:
    """
    Per (statistic, arm, step, frame, band): real against its own
    calibration, and whether the cell's labels may be released.

    A cell is released only if (1) its calibration admits in at most
    ``RELEASE_BOUND`` of its records, (2) it is not the control step, and
    (3) the control step, same statistic / arm / frame / band, also admits
    in at most ``RELEASE_BOUND`` of its records. A file without the control
    releases nothing. ``withheld`` says which condition failed.
    """
    if not cal.get("calibrate") or real.get("calibrate"):
        raise ValueError("need a real file and a --calibrate file, in that order")
    for k in ("seed", "alpha", "min_cluster_sizes", "n_draws", "min_position"):
        if real.get(k, 0 if k == "min_position" else None) != cal.get(
                k, 0 if k == "min_position" else None):
            raise ValueError(f"real and calibration differ in {k}; a calibration is "
                             "only read against its own settings")
    if {Path(p).name for p in real["inputs"]} != {Path(p).name for p in cal["inputs"]}:
        raise ValueError("real and calibration were run on different prompt sets")

    def keys(d):
        return {(r["step"], r["prompt"], r["layer"], r["info"]["frame"]) for r in d["records"]}

    if keys(real) != keys(cal):
        raise ValueError("real and calibration cover different (step, prompt, layer, frame) "
                         f"records ({len(keys(real) ^ keys(cal))} differ)")

    def select(d, step, frame, band):
        return [r for r in d["records"] if r["step"] == step
                and r["info"]["frame"] == frame and band_of(r["layer"]) == band]

    rows = []
    for stat in STATISTICS:
        for arm in (str(m) for m in real["min_cluster_sizes"]):
            for step in sorted({r["step"] for r in real["records"]}):
                for frame in FRAMES:
                    for band in BANDS:
                        rr = select(real, step, frame, band)
                        if rr:
                            rows.append({"stat": stat, "arm": arm, "step": step, "frame": frame,
                                         "band": band, "real": _cell(rr, arm, stat),
                                         "cal": _cell(select(cal, step, frame, band), arm, stat)})
    control = {(r["stat"], r["arm"], r["frame"], r["band"]): r["real"]
               for r in rows if r["step"] == CONTROL_STEP}
    for r in rows:
        c, k = r["cal"], control.get((r["stat"], r["arm"], r["frame"], r["band"]))
        if c["records_admitting"] > RELEASE_BOUND * c["n"]:
            why = f"calibration admits in {c['records_admitting']} of {c['n']} records"
        elif r["step"] == CONTROL_STEP:
            why = "the control step"
        elif k is None:
            why = f"no {CONTROL_STEP} control in this file"
        elif k["records_admitting"] > RELEASE_BOUND * k["n"]:
            why = f"{CONTROL_STEP} control admits in {k['records_admitting']} of {k['n']} records"
        else:
            why = None
        r["released"], r["withheld"] = why is None, why
    return rows


def labels_out(real: Dict, rows: List[Dict], stat: str = "excess") -> Dict:
    """Full-length labels per record, for released (step, frame, arm, band) cells only."""
    released = {(r["arm"], r["step"], r["frame"], r["band"]): r for r in rows if r["stat"] == stat}
    out = []
    for r in real["records"]:
        for arm in r["arms"]:
            cell = released.get((arm, r["step"], r["info"]["frame"], band_of(r["layer"])))
            base = {"run_dir": r["run_dir"], "step": r["step"], "prompt": r["prompt"],
                    "layer": r["layer"], "frame": r["info"]["frame"], "arm": int(arm)}
            if cell is None or not cell["released"]:
                out.append({**base, "withheld": cell["withheld"] if cell else "no cell"})
                continue
            lab = np.full(r["n_tokens"], NOT_TESTED, dtype=int)
            lab[:r.get("min_position", 0)] = BEFORE_MIN_POSITION
            lab[r["keep"]] = NOT_ADMITTED
            keep = np.asarray(r["keep"])
            for g in r["arms"][arm]["groups"]:
                if g[f"admitted_{stat}"]:
                    lab[keep[g["members"]]] = g["label"]
            out.append({**base, "labels": lab.tolist()})
    return {"statistic": stat, "codes": {">=0": "admitted group (level-set HDBSCAN label)",
                                         str(NOT_TESTED): "not tested (later occurrence, deduped)",
                                         str(NOT_ADMITTED): "tested, not admitted",
                                         str(BEFORE_MIN_POSITION): "not tested (before min_position)"},
            "min_position": real.get("min_position", 0),
            "release_bound": RELEASE_BOUND, "records": out}


def table_text(rows: List[Dict]) -> str:
    lines = [f"Admission (max statistic, alpha {ALPHA}) against calibration; release if "
             f"calibration and the {CONTROL_STEP} control each admit in <= {RELEASE_BOUND:.0%} "
             "of records (the control itself is never released).",
             "rec = records with >= 1 admitted group; adm = admitted groups (pairs); "
             "grp = all HDBSCAN groups"]
    head = None
    for r in rows:
        if (r["stat"], r["arm"]) != head:
            head = (r["stat"], r["arm"])
            lines.append(f"\n[{r['stat']}, min_cluster_size {r['arm']}]  step        frame    band    "
                         f"  n | real rec  adm (pairs)   grp | cal rec  adm   grp | release")
        a, c = r["real"], r["cal"]
        lines.append(f"  {'':<34}{r['step']:<11} {r['frame']:<8} {r['band']:<7} {a['n']:>3} | "
                     f"{a['records_admitting']:>8} {a['admitted']:>4} ({a['admitted_pairs']:>4}) "
                     f"{a['groups']:>5} | {c['records_admitting']:>7} {c['admitted']:>4} {c['groups']:>5} | "
                     f"{'yes' if r['released'] else 'no: ' + r['withheld']}")
    return "\n".join(lines)


def report(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="admit report")
    ap.add_argument("--real", type=Path, required=True)
    ap.add_argument("--calibrate", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="report JSON (.txt and labels beside it)")
    args = ap.parse_args(argv)
    real, cal = load(args.real), load(args.calibrate)
    rows = table(real, cal)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"real": str(args.real), "calibrate": str(args.calibrate),
                                    "rows": rows}, indent=1))
    args.out.with_name(args.out.stem + "_labels.json").write_text(json.dumps(labels_out(real, rows)))
    text = table_text(rows)
    args.out.with_suffix(".txt").write_text(text + "\n")
    print(text)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("run", "report"):
        print("usage: python -m p1d_cluster_ensemble.admit {run,report} ...", file=sys.stderr)
        return 2
    return (run if argv[0] == "run" else report)(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
