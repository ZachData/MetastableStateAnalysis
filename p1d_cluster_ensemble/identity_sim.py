"""
p1d_cluster_ensemble/identity_sim.py — the identity-weights positive control.

`design-1d.md` "Identity-weights positive control"; literature `lit-1d.md` §9.
The theory's own dynamics, run from Phase 1's deduplicated L0 rows, with
nothing of Pythia in them but the starting points:

    x_k' = P_{x_k}( sum_{j in S(k)} e^{beta <x_k, x_j>} x_j / Z_k ),
    Z_k  = sum_{j in S(k)} e^{beta <x_k, x_j>}

``S(k) = {j <= k}`` under the **causal** mask (2411.04990's (CSA) with
``Q = K = V = I``, self included) and all tokens under the **full** mask
(Geshkovski et al.'s (SA), whose orthogonal-start closed form is
`p1c_frames/gamma_ode.py`'s eq. (6.9)). One head, the same weights at every
time, no MLP, no RoPE, no LayerNorm.

Three stages, each its own subcommand:

- ``simulate`` integrates every (input, mask, beta) to the time grid and
  writes the snapshots, the theory's clusters (components of
  ``1 - cos <= eta``), each token's cosine to the first token, and (6.9)'s
  reference times.
- ``admit`` runs `admit.admit_record`, unchanged, on every snapshot above
  the float floor, real and ``--calibrate``.
- ``report`` reads recovery (theory clusters against admitted groups), the
  opening (the theory cluster and admitted groups holding position 0), and
  step 0's simulated opening against its real admitted groups.

Coordinates: the dynamics never leave the start rows' span (every velocity
is a combination of the ``x_j``), so they are integrated in an orthonormal
basis of it (`span_coordinates`): exact, and ``n`` wide instead of 1024.
Tier 1: exploratory, unregistered.
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

from .admit import FRAMES, MIN_CLUSTER_SIZES, admit_record
from .gaussian_null import (MIN_TOKENS, _step_prompt, first_occurrences, input_fingerprint,
                            run_tokens, span_coordinates)

MASKS = ("causal", "full")
#: `design-1d.md`: 0.43 = 3.46 / 8 and 3.46 [1.55, 5.57] are Blocked 9's two
#: conventions; 0.2 and 8 bracket them; 16 and 64 put delta = 4 beta^-1/2
#: below the typical angle between tokens.
BETAS = (0.0, 0.2, 0.43, 1.0, 2.0, 3.46, 5.57, 8.0, 16.0, 64.0)
#: Brackets (6.9)'s t* ~ 4.2 at n = 467 from 8x below to 4x above.
TIMES = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
#: PLACED: the theory's clusters are components of 1 - cos <= eta.
ETAS = (1e-2, 1e-3, 1e-4)
ETA = 1e-3
#: PLACED: a snapshot whose smallest pairwise 1 - cos is below this is not
#: admitted (admission reads float32 rows; collapsed pairs become ties at 0).
FLOAT_FLOOR = 1e-9
#: PLACED: dt is halved until no snapshot's Gram moves by more than this.
GRAM_TOL = 1e-6
MAX_HALVINGS = 6
#: PLACED: a theory cluster is recovered by an admitted group at Jaccard >= this.
MATCH_JACCARD = 0.5


# ---------------------------------------------------------------------------
# The dynamics
# ---------------------------------------------------------------------------

def _unit(X: np.ndarray) -> np.ndarray:
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def attention(X: np.ndarray, beta: float, mask: str) -> np.ndarray:
    """Row-stochastic ``softmax(beta <x_k, x_j>)`` over ``S(k)``; ``X`` unit rows."""
    if mask not in MASKS:
        raise ValueError(f"mask must be one of {MASKS}, got {mask!r}")
    L = float(beta) * (X @ X.T)
    if mask == "causal":
        L = np.where(np.tri(L.shape[0], dtype=bool), L, -np.inf)
    L = L - L.max(axis=1, keepdims=True)
    A = np.exp(L)
    return A / A.sum(axis=1, keepdims=True)


def velocity(X: np.ndarray, beta: float, mask: str) -> np.ndarray:
    """The field at ``X``, extended off the sphere by ``x -> x / |x|`` so RK4's
    intermediate stages see the sphere's field."""
    U = _unit(X)
    F = attention(U, beta, mask) @ U
    return F - np.sum(F * U, axis=1, keepdims=True) * U


def integrate(X0: np.ndarray, beta: float, mask: str, times: Sequence[float],
              dt: float) -> np.ndarray:
    """RK4 from unit rows ``X0``, renormalised every step; snapshots at ``times``,
    each of which must be a whole number of steps."""
    X = _unit(np.asarray(X0, dtype=np.float64))
    steps = [t / dt for t in times]
    if any(abs(s - round(s)) > 1e-9 for s in steps):
        raise ValueError(f"times {times} are not multiples of dt {dt}")
    steps = [int(round(s)) for s in steps]
    if steps != sorted(steps):
        raise ValueError("times must be increasing")
    out, done = [], 0
    for s in steps:
        for _ in range(s - done):
            k1 = velocity(X, beta, mask)
            k2 = velocity(X + 0.5 * dt * k1, beta, mask)
            k3 = velocity(X + 0.5 * dt * k2, beta, mask)
            k4 = velocity(X + dt * k3, beta, mask)
            X = _unit(X + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))
        done = s
        out.append(X.copy())
    return np.stack(out)


def initial_dt(beta: float) -> float:
    """The largest power of two at or below ``min(0.05, 0.25 / beta)``, so every
    grid time is a whole number of steps at every halving."""
    cap = min(0.05, 0.25 / beta) if beta > 0 else 0.05
    return float(2.0 ** np.floor(np.log2(cap)))


def integrate_converged(X0: np.ndarray, beta: float, mask: str,
                        times: Sequence[float] = TIMES, tol: float = GRAM_TOL,
                        max_halvings: int = MAX_HALVINGS) -> Tuple[np.ndarray, Dict]:
    """`integrate`, halving ``dt`` until no snapshot's Gram moves by more than
    ``tol``; ``info`` records the last ``dt``, the change and whether it converged,
    so an unconverged trajectory is visible rather than silently the coarsest."""
    dt = initial_dt(beta)
    S = integrate(X0, beta, mask, times, dt)
    info = {"dt": dt, "halvings": 0, "gram_change": float("nan"), "converged": False}
    for k in range(1, max_halvings + 1):
        dt /= 2.0
        S2 = integrate(X0, beta, mask, times, dt)
        change = float(max(np.max(np.abs(a @ a.T - b @ b.T)) for a, b in zip(S, S2)))
        S = S2
        info.update(dt=dt, halvings=k, gram_change=change)
        if change <= tol:
            info["converged"] = True
            break
    return S, info


# ---------------------------------------------------------------------------
# What a snapshot shows
# ---------------------------------------------------------------------------

def distances(X: np.ndarray) -> np.ndarray:
    """``1 - cos`` in float64, zero diagonal."""
    U = _unit(np.asarray(X, dtype=np.float64))
    D = np.clip(1.0 - U @ U.T, 0.0, None)
    np.fill_diagonal(D, 0.0)
    return 0.5 * (D + D.T)


def theory_clusters(X: np.ndarray, eta: float) -> np.ndarray:
    """Labels of the components of ``1 - cos <= eta`` with >= 2 tokens, numbered
    by smallest member; ``-1`` for a token alone."""
    from scipy.sparse.csgraph import connected_components
    D = distances(X)
    _, comp = connected_components(D <= eta, directed=False)
    labels = np.full(len(comp), -1, dtype=int)
    g = 0
    for c in dict.fromkeys(comp):   # first-seen order = smallest member first
        idx = np.flatnonzero(comp == c)
        if idx.size >= 2:
            labels[idx] = g
            g += 1
    return labels


def min_distance(X: np.ndarray) -> float:
    D = distances(X)
    return float(D[~np.eye(D.shape[0], dtype=bool)].min()) if D.shape[0] > 1 else float("inf")


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def groups_of(labels: Sequence[int]) -> List[List[int]]:
    labels = np.asarray(labels)
    return [np.flatnonzero(labels == g).tolist() for g in sorted(set(labels[labels >= 0]))]


def recovery(theory: Sequence[int], admitted: List[List[int]], min_size: int) -> Dict:
    """Per theory cluster of >= ``min_size`` tokens, its best Jaccard to an admitted
    group; recall and precision at `MATCH_JACCARD`; ARI with singletons and
    unadmitted tokens each in a group of their own."""
    from sklearn.metrics import adjusted_rand_score
    truth = [g for g in groups_of(theory) if len(g) >= min_size]
    best = [max((jaccard(t, a) for a in admitted), default=0.0) for t in truth]
    prec = [max((jaccard(a, t) for t in groups_of(theory)), default=0.0) for a in admitted]
    n = len(theory)

    def singletons(gs):
        lab = np.arange(n) + n
        for i, g in enumerate(gs):
            lab[g] = i
        return lab
    return {"n_theory": len(truth), "n_admitted": len(admitted), "best_jaccard": best,
            "recall": (float(np.mean([b >= MATCH_JACCARD for b in best])) if best else None),
            "precision": (float(np.mean([p >= MATCH_JACCARD for p in prec])) if prec else None),
            "ari": float(adjusted_rand_score(singletons(groups_of(theory)), singletons(admitted)))}


def gamma_reference(n: int, beta: float) -> Dict[str, float]:
    """(6.9)'s ``t_0.5`` and ``t_0.9`` at this ``n`` and beta: the full-attention,
    orthogonal-start collapse times, as a scale for ``t`` (``inf`` if not reached
    by ``t = 64``). One fixed step of 1e-2: a reference scale, not a result, so
    `collapse_time`'s step halving (slow where gamma never reaches the target)
    is not needed."""
    from p1c_frames.gamma_ode import integrate_gamma, time_to_threshold
    t, g = integrate_gamma(n, beta, t_max=64.0, dt=1e-2)
    return {f"t_{q}": time_to_threshold(t, g, q) for q in (0.5, 0.9)}


def snapshot_summary(X: np.ndarray) -> Dict:
    return {"min_dist": min_distance(X),
            "theory": {str(e): theory_clusters(X, e).tolist() for e in ETAS},
            "cos_to_first": (_unit(X) @ _unit(X)[0]).tolist()}


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------

def load_start(run_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Deduplicated L0 rows (first occurrences) of a Phase 1 run, in span
    coordinates, and their absolute positions."""
    acts = np.load(Path(run_dir) / "activations.npz")["activations"]
    tokens = run_tokens(Path(run_dir))
    if len(tokens) != acts.shape[1]:
        raise ValueError(f"{run_dir}: {len(tokens)} token strings for {acts.shape[1]} rows")
    keep = first_occurrences(tokens)
    return span_coordinates(_unit(np.asarray(acts[0][keep], dtype=np.float64))), keep


def traj_name(run_dir: str, mask: str, beta: float) -> str:
    step, prompt = _step_prompt(Path(run_dir))
    return f"{step}__{prompt}__{mask}__b{beta:g}"


def _sim_job(args: Tuple) -> Dict:
    run_dir, mask, beta, times, out_dir = args
    X0, keep = load_start(Path(run_dir))
    step, prompt = _step_prompt(Path(run_dir))
    base = {"run_dir": run_dir, "step": step, "prompt": prompt, "mask": mask,
            "beta": float(beta), "n_kept": int(keep.size), "keep": keep.tolist(),
            "times": list(times)}
    if keep.size < MIN_TOKENS:
        return {**base, "skipped": f"{keep.size} distinct strings < {MIN_TOKENS}"}
    t0 = time.time()
    S, info = integrate_converged(X0, beta, mask, times)
    path = Path(out_dir) / "traj" / f"{traj_name(run_dir, mask, beta)}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, snaps=S, times=np.asarray(times), keep=keep)
    return {**base, "integration": info, "traj": str(path),
            "gamma_ref": gamma_reference(int(keep.size), beta),
            "snapshots": [snapshot_summary(X) for X in S],
            "seconds": round(time.time() - t0, 1)}


def _pool_map(fn, jobs, workers):
    if workers > 1 and len(jobs) > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(workers) as pool:
            yield from pool.imap_unordered(fn, jobs, chunksize=1)
    else:
        for j in jobs:
            yield fn(j)


def _resumable(jobs, part_of, settings, fn, workers, label):
    """Run ``fn`` over ``jobs``, writing each result to ``part_of(job)`` as it
    finishes and reusing a part only when its settings match."""
    done, todo = [], []
    for j in jobs:
        p = part_of(j)
        if p.exists():
            rec = json.loads(p.read_text())
            if rec.get("_settings") == settings(j):
                done.append(rec)
                continue
        todo.append(j)
    print(f"  {label}: {len(done)} of {len(jobs)} already done; running {len(todo)}", flush=True)
    t0 = time.time()
    by_part = {str(part_of(j)): j for j in todo}
    for rec in _pool_map(fn, todo, workers):
        j = by_part[rec["_part"]]
        rec["_settings"] = settings(j)
        tmp = part_of(j).with_suffix(".tmp")
        tmp.write_text(json.dumps(rec))
        tmp.replace(part_of(j))
        done.append(rec)
        if len(done) % 20 == 0 or len(done) == len(jobs):
            print(f"  {label}: {len(done)}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)
    return done


def _sim_job_tagged(args):
    *job, part = args
    return {**_sim_job(tuple(job)), "_part": part}


def simulate(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="identity_sim simulate")
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--betas", type=float, nargs="*", default=list(BETAS))
    ap.add_argument("--masks", nargs="*", default=list(MASKS), choices=MASKS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    add_holdout_args(ap)
    args = ap.parse_args(argv)
    runs, holdout = refuse_held_out(list(args.runs), allow=args.allow_holdout,
                                    drop=args.v1_only, context="identity_sim")
    missing = [r for r in runs if not (r / "activations.npz").exists()]
    if missing or not runs:
        print(f"refusing: no runs, or no activations.npz in {missing}", file=sys.stderr)
        return 1
    parts = args.out / "simulate.parts"
    parts.mkdir(parents=True, exist_ok=True)
    jobs = [(str(r), m, float(b), list(TIMES), str(args.out))
            for r in runs for m in args.masks for b in args.betas]

    def part_of(j):
        return parts / f"{traj_name(j[0], j[1], j[2])}.json"

    def settings(j):
        return {"times": list(TIMES), "etas": list(ETAS), "tol": GRAM_TOL,
                "input": input_fingerprint(j[0], ("activations.npz", "geometry.json"))}
    tagged = [(*j, str(part_of(j))) for j in jobs]
    done = _resumable(tagged, lambda j: Path(j[-1]), lambda j: settings(j[:-1]),
                      _sim_job_tagged, args.workers, "simulate")
    unconverged = [d["_part"] for d in done if not d.get("skipped")
                   and not d["integration"]["converged"]]
    summary = {"betas": args.betas, "masks": args.masks, "times": list(TIMES),
               "etas": list(ETAS), "float_floor": FLOAT_FLOOR, "gram_tol": GRAM_TOL,
               "holdout": holdout, "inputs": [str(r) for r in runs],
               "parts": sorted(d["_part"] for d in done), "unconverged": unconverged}
    (args.out / "simulate.json").write_text(json.dumps(summary, indent=1))
    print(f"  {len(done)} trajectories; {len(unconverged)} not converged")
    return 0


# ---------------------------------------------------------------------------
# admit
# ---------------------------------------------------------------------------

def _admit_job(args: Tuple) -> Dict:
    traj, t_index, frame, n_draws, seed, calibrate, part = args
    z = np.load(traj)
    X = z["snaps"][t_index]
    base = {"traj": traj, "t_index": int(t_index), "t": float(z["times"][t_index]),
            "frame": frame, "calibrate": bool(calibrate), "_part": part}
    md = min_distance(X)
    D = distances(X)
    base["dist_q01"] = float(np.quantile(D[np.triu_indices(D.shape[0], 1)], 0.01))
    if md < FLOAT_FLOOR:
        return {**base, "skipped": f"smallest 1 - cos {md:.3g} < float floor {FLOAT_FLOOR:g}",
                "min_dist": md}
    try:
        rec = admit_record(X, frame, n_draws, seed, calibrate=calibrate)
    except ValueError as e:
        # The floor bounds the snapshot, not its null: a nearly collapsed
        # snapshot's Gaussian draws are as tight as it is, and the level-set
        # code refuses a draw with a pair below float32 resolution. Refused
        # here too, with the reason, rather than dropped.
        if "zero" not in str(e):
            raise
        return {**base, "skipped": f"a null draw fell below float resolution ({e})",
                "min_dist": md}
    for arm in rec["arms"].values():   # the null's per-draw arrays: not read here
        arm.pop("null", None)
    return {**base, **rec, "min_dist": md}


def admit(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="identity_sim admit")
    ap.add_argument("--out", type=Path, required=True, help="the simulate directory")
    ap.add_argument("--frames", nargs="*", default=list(FRAMES), choices=FRAMES)
    ap.add_argument("--n-draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to trajectories whose name contains one of these")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args(argv)
    sims = [json.loads(Path(p).read_text())
            for p in json.loads((args.out / "simulate.json").read_text())["parts"]]
    sims = [s for s in sims if "skipped" not in s and s["integration"]["converged"]]
    if args.only:
        sims = [s for s in sims if any(o in Path(s["traj"]).stem for o in args.only)]
    tag = "calibrate" if args.calibrate else "real"
    parts = args.out / f"admit_{tag}.parts"
    parts.mkdir(parents=True, exist_ok=True)
    jobs, seen_start = [], set()
    for s in sims:
        for i, t in enumerate(s["times"]):
            if t == 0.0:   # the start is the same for every mask and beta: admit it once
                if s["run_dir"] in seen_start:
                    continue
                seen_start.add(s["run_dir"])
            for f in args.frames:
                name = f"{Path(s['traj']).stem}__t{i}__{f}.json"
                jobs.append((s["traj"], i, f, args.n_draws, args.seed, args.calibrate,
                             str(parts / name)))
    settings = {"n_draws": args.n_draws, "seed": args.seed, "calibrate": args.calibrate,
                "min_cluster_sizes": list(MIN_CLUSTER_SIZES), "float_floor": FLOAT_FLOOR}
    done = _resumable(jobs, lambda j: Path(j[-1]), lambda j: settings, _admit_job,
                      args.workers, f"admit {tag}")
    (args.out / f"admit_{tag}.json").write_text(json.dumps(
        {**settings, "frames": args.frames, "parts": sorted(d["_part"] for d in done)}, indent=1))
    print(f"  {len(done)} records, {sum('skipped' in d for d in done)} below the float floor")
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _admitted(rec: Dict, arm: str, stat: str = "excess") -> List[List[int]]:
    return [g["members"] for g in rec["arms"][arm]["groups"] if g[f"admitted_{stat}"]]


def report_rows(sims: List[Dict], real: Dict[Tuple, Dict], cal: Dict[Tuple, Dict],
                eta: float = ETA) -> List[Dict]:
    """One row per (trajectory, t, frame, arm): theory clusters, recovery, the
    opening, the calibration's admitted count on the same snapshot, and #123's
    position check on the admitted groups (against random same-size groups of
    the record's kept positions)."""
    from .position_check import ALPHA as POS_ALPHA, N_DRAWS, group_position, random_groups
    rng = np.random.default_rng(0)
    nulls: Dict[Tuple, Dict] = {}

    def position(members, kept, run_dir):
        k = (run_dir, len(members))
        if k not in nulls:
            nulls[k] = random_groups(kept, len(members), N_DRAWS, rng)
        return group_position(members, kept, nulls[k])

    rows = []
    for s in sims:
        stem = Path(s["traj"]).stem
        kept = np.asarray(s["keep"])
        start = (s["run_dir"], 0)
        for i, t in enumerate(s["times"]):
            snap = s["snapshots"][i]
            theory = np.asarray(snap["theory"][str(eta)])
            open_lab = theory[0]
            open_members = np.flatnonzero(theory == open_lab).tolist() if open_lab >= 0 else [0]
            for f in FRAMES:
                key = (start, f) if t == 0.0 else ((stem, i), f)
                r, c = real.get(key), cal.get(key)
                if r is None:
                    continue
                for arm in (r.get("arms") or {"skipped": None}):
                    row = {"step": s["step"], "prompt": s["prompt"], "mask": s["mask"],
                           "beta": s["beta"], "t": t, "frame": f, "arm": arm,
                           "n": s["n_kept"], "min_dist": snap["min_dist"],
                           "open_size": len(open_members),
                           "open_max_rank": max(open_members),
                           "cos_first_early": float(np.mean(snap["cos_to_first"][1:9])),
                           "cos_first_late": float(np.mean(
                               snap["cos_to_first"][len(snap["cos_to_first"]) // 2:]))}
                    if "skipped" in r:
                        rows.append({**row, "skipped": r["skipped"]})
                        continue
                    adm = _admitted(r, arm)
                    rec = recovery(theory, adm, int(arm))
                    with0 = [g for g in adm if 0 in g]
                    pos = [position(g, kept, s["run_dir"]) for g in adm]
                    first_pos = position(with0[0], kept, s["run_dir"]) if with0 else {}
                    # Post hoc (after the first full read): recall against every
                    # HDBSCAN group, admitted or not, separates "the group step never
                    # formed it" from "the null rejected it"; and the largest theory
                    # cluster's share of tokens, since a cluster that is the whole cloud
                    # cannot beat the cloud's own Gaussian by construction.
                    every = [g["members"] for g in r["arms"][arm]["groups"]]
                    tsizes = [len(g) for g in groups_of(theory)]
                    row.update(recall_any_group=recovery(theory, every, int(arm))["recall"],
                               theory_largest_share=max(tsizes, default=0) / len(theory))
                    row.update(n_admitted=len(adm), recall=rec["recall"],
                               precision=rec["precision"], ari=rec["ari"],
                               n_theory=rec["n_theory"],
                               admitted_has_first=bool(with0),
                               admitted_first_size=len(with0[0]) if with0 else 0,
                               admitted_first_jaccard_open=(jaccard(with0[0], open_members)
                                                            if with0 else 0.0),
                               # 1.0 = exactly the first |g| kept tokens (an opening run)
                               admitted_first_opening_share=(
                                   float(np.mean(np.asarray(with0[0]) < len(with0[0])))
                                   if with0 else None),
                               admitted_first_max_rank=max(with0[0]) if with0 else None,
                               admitted_first_p_near=first_pos.get("p_near"),
                               admitted_first_p_span=first_pos.get("p_span"),
                               admitted_first_quarter_share=first_pos.get("first_quarter_share"),
                               n_admitted_positional=sum(q["p_near"] <= POS_ALPHA for q in pos),
                               n_admitted_early=sum(q["p_span"] <= POS_ALPHA
                                                    and q["first_quarter_share"] >= 0.5
                                                    for q in pos),
                               cal_admitted=(None if c is None or "skipped" in c
                                             else len(_admitted(c, arm))))
                    rows.append(row)
    return rows


def step0_real_match(sims: List[Dict], real: Dict[Tuple, Dict], admit_real: Path,
                     betas=(0.0, 0.43, 3.46), eta: float = ETA) -> List[Dict]:
    """For step 0: the simulated opening (theory cluster holding position 0, and
    the admitted group holding it) against step 0's real admitted groups holding
    position 0 at L1-24, same frame and arm; best Jaccard over layers."""
    d = json.loads(Path(admit_real).read_text())
    by = {}
    for r in d["records"]:
        if r["step"] != "step0" or "skipped" in r:
            continue
        for arm, a in r["arms"].items():
            for g in a["groups"]:
                if g["admitted_excess"] and 0 in g["members"]:
                    by.setdefault((r["prompt"], r["info"]["frame"], arm), []).append(
                        (r["layer"], g["members"]))
    out = []
    for s in sims:
        if s["step"] != "step0" or s["beta"] not in betas:
            continue
        stem = Path(s["traj"]).stem
        for i, t in enumerate(s["times"]):
            if t == 0.0:
                continue
            theory = np.asarray(s["snapshots"][i]["theory"][str(eta)])
            open_members = np.flatnonzero(theory == theory[0]).tolist() if theory[0] >= 0 else [0]
            for f in FRAMES:
                r = real.get(((stem, i), f))
                for arm in MIN_CLUSTER_SIZES:
                    arm = str(arm)
                    reals = by.get((s["prompt"], f, arm), [])
                    sim_adm = ([g for g in _admitted(r, arm) if 0 in g]
                               if r and "skipped" not in r else [])

                    def best(members):
                        if not reals:
                            return (0.0, None)
                        j, L = max((jaccard(members, m), L) for L, m in reals)
                        return (j, L)
                    jo, Lo = best(open_members)
                    ja, La = best(sim_adm[0]) if sim_adm else (0.0, None)
                    out.append({"prompt": s["prompt"], "mask": s["mask"], "beta": s["beta"],
                                "t": t, "frame": f, "arm": arm, "n_real_layers": len(reals),
                                "open_size": len(open_members),
                                "jaccard_open": jo, "layer_open": Lo,
                                "jaccard_admitted": ja, "layer_admitted": La})
    return out


def _load_admit(out: Path, tag: str, sims: List[Dict]) -> Dict[Tuple, Dict]:
    meta = out / f"admit_{tag}.json"
    if not meta.exists():
        return {}
    run_of = {Path(s["traj"]).stem: s["run_dir"] for s in sims if "traj" in s}
    res = {}
    for p in json.loads(meta.read_text())["parts"]:
        r = json.loads(Path(p).read_text())
        stem = Path(r["traj"]).stem
        if r["t"] == 0.0:
            res[((run_of[stem], 0), r["frame"])] = r
        else:
            res[((stem, r["t_index"]), r["frame"])] = r
    return res


def report(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="identity_sim report")
    ap.add_argument("--out", type=Path, required=True, help="the simulate directory")
    ap.add_argument("--step0-admit", type=Path, default=None,
                    help="#122's real.json, for step 0's real groups")
    args = ap.parse_args(argv)
    every = [json.loads(Path(p).read_text())
             for p in json.loads((args.out / "simulate.json").read_text())["parts"]]
    sims = [s for s in every if "skipped" not in s and s["integration"]["converged"]]
    real, cal = _load_admit(args.out, "real", every), _load_admit(args.out, "calibrate", every)
    if not real:
        print("refusing: no admit_real.json", file=sys.stderr)
        return 1
    out = {"eta": ETA, "match_jaccard": MATCH_JACCARD,
           "rows": report_rows(sims, real, cal),
           "eta_sensitivity": {str(e): [{"step": s["step"], "prompt": s["prompt"],
                                         "mask": s["mask"], "beta": s["beta"], "t": t,
                                         "n_theory": len(groups_of(sn["theory"][str(e)])),
                                         "sizes": sorted((len(g) for g in groups_of(
                                             sn["theory"][str(e)])), reverse=True)[:5]}
                                        for s in sims for t, sn in zip(s["times"], s["snapshots"])]
                                   for e in ETAS}}
    if args.step0_admit:
        out["step0_real_match"] = step0_real_match(sims, real, args.step0_admit)
    (args.out / "report.json").write_text(json.dumps(out))
    (args.out / "report.txt").write_text(summary_text(out))
    print(f"  wrote {args.out / 'report.json'} ({len(out['rows'])} rows) and report.txt")
    return 0


def _cells(rows, step, mask, beta, t, frame, arm):
    return [r for r in rows if r["step"] == step and r["mask"] == mask and r["beta"] == beta
            and r["t"] == t and r["frame"] == frame and r["arm"] == str(arm)]


def summary_text(rep: Dict) -> str:
    """The tables `status-1d.md` "Identity-weights positive control" reads: per
    (step, mask, beta, t), counts over the 7 prompts."""
    rows, times = rep["rows"], [t for t in TIMES if t > 0]
    out = []
    for frame in FRAMES:
        for arm in (4, 2):
            out.append(f"\n## Recovery, {frame}, min_cluster_size {arm}, eta {rep['eta']}. "
                       "Per t: records with a theory cluster >= arm / mean recall over them / "
                       "records admitting (real) / records admitting (calibration). "
                       "'-' = every record skipped (float floor)")
            for step in ("step0", "step143000"):
                for mask in MASKS:
                    out.append(f"{step} {mask}")
                    for b in BETAS:
                        line = []
                        for t in times:
                            ok = [r for r in _cells(rows, step, mask, b, t, frame, arm)
                                  if "skipped" not in r]
                            if not ok:
                                line.append(f"{'-':>16}")
                                continue
                            wt = [r["recall"] for r in ok if r["n_theory"] > 0]
                            adm = sum(r["n_admitted"] > 0 for r in ok)
                            cal = sum((r["cal_admitted"] or 0) > 0 for r in ok)
                            rec = f"{np.mean(wt):.2f}" if wt else "  - "
                            line.append(f"{len(wt)}/{rec}/{adm}/{cal}".rjust(16))
                        out.append(f"  b={b:<5}" + "".join(line))
    for frame in FRAMES:
        out.append(f"\n## Opening, {frame}, min_cluster_size 2. Per t: records with an "
                   "admitted group holding position 0 / its median size / median share of "
                   "its members among the first |g| kept tokens / mean cos to x1 at kept "
                   "ranks 1-8 vs the second half")
        for step in ("step0", "step143000"):
            for mask in MASKS:
                out.append(f"{step} {mask}")
                for b in BETAS:
                    line = []
                    for t in times:
                        ok = [r for r in _cells(rows, step, mask, b, t, frame, 2)
                              if "skipped" not in r]
                        if not ok:
                            line.append(f"{'-':>26}")
                            continue
                        h = [r for r in ok if r["admitted_has_first"]]
                        sz = np.median([r["admitted_first_size"] for r in h]) if h else 0
                        sh = (f"{np.median([r['admitted_first_opening_share'] for r in h]):.2f}"
                              if h else "  - ")
                        e = np.mean([r["cos_first_early"] for r in ok])
                        late = np.mean([r["cos_first_late"] for r in ok])
                        line.append(f"{len(h)}/{sz:.0f}/{sh}/{e:.2f}v{late:.2f}".rjust(26))
                    out.append(f"  b={b:<5}" + "".join(line))
    m = rep.get("step0_real_match")
    if m:
        out.append("\n## Step 0: the simulated opening against step 0's real admitted groups "
                   "holding position 0 (best Jaccard over L1-24, median over prompts). "
                   "Per t: theory cluster holding position 0 / admitted group holding it")
        for frame in FRAMES:
            for arm in ("2", "4"):
                out.append(f"{frame}, min_cluster_size {arm}")
                for mask in MASKS:
                    for b in sorted({r["beta"] for r in m}):
                        line = []
                        for t in times:
                            rs = [r for r in m if r["mask"] == mask and r["beta"] == b
                                  and r["t"] == t and r["frame"] == frame and r["arm"] == arm]
                            line.append((f"{np.median([r['jaccard_open'] for r in rs]):.2f}/"
                                         f"{np.median([r['jaccard_admitted'] for r in rs]):.2f}"
                                         if rs else "-").rjust(11))
                        out.append(f"  {mask:6} b={b:<5}" + "".join(line))
    return "\n".join(out) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmds = {"simulate": simulate, "admit": admit, "report": report}
    if not argv or argv[0] not in cmds:
        print(f"usage: python -m p1d_cluster_ensemble.identity_sim {{{','.join(cmds)}}} ...",
              file=sys.stderr)
        return 2
    return cmds[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
