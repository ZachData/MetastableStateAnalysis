"""
p1e_energy_field/u3_saddles.py — U3 at β 10, crests and saddles between U1 (b)'s wells
(`design-1e.md` "U3 at β 10", fixed before any U3 output).

Per (step, long passage, layer ℓ ∈ {4, 12, 20}), in β's frame (unit LN1 rows, states fixed), at β 10:

- **wells**: U1 (b)'s mean shift, recomputed with its modes and refused unless the labels equal
  (b)'s saved ones; every well counts, singletons too;
- **merge tree** (primary): a 15-nearest-neighbour graph over the targets (where it leaves wells
  apart, each apart component's targets also joined to their 15 nearest outside it), edge height ``min(h(u_i), h(unit(u_i + u_j)), h(u_j))`` with
  ``h = log φ_β``; edges from the highest down, and where an edge joins two components whose
  highest wells differ, the lower well dies (elder rule): persistence ``peak − edge height``;
- **NEB** (beside): a climbing-image nudged elastic band on the cell's 8 most persistent deaths,
  its saddle height against the graph's;
- **null**: U1's 4 matched Gaussians (U1's seeds), their merge trees (graph only).

Not computed (fences): inner products between modes or any statistic of their configuration
(P-S1); the modes are path ends only and are never stored. Tier 1: exploratory, unregistered.
    python -m p1e_energy_field.u3_saddles run --wells <U1 (b) dir> --runs <p1e_long8 dir> \
        --out <dir> [--device cuda|cpu] [--first-only]
    python -m p1e_energy_field.u3_saddles report --out <dir>
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .u1_field import (BANDS, FIRST, N_DRAW, SEED, agreement, gaussian_draw, mean_shift,
                       plan, token_classes, well_stats)
from .u2_block import code_sha, load_ln1, unit_rows

BETA = 10.0
#: Placed by the rule, not calibrated.
KNN, MAX_BRIDGE_ROUNDS = 15, 50
NEB_IMAGES, NEB_ITERS, NEB_HTOL, NEB_TOP = 24, 3000, 1e-4, 8
NEB_WARM, NEB_EVERY = 200, 500
#: Amended after the first cell's check (`design-1e.md` "U3 at β 10"): one layer per trained band.
U3_LAYERS = (4, 12, 20)
AGREE_MIN, NEB_CONV_MIN, P_TOL = 0.999, 0.9, 1e-9
CHECK_LAYERS = (4, 12, 20)
DEEP = 1.0


# ---------------------------------------------------------------- pure: heights and the tree

def heights(Y: np.ndarray, U: np.ndarray, beta: float, chunk: int = 4096) -> np.ndarray:
    """``log φ_β(y) = log Σ_j exp(β⟨y, u_j⟩)`` at each row of ``Y`` (float64)."""
    from scipy.special import logsumexp
    Y, U = np.asarray(Y, dtype=np.float64), np.asarray(U, dtype=np.float64)
    return np.concatenate([logsumexp(beta * (Y[a:a + chunk] @ U.T), axis=1)
                           for a in range(0, len(Y), chunk)]) if len(Y) else np.zeros(0)


def unit(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-300)


def knn_edges(U: np.ndarray, k: int) -> np.ndarray:
    """Undirected edges ``(i, j)``, i < j, joining each row to its ``k`` nearest by inner product."""
    S = U @ U.T
    np.fill_diagonal(S, -np.inf)
    k = min(k, len(U) - 1)
    nn = np.argpartition(-S, k - 1, axis=1)[:, :k]
    i = np.repeat(np.arange(len(U)), k)
    e = np.stack([np.minimum(i, nn.ravel()), np.maximum(i, nn.ravel())], axis=1)
    return np.unique(e, axis=0)


class _UF:
    def __init__(self, n: int):
        self.p = np.arange(n)

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a


def merge_tree(U: np.ndarray, wells: np.ndarray, peaks: np.ndarray, beta: float,
               k: int = KNN) -> Dict:
    """
    The elder-rule merge tree of ``φ_β``'s wells on a k-NN graph over the rows ``U``, each basin
    contracted to one node (a basin is connected in the continuum; on the graph it may not be):
    two wells are joined at the highest token edge between their basins, and joins are taken from
    the highest down. Where the k-NN graph leaves wells apart, each apart component's rows are
    also joined to their ``k`` nearest rows outside it, until every well is joined (``bridges``
    counts the rounds). Returns ``deaths`` (one per well but the highest: the well, the well whose
    component it joins, the saddle edge's ends with ``i`` in the dying basin, its height, the
    persistence) and ``bridges``.
    """
    wells = np.asarray(wells, dtype=int)
    kw = int(wells.max()) + 1 if wells.size else 0
    h_tok = heights(U, U, beta)
    E = knn_edges(U, k)
    best: Dict = {}
    bridges = 0
    while True:
        E = E[wells[E[:, 0]] != wells[E[:, 1]]]
        h_mid = heights(unit(U[E[:, 0]] + U[E[:, 1]]), U, beta) if len(E) else np.zeros(0)
        eh = np.minimum(np.minimum(h_tok[E[:, 0]], h_mid), h_tok[E[:, 1]]) if len(E) else h_mid
        for e in range(len(E)):
            a, b = int(E[e, 0]), int(E[e, 1])
            key = (min(wells[a], wells[b]), max(wells[a], wells[b]))
            if key not in best or eh[e] > best[key][0]:
                best[key] = (float(eh[e]), a, b)
        uf = _UF(kw)
        for wa, wb in best:
            uf.p[uf.find(wa)] = uf.find(wb)
        comp = np.asarray([uf.find(w) for w in range(kw)])
        if kw <= 1 or len(set(comp.tolist())) == 1:
            break
        if bridges >= MAX_BRIDGE_ROUNDS:
            raise ValueError(f"wells still apart after {bridges} bridging rounds")
        bridges += 1
        tok = comp[wells]
        new = []
        for c in sorted(set(tok.tolist())):
            inside, outside = np.flatnonzero(tok == c), np.flatnonzero(tok != c)
            S = U[inside] @ U[outside].T
            kk = min(k, outside.size)
            nn = outside[np.argpartition(-S, kk - 1, axis=1)[:, :kk]]
            ii = np.repeat(inside, kk)
            new.append(np.stack([np.minimum(ii, nn.ravel()), np.maximum(ii, nn.ravel())], axis=1))
        E = np.unique(np.concatenate(new), axis=0)
    uf, top = _UF(kw), {w: w for w in range(kw)}
    deaths = []
    for (wa, wb), (h, a, b) in sorted(best.items(), key=lambda kv: (-kv[1][0], kv[0])):
        ra, rb = uf.find(wa), uf.find(wb)
        if ra == rb:
            continue
        ta, tb = top.pop(ra), top.pop(rb)
        uf.p[ra] = rb
        die, keep = (ta, tb) if (peaks[ta], -ta) < (peaks[tb], -tb) else (tb, ta)
        top[rb] = keep
        # i on the dying well's side: the edge's end in the component the dying well tops
        i, j = (a, b) if (die == ta) == (wells[a] == wa) else (b, a)
        deaths.append({"well": int(die), "into": int(keep), "i": int(i), "j": int(j), "h": h,
                       "p": float(peaks[die] - h)})
    return {"deaths": deaths, "bridges": bridges}


# ---------------------------------------------------------------- NEB on the sphere

def _slerp_path(points: np.ndarray, m: int) -> np.ndarray:
    """``m`` points equally spaced in arc length along the geodesic polyline through ``points``."""
    P = unit(np.asarray(points, dtype=np.float64))
    ang = np.arccos(np.clip(np.sum(P[:-1] * P[1:], axis=1), -1, 1))
    cum = np.concatenate([[0.0], np.cumsum(ang)])
    out = []
    for t in np.linspace(0, cum[-1], m):
        s = min(int(np.searchsorted(cum, t, side="right")) - 1, len(ang) - 1)
        w = ang[s]
        if w < 1e-12:
            out.append(P[s])
            continue
        f = (t - cum[s]) / w
        out.append((np.sin((1 - f) * w) * P[s] + np.sin(f * w) * P[s + 1]) / np.sin(w))
    return unit(np.asarray(out))


def neb(a: np.ndarray, b: np.ndarray, via: Sequence[np.ndarray], U: np.ndarray, beta: float,
        images: int = NEB_IMAGES, iters: int = NEB_ITERS, htol: float = NEB_HTOL,
        device: str = "cpu") -> Dict:
    """
    Climbing-image NEB for the pass of ``h = log φ_β`` between modes ``a`` and ``b`` (ascent of
    ``h`` off the path; the climbing image descends ``h`` along it), started on the geodesic
    polyline ``a → via → b``. float32 iterations, the path's heights in float64. Converged when
    the climbing image's height moves < ``htol`` nats over ``NEB_EVERY`` iterations (amended after
    the first cell: the largest force on the band oscillates over token spikes while the pass
    height is fixed to 1e-4 by 500 iterations); the last largest true force is reported beside.
    """
    import torch
    dev = torch.device(device)
    Y = torch.as_tensor(_slerp_path([a, *via, b], images + 2), device=dev, dtype=torch.float32)
    Ut = torch.as_tensor(U, device=dev, dtype=torch.float32)
    eta, ks = 0.05 / beta, beta
    conv, t, last, fmax = False, 0, None, float("nan")
    for t in range(1, iters + 1):
        S = beta * (Y @ Ut.T)
        W = torch.softmax(S, dim=1)
        m = W @ Ut
        g = beta * (m - (m * Y).sum(1, keepdim=True) * Y)
        h = torch.logsumexp(S, dim=1)
        tau = Y[2:] - Y[:-2]
        y = Y[1:-1]
        tau = tau - (tau * y).sum(1, keepdim=True) * y
        tau = tau / tau.norm(dim=1, keepdim=True).clamp_min(1e-12)
        gi = g[1:-1]
        gt = (gi * tau).sum(1, keepdim=True)
        F = gi - gt * tau
        d_f = (Y[2:] - Y[1:-1]).norm(dim=1, keepdim=True)
        d_b = (Y[1:-1] - Y[:-2]).norm(dim=1, keepdim=True)
        F = F + ks * (d_f - d_b) * tau
        if t > NEB_WARM:
            ci = int(torch.argmin(h[1:-1]))
            F[ci] = gi[ci] - 2 * gt[ci] * tau[ci]
            if t % NEB_EVERY == 0:
                Fp = gi - gt * tau
                Fp[ci] = F[ci]
                fmax = float(Fp.norm(dim=1).max())
                hc = float(h[1:-1].min())
                if last is not None and abs(hc - last) < htol:
                    conv = True
                    break
                last = hc
        Y = torch.cat([Y[:1], torch.nn.functional.normalize(y + eta * F, dim=1), Y[-1:]])
    path = Y.double().cpu().numpy()
    hp = heights(path, U, beta)
    return {"h": float(hp[1:-1].min()), "converged": conv, "iters": t, "fmax": fmax}


# ---------------------------------------------------------------- one cell

def wells_and_modes(X: np.ndarray, beta: float, device: str) -> Dict:
    """U1's mean shift with its modes, the well labels made contiguous (U1's can skip an id when
    merged rows leave a mode with no row; a skipped id would be a well no edge can join)."""
    ms = mean_shift(X, beta, device, modes=True)
    ids, w = np.unique(ms["wells"], return_inverse=True)
    return {**ms, "wells": w, "modes": ms["modes"][ids]}


def read_cell(U: np.ndarray, beta: float, device: str, ref_wells: Optional[np.ndarray],
              cls: np.ndarray, rng_key, neb_top: int = NEB_TOP, draws: bool = True) -> tuple:
    ms = wells_and_modes(U, beta, device)
    w, modes = ms["wells"], ms["modes"]
    agree = None if ref_wells is None else agreement(w, ref_wells)
    peaks = heights(modes, U, beta)
    tree = merge_tree(U, w, peaks, beta)
    ds = tree["deaths"]
    ps = np.asarray([d["p"] for d in ds])
    rec = {"beta": beta, "n": int(len(U)), **well_stats(w), "unconverged": ms["unconverged"],
           "agree_b": agree, "bridges": tree["bridges"], "deaths": len(ds),
           "P": float(ps.sum()) if ps.size else 0.0, "p_max": float(ps.max()) if ps.size else 0.0,
           "deep": int((ps > DEEP).sum()), "p_min": float(ps.min()) if ps.size else 0.0}
    nebs = []
    dev = "cuda" if device == "cuda" else "cpu"
    for d in sorted(ds, key=lambda d: -d["p"])[:neb_top]:
        r = neb(modes[d["well"]], modes[d["into"]], [U[d["i"]], U[d["j"]]], U, beta, device=dev)
        nebs.append({"well": d["well"], "graph_h": d["h"], "neb_h": r["h"], "gap": r["h"] - d["h"],
                     "converged": r["converged"], "iters": r["iters"], "fmax": r["fmax"]})
    if nebs:
        gaps = np.asarray([x["gap"] for x in nebs])
        rec.update(neb_n=len(nebs), neb_conv=int(sum(x["converged"] for x in nebs)),
                   neb_gap_median=float(np.median(gaps)), neb_gap_min=float(gaps.min()),
                   neb_gap_max=float(gaps.max()))
    if ds:
        ends = Counter(cls[[x for d in ds for x in (d["i"], d["j"])]].tolist())
        tot = sum(ends.values())
        allc = Counter(cls.tolist())
        rec["crest_mix"] = {c: [ends.get(c, 0) / tot, allc[c] / len(cls)] for c in sorted(allc)}
    if draws:
        rng = np.random.default_rng(rng_key)
        gauss = [gaussian_draw(U, rng) for _ in range(N_DRAW)]   # U1's draws, in U1's order
        PG, kG, bG = [], [], []
        for Y in gauss:
            mg = wells_and_modes(Y, beta, device)
            tg = merge_tree(Y, mg["wells"], heights(mg["modes"], Y, beta), beta)
            PG.append(float(sum(d["p"] for d in tg["deaths"])))
            bG.append(tg["bridges"])
            kG.append(well_stats(mg["wells"])["k"])
        rec.update(P_G=PG, k_G=kG, bridges_G=bG, Xp=float(np.log1p(rec["P"]) - np.mean(np.log1p(PG))))
    saved = {"wells": w.astype(np.int32),
             "deaths": np.asarray([[d["well"], d["into"], d["i"], d["j"]] for d in ds],
                                  dtype=np.int32).reshape(-1, 4),
             "death_hp": np.asarray([[d["h"], d["p"]] for d in ds], dtype=np.float64).reshape(-1, 2)}
    return rec, saved, nebs


def read_run(run_dir: Path, ln1: Dict, targets: Dict[str, np.ndarray], key: str, step: str,
             ref: Optional[Dict[str, np.ndarray]], device: str,
             layers: Sequence[int] = U3_LAYERS) -> tuple:
    z = np.load(run_dir / "activations.npz")
    acts, norms = z["activations"], z["norms"]
    tokens = json.loads((run_dir / "geometry.json").read_text())["tokens"]
    cls_all = token_classes(tokens)
    out, saved, nebs = [], {}, []
    for L in layers:
        for name, t in targets.items():
            U = unit_rows(acts[L] * norms[L][:, None], ln1["w"][L], ln1["b"][L], ln1["eps"])[t]
            rw = None if ref is None else ref[f"{name}/wells_{BETA:g}"][L]
            # U1's seed for this cell, so the Gaussians are the ones (b) read
            rng_key = [SEED, zlib.crc32(f"{key}|{step}|{L}|{name}".encode())]
            rec, sv, nb = read_cell(U, BETA, device, rw, cls_all[t], rng_key)
            if rw is not None and rec["agree_b"] < AGREE_MIN:
                raise SystemExit(f"refusing: {key} step {step} L{L}: wells agree with (b)'s at "
                                 f"{rec['agree_b']:.4f} < {AGREE_MIN}")
            if rec["p_min"] < -P_TOL:
                raise SystemExit(f"refusing: {key} step {step} L{L}: persistence {rec['p_min']:.2e} < 0")
            out.append({"layer": L, "targets": name, **rec})
            nebs += [{"layer": L, "targets": name, **x} for x in nb]
            for k, v in sv.items():
                saved[f"{name}/L{L}/{k}"] = v
    return out, saved, nebs


# ---------------------------------------------------------------- run

def _primary_only(job):
    kind, step, key, rd, rev, tg, _ = job
    return kind, step, key, rd, rev, {k: v for k, v in tg.items() if k == "t12"}


def _job(job, wells_dir: Path, out: Path, code: str, device: str) -> str:
    kind, step, key, rd, rev, tg = _primary_only(job)
    path = out / "records" / kind / f"step{step}_{key}.json"
    if path.exists():
        had = json.loads(path.read_text()).get("code")
        if had != code:
            raise SystemExit(f"refusing to resume: {path} was written by {had}, this is {code}")
        return f"have {path.name}"
    src = wells_dir / "records" / kind / f"step{step}_{key}.npz"
    meta = json.loads(src.with_suffix(".json").read_text())
    if meta.get("opts", {}).get("primary_beta") != BETA or BETA not in meta["opts"]["betas"]:
        raise SystemExit(f"refusing: {src} is not a β {BETA:g} U1 record ({meta.get('opts')})")
    zb = np.load(src)
    for k, v in tg.items():
        if not np.array_equal(zb[f"{k}/positions"], v):
            raise SystemExit(f"refusing: {src}'s {k} positions are not this run's targets")
    t0 = time.monotonic()
    recs, saved, nebs = read_run(rd, load_ln1(wells_dir / "ln1" / f"{rev}.npz"), tg, key, step,
                                 {k: zb[k] for k in zb.files}, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), **saved)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"kind": kind, "step": step, "passage": key, "run": str(rd),
                               "wells": str(src), "code": code, "device": device,
                               "seconds": round(time.monotonic() - t0, 1), "cells": recs,
                               "neb": nebs}) + "\n")
    tmp.rename(path)
    return f"done {path.name} {time.monotonic() - t0:.0f}s"


def check_first(path: Path) -> Dict:
    rec = json.loads(path.read_text())
    cells = [c for c in rec["cells"] if c["targets"] == "t12"]
    bad = [c["layer"] for c in cells if c["deaths"] != c["k"] - 1 or c["agree_b"] < AGREE_MIN]
    nb = [x for x in rec["neb"] if x["layer"] in CHECK_LAYERS and x["targets"] == "t12"]
    conv = (sum(x["converged"] for x in nb) / len(nb)) if nb else 1.0
    print(f"first cell {path.name}: {len(cells)} layers, wells agree with (b) "
          f"(min {min(c['agree_b'] for c in cells):.4f}), deaths = k − 1 except {bad}; NEB at "
          f"L{'/'.join(map(str, CHECK_LAYERS))}: {sum(x['converged'] for x in nb)} of {len(nb)} "
          f"converged", flush=True)
    if bad or {c["layer"] for c in cells} != set(U3_LAYERS) or conv < NEB_CONV_MIN:
        raise SystemExit(f"refusing: first cell not populated (layers {bad}, NEB conv {conv:.2f})")
    return {"neb_converged": conv, "neb_n": len(nb)}


def run(a) -> int:
    out, code = a.out, code_sha()
    jobs, _, meta = plan(a.runs, None, None)       # long passages only (amended, cost)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps({**meta, "code": code, "device": a.device,
                                               "beta": BETA, "wells": str(a.wells)}, indent=1) + "\n")
    first = next(j for j in jobs if j[0] == "long" and (j[1], j[2]) == FIRST)
    print(_job(first, a.wells, out, code, a.device), flush=True)
    chk = check_first(out / "records" / "long" / f"step{FIRST[0]}_{FIRST[1]}.json")
    (out / "first_check.json").write_text(json.dumps(chk, indent=1) + "\n")
    if a.first_only:
        return 0
    for j in jobs:
        if j is not first and (not a.steps or j[1] in a.steps):
            print(_job(j, a.wells, out, code, a.device), flush=True)
    return 0


# ---------------------------------------------------------------- report

def report(out: Path) -> int:
    from .extract_long8 import STEPS
    from .u1_report import TRAINED, sign_label
    from .u2_report import chance
    full = {}
    for kind in ("long", "v1"):
        files = sorted((out / "records" / kind).glob("*.json"))
        if not files:
            continue
        acc, beside, nebs = {}, {}, []
        for f in files:
            r = json.loads(f.read_text())
            nebs += r["neb"]
            for c in r["cells"]:
                for band, layers in BANDS.items():
                    if c["layer"] in layers:
                        d = acc.setdefault((int(r["step"]), band), {}).setdefault(r["passage"], {})
                        for s in ("Xp", "P", "p_max", "deep", "k", "k2"):
                            d.setdefault(s, []).append(c[s])
        n_pass = len({json.loads(f.read_text())["passage"] for f in files})
        tab = {}
        for (step, band), by in sorted(acc.items()):
            xs = [float(np.mean(v["Xp"])) for v in by.values()]
            if len(xs) != n_pass:
                raise SystemExit(f"refusing: {kind} {step} {band} has {len(xs)} passages, expected {n_pass}")
            tab[f"{step}|{band}"] = {"label": sign_label(xs, ("deeper", "shallower")),
                                     "median": float(np.median(xs)), "n": len(xs)}
            beside[f"{step}|{band}"] = {s: float(np.median([np.mean(v[s]) for v in by.values()]))
                                        for s in ("P", "p_max", "deep", "k", "k2")}
        gaps = np.asarray([x["gap"] for x in nebs]) if nebs else np.zeros(0)
        full[kind] = {"n_passages": n_pass, "chance_per_54": chance(n_pass, 54), "Xp": tab,
                      "beside": beside,
                      "neb": {"n": len(nebs), "converged": int(sum(x["converged"] for x in nebs)),
                              "gap_quantiles": [float(q) for q in np.quantile(gaps, [0, .1, .5, .9, 1])]
                              if gaps.size else None,
                              "below_graph": int((gaps < -1e-6).sum())}}
        print(f"\n===== {kind} ({n_pass} passages): (5) Xp, β {BETA:g} =====")
        print(f"{'step':>7} " + " ".join(f"{b:>18}" for b in TRAINED + ('L0',)))
        count = Counter()
        for s in STEPS:
            row = []
            for b in TRAINED + ("L0",):
                c = tab.get(f"{s}|{b}")
                if c is None:
                    row.append(" " * 18)
                    continue
                if b in TRAINED:
                    count[c["label"]] += 1
                x = beside[f"{s}|{b}"]
                row.append(f"{c['label'][:9]:>9} {c['median']:+.2f} {x['k']:4.0f}"[:18].rjust(18))
            print(f"{s:>7} " + " ".join(row))
        print("  trained cells:", dict(count), "| chance per 54:", full[kind]["chance_per_54"])
        print("  NEB:", json.dumps(full[kind]["neb"]))
    (out / "labels.json").write_text(json.dumps(full, indent=1) + "\n")
    print(f"\nwrote {out / 'labels.json'}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--wells", type=Path, required=True, help="U1 (b)'s output dir (β 10)")
    r.add_argument("--runs", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--steps", nargs="*", default=None)
    r.add_argument("--first-only", action="store_true")
    r.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p = sub.add_parser("report")
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "report":
        return report(a.out)
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
