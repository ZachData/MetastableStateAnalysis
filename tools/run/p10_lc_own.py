"""LCo: the members' own-token surprisal, adjusted (`p10_cluster_function/design-10.md` "LCo", fixed
before any statistic on a real record).

Are c3x's members' own tokens less surprising than the other kept tokens at the same positions,
beyond their own Pile frequency and class? No pass: LC's records give ``s_own[i] = s[i−1] =
−log p(x_i | x_<i)``. Per (step, passage) ``ln s_own`` is residualised, membership-blind, on a cubic
in log(i + 1), the token's own lf, lf², word_start and continuation (``r``); per (layer, column)
``O`` is F13's binned difference of r, members − rest, and ``O_A`` LC's rank gap on r (labels: sign
from ``O_A``, size from ``O``). Beside: ``O0`` (position alone), ``O~`` (least squares), ``O_s`` (the
s scale), ``D_own`` as LC stored it. ``calibrate`` runs the reading on planted s_own over the real
design points and tokens. Tier 1: exploratory, unregistered.
    python tools/run/p10_lc_own.py calibrate --labels <R9 source> --out <dir> [--only N0]
    python tools/run/p10_lc_own.py stats --labels <R9 source> --records <LC records> --out <dir>
    python tools/run/p10_lc_own.py report --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Optional, Sequence

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[2])))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))

import numpy as np

from tools.run import p10_f13_centres as f
from tools.run import p10_lc_loss as lc
from tools.run.p10_f13_rare import ols_centre

STEPS, PRIMARY_STEPS, PASSAGES, LAYERS, BANDS = lc.STEPS, lc.PRIMARY_STEPS, lc.PASSAGES, lc.LAYERS, lc.BANDS
COLUMNS = lc.COLUMNS
LC_CODE = "e6ab408ec7a5"                                   # LC's records (plan.json)
# (label stat, its sign stat or None for its own sign)
ROWS = {"O": "O_A", "O0": "O0_A", "O~": None, "O_s": "O_s_A", "D_own": None}
# calibration (the rule's)
N_DRAWS = 200
DELTAS = (0.1, 0.2, 0.3, 0.5)
K_CLS = {"continuation": 0.3, "word_start": 1.0, "rest": 0.7}
LF_SCALE = 4.5
FULL_MIN = 7                                                # amended at the gate: a full label's null rate ≤ 2 %


class LCoError(RuntimeError):
    pass


# ---------------------------------------------------------------- pure pieces
def own_features(tokens: Sequence[str], counts: np.ndarray, vocab: dict, added: set) -> Dict[str, np.ndarray]:
    """Per position i, the token's *own* lf and word_start / continuation indicators (not shifted)."""
    from tools.run import p10_token_composition as tc
    tf = tc.token_features(np.asarray(tokens), vocab, added)
    ids = np.array([t["id"] for t in tf])
    cls = np.array([t["cls"] for t in tf])
    return {"lf": np.log10((counts[ids] + 0.5) / counts.sum()),
            "ws": (cls == "word_start").astype(float), "ct": (cls == "continuation").astype(float)}


def covariates(dom: np.ndarray, feat: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
    """Design over the domain: 1, a cubic in log(i + 1), and (feat given) lf, lf², word_start, continuation."""
    lp = np.log(dom + 1.0)
    cols = [np.ones_like(lp), lp, lp ** 2, lp ** 3]
    if feat is not None:
        lf = feat["lf"][dom]
        cols += [lf, lf ** 2, feat["ws"][dom], feat["ct"][dom]]
    return np.column_stack(cols)


def resid(v: np.ndarray, dom: np.ndarray, X: np.ndarray) -> np.ndarray:
    """v less its membership-blind least-squares fit on X over the domain (NaN off it)."""
    y = v[dom]
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    out = np.full(v.shape, np.nan)
    out[dom[ok]] = y[ok] - X[ok] @ np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
    return out


def record_resids(so: np.ndarray, dom: np.ndarray, feat: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """The layer-free residuals: r (log, full), r0 (log, position), rs (s scale, full)."""
    y = np.log(so)
    Xf, X0 = covariates(dom, feat), covariates(dom, None)
    return {"r": resid(y, dom, Xf), "r0": resid(y, dom, X0), "rs": resid(so, dom, Xf)}


def layer_stats(so: np.ndarray, res: Dict[str, np.ndarray], dom: np.ndarray, lab_dom: np.ndarray,
                feat: Dict[str, np.ndarray]) -> Dict[str, Optional[float]]:
    """``lab_dom``: the column's labels at ``dom`` (≥ 0 member)."""
    mem = lab_dom >= 0
    A, B = dom[mem], dom[~mem]
    bins = f.bins_of(dom, lc.N_BINS)
    bd = lambda v: f.binned_diff(v, A, B, bins, lc.MIN_A, lc.MIN_B)      # noqa: E731
    scale = float(np.nanmean(so[dom]))
    ds = bd(res["rs"])
    lf = feat["lf"]
    ols = (ols_centre(np.log(so[dom]), mem, [lf[dom], lf[dom] ** 2, feat["ws"][dom], feat["ct"][dom]],
                      dom.astype(float)) if mem.sum() >= lc.MIN_A and (~mem).sum() >= lc.MIN_B else None)
    return {"O": bd(res["r"]), "O_A": lc.rank_gap_resid(res["r"], dom, A, B),
            "O0": bd(res["r0"]), "O0_A": lc.rank_gap_resid(res["r0"], dom, A, B),
            "O~": ols,
            "O_s": None if ds is None else ds / scale, "O_s_A": lc.rank_gap_resid(res["rs"], dom, A, B),
            "D_own": lc.rel_binned_resid(so, dom, A, B),
            "n_members": int(mem.sum()), "n_rest": int((~mem).sum())}


def floor_label(vals: Sequence[Optional[float]], sizes: Optional[Sequence[Optional[float]]] = None) -> str:
    """LC's ``floor_label``, with a full label demoted to a lean below FULL_MIN readable passages
    (amended at the gate: the sign rule's null rate 2·2⁻ⁿ is 3.1 % at 6)."""
    lab = lc.floor_label(vals, sizes)
    n = sum(x is not None and np.isfinite(x) for x in vals)
    for side in lc.NAMES:
        if n < FULL_MIN and lab.startswith(side):
            return "leans " + lab
    return lab


def labels(grid: Dict, sizes: Optional[Dict] = None) -> Dict[str, Dict[int, str]]:
    """LC's ``labels`` (isolated leans, at-init marks) on this module's ``floor_label``."""
    out = {}
    for band, row in grid.items():
        labs = {s: floor_label(v, None if sizes is None else sizes[band][s]) for s, v in row.items()}
        labs = f.mark_isolated(labs)
        if 0 in labs:
            base = labs[0]
            for s in labs:
                if s != 0 and labs[s] == base and base not in ("mixed", "too few"):
                    labs[s] += " (at init)"
        out[band] = labs
    return out


def row_labels(vals: Dict, steps: Sequence[int], stat: str, column: str) -> Dict:
    """Labels for one row: sign from its rank stat (if any), size from the row itself."""
    g = lc.label_grid(vals, steps, stat, column)
    sign = ROWS[stat]
    return labels(lc.label_grid(vals, steps, sign, column), g) if sign else labels(g)


def readings(labs: Dict[str, Dict]) -> Dict:
    """The rule's readings (1)–(4) on c3x at the primary steps; ``labs`` keyed by row."""
    O, O0 = labs["O"], labs["O0"]
    out: Dict = {"trained": {b: O[b][143000] for b in BANDS}, "onset": {b: lc.onset(O[b]) for b in BANDS},
                 "beyond": [], "carried_by_token": []}
    for b in BANDS:
        for s in PRIMARY_STEPS:
            if core_floor(O0[b][s]) in ("lower", "leans lower"):
                key = "beyond" if core_floor(O[b][s]) in ("lower", "leans lower") else "carried_by_token"
                out[key].append([b, s, O0[b][s], O[b][s]])
    any_onset = any(o[0] is not None for o in out["onset"].values())
    o0_full = any(lc.core(O0[b][s]) == "lower" for b in BANDS for s in PRIMARY_STEPS)
    out["headline"] = ("beyond frequency and class" if any_onset else
                       "carried by the token" if o0_full else "no own-token gap")
    return out


def core_floor(lab: str) -> str:
    """``lc.core`` with labels under the floor (or whose size disagrees) read as not labelled."""
    lab = lc.core(lab)
    return "mixed" if "(" in lab else lab


# ---------------------------------------------------------------- the design points
def design(labels_src: Path, counts_dir: Path) -> Dict:
    """LC's design points with the token's own features in place of the next token's."""
    from tools.run import p10_token_composition as tc
    dps = lc.design(labels_src, None)
    counts = np.load(Path(counts_dir) / "counts.npy")
    vocab, added = tc.load_vocab(tc.find_tokenizer(Path(os.environ.get("HF_HOME", str(DATA / "hf")))))
    for dp in dps.values():
        dp["feat"] = own_features(dp["tokens"], counts, vocab, added)
        if not np.isfinite(dp["feat"]["lf"][dp["dom"]]).all():
            raise LCoError(f"{dp['run']}: lf not finite on the domain")
    return dps


# ---------------------------------------------------------------- calibration (planted, no pass)
def planted_scale(dp: Dict, arm: str) -> np.ndarray:
    n = dp["n"]
    i = np.arange(n)
    base = 3 + 5 * (1 + i) ** -0.4
    if arm == "N0":
        return base
    ft = dp["feat"]
    k = np.where(ft["ct"] == 1, K_CLS["continuation"], np.where(ft["ws"] == 1, K_CLS["word_start"], K_CLS["rest"]))
    return base * (-ft["lf"] / LF_SCALE) * k


def _cal_job(args) -> Dict:
    """One planted (step, passage): s_own = scale·Exp(1); E / O scale each layer's members by (1 − δ)."""
    key, dp, arm, delta, start, draw = args
    step, passage = key
    rng = np.random.default_rng([zlib.crc32(f"LCo|{arm}|{delta}|{start}|{step}|{passage}".encode()), draw])
    base = planted_scale(dp, "N0" if arm == "N0" else "Nf") * rng.exponential(1.0, dp["n"])
    dom, ft = dp["dom"], dp["feat"]
    res0 = record_resids(base, dom, ft)
    out = {"c3x": {}}
    for L, lab in dp["cols"]["c3x"].items():
        so, res = base, res0
        if arm in ("E", "O") and step >= start:
            so = base.copy()
            mem = dom[lab >= 0]
            so[mem] = so[mem] * (1 - delta)
            res = record_resids(so, dom, ft)
        st = layer_stats(so, res, dom, lab, ft)
        out["c3x"][str(L)] = {k: st[k] for k in ("O", "O_A", "O0", "O0_A")}
    return out


def calibrate(a) -> int:
    dps = design(a.labels, a.counts)
    arms = ([("N0", 0.0, 0), ("Nf", 0.0, 0)] + [("E", d, 0) for d in DELTAS]
            + [("O", 0.3, 2000), ("O", 0.3, 256)])
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / "calibration.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    keys = [k for k in dps if k[0] in PRIMARY_STEPS]
    slim = {k: {"n": dps[k]["n"], "dom": dps[k]["dom"], "cols": {"c3x": dps[k]["cols"]["c3x"]},
                "feat": dps[k]["feat"]} for k in keys}
    with Pool(a.workers) as pool:
        for arm, delta, start in arms:
            tag = arm + (f"_d{delta}" if arm in ("E", "O") else "") + (f"_from{start}" if arm == "O" else "")
            if a.only and tag not in a.only:
                continue
            t0 = time.monotonic()
            cells, cells0, onsets = [], [], []
            for draw in range(a.draws):
                jobs = [(k, slim[k], arm, delta, start, draw) for k in keys]
                vals = dict(zip(keys, pool.map(_cal_job, jobs, chunksize=4)))
                labs, labs0 = (row_labels(vals, PRIMARY_STEPS, st, "c3x") for st in ("O", "O0"))
                for b in BANDS:
                    for s in PRIMARY_STEPS:
                        cells.append(lc.core(labs[b][s]))
                        cells0.append(lc.core(labs0[b][s]))
                    onsets.append(lc.onset(labs[b]))
            ca, c0 = np.array(cells), np.array(cells0)
            res[tag] = {
                "cells": len(cells),
                "full_lower": float((ca == "lower").mean()), "full_higher": float((ca == "higher").mean()),
                "lower_or_lean": float(np.isin(ca, ["lower", "leans lower"]).mean()),
                "O0_full_lower": float((c0 == "lower").mean()), "O0_full_higher": float((c0 == "higher").mean()),
                "O0_lower_or_lean": float(np.isin(c0, ["lower", "leans lower"]).mean()),
                "onset_categories": {c: sum(o[1] == c for o in onsets) / len(onsets)
                                     for c in ("precedes", "co-locates", "follows", "no gap")},
                "onset_steps": {str(s): sum(o[0] == s for o in onsets) / len(onsets) for s in PRIMARY_STEPS},
                "seconds": round(time.monotonic() - t0, 1)}
            print(tag, json.dumps(res[tag]), flush=True)
            path.write_text(json.dumps(res, indent=1) + "\n")
    verdict = {}
    for tag in ("N0", "Nf"):
        if tag in res:
            r = res[tag]
            verdict[tag] = (r["full_lower"] + r["full_higher"] <= lc.NULL_CELL_BAR
                            and r["onset_categories"]["no gap"] >= lc.NULL_ONSET_BAR)
            print(f"{tag} bar: {'passes' if verdict[tag] else 'FAILS'}", flush=True)
    return 0 if all(verdict.values()) else 2


# ---------------------------------------------------------------- stats and report (CPU)
def code_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def stats(a) -> int:
    dps = design(a.labels, a.counts)
    out = {}
    for (s, p), dp in dps.items():
        z = np.load(a.records / f"step{s}_{p}.npz")
        if len(z["s"]) != dp["n"] - 1 or str(z["code"]) != LC_CODE:
            raise LCoError(f"{s} {p}: {len(z['s'])} surprisals for n {dp['n']}, code {z['code']} (LC {LC_CODE})")
        so = lc.own(z["s"])
        if not (np.isfinite(so[dp["dom"]]).all() and (so[dp["dom"]] > 0).all()):
            raise LCoError(f"{s} {p}: s_own not finite and positive on the domain")
        res = record_resids(so, dp["dom"], dp["feat"])
        out[f"{s}|{p}"] = {c: {str(L): layer_stats(so, res, dp["dom"], lab, dp["feat"])
                               for L, lab in dp["cols"][c].items()} for c in COLUMNS}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "plan.json").write_text(json.dumps({"rule": 'design-10.md "LCo"', "code": code_sha(),
                                                 "lc_records": str(a.records), "lc_code": LC_CODE,
                                                 "labels": str(a.labels)}, indent=1) + "\n")
    (a.out / "stats.json").write_text(json.dumps(out) + "\n")
    print(f"stats: {len(out)} records", flush=True)
    return 0


def report(a) -> int:
    raw = json.loads((a.out / "stats.json").read_text())
    vals = {(int(k.split("|")[0]), k.split("|")[1]): v for k, v in raw.items()}
    rep: Dict = {"labels": {}, "medians": {}, "boot": {}}
    for c in COLUMNS:
        for st in ROWS:
            g = lc.label_grid(vals, STEPS, st, c)
            rep["labels"][f"{c}|{st}"] = row_labels(vals, STEPS, st, c)
            rep["medians"][f"{c}|{st}"] = {b: {str(s): (float(np.median([x for x in v if x is not None]))
                                                          if any(x is not None for x in v) else None)
                                               for s, v in row.items()} for b, row in g.items()}
            if st == "O":
                rep["boot"][c] = {b: {str(s): lc.boot_share(v, zlib.crc32(f"LCo|{c}|{b}|{s}".encode()))
                                      for s, v in row.items()} for b, row in g.items()}
    rep["readings"] = readings({st: rep["labels"][f"c3x|{st}"] for st in ROWS})
    (a.out / "report.json").write_text(json.dumps(rep, indent=1) + "\n")
    lines = ["# LCo report", "", "label (passage median); O, O0, O_s signs from their rank gaps", ""]
    lines += ["| row | band | " + " | ".join(str(s) for s in STEPS) + " |", "|" + "---|" * (len(STEPS) + 2)]
    for c in COLUMNS:
        for st in ROWS:
            for b in BANDS:
                labs, meds = rep["labels"][f"{c}|{st}"][b], rep["medians"][f"{c}|{st}"][b]
                cells = [f"{labs[s]} ({meds[str(s)]:+.3f})" if meds[str(s)] is not None else labs[s] for s in STEPS]
                lines.append(f"| {c} {st} | {b} | " + " | ".join(cells) + " |")
    lines += ["", "readings: " + json.dumps(rep["readings"])]
    (a.out / "report.txt").write_text("\n".join(lines) + "\n")
    print(lines[-1])
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("calibrate", "stats", "report"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        if name != "report":
            p.add_argument("--labels", type=Path, required=True)
            p.add_argument("--counts", type=Path, default=DATA / "p10" / "pile_unigram_2026-10-10")
        if name == "stats":
            p.add_argument("--records", type=Path, default=DATA / "p10" / "lc_loss_2026-10-11" / "records")
        if name == "calibrate":
            p.add_argument("--draws", type=int, default=N_DRAWS)
            p.add_argument("--workers", type=int, default=12)
            p.add_argument("--only", nargs="*", help="arm tags to run, merged into an existing calibration.json")
    a = ap.parse_args(argv)
    return {"calibrate": calibrate, "stats": stats, "report": report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
