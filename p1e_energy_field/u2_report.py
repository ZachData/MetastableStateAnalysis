"""
p1e_energy_field/u2_report.py — U2's block arm: the label table from `u2_block`'s records,
by the rule in `design-1e.md` "U2's block arm: the rule" (row *label per (step, band)*).

Per (target set, source, β, step, band): each passage's ``X`` averaged over the band's blocks;
**ascends** if every passage is > 0, **leans ascends** if all but one, the same below 0 for
**descends**, else **mixed** (8 long passages: p = 1/256 and 9/256 one-sided; v1's 7: 7 of 7 and
6 of 7). Beside: chance counts, isolated leans, β-robustness, step 0, cells with rank ≤ 0.05,
the label without the lowest-NLL passage at 143000, ``Xt`` (mean move out) by the same rule,
and the median Spearman of ``a_i`` with log offset. Tier 1: exploratory, unregistered.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .extract_long8 import STEPS
from .u2_block import BANDS, BETAS, SOURCES

PRIMARY = {"long": ("t12", "causal", 3.5), "v1": ("r0", "causal", 3.5)}
TRAINED_BANDS = ("L1-8", "L9-16", "L17-22")
SHARED_ROWS = (("causal:r1out", 3.5), ("causal:residout", 3.5), ("causal:resid", 3.5),
               ("causal:ambout", 3.5), ("mean0:frozen", 0.0), ("mean0:r1out", 0.0),
               ("mean0:residout", 0.0))
SHORT = {"ascends": "ASC", "leans ascends": "asc", "descends": "DES", "leans descends": "des",
         "mixed": "·"}


def label(xs: Sequence[float]) -> str:
    """The sign rule over passages: all, all but one, else mixed."""
    xs = np.asarray(xs, dtype=float)
    n, pos, neg = xs.size, int(np.sum(xs > 0)), int(np.sum(xs < 0))
    if pos == n:
        return "ascends"
    if neg == n:
        return "descends"
    if pos == n - 1:
        return "leans ascends"
    if neg == n - 1:
        return "leans descends"
    return "mixed"


def direction(lab: str) -> int:
    return 1 if "ascends" in lab else (-1 if "descends" in lab else 0)


def chance(n_passages: int, n_cells: int) -> Dict[str, float]:
    """Expected labels by chance (each passage's sign a fair coin, passages independent)."""
    full, lean = 2 / 2 ** n_passages, 2 * n_passages / 2 ** n_passages
    return {"ascends_or_descends": n_cells * full, "leans": n_cells * lean}


def load(out: Path, kind: str, sub: str = "records") -> List[Dict]:
    recs = []
    for p in sorted((out / sub / kind).glob("*.json")):
        r = json.loads(p.read_text())
        for c in r["cells"]:
            recs.append({"step": int(r["step"]), "passage": r["passage"], **c})
    return recs


def band_values(recs: List[Dict], stat: str = "X") -> Dict:
    """``{(targets, source, β, step, band): {passage: band mean of stat}}`` (+ rank counts)."""
    acc = defaultdict(lambda: defaultdict(list))
    ranks = defaultdict(lambda: [0, 0, 0])
    rho = defaultdict(list)
    for c in recs:
        for band, blocks in BANDS.items():
            if c["block"] in blocks:
                k = (c["targets"], c["source"], c["beta"], c["step"], band)
                acc[k][c["passage"]].append(c[stat])
                ranks[k][0] += c["p_hi"] <= 0.05
                ranks[k][1] += c["p_lo"] <= 0.05
                ranks[k][2] += 1
                rho[k].append(c["rho_logpos"])
    vals = {k: {p: float(np.mean(v)) for p, v in d.items()} for k, d in acc.items()}
    return vals, dict(ranks), {k: float(np.nanmedian(v)) for k, v in rho.items()}


def labels_for(vals: Dict, n_passages: int) -> Dict:
    out = {}
    for k, d in vals.items():
        if len(d) != n_passages:
            raise SystemExit(f"refusing: {k} has {len(d)} passages, expected {n_passages}")
        out[k] = label(list(d.values()))
    return out


def isolated(labs: Dict, k: tuple) -> bool:
    """A lean not shared (same direction) by an adjacent step."""
    t, s, b, step, band = k
    i = STEPS.index(step)
    near = [STEPS[j] for j in (i - 1, i + 1) if 0 <= j < len(STEPS)]
    return not any(direction(labs.get((t, s, b, n, band), "mixed")) == direction(labs[k])
                   for n in near)


def summarise(out: Path, kind: str, nll: Dict = None, sub: str = "records") -> Dict:
    recs = load(out, kind, sub)
    if not recs:
        return {}
    n_pass = len({c["passage"] for c in recs})
    vals, ranks, rho = band_values(recs)
    valst, _, _ = band_values(recs, "Xt")
    labs, labst = labels_for(vals, n_pass), labels_for(valst, n_pass)
    rows = {}
    for k, lab in labs.items():
        t, s, b, step, band = k
        at_b = [labs[(t, s, bb, step, band)] for bb in BETAS if (t, s, bb, step, band) in labs]
        robust = len(at_b) > 1 and len(set(at_b)) == 1
        at0 = labs.get((t, s, b, 0, band))
        rows["|".join(map(str, k))] = {
            "label": lab, "label_Xt": labst[k], "mean_X": float(np.mean(list(vals[k].values()))),
            "X": vals[k], "beta_robust": robust, "step0_same": step != 0 and at0 == lab and lab != "mixed",
            "isolated": "leans" in lab and isolated(labs, k),
            "rank_hi": ranks[k][0], "rank_lo": ranks[k][1], "cells": ranks[k][2],
            "rho_logpos_median": rho[k]}
    counts = defaultdict(lambda: defaultdict(int))
    for k, lab in labs.items():
        t, s, b, step, band = k
        if band in TRAINED_BANDS:
            counts[f"{t}|{s}|{b}"][lab] += 1
    rec = {"kind": kind, "n_passages": n_pass, "rows": rows,
           "counts": {k: dict(v) for k, v in counts.items()},
           "chance": chance(n_pass, len(STEPS) * len(TRAINED_BANDS))}
    if nll and kind == "long":
        low = min(nll["143000"], key=nll["143000"].get)
        drop = {}
        for k, d in vals.items():
            if k[3] == 143000:
                drop["|".join(map(str, k))] = label([v for p, v in d.items() if p != low])
        rec["drop_lowest_nll"] = {"passage": low, "nll": nll["143000"][low], "labels": drop}
    return rec


def table(rec: Dict, t: str, s: str, b: float) -> List[str]:
    lines = [f"{'step':>7s} " + " ".join(f"{band:>22s}" for band in BANDS)]
    for step in STEPS:
        cells = []
        for band in BANDS:
            r = rec["rows"].get(f"{t}|{s}|{b}|{step}|{band}")
            if r is None:
                cells.append(f"{'—':>22s}")
                continue
            mark = ("β" if r["beta_robust"] else "") + ("0" if r["step0_same"] else "") + \
                   ("i" if r["isolated"] else "")
            cells.append(f"{SHORT[r['label']]:>3s} {r['mean_X']:+.3f} ({SHORT[r['label_Xt']]:>3s}) {mark:<3s}"
                         .rjust(22))
        lines.append(f"{step:>7d} " + " ".join(cells))
    return lines


def report(out: Path) -> int:
    nll_path = out / "nll.json"
    nll = json.loads(nll_path.read_text()) if nll_path.exists() else None
    full, text = {}, []
    for kind in ("long", "v1"):
        rec = summarise(out, kind, nll)
        if not rec:
            continue
        full[kind] = rec
        t, s, b = PRIMARY[kind]
        text += [f"== {kind}: {rec['n_passages']} passages; label on X (Xt's label in brackets); "
                 f"marks: β = same at 1.6/3.5/5.6, 0 = step 0 carries it, i = isolated lean"]
        for tt, ss, bb in [(t, s, b)] + ([("t123", "causal", 3.5)] if kind == "long" else []) + \
                          [(t, "nosink", 3.5), (t, "full", 3.5), (t, "local", 3.5),
                           (t, "causal", 1.6), (t, "causal", 5.6)]:
            text += [f"-- targets {tt}, source {ss}, β {bb}"] + table(rec, tt, ss, bb)
        ch = rec["chance"]
        text += [f"chance over the {len(STEPS) * len(TRAINED_BANDS)} L1–22 cells per (targets, "
                 f"source, β): {ch['ascends_or_descends']:.2f} ascends/descends, {ch['leans']:.1f} leans"]
        for k in sorted(rec["counts"]):
            text.append(f"  {k}: {rec['counts'][k]}")
        if "drop_lowest_nll" in rec:
            d = rec["drop_lowest_nll"]
            text.append(f"143000 without {d['passage']} (NLL {d['nll']:.2f}), {t}|{s}|{b}: " + ", ".join(
                f"{band} {SHORT[d['labels'][f'{t}|{s}|{b}|143000|{band}']]}" for band in BANDS))
    for kind in ("long", "v1"):
        rec = summarise(out, kind, None, "records_shared")
        if not rec:
            continue
        full[f"shared_{kind}"] = rec
        t = PRIMARY[kind][0]
        text += [f"== {kind}, beside after /challenge-pr on #158 (finding 1): the shared update",
                 "   r1out: each token's residual update less its component along the shared "
                 "direction; residout: the residual update's mean removed; resid: that mean "
                 "alone; ambout: unit-frame mean move out (#158's review); mean0: the β = 0 field"]
        for ss, bb in SHARED_ROWS:
            text += [f"-- targets {t}, {ss}, β {bb}"] + table(rec, t, ss, bb)
        for k in sorted(rec["counts"]):
            text.append(f"  {k}: {rec['counts'][k]}")
    if "long" in full:
        diff = [k for k, r in full["long"]["rows"].items() if k.startswith("t12|")
                and full["long"]["rows"]["t123|" + k[4:]]["label"] != r["label"]]
        full["long"]["t12_vs_t123"] = diff
        text.append(f"T1+T2 against T1–T3: {len(diff)} of "
                    f"{sum(k.startswith('t12|') for k in full['long']['rows'])} (targets-free) cells differ")
    if nll:
        full["nll"] = nll
    (out / "report.json").write_text(json.dumps(full, indent=1) + "\n")
    (out / "report.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text))
    return 0
