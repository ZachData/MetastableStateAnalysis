"""
p1e_energy_field/u2_heads_report.py — U2's per-head arm: label tables for attention's move against
the heads' own kernels, the per-window reading, per-head counts, and Blocked 29's ascent shares
(`design-1e.md` "U2's per-head arm: the rule", rows *labels*, *reading*, *Blocked 29's ascent
reading*). Tier 1: exploratory, unregistered.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from .extract_long8 import STEPS
from .u2_block import BANDS
from .u2_heads import ATTN_CELLS, PLAYERS
from .u2_report import SHORT, TRAINED_BANDS, direction, label, summarise, table

PRIMARY_T = {"long": "t12", "v1": "r0"}
ATTN_ARM, BLOCK_ARM = "u2_attn_2026-10-07", "u2_block_2026-10-07"
HEAD_ROWS = ("kernns_h:keys_h:r1out", "kern_h:head_h:frozen")
#: Ascent shares closer than this are a tie (the rule's 0.1).
TIE = 0.1


def beta_of(field: str) -> float:
    return 3.5 if field == "causal" else 0.0


def verdict(lab: str) -> str:
    return {1: "follows its own kernel (φ_β the wrong field)", -1: "against its own kernel (repels)",
            0: "the real kernel does not sign it"}[direction(lab)]


def head_labels(out: Path, kind: str) -> Dict:
    """Per (source, step, block, head): the sign label over passages, and whether any is short."""
    acc = defaultdict(dict)
    short = defaultdict(bool)
    stats = defaultdict(lambda: defaultdict(list))
    n_pass = set()
    for p in sorted((out / "records" / kind).glob("*.json")):
        r = json.loads(p.read_text())
        n_pass.add(r["passage"])
        for c in r["head_cells"]:
            k = (c["source"], int(r["step"]), c["block"], c["head"])
            acc[k][r["passage"]] = c["X"]
            short[k] |= c["n"] < 0.9 * r["targets"]
        for s in r["heads"]:
            for f in ("align_phi", "a0"):
                stats[(int(r["step"]), s["block"], s["head"])][f].append(s[f])
    labs = {}
    for k, d in acc.items():
        if len(d) != len(n_pass):
            raise SystemExit(f"refusing: {k} has {len(d)} passages, expected {len(n_pass)}")
        labs[k] = label(list(d.values()))
    return {"labels": labs, "short": dict(short), "stats": stats, "n_passages": len(n_pass)}


def head_counts(h: Dict, src: str) -> Dict:
    """Per (step, band): counts of (block, head) units by label, short ones kept out and counted."""
    res = {}
    for step in STEPS:
        for band in BANDS:
            c = defaultdict(int)
            al, a0 = [], []
            for (s, st, b, hh), lab in h["labels"].items():
                if s != src or st != step or b not in BANDS[band]:
                    continue
                if h["short"][(s, st, b, hh)]:
                    c["short"] += 1
                    continue
                c[lab] += 1
                c["units"] += 1
                al.append(np.median(h["stats"][(st, b, hh)]["align_phi"]))
                a0.append(np.median(h["stats"][(st, b, hh)]["a0"]))
            if c:
                P = h["n_passages"]
                c["chance_full"] = c["units"] * 2 / 2 ** P
                c["chance_lean"] = c["units"] * 2 * P / 2 ** P
                c["align_phi_median"] = float(np.median(al)) if al else float("nan")
                c["a0_median"] = float(np.median(a0)) if a0 else float("nan")
                res[(step, band)] = dict(c)
    return res


def ascent_table(out: Path, kind: str) -> Dict:
    """Per (step, band): each part's ascent share (median over passages of band sums' ratio)."""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for p in sorted((out / "ascent" / kind).glob("*.json")):
        r = json.loads(p.read_text())
        for L, v in r["blocks"].items():
            for band, blocks in BANDS.items():
                if int(L) in blocks:
                    a = acc[(int(r["step"]), band)][r["passage"]]
                    a["F_all"] += v["F_all"]
                    for k in PLAYERS:
                        a[k] += v["shapley"][k]
    res = {}
    for k, per in acc.items():
        res[k] = {"share": {p: float(np.median([v[p] / v["F_all"] for v in per.values()])) for p in PLAYERS},
                  "F_all_median": float(np.median([v["F_all"] for v in per.values()])),
                  "F_all_positive": int(sum(v["F_all"] > 0 for v in per.values())),
                  "n": len(per)}
    return res


def blocked29_ascent(asc: Dict, block_rows: Dict, attn_b29: Dict, t: str) -> Dict:
    out = {}
    for step in STEPS:
        for band in TRAINED_BANDS:
            blab = block_rows.get(f"{t}|causal:resid|3.5|{step}|{band}", {}).get("label")
            if blab != "ascends" or (step, band) not in asc:
                continue
            sh = asc[(step, band)]["share"]
            ranked = sorted(sh.items(), key=lambda kv: -kv[1])
            if ranked[0][1] - ranked[1][1] < TIE:
                who, v = f"{ranked[0][0]}≈{ranked[1][0]}", "tied"
            else:
                who = ranked[0][0]
                v = "(a) follows the field" if who == "keys" else "(b) not interaction"
            out[f"{step}|{band}"] = {"carrier": who, "verdict": v, "shares": sh,
                                     "F_all_positive": asc[(step, band)]["F_all_positive"],
                                     "length_carrier": attn_b29.get(f"{step}|{band}", {}).get("carrier")}
    return out


def report(out: Path) -> int:
    attn_rep, block_rep = out.parent / ATTN_ARM / "report.json", out.parent / BLOCK_ARM / "report.json"
    for p in (attn_rep, block_rep):
        if not p.exists():
            raise SystemExit(f"refusing: no {p}; the reading needs the attention and block arms' labels")
    attn, block = json.loads(attn_rep.read_text()), json.loads(block_rep.read_text())
    full, text = {}, []
    for kind in ("long", "v1"):
        rec = summarise(out, kind, None, "records")
        if not rec:
            continue
        t = PRIMARY_T[kind]
        arows = attn[kind]["labels"]["rows"]
        full[kind] = {"labels": rec}
        text += [f"== {kind}: {rec['n_passages']} passages, targets {t}; label on X (Xt's in brackets); "
                 "kernel rows have no β (shown as 0.0); marks: 0 = step 0 carries it, i = isolated lean"]
        for f, c, r in ATTN_CELLS:
            text += [f"-- {f}:{c}:{r}"] + table(rec, t, f"{f}:{c}:{r}", beta_of(f))
        ch = rec["chance"]
        text.append(f"chance over 54 L1–22 cells: {ch['ascends_or_descends']:.2f} ascends/descends, "
                    f"{ch['leans']:.1f} leans")
        for k in sorted(rec["counts"]):
            text.append(f"  {k}: {rec['counts'][k]}")
        # the check row reproduces the attention arm's band values, passage by passage
        dev = max(abs(x - arows[k]["X"][p]) for k, v in rec["rows"].items()
                  if "|causal:keys:r1out|" in k for p, x in v["X"].items())
        if dev > 1e-6:
            raise SystemExit(f"refusing: check row differs from the attention arm's by {dev:.1e}")
        text.append(f"check row against the attention arm's causal:keys:r1out: max |ΔX| {dev:.1e}")
        h = head_labels(out, kind)
        hc = {src: head_counts(h, src) for src in HEAD_ROWS}
        full[kind]["heads"] = {src: {f"{s}|{b}": v for (s, b), v in d.items()} for src, d in hc.items()}
        # the reading, per window
        win, missing = {}, []
        for step in STEPS:
            for band in TRAINED_BANDS:
                a = arows.get(f"{t}|causal:keys:r1out|3.5|{step}|{band}", {}).get("label", "mixed")
                if direction(a) != -1:
                    continue
                if f"{t}|kernns:keys:r1out|0.0|{step}|{band}" not in rec["rows"]:
                    missing.append(f"{step}|{band}")
                    continue
                k = rec["rows"][f"{t}|kernns:keys:r1out|0.0|{step}|{band}"]["label"]
                k2 = rec["rows"][f"{t}|kern:attn:r1out|0.0|{step}|{band}"]["label"]
                win[f"{step}|{band}"] = {"phi": a, "kernns": k, "kern_attn": k2, "verdict": verdict(k),
                                         "heads": hc["kernns_h:keys_h:r1out"].get((step, band), {})}
        full[kind]["reading"] = win
        text += [f"-- the reading: where attention's keys:r1out descends φ_β (attention arm), "
                 f"kernns:keys:r1out against the heads' own kernels; per-head units (keys r1out "
                 f"against own kernel): ASC / DES / asc / des / mixed / short, chance full / lean"]
        tally = defaultdict(int)
        for k, v in win.items():
            tally[v["verdict"]] += 1
            hh = v["heads"]
            text.append(f"  {k:>14s}: φ {SHORT[v['phi']]:>3s}  kernns {SHORT[v['kernns']]:>3s}  "
                        f"kern:attn {SHORT[v['kern_attn']]:>3s}  → {v['verdict']:<46s} heads "
                        + " / ".join(str(hh.get(x, 0)) for x in ("ascends", "descends", "leans ascends",
                                                                 "leans descends", "mixed", "short"))
                        + f" of {hh.get('units', 0)} (chance {hh.get('chance_full', 0):.1f} / "
                          f"{hh.get('chance_lean', 0):.1f}); align {hh.get('align_phi_median', float('nan')):+.2f}")
        text.append(f"  tally: {dict(tally)}" + (f"; NOT YET RUN: {len(missing)} windows" if missing else ""))
        full[kind]["reading_missing"] = missing
        for src in HEAD_ROWS:
            text += [f"-- per-head units, {src}: ASC DES asc des mixed short | units, chance full/lean | "
                     "median align with φ_β, median attention to key 0"]
            for step in STEPS:
                for band in TRAINED_BANDS:
                    v = hc[src].get((step, band))
                    if v:
                        text.append(f"{step:>7d} {band:>7s} " + " ".join(
                            f"{v.get(x, 0):4d}" for x in ("ascends", "descends", "leans ascends",
                                                         "leans descends", "mixed", "short"))
                            + f" | {v['units']:4d} {v['chance_full']:.1f}/{v['chance_lean']:.1f} | "
                              f"{v['align_phi_median']:+.2f} {v['a0_median']:.2f}")
        # Blocked 29: who supplies the shared ascent
        asc = ascent_table(out, kind)
        if asc:
            brows = block.get(f"shared_{kind}", {}).get("rows", {})
            b29 = blocked29_ascent(asc, brows, attn[kind].get("blocked29", {}), t)
            full[kind]["ascent"] = {f"{s}|{b}": v for (s, b), v in asc.items()}
            full[kind]["blocked29_ascent"] = b29
            text += ["-- Blocked 29 by ascent: where the block's resid ascends, each part's Shapley share "
                     "of the shared move's projection on the force (median over passages; tie < 0.1); "
                     "the attention arm's length carrier beside"]
            tally = defaultdict(int)
            for k, v in b29.items():
                tally[v["verdict"]] += 1
                text.append(f"  {k:>14s}: {v['carrier']:>12s} {v['verdict']:<22s} " +
                            ", ".join(f"{p} {v['shares'][p]:+.2f}" for p in PLAYERS) +
                            f"  (F > 0 in {v['F_all_positive']}; length: {v['length_carrier']})")
            text.append(f"  tally: {dict(tally)}")
    (out / "report.json").write_text(json.dumps(full, indent=1) + "\n")
    (out / "report.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text))
    return 0
