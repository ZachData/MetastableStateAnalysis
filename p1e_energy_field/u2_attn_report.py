"""
p1e_energy_field/u2_attn_report.py — U2's attention arm: label tables per component and reading
(`u2_report`'s rule), the block's shared update split by component, and the per-window reading for
Blocked 29 (`design-1e.md` "U2's attention arm: the rule", row *reading for Blocked 29*).
Tier 1: exploratory, unregistered.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np

from .extract_long8 import STEPS
from .u2_attn import COMPONENTS, PARTS
from .u2_block import BANDS
from .u2_report import TRAINED_BANDS, direction, summarise, table

PRIMARY_T = {"long": "t12", "v1": "r0"}
ROWS = [(f"causal:{c}:frozen", 3.5) for c in COMPONENTS] + \
       [(f"causal:{c}:r1out", 3.5) for c in COMPONENTS if c not in ("bias", "block")] + \
       [(f"causal:{c}:resid", 3.5) for c in COMPONENTS if c not in ("bias", "block")] + \
       [(f"mean0:{c}:r1out", 0.0) for c in COMPONENTS if c not in ("bias", "block")]
#: Shares closer than this are a tie (the rule's 0.1).
TIE = 0.1
#: The block arm's output, whose ``resid`` labels the rule reads.
BLOCK_ARM = "u2_block_2026-10-07"


def share_table(out: Path, kind: str) -> Dict:
    """``{(step, band): {"share": {part: median}, "sharedness": {...}, "a0": median}}``."""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for p in sorted((out / "records" / kind).glob("*.json")):
        r = json.loads(p.read_text())
        for L, s in r["shares"].items():
            for band, blocks in BANDS.items():
                if int(L) in blocks:
                    a = acc[(int(r["step"]), band)]
                    for k, v in s["share"].items():
                        a["share:" + k][r["passage"]].append(v)
                    for k, v in s["sharedness"].items():
                        a["sharedness:" + k][r["passage"]].append(v)
                    a["a0"][r["passage"]].append(s["a0_mean"])
    res = {}
    for k, a in acc.items():
        med = {name: float(np.median([np.mean(v) for v in per.values()])) for name, per in a.items()}
        res[k] = {"share": {p: med["share:" + p] for p in PARTS},
                  "sharedness": {c: med["sharedness:" + c] for c in COMPONENTS if c != "bias"},
                  "a0": med["a0"]}
    return res


def blocked29(rec: Dict, shares: Dict, block_rows: Dict, t: str) -> Dict:
    """Per trained (step, band) where the block's ``resid`` ascends: which part carries it."""
    out = {}
    for step in STEPS:
        for band in TRAINED_BANDS:
            blab = block_rows.get(f"{t}|causal:resid|3.5|{step}|{band}", {}).get("label")
            if blab != "ascends":
                continue
            sh = shares[(step, band)]["share"]
            cand = {}
            for p in PARTS:
                src = f"causal:{p}:frozen" if p == "bias" else f"causal:{p}:resid"
                lab = rec["rows"][f"{t}|{src}|3.5|{step}|{band}"]["label"]
                if direction(lab) == direction(blab):
                    cand[p] = sh[p]
            ranked = sorted(cand.items(), key=lambda kv: -kv[1])
            if not ranked:
                who, verdict = None, "undecided (no part's resid ascends)"
            elif len(ranked) > 1 and ranked[0][1] - ranked[1][1] < TIE:
                who, verdict = f"{ranked[0][0]}≈{ranked[1][0]}", "undecided (tied)"
            else:
                who = ranked[0][0]
                verdict = "(a) follows the field" if who == "keys" else "(b) not interaction"
            out[f"{step}|{band}"] = {"block_resid": blab, "carrier": who, "verdict": verdict,
                                     "shares": sh, "candidates": cand}
    return out


def report(out: Path) -> int:
    block_rep = out.parent / BLOCK_ARM / "report.json"
    block = json.loads(block_rep.read_text()) if block_rep.exists() else {}
    full, text = {}, []
    for kind in ("long", "v1"):
        rec = summarise(out, kind, None, "records")
        if not rec:
            continue
        t = PRIMARY_T[kind]
        sh = share_table(out, kind)
        full[kind] = {"labels": rec, "shares": {f"{s}|{b}": v for (s, b), v in sh.items()}}
        text += [f"== {kind}: {rec['n_passages']} passages, targets {t}; label on X (Xt's in brackets); "
                 "marks: β = same at 1.6/3.5/5.6, 0 = step 0 carries it, i = isolated lean"]
        for src, b in ROWS:
            text += [f"-- {src}, β {b}"] + table(rec, t, src, b)
        ch = rec["chance"]
        text.append(f"chance over 54 L1–22 cells: {ch['ascends_or_descends']:.2f} ascends/descends, "
                    f"{ch['leans']:.1f} leans")
        for k in sorted(rec["counts"]):
            text.append(f"  {k}: {rec['counts'][k]}")
        text += [f"-- the block's shared update c̄ split by part (median over passages of the band "
                 f"mean; sums to 1), sharedness |mean c|/mean |c|, attention to key 0",
                 f"{'step':>7s} {'band':>7s} " + " ".join(f"{p:>6s}" for p in PARTS) + "  | " +
                 " ".join(f"{c:>6s}" for c in COMPONENTS if c != "bias") + " |     a0"]
        for step in STEPS:
            for band in BANDS:
                v = sh.get((step, band))
                if v:
                    text.append(f"{step:>7d} {band:>7s} " + " ".join(f"{v['share'][p]:+6.2f}" for p in PARTS)
                                + "  | " + " ".join(f"{v['sharedness'][c]:6.2f}" for c in COMPONENTS if c != "bias")
                                + f" | {v['a0']:6.3f}")
        brows = block.get(f"shared_{kind}", {}).get("rows", {})
        if brows:
            b29 = blocked29(rec, sh, brows, t)
            full[kind]["blocked29"] = b29
            text += ["-- Blocked 29: where the block's resid ascends, the part with the largest share "
                     "whose own resid ascends too (tie < 0.1)"]
            for k, v in b29.items():
                text.append(f"  {k:>16s}: {str(v['carrier']):>10s}  {v['verdict']}   shares " +
                            ", ".join(f"{p} {v['shares'][p]:+.2f}" for p in PARTS))
            tally = defaultdict(int)
            for v in b29.values():
                tally[v["verdict"]] += 1
            text.append(f"  tally: {dict(tally)}")
        else:
            raise SystemExit(f"refusing: no block-arm report at {block_rep}; Blocked 29's table "
                             "needs its resid labels (`/challenge-pr` on #160, finding 5)")
    (out / "report.json").write_text(json.dumps(full, indent=1) + "\n")
    (out / "report.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text))
    return 0
