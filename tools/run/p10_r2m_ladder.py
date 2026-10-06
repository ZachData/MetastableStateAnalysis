"""Phase 10 re-read, R2m: F12 against a matched control, and §1.5 again, down the ladder
(`p10_cluster_function/design-10.md` "R2m"; `STATE.md` Blocked 25, option (iv) taken).

Reads, per column, R2's trained record (``<r2>/f12_<column>.json``), the matched control
(``<dir>/f12m_<column>.json``: the same labels scored on step 0's activations, written by
``p10_partition_function.py --activations-step 0``) and R2's F1 record (``<r2>/f1_<column>.json``).

====== ====================================================================
F12m   per unit (step, prompt, layer, beta), matched Δ = trained − control;
       per step, the mean over paired units: **below / as / above control**
       at ±``GAP_FLOOR``. Beside it: the raw sign (trained mean: members
       **below / as / above the rest** at ±``GAP_FLOOR``), the control's mean
       and median p, and the prompts whose mean Δ is negative, with a two-sided
       sign-test p over prompts (`sign_p`; beside, not a label: added after
       `/challenge-pr` on #149)
§1.5   per step, **parked** iff R2's F1 label is negative and F12m is below
       control, else **not**
====== ====================================================================

Primary column per step and "holds" are R2's (`p10_r2_ladder.primary`, `p10_r1_ladder.holds`).
Checks before any reading: every record names the same label source and summary sha256; the
control read step 0's activations and the same readable records as the trained record; every
trained unit has its control; at step 0 every Δ is exactly 0 (same activations, labels and
generator), which fails if the two runs are not the pair they claim to be.

Tier 1: exploratory, unregistered. Run:
    python tools/run/p10_r2m_ladder.py --dir <R2m out> --r2 <R2 out> --labels <R0 labels>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from math import comb
from pathlib import Path
from statistics import mean, median
from typing import Dict, Optional, Sequence

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))

from tools.run.p10_r1_ladder import LadderError, _fmt, arms_differ, holds
from tools.run.p10_r2_ladder import COLUMNS, GAP_FLOOR, f1_label, primary


def _read(f: Path, labels: Path, col: str) -> Dict:
    if not f.exists():
        raise LadderError(f"no record {f.name}")
    d = json.loads(f.read_text())
    ls = d["label_source"]
    if Path(ls["labels"]) != labels or ls["column"] != col:
        raise LadderError(f"{f.name}: read {ls['labels']} column {ls['column']}")
    return d


def load(dirpath: Path, r2: Path, labels: Path) -> Dict:
    recs: Dict = {"f12": {}, "f12m": {}, "f1": {}}
    shas = set()
    for col in COLUMNS:
        t = _read(r2 / f"f12_{col}.json", labels, col)
        c = _read(dirpath / f"f12m_{col}.json", labels, col)
        f1 = _read(r2 / f"f1_{col}.json", labels, col)
        shas |= {x["label_source"]["summary_sha256"] for x in (t, c, f1)}
        if c.get("activations_step") != 0:
            raise LadderError(f"f12m_{col}: activations from step {c.get('activations_step')}, not 0")
        if c["records_readable"] != t["records_readable"]:
            raise LadderError(f"f12m_{col}: readable records differ from the trained record's")
        recs["f12"][col], recs["f12m"][col] = t, c
        recs["f1"][col] = {int(s): x for s, x in f1["by_step"].items()}
    if len(shas) != 1:
        raise LadderError(f"records name {len(shas)} label source summaries: {sorted(shas)}")
    summ = json.loads((labels / "summary.json").read_text())
    return {"recs": recs, "summary": summ, "summary_sha256": shas.pop()}


def paired(trained: Dict, control: Dict) -> Dict[int, list]:
    """{step: [(prompt, trained, control, control p)]} over units both read; refuses a trained
    unit without its control, and a non-zero Δ at step 0."""
    out: Dict[int, list] = defaultdict(list)
    for key, rows in trained.items():
        ctrl = {(u["layer"], u["beta"]): u for u in control.get(key, [])}
        s, prompt = key.split("|", 1)
        for u in rows:
            c = ctrl.get((u["layer"], u["beta"]))
            if c is None:
                raise LadderError(f"{key} L{u['layer']} β{u['beta']}: no control unit")
            if int(s) == 0 and u["stat"] != c["stat"]:
                raise LadderError(f"{key} L{u['layer']} β{u['beta']}: Δ {u['stat'] - c['stat']} at step 0")
            out[int(s)].append((prompt, u["stat"], c["stat"], c["p"]))
    return dict(sorted(out.items()))


def word(d: Optional[float], what: str) -> str:
    return ("unavailable" if d is None else f"above {what}" if d > GAP_FLOOR
            else f"below {what}" if d < -GAP_FLOOR else f"as {what}")


def sign_p(k: int, n: int) -> float:
    """Two-sided sign-test p for ``k`` of ``n`` prompts on one side (exact binomial, 1/2)."""
    if n == 0:
        return 1.0
    m = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(m + 1)) / 2 ** n)


def step_cells(units: Dict[int, list]) -> Dict[int, Dict]:
    out = {}
    for s, us in units.items():
        t, c = mean(u[1] for u in us), mean(u[2] for u in us)
        by_prompt: Dict[str, list] = defaultdict(list)
        for p, a, b, _ in us:
            by_prompt[p].append(a - b)
        d = round(t - c, 4)
        neg = sum(mean(v) < 0 for v in by_prompt.values())
        pos = sum(mean(v) > 0 for v in by_prompt.values())   # a prompt at exactly 0 has no sign
        out[s] = {"n": len(us), "trained": round(t, 4), "control": round(c, 4), "delta": d,
                  "label": word(d, "control"), "raw": word(round(t, 4), "rest"),
                  "control_median_p": round(median(u[3] for u in us), 4),
                  "prompts_negative": neg, "prompts": len(by_prompt),
                  "sign_p": round(sign_p(neg, neg + pos), 4)}
    return out


def columns(data: Dict) -> Dict:
    """Per row, per column: {(quantity, layer): {step: (value, label)}}, and the cells."""
    cells = {c: step_cells(paired(data["recs"]["f12"][c]["runs"], data["recs"]["f12m"][c]["runs"]))
             for c in COLUMNS}
    f12m = {c: {("F12m", "all"): {s: (x["delta"], x["label"]) for s, x in cells[c].items()}} for c in COLUMNS}
    raw = {c: {("F12 raw", "all"): {s: (x["trained"], x["raw"]) for s, x in cells[c].items()}} for c in COLUMNS}
    park = {}
    for c in COLUMNS:
        f1 = data["recs"]["f1"][c]
        park[c] = {("§1.5", "all"): {s: (None, "parked" if f1_label(f1.get(s, {})) == "negative"
                                         and x["label"] == "below control" else "not")
                                     for s, x in cells[c].items()}}
    return {"F12m": f12m, "F12 raw": raw, "§1.5": park, "cells": cells}


def window(cells: Dict, want: str, prim: Optional[Dict[int, str]] = None, col: str = "c3") -> list:
    q = next(iter(cells[col]))
    return [s for s in sorted(cells[col][q])
            if cells[prim.get(s, "c3") if prim else col].get(q, {}).get(s, (None, ""))[1] == want]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", type=Path, required=True, help="the R2m records (f12m_<column>.json)")
    ap.add_argument("--r2", type=Path, required=True, help="the R2 records (f12_ and f1_<column>.json)")
    ap.add_argument("--labels", type=Path, required=True, help="the R0 label source both were read on")
    ap.add_argument("--out", type=Path, default=None, help="default <dir>/ladder.json")
    args = ap.parse_args(argv)
    data = load(args.dir, args.r2, args.labels)
    prim = primary(data["summary"])
    cols = columns(data)
    res = {"label_source": str(args.labels), "summary_sha256": data["summary_sha256"], "primary": prim,
           "gap_floor": GAP_FLOOR, "cells": cols["cells"], "rows": {}}
    for row in ("F12m", "F12 raw", "§1.5"):
        h = holds(cols[row], prim, skip0=True)
        res["rows"][row] = {"holds": h, "arms": arms_differ(cols[row], prim)}
        print(f"\n{row}: {h['agree']} of {h['n']} step labels agree, primary vs c0; the rest first changed "
              f"at {h['first_changed_at'] or 'none'}, settled at {h['settled_at'] or 'none'}")
        for a, x in res["rows"][row]["arms"].items():
            print(f"  arm {a}: differs from c3 in {len(x['differ'])} of {x['n']}")
    res["windows"] = {
        row: {c: window(cols[row], want, col=c) for c in ("c0", "c2", "c3")}
        | {"primary": window(cols[row], want, prim)}
        for row, want in (("F12m", "below control"), ("F12 raw", "below rest"), ("§1.5", "parked"))}
    cc = cols["cells"]
    print("\nstep prim | F12m Δ = trained − control (prompts Δ<0 of n), control median p: c0 / c2 / prim")
    for s, pc in prim.items():
        def f(c):
            x = cc[c].get(s)
            return "—" if x is None else (f"{_fmt(x['delta'])} = {_fmt(x['trained'])} − {_fmt(x['control'])} "
                                          f"({x['prompts_negative']}/{x['prompts']}, sign p {x['sign_p']:.3f}) p{x['control_median_p']:.3f}")
        print(f"{s:>6} {pc:>3} | {f('c0')} | {f('c2')} | {f(pc)}")
    for row, w in res["windows"].items():
        print(f"window {row}: {w}")
    out = args.out or args.dir / "ladder.json"
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
