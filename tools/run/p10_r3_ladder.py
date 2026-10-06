"""Phase 10 re-read, R3: A0 (the attention flip against the causal mask) down the ladder, c1 onward
under T4 (`p10_cluster_function/design-10.md` "Row by row", "Order").

Reads the record `p10_attention_baseline.py --labels <R0 label source>` writes (every column in
one file) and applies A0's rule, unchanged, to every column. The flip's **gap** keeps A0's
published sign: rest − members (HDBSCAN's noise − clustered on c0), in "× layer average" units of
the column's domain, raw and mask-corrected.

======== ==================================================================
share    over the sweep (every unit of the column, pooled as published):
         mask share = 1 − corrected gap / raw gap; **≥ 0.9** or **< 0.9**
         (0.9 placed, below c0's ~0.94); **no flip** where the raw gap is
         ≤ 0 (the rest receives no more than the members: nothing for the
         mask to explain)
residual per step: **residual** where the corrected gap is ≥ ``RESIDUAL``,
         else **none**. Placed at 0.05, as F12's and §1.9's floors; the
         published reading ("appears at 2000–4000") was a size reading (no
         step's merged E rejects on c0), and 0.05 gives that interval on
         the published record (0.019 at 2000, 0.245 at 4000)
appears  the interval (last step before, first step) where the per-step
         label is first **residual**; ``never`` if it never is
persists **yes** iff every step from the first residual one to 143000 is
         residual
======== ==================================================================

**A row holds** where the primary's three labels equal **c1's** (T4 changes the statistic, so
c3 is compared with c1, not c0; `design-10.md`). The per-step residual labels are compared the
same way, naming the first column after c1 whose label differs. c0 → c1 (T4 and the token
rules together) is printed beside, not judged. THE PRIMARY per step: c3, unless fewer than half
its records are readable there (the label source's own count), then c2 (`p10_r2_ladder.primary`).
The primary's sweep share pools c2's units at its c2 steps and c3's elsewhere.

There is no step-0 baseline in A0's rule (the mask baseline is closed-form), so `STATE.md`
Blocked 25 does not bear on it. Significance is printed beside every step (median corrected p,
merged E), never used by a label.

Checks before any reading: the record names this label source and its ``summary.json`` hash,
holds every step (the learned split: 143000 only), and c3's readable counts equal the source's.
``--published <file>``: c0's per-unit values against the published A0 record by run-dir name and
layer (that record read the earlier WDS sweep, 2026-08-31 dirs, on 8 prompts; units both read).

Tier 1: exploratory, unregistered. Run:
    python tools/run/p10_r3_ladder.py --record <a0.json> --labels <R0 labels> \
        [--published data/analysis/p10_row_a0.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))

from tools.run.p10_r1_ladder import ARMS, LADDER, LEARNED, LadderError, _fmt
from tools.run.p10_r2_ladder import primary

SHARE_BAR = 0.9   # placed (design-10.md "Row by row", A0)
RESIDUAL = 0.05   # placed, as F12's and §1.9's floors
COLUMNS = (*LADDER, *ARMS, *LEARNED)
AFTER_C1 = LADDER[LADDER.index("c1"):]
PUBLISHED_FIELDS = ("raw_noise", "raw_clustered", "corrected_noise", "corrected_clustered",
                    "position_bias", "n_tokens", "noise_fraction")


def load(record: Path, labels: Path) -> Dict:
    d = json.loads(Path(record).read_text())
    ls = d["label_source"]
    if Path(ls["labels"]) != Path(labels):
        raise LadderError(f"{record}: read {ls['labels']}, not {labels}")
    sha = hashlib.sha256((Path(labels) / "summary.json").read_bytes()).hexdigest()[:16]
    if ls["summary_sha256"] != sha:
        raise LadderError(f"{record}: label source summary {ls['summary_sha256']}, on disk {sha}")
    summ = json.loads((Path(labels) / "summary.json").read_text())
    steps = {int(k.removeprefix("step")) for k in summ if k.startswith("step")}
    missing = set(COLUMNS) - set(d["columns"])
    if missing:
        raise LadderError(f"{record}: no column {sorted(missing)}")
    for col in COLUMNS:
        got = {int(s) for s in d["columns"][col]["records_readable"]}
        want = {max(steps)} if col in LEARNED else steps
        if got != want:
            raise LadderError(f"{col}: steps {sorted(got ^ want)} missing or extra")
    for s, rec in d["columns"]["c3"]["records_readable"].items():
        src = summ[f"step{s}"]["columns"]["c3"]
        if (rec["n"], rec["readable"]) != (src["records"], src["readable"]):
            raise LadderError(f"c3 step {s}: readable {rec} against the source's {src}")
    return {"record": d, "summary": summ}


def units_by_step(runs: Dict) -> Dict[int, List[Dict]]:
    out: Dict[int, List[Dict]] = {}
    for key, rows in runs.items():
        out.setdefault(int(key.split("|")[0]), []).extend(rows)
    return dict(sorted(out.items()))


def gaps(rows: List[Dict]) -> Dict:
    """A0's published pooling: gap = mean(rest) − mean(members), raw and corrected."""
    if not rows:
        return {"n": 0, "raw_gap": None, "corrected_gap": None, "share": None,
                "position_bias": None, "median_p": None, "frac_below_05": None}
    m = lambda k: float(np.mean([r[k] for r in rows]))  # noqa: E731
    raw = m("raw_noise") - m("raw_clustered")
    cor = m("corrected_noise") - m("corrected_clustered")
    ps = np.array([r["corrected_p"] for r in rows])
    return {"n": len(rows), "raw_gap": round(raw, 4), "corrected_gap": round(cor, 4),
            "share": round(1 - cor / raw, 4) if raw > 0 else None,
            "position_bias": round(m("position_bias"), 4),
            "median_p": round(float(np.median(ps)), 4), "frac_below_05": round(float(np.mean(ps < 0.05)), 4)}


def share_label(g: Dict) -> str:
    if not g or g["n"] == 0:
        return "n/a"
    if g["raw_gap"] is None or g["raw_gap"] <= 0:
        return "no flip"
    return "≥ 0.9" if g["share"] >= SHARE_BAR else "< 0.9"


def residual_label(g: Dict) -> str:
    if not g or g["n"] == 0:
        return "n/a"
    return "residual" if g["corrected_gap"] >= RESIDUAL else "none"


def appears(labels: Dict[int, str]) -> Dict:
    steps = sorted(labels)
    first = next((s for s in steps if labels[s] == "residual"), None)
    if first is None:
        return {"appears": "never", "persists": "n/a"}
    i = steps.index(first)
    before = steps[i - 1] if i else None
    persists = all(labels[s] == "residual" for s in steps[i:])
    return {"appears": f"{before}–{first}" if before is not None else f"at {first}",
            "persists": "yes" if persists else "no"}


def read_column(units: Dict[int, List[Dict]]) -> Dict:
    per = {s: gaps(rows) for s, rows in units.items()}
    labels = {s: residual_label(g) for s, g in per.items()}
    sweep = gaps([r for rows in units.values() for r in rows])
    return {"per_step": per, "residual": labels, "sweep": sweep, "share": share_label(sweep),
            **appears(labels)}


def read_all(data: Dict) -> Dict:
    cols = data["record"]["columns"]
    units = {c: units_by_step(cols[c]["runs"]) for c in COLUMNS}
    out = {c: read_column(units[c]) for c in COLUMNS}
    prim = primary(data["summary"])
    pu = {s: units[prim[s]].get(s, []) for s in prim}
    out["primary"] = read_column(pu)
    return {"columns": out, "primary": prim}


def holds(cols: Dict, prim: Dict[int, str]) -> Dict:
    """The three labels, primary vs c1; per step, the residual label vs c1 and the first column
    after c1 that changed it."""
    p, c1 = cols["primary"], cols["c1"]
    rule = {k: {"c1": c1[k], "primary": p[k], "same": c1[k] == p[k]} for k in ("share", "appears", "persists")}
    differ = []
    for s, lab1 in c1["residual"].items():
        lab = p["residual"].get(s)
        if lab == lab1:
            continue
        ladder = {c: cols[c]["residual"].get(s) for c in AFTER_C1}
        first = next(c for c in AFTER_C1 if ladder[c] != lab1)
        differ.append({"step": s, "primary": prim[s], "c1": lab1, "label": lab, "first_changed_at": first,
                       "ladder": ladder})
    n = len(c1["residual"])
    return {"rule": rule, "holds": all(x["same"] for x in rule.values()),
            "steps": {"n": n, "agree": n - len(differ), "differ": differ}}


def arms_and_learned(cols: Dict, prim: Dict[int, str]) -> Dict:
    out = {}
    for arm in ARMS:
        d = [s for s, c in prim.items() if c == "c3" and cols[arm]["residual"].get(s) != cols["c3"]["residual"][s]]
        out[arm] = {"n": sum(c == "c3" for c in prim.values()), "differ": d,
                    "share": cols[arm]["share"], "appears": cols[arm]["appears"], "persists": cols[arm]["persists"]}
    s = max(prim)
    out["learned_at_143000"] = {c: cols[c]["per_step"].get(s) for c in LEARNED}
    return out


def published_check(rec: Dict, pub: Path) -> Dict:
    d = json.loads(Path(pub).read_text())
    old = {(r["run_dir"], u["layer"]): u for r in d["directories"] for u in r.get("layers", [])}
    inputs = dict(rec["inputs"])
    n = same = 0
    worst = 0.0
    for k, rows in rec["columns"]["c0"]["runs"].items():
        name = Path(inputs[k]).name
        for u in rows:
            o = old.get((name, u["layer"]))
            if o is None:
                continue
            n += 1
            same += all(o[f] == u[f] for f in PUBLISHED_FIELDS)
            worst = max(worst, max(abs(o[f] - u[f]) for f in PUBLISHED_FIELDS))
    if n == 0:
        raise LadderError(f"{pub}: no unit of c0 matched the published record (compared 0)")
    return {"compared": n, "identical": same, "max_abs_diff": round(worst, 6), "fields": list(PUBLISHED_FIELDS),
            "published": str(pub)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--record", type=Path, required=True, help="the R3 record (p10_attention_baseline --labels)")
    ap.add_argument("--labels", type=Path, required=True, help="the R0 label source it was read on")
    ap.add_argument("--published", type=Path, default=None, help="the published A0 record: check c0 against it")
    ap.add_argument("--out", type=Path, default=None, help="default <record dir>/ladder.json")
    args = ap.parse_args(argv)
    data = load(args.record, args.labels)
    r = read_all(data)
    cols, prim = r["columns"], r["primary"]
    h = holds(cols, prim)
    res = {"record": str(args.record), "label_source": str(args.labels),
           "summary_sha256": data["record"]["label_source"]["summary_sha256"], "primary": prim,
           "share_bar": SHARE_BAR, "residual_floor": RESIDUAL, "holds": h,
           "arms_learned": arms_and_learned(cols, prim),
           "columns": {c: {**v, "per_step": {str(s): g for s, g in v["per_step"].items()},
                           "residual": {str(s): x for s, x in v["residual"].items()}} for c, v in cols.items()}}
    print("column   | sweep raw / corrected gap, share → label | appears | persists")
    for c in (*COLUMNS, "primary"):
        v = cols[c]
        g = v["sweep"]
        print(f"{c:<12} | {_fmt(g['raw_gap'])} / {_fmt(g['corrected_gap'])}, {_fmt(g['share'])} → {v['share']:<8}"
              f" | {v['appears']:<10} | {v['persists']}")
    print("\nstep  prim | corrected gap (raw gap) c0 / c1 / c2 / primary | primary median p, position bias")
    for s, pc in prim.items():
        def cell(c, s=s):
            g = cols[c]["per_step"].get(s)
            return "—" if not g or g["n"] == 0 else f"{_fmt(g['corrected_gap'])} ({_fmt(g['raw_gap'])})"
        g = cols["primary"]["per_step"][s]
        print(f"{s:>6} {pc:>3} | {cell('c0')} / {cell('c1')} / {cell('c2')} / {cell('primary')} | "
              f"{_fmt(g['median_p'])}, {_fmt(g['position_bias'])}")
    print(f"\nholds (primary vs c1): {h['holds']}; " + "; ".join(
        f"{k}: c1 {x['c1']}, primary {x['primary']}" for k, x in h["rule"].items()))
    print(f"per-step residual labels: {h['steps']['agree']} of {h['steps']['n']} agree; first changed at "
          f"{sorted({x['first_changed_at'] for x in h['steps']['differ']})}")
    for a, x in res["arms_learned"].items():
        print(f"  {a}: {x}")
    if args.published is not None:
        res["published_check"] = published_check(data["record"], args.published)
        x = res["published_check"]
        print(f"c0 against the published A0: {x['identical']} of {x['compared']} units identical, "
              f"max |diff| {x['max_abs_diff']}")
    out = args.out or args.record.parent / "ladder.json"
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
