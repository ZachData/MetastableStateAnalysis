"""Phase 10 re-read, R2: F1, F12's gap against step 0, and §1.5 read off both, down the ladder
(`p10_cluster_function/design-10.md` "Row by row", "Order").

Reads the per-column records that `transport.py` and `p10_partition_function.py` write under
``--labels <R0 label source> --column <c>`` (``<dir>/{f1,f12}_<column>.json``) and applies each
row's own rule, unchanged, to every column:

====== ====================================================================
F1     per step, members − rest in per-particle step (boundaries from L1):
       **negative** (median p ≤ ``ALPHA``, mean < 0), **positive** (median
       p ≤ ``ALPHA``, mean > 0), else **none**
F12    per step, Δ = mean members − rest in corrected log Z (pooled over
       betas, as §1.4) minus the baseline's step-0 value: **below / as /
       above baseline** at ±``GAP_FLOOR``. The raw value and the baseline
       are kept beside every Δ (`STATE.md` Blocked 24 (a))
§1.5   per step, **parked** iff F1 is negative and F12 below baseline, else
       **not**; no reader of its own
====== ====================================================================

THE BASELINE (`STATE.md` Blocked 24 (b), decided 2026-10-05): c3, the arms and the learned
split take **c2's step-0 value**; c0 to c2 take their own, so c0 stays the old reading on the
re-read's inputs. THE PRIMARY COLUMN per step: c3, unless fewer than half its (prompt, layer)
records are readable there (the label source's own count), when the step is read on c2 and
labelled so. A window that spans the switch (32 → 64) is read on the primary, and on c2 at
every step beside it. **A row holds** where the primary's label equals c0's at every step
(`p10_r1_ladder.holds`: the first column that changed a label, and where it settled).

Checks before any reading: every record names the same label source and summary sha256, holds
every step of the source (the learned split: 143000 only), and c3's readable counts equal the
source's ``summary.json`` (and the lead's). ``--published <dir>``: c0's per-unit
statistic against the published records (``p10_f1_transport.json``, ``p10_f12_z.json``) on the
units both read (same run-dir name, layer, beta). Those records read the **earlier WDS sweep**
(2026-08-31 dirs), not Stage 0, so this compares two sweeps of the same prompts and steps: a
difference can be the stored labels or the activations, not this re-read's code.

``--lead c3x`` (R9, `design-10.md` "R9"): c3x is appended to the ladder after c3 and is the
primary column (c2 where under half its records are readable), its learned split beside; its
F12 Δ takes c2's step 0, as c3's. Beside, per row, the step labels where c3x differs from c3 at
steps both read on their own column (`p10_r1_ladder.c3_to_c3x`). ``--reproduce <R2 dir>
--reproduce-labels <its source>``: refuse unless every c0–c3 record (and c3's arms and split)
has the same per-step summary and per-unit rows as that run's (R9's first check).

Tier 1: exploratory, unregistered. Run:
    python tools/run/p10_r2_ladder.py --dir <R2 out> --labels <R0 labels> [--published data/analysis]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))

from tools.run import p10_r9_lead as lead_args
from tools.run.p10_r1_ladder import (ARMS, LADDER, LEADS, LEARNED, ON_C2_BASE, LadderError, _fmt, arms_differ,
                                     c3_to_c3x, holds)

READERS = {"f1": "F1", "f12": "F12"}
ALPHA = 0.05       # F1's label: median p (design-10.md "Row by row")
GAP_FLOOR = 0.05   # F12's label: placed, as §1.9's floor (design-10.md)
COLUMNS = (*LADDER, *ARMS, *LEARNED)


def columns_of(lead: str = "c3") -> tuple:
    ladder, learned = LEADS[lead]
    return (*ladder, *ARMS, *learned)


def load(dirpath: Path, labels: Path, lead: str = "c3") -> Dict:
    recs: Dict = {}
    want_sha = None
    learned = LEADS[lead][1]
    for r in READERS:
        for col in columns_of(lead):
            f = dirpath / f"{r}_{col}.json"
            if not f.exists():
                raise LadderError(f"no record {f.name}")
            d = json.loads(f.read_text())
            ls = d["label_source"]
            if Path(ls["labels"]) != labels or ls["column"] != col:
                raise LadderError(f"{f.name}: read {ls['labels']} column {ls['column']}")
            want_sha = want_sha or ls["summary_sha256"]
            if ls["summary_sha256"] != want_sha:
                raise LadderError(f"{f.name}: another label source summary ({ls['summary_sha256']})")
            recs.setdefault(r, {})[col] = {
                "by_step": {int(s): x for s, x in d["by_step"].items()},
                "records": {int(s): x for s, x in d["records_readable"].items()},
                "runs": d["runs"], "inputs": dict(d["inputs"])}
    summ = json.loads((labels / "summary.json").read_text())
    steps = {int(k.removeprefix("step")) for k in summ if k.startswith("step")}
    for r in READERS:
        for col, rec in recs[r].items():
            want = {max(steps)} if col in learned else steps
            if set(rec["records"]) != want:
                raise LadderError(f"{r}_{col}: steps {sorted(set(rec['records']) ^ want)} missing or extra")
        for col in sorted({"c3", lead}):
            for s, rec in recs[r][col]["records"].items():
                src = summ[f"step{s}"]["columns"][col]
                if (rec["n"], rec["readable"]) != (src["records"], src["readable"]):
                    raise LadderError(f"{r} {col} step {s}: readable {rec} against the source's {src}")
    return {"recs": recs, "summary": summ, "summary_sha256": want_sha, "lead": lead}


def primary(summ: Dict, lead: str = "c3") -> Dict[int, str]:
    """``lead`` where at least half its records are readable (the source's count), else c2."""
    out = {}
    for k, v in summ.items():
        if k.startswith("step"):
            c = v["columns"][lead]
            out[int(k.removeprefix("step"))] = lead if 2 * c["readable"] >= c["records"] else "c2"
    return dict(sorted(out.items()))


def reproduce(data: Dict, other: Dict) -> list:
    """Records whose per-step summary or per-unit rows differ from another run's; [] if none."""
    return [f"{r}_{col}" for r in READERS for col in COLUMNS
            if any(data["recs"][r][col][k] != other["recs"][r][col][k] for k in ("by_step", "runs", "records"))]


def f1_label(x: Dict) -> str:
    if not x or x.get("n", 0) == 0 or x.get("mean") is None:
        return "n/a"
    if x["median_p"] <= ALPHA:
        return "negative" if x["mean"] < 0 else "positive" if x["mean"] > 0 else "none"
    return "none"


def gap_word(d: Optional[float]) -> str:
    return ("unavailable" if d is None else "above baseline" if d > GAP_FLOOR
            else "below baseline" if d < -GAP_FLOOR else "as baseline")


def f1_cells(rec: Dict) -> Dict:
    return {("F1", "all"): {s: (x["mean"], f1_label(x)) for s, x in rec["by_step"].items()}}


def f12_cells(rec: Dict, base_rec: Dict) -> Dict:
    """Δ per step with its raw value and baseline beside: {step: (Δ, label)}, and ``beside``."""
    b = base_rec["by_step"].get(0, {}).get("mean")
    cells, beside = {}, {}
    for s, x in rec["by_step"].items():
        a = x.get("mean")
        d = None if a is None or b is None else round(a - b, 4)
        cells[s] = (d, gap_word(d))
        beside[s] = {"raw": a, "baseline": b}
    return {("F12", "all"): cells}, beside


def columns(data: Dict) -> Dict:
    """Per row, per column: {(quantity, layer): {step: (value, label)}}; F12 also ``beside``."""
    cols = columns_of(data.get("lead", "c3"))
    f1 = {c: f1_cells(data["recs"]["f1"][c]) for c in cols}
    f12, beside = {}, {}
    for c in cols:
        base = "c2" if c in ON_C2_BASE else c
        f12[c], beside[c] = f12_cells(data["recs"]["f12"][c], data["recs"]["f12"][base])
    park = {}
    for c in cols:
        a, g = f1[c][("F1", "all")], f12[c][("F12", "all")]
        park[c] = {("§1.5", "all"): {s: (None, "parked" if a[s][1] == "negative" and g.get(s, (0, ""))[1]
                                         == "below baseline" else "not") for s in a}}
    return {"F1": f1, "F12": f12, "§1.5": park, "beside": beside}


def window(cells: Dict, want: str, prim: Optional[Dict[int, str]] = None, col: str = "c3") -> list:
    """Steps whose label is ``want``, on ``prim``'s column per step (or ``col`` throughout)."""
    q = next(iter(cells[col]))
    return [s for s in sorted(cells[col][q])
            if cells[prim.get(s, col) if prim else col].get(q, {}).get(s, (None, ""))[1] == want]


def published_check(data: Dict, pub: Path) -> Dict:
    """c0's per-unit statistic against the published per-unit records on the units both read."""
    out = {}
    for r, fname, key in (("f1", "p10_f1_transport.json", "clustered_minus_noise_step"),
                          ("f12", "p10_f12_z.json", "clustered_minus_noise")):
        d = json.loads((pub / fname).read_text())
        old = {}
        for run in d["directories"]:
            rows = run.get("boundaries") if r == "f1" else run.get("rows")
            for u in rows or []:
                if u.get(key) is not None:
                    old[(run["run_dir"], u["layer"], u.get("beta"))] = u[key]
        c0 = data["recs"][r]["c0"]
        n = same = big = 0
        worst = 0.0
        for k, rows in c0["runs"].items():
            p = Path(c0["inputs"][k])
            for u in rows:
                o = old.get((p.name, u["layer"], u.get("beta")))
                if o is None:
                    continue
                n += 1
                same += o == u["stat"]
                worst = max(worst, abs(o - u["stat"]))
                big += abs(o - u["stat"]) > 0.01
        if n == 0:
            raise LadderError(f"{fname}: no unit of c0 matched the published record (compared 0)")
        out[READERS[r]] = {"compared": n, "identical": same, "over_0.01": big, "max_abs_diff": round(worst, 6),
                           "published": str(pub / fname)}
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", type=Path, required=True, help="the R2 records ({f1,f12}_<column>.json)")
    ap.add_argument("--labels", type=Path, required=True, help="the R0 label source they were read on")
    ap.add_argument("--published", type=Path, default=None,
                    help="dir holding p10_f1_transport.json and p10_f12_z.json: check c0 against them")
    ap.add_argument("--out", type=Path, default=None, help="default <dir>/ladder.json")
    lead_args.add_args(ap, "R2")
    args = ap.parse_args(argv)
    lead_args.reproduce_inputs(args)
    lead = args.lead
    data = load(args.dir, args.labels, lead)
    reproduces = lead_args.check_reproduces(args, data, load, reproduce, "per-step summary and per-unit rows")
    ladder = LEADS[lead][0]
    prim, prim_c3 = primary(data["summary"], lead), primary(data["summary"], "c3")
    cols = columns(data)
    res = {**lead_args.header(args.labels, data["summary_sha256"], lead, reproduces, prim, data["summary"]),
           "rows": {}}
    for row, skip0 in (("F1", False), ("F12", True), ("§1.5", True)):
        h = holds(cols[row], prim, skip0, ladder=ladder)
        res["rows"][row] = {"holds": h, "arms": arms_differ(cols[row], prim_c3),
                            "columns": {c: {s: list(v) for s, v in next(iter(cc.values())).items()}
                                        for c, cc in cols[row].items()}}
        print(f"\n{row}: {h['agree']} of {h['n']} step labels agree, primary vs c0; the rest first changed "
              f"at {h['first_changed_at'] or 'none'}, settled at {h['settled_at'] or 'none'}")
        for a, x in res["rows"][row]["arms"].items():
            print(f"  arm {a}: differs from c3 in {len(x['differ'])} of {x['n']}")
        if lead == "c3x":
            x = res["rows"][row]["c3_to_c3x"] = c3_to_c3x(cols[row], prim_c3, prim, skip0)
            print(f"  c3 → c3x: {len(x['differ'])} of {x['n']} step labels change"
                  + "".join(f"; {d['step']} {d['c3']} → {d['c3x']}" for d in x["differ"]))
    res["rows"]["F12"]["beside"] = {c: {s: v for s, v in b.items()} for c, b in cols["beside"].items()}
    res["windows"] = {
        row: {"c0": window(cols[row], want, col="c0"), "c2": window(cols[row], want, col="c2"),
              "c3": window(cols[row], want, col="c3"),
              **({lead: window(cols[row], want, col=lead)} if lead != "c3" else {}),
              "primary": window(cols[row], want, prim, col=lead)}
        for row, want in (("F1", "negative"), ("F12", "below baseline"), ("§1.5", "parked"))}
    print("\nstep  prim | F1 c0 / c2 / c3 / prim (mean, label) | F12 Δ c0 / c2 / c3 / prim (raw − baseline)")
    f1c, f12c, bes = cols["F1"], cols["F12"], cols["beside"]
    for s, pc in prim.items():
        def f1s(c):
            v = f1c[c][("F1", "all")].get(s)
            return "—" if v is None else f"{_fmt(v[0])} {v[1][:3]}"

        def f12s(c):
            v = f12c[c][("F12", "all")].get(s)
            b = bes[c].get(s)
            return "—" if v is None else f"{_fmt(v[0])} ({_fmt(b['raw'])} − {_fmt(b['baseline'])})"
        print(f"{s:>6} {pc:>3} | {f1s('c0')} / {f1s('c2')} / {f1s('c3')} / {f1s(pc)} | "
              f"{f12s('c0')} / {f12s('c2')} / {f12s('c3')} / {f12s(pc)}")
    for row, w in res["windows"].items():
        print(f"window {row}: {w}")
    if args.published is not None:
        res["published_check"] = published_check(data, args.published)
        for row, x in res["published_check"].items():
            print(f"c0 against the published {row}: {x['identical']} of {x['compared']} units identical, "
                  f"{x['over_0.01']} differ by > 0.01, max |diff| {x['max_abs_diff']}")
    out = args.out or args.dir / "ladder.json"
    out.write_text(json.dumps(res, indent=1) + "\n")
    print("\n" + lead_args.floor_line(prim, data["summary"], lead))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
