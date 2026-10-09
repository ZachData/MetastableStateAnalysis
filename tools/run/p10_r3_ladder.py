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
holds every step (the learned split: 143000 only), and c3's readable counts equal the source's
(and the lead's). ``--published <file>``: c0's per-unit values against the published A0 record
by run-dir name and layer (that record read the earlier WDS sweep, 2026-08-31 dirs, on 8 prompts;
units both read).

``--lead c3x`` (R9, `design-10.md` "R9"): c3x is appended to the ladder after c3 and is the
primary column (c2 where under half its records are readable), its learned split beside; the
rule is still judged against c1. Beside (``c3_to_c3x``): the three labels on the primary with c3
leading against c3x leading, each column's own three labels, and the per-step residual labels
where c3x's differs from c3's at steps both read on their own column, with the corrected gaps.
The arms stay c3's. ``--reproduce <R3 record> --reproduce-labels <its source>``: refuse unless
every c0–c3 column (and c3's arms and split) has the same readable counts, per-step summary and
per-unit rows as that run's (R9's first check; `p10_r9_lead`).

Tier 1: exploratory, unregistered. Run:
    python tools/run/p10_r3_ladder.py --record <a0.json> --labels <R0 labels> \
        [--published data/analysis/p10_row_a0.json] \
        [--lead c3x --reproduce <R3 a0.json> --reproduce-labels <R0 labels>]
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

from tools.run import p10_r9_lead as lead_args
from tools.run.p10_r1_ladder import ARMS, LADDER, LEADS, LEARNED, LadderError, _fmt
from tools.run.p10_r2_ladder import columns_of, primary

SHARE_BAR = 0.9   # placed (design-10.md "Row by row", A0)
RESIDUAL = 0.05   # placed, as F12's and §1.9's floors
COLUMNS = (*LADDER, *ARMS, *LEARNED)
RECORD_KEYS = ("records_readable", "summary", "runs")
RESAMPLES = 2000  # prompt bootstrap draws beside c3 → c3x (`/challenge-pr` on #171, finding 1)
PUBLISHED_FIELDS = ("raw_noise", "raw_clustered", "corrected_noise", "corrected_clustered",
                    "position_bias", "n_tokens", "noise_fraction")


def load(record: Path, labels: Path, lead: str = "c3") -> Dict:
    d = json.loads(Path(record).read_text())
    ls = d["label_source"]
    if Path(ls["labels"]) != Path(labels):
        raise LadderError(f"{record}: read {ls['labels']}, not {labels}")
    sha = hashlib.sha256((Path(labels) / "summary.json").read_bytes()).hexdigest()[:16]
    if ls["summary_sha256"] != sha:
        raise LadderError(f"{record}: label source summary {ls['summary_sha256']}, on disk {sha}")
    summ = json.loads((Path(labels) / "summary.json").read_text())
    steps = {int(k.removeprefix("step")) for k in summ if k.startswith("step")}
    missing = set(columns_of(lead)) - set(d["columns"])
    if missing:
        raise LadderError(f"{record}: no column {sorted(missing)}")
    for col in columns_of(lead):
        got = {int(s) for s in d["columns"][col]["records_readable"]}
        want = {max(steps)} if col in LEADS[lead][1] else steps
        if got != want:
            raise LadderError(f"{col}: steps {sorted(got ^ want)} missing or extra")
    for col in sorted({"c3", lead}):
        for s, rec in d["columns"][col]["records_readable"].items():
            src = summ[f"step{s}"]["columns"][col]
            if (rec["n"], rec["readable"]) != (src["records"], src["readable"]):
                raise LadderError(f"{col} step {s}: readable {rec} against the source's {src}")
    return {"record": d, "summary": summ, "lead": lead}


def reproduce(data: Dict, other: Dict) -> list:
    """c0–c3 columns (and c3's arms and split) whose readable counts, per-step summary or per-unit
    rows differ from another R3 run's; [] if none."""
    a, b = data["record"]["columns"], other["record"]["columns"]
    return [col for col in COLUMNS if any(a[col][k] != b[col][k] for k in RECORD_KEYS)]


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
    """Every column read; ``primary`` pools the lead's units (c2's under half readable), and with
    c3x leading ``primary_c3`` is the same pooling with c3 leading."""
    lead = data.get("lead", "c3")
    cols = data["record"]["columns"]
    units = {c: units_by_step(cols[c]["runs"]) for c in columns_of(lead)}
    out = {c: read_column(units[c]) for c in columns_of(lead)}
    prim = primary(data["summary"], lead)
    out["primary"] = read_column({s: units[prim[s]].get(s, []) for s in prim})
    res = {"columns": out, "primary": prim}
    if lead != "c3":
        p3 = res["primary_c3"] = primary(data["summary"], "c3")
        out["primary_c3"] = read_column({s: units[p3[s]].get(s, []) for s in p3})
    return res


RULE_LABELS = ("share", "appears", "persists")


def holds(cols: Dict, prim: Dict[int, str], ladder: Sequence[str] = LADDER) -> Dict:
    """The three labels, primary vs c1; per step, the residual label vs c1 and the first column
    after c1 that changed it."""
    after_c1 = ladder[ladder.index("c1"):]
    p, c1 = cols["primary"], cols["c1"]
    rule = {k: {"c1": c1[k], "primary": p[k], "same": c1[k] == p[k]} for k in RULE_LABELS}
    differ = []
    for s, lab1 in c1["residual"].items():
        lab = p["residual"].get(s)
        if lab == lab1:
            continue
        labs = {c: cols[c]["residual"].get(s) for c in after_c1}
        first = next(c for c in after_c1 if labs[c] != lab1)
        differ.append({"step": s, "primary": prim[s], "c1": lab1, "label": lab, "first_changed_at": first,
                       "ladder": labs})
    n = len(c1["residual"])
    return {"rule": rule, "holds": all(x["same"] for x in rule.values()),
            "steps": {"n": n, "agree": n - len(differ), "differ": differ}}


def arms_and_learned(cols: Dict, prim: Dict[int, str], learned: Sequence[str] = LEARNED) -> Dict:
    """Arms against c3 at the steps c3 leads (``prim``: c3's primary); the learned split at 143000."""
    out = {}
    for arm in ARMS:
        d = [s for s, c in prim.items() if c == "c3" and cols[arm]["residual"].get(s) != cols["c3"]["residual"][s]]
        out[arm] = {"n": sum(c == "c3" for c in prim.values()), "differ": d,
                    "share": cols[arm]["share"], "appears": cols[arm]["appears"], "persists": cols[arm]["persists"]}
    s = max(prim)
    out["learned_at_143000"] = {c: cols[c]["per_step"].get(s) for c in learned}
    return out


def prompt_resample(runs: Dict, steps: Sequence[int], n: int = RESAMPLES, seed: int = 0) -> Dict:
    """Per step, how far 7 prompts settle the residual label: how many leave-one-prompt-out pools
    flip it, and the share of prompt bootstrap pools (``n`` draws, seeded) that read residual."""
    rng = np.random.default_rng(seed)
    out = {}
    for s in steps:
        by: Dict[str, List[Dict]] = {}
        for key, rows in runs.items():
            st, prompt = key.split("|")
            if int(st) == s and rows:
                by.setdefault(prompt, []).extend(rows)
        ps = sorted(by)
        lab = residual_label(gaps([r for q in ps for r in by[q]]))
        loo = sum(residual_label(gaps([r for q in ps if q != p for r in by[q]])) != lab for p in ps)
        boot = [residual_label(gaps([r for q in rng.choice(ps, len(ps)) for r in by[q]])) == "residual"
                for _ in range(n)]
        out[s] = {"prompts": len(ps), "label": lab, "loo_flips": int(loo), "p_residual": round(float(np.mean(boot)), 2)}
    return out


def c3_to_c3x(cols: Dict, prim_c3: Dict[int, str], prim_x: Dict[int, str]) -> Dict:
    """What the column changed: the three labels on the primary (c3 leading against c3x leading)
    and on each column; per step, the residual labels and corrected gaps where both lead."""
    p3, px = cols["primary_c3"], cols["primary"]
    rule = {k: {"c3_leading": p3[k], "c3x_leading": px[k], "c3": cols["c3"][k], "c3x": cols["c3x"][k]}
            for k in RULE_LABELS}
    both = [s for s in prim_x if prim_c3[s] == "c3" and prim_x[s] == "c3x"]
    steps = {s: {"c3": cols["c3"]["residual"][s], "c3x": cols["c3x"]["residual"][s],
                 "c3_gap": cols["c3"]["per_step"][s]["corrected_gap"],
                 "c3x_gap": cols["c3x"]["per_step"][s]["corrected_gap"]} for s in both}
    moves = [abs(x["c3x_gap"] - x["c3_gap"]) for x in steps.values()
             if x["c3_gap"] is not None and x["c3x_gap"] is not None]
    return {"rule": rule, "rule_differ": [k for k, x in rule.items() if x["c3_leading"] != x["c3x_leading"]],
            "n": len(both), "differ": [{"step": s, **x} for s, x in steps.items() if x["c3"] != x["c3x"]],
            "steps": steps, "max_gap_move": round(max(moves), 4) if moves else None}


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
    lead_args.add_args(ap, "R3")
    args = ap.parse_args(argv)
    lead_args.reproduce_inputs(args)
    lead = args.lead
    ladder, learned = LEADS[lead]
    data = load(args.record, args.labels, lead)
    reproduces = lead_args.check_reproduces(args, data, load, reproduce,
                                            "readable counts, per-step summary and per-unit rows")
    r = read_all(data)
    cols, prim = r["columns"], r["primary"]
    prim_c3 = r.get("primary_c3", prim)
    h = holds(cols, prim, ladder)
    res = {"record": str(args.record),
           **lead_args.header(args.labels, data["record"]["label_source"]["summary_sha256"], lead, reproduces,
                              prim, data["summary"]),
           "share_bar": SHARE_BAR, "residual_floor": RESIDUAL, "holds": h,
           "arms_learned": arms_and_learned(cols, prim_c3, learned),
           "columns": {c: {**v, "per_step": {str(s): g for s, g in v["per_step"].items()},
                           "residual": {str(s): x for s, x in v["residual"].items()}} for c, v in cols.items()}}
    if lead == "c3x":
        x = res["c3_to_c3x"] = c3_to_c3x(cols, prim_c3, prim)
        rec = data["record"]["columns"]
        x["prompts"] = {c: prompt_resample(rec[c]["runs"], sorted(x["steps"])) for c in ("c3", "c3x")}
    print("column   | sweep raw / corrected gap, share → label | appears | persists")
    for c in (*columns_of(lead), "primary", *(("primary_c3",) if lead != "c3" else ())):
        v = cols[c]
        g = v["sweep"]
        print(f"{c:<12} | {_fmt(g['raw_gap'])} / {_fmt(g['corrected_gap'])}, {_fmt(g['share'])} → {v['share']:<8}"
              f" | {v['appears']:<10} | {v['persists']}")
    print(f"\nstep  prim | corrected gap (raw gap) c0 / c1 / c2 / {'c3 / ' if lead != 'c3' else ''}primary"
          " | primary median p, position bias")
    for s, pc in prim.items():
        def cell(c, s=s):
            g = cols[c]["per_step"].get(s)
            return "—" if not g or g["n"] == 0 else f"{_fmt(g['corrected_gap'])} ({_fmt(g['raw_gap'])})"
        g = cols["primary"]["per_step"][s]
        mid = f"{cell('c3')} / " if lead != "c3" else ""
        print(f"{s:>6} {pc:>3} | {cell('c0')} / {cell('c1')} / {cell('c2')} / {mid}{cell('primary')} | "
              f"{_fmt(g['median_p'])}, {_fmt(g['position_bias'])}")
    print(f"\nholds (primary vs c1): {h['holds']}; " + "; ".join(
        f"{k}: c1 {x['c1']}, primary {x['primary']}" for k, x in h["rule"].items()))
    print(f"per-step residual labels: {h['steps']['agree']} of {h['steps']['n']} agree; first changed at "
          f"{sorted({x['first_changed_at'] for x in h['steps']['differ']})}")
    for a, x in res["arms_learned"].items():
        print(f"  {a}: {x}")
    if lead == "c3x":
        x = res["c3_to_c3x"]
        print("c3 → c3x: " + "; ".join(f"{k}: c3 leading {v['c3_leading']}, c3x leading {v['c3x_leading']} "
                                         f"(columns {v['c3']} / {v['c3x']})" for k, v in x["rule"].items()))
        print(f"  per-step residual labels: {len(x['differ'])} of {x['n']} change"
              + "".join(f"; {d['step']} {d['c3']} → {d['c3x']} ({_fmt(d['c3_gap'])} → {_fmt(d['c3x_gap'])})"
                        for d in x["differ"]) + f"; corrected gaps move ≤ {x['max_gap_move']}")
        for c, by in x["prompts"].items():
            print(f"  {c} per step, leave-one-prompt-out flips / bootstrap P(residual): "
                  + " ".join(f"{s}:{v['loo_flips']}/{v['prompts']},{v['p_residual']:.2f}" for s, v in by.items()))
        print(lead_args.floor_line(prim, data["summary"], lead))
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
