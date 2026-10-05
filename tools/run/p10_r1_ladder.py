"""Phase 10 re-read, R1: §1.7 (unique tokens), §1.9 and §1.10 read down the ladder
(`p10_cluster_function/design-10.md` "The ladder", "Row by row", "Order").

Reads the per-column records that `p10_token_composition`, `p10_comembership` and
`p10_lexical_carry` write under ``--labels <R0 label source> --column <c>``
(``<dir>/{tc,cm,lc}_<column>.json``) and applies each row's own rule, unchanged, to
every column:

====== ====================================================================
§1.7   per step, the unique-token contrasts (``freq``, ``class``), each
       ``+`` / ``0`` / ``−`` at ±``CONTRAST_FLOOR``, and the trash-collection
       verdict (`p10_token_composition.contrast_verdict`)
§1.9   per step, layer 12, 24 and the mean over L1–24: Δ = lift(step) −
       baseline, "above / as / below step 0" at ±``DELTA_FLOOR``, for the
       all-positions draw (pre-stated); ``emb_pct`` (frozen frame) not read
§1.10  as §1.9 for ``class_given_emb`` (10 bins), ``emb_given_class``,
       ``emb_same``, ``emb_cross``; carry's gap (focal − unclustered
       ``self_pct``) as a level, ±0.05; and the deciding post hoc reading,
       ``class_given_emb_40`` minus its kNN lexical control, ±0.05
====== ====================================================================

THE BASELINE (`design-10.md`, after `/challenge-pr` on #144): c3 and every column built
on it (the arms, the learned split) take **c2's step-0 lift**; c0 to c2 take their own
step 0, so c0 stays the old reading on the re-read's inputs. c3's own step-0 value is
reported as the floor reading beside the count (62 records).

THE PRIMARY COLUMN per step: c3, unless fewer than half its (prompt, layer) records are
readable there, when the step is read on c2 and labelled so (`design-10.md` "Readable").
**A row holds** where the primary's label equals c0's; where it does not, the first
column (c0f, c1, c1c, c2a, c2b, c2, c3) whose label differs from c0's is named, and
beside it (added at R1: c1, the raw frame, swings and later columns swing back) the
column the label **settled at**, the first from which every column through the primary
carries the primary's label. Step 0 is not counted for the Δ rows, where it is 0 by
construction in both.

Checks before any reading: every record names the same label source and summary
sha256, and c3's readable counts equal the label source's own ``summary.json``.

Tier 1: exploratory, unregistered. Run:
    python tools/run/p10_r1_ladder.py --dir <R1 out> --labels <R0 labels>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Sequence

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))

from tools.run.p10_comembership import DELTA_FLOOR, NOT_READ, PROPS
from tools.run.p10_token_composition import CONTRAST_FLOOR

LADDER = ("c0", "c0f", "c1", "c1c", "c2a", "c2b", "c2", "c3")
ARMS = ("c3_c4", "c3_r2")
LEARNED = ("c3_learned", "c3_unlearned")
ON_C2_BASE = ("c3", *ARMS, *LEARNED)
LAYERS = (12, 24, "mean")
READERS = {"tc": "§1.7", "cm": "§1.9", "lc": "§1.10"}
CM_READ = tuple(p for p in PROPS if p not in NOT_READ)
LC_DELTA = ("class_given_emb", "emb_given_class", "emb_same", "emb_cross")
LEVEL_FLOOR = 0.05   # §1.10's carry gap and kNN excess, as its docstring and status-10 §1.10


class LadderError(RuntimeError):
    pass


def _keys(x):
    """JSON's string keys back to ints where they are ints."""
    if isinstance(x, dict):
        return {(int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k): _keys(v) for k, v in x.items()}
    return x


def sign(x: Optional[float], floor: float) -> str:
    return "n/a" if x is None else "+" if x > floor else "−" if x < -floor else "0"


def delta_word(d: Optional[float]) -> str:
    return ("unavailable" if d is None else "above step 0" if d > DELTA_FLOOR
            else "below step 0" if d < -DELTA_FLOOR else "as step 0")


def load(dirpath: Path, labels: Path) -> Dict:
    recs: Dict = {}
    want_sha = None
    for f in sorted(dirpath.glob("*_*.json")):
        r, _, col = f.stem.partition("_")
        if r not in READERS:
            continue
        d = json.loads(f.read_text())
        ls = d["label_source"]
        if Path(ls["labels"]) != labels or ls["column"] != col:
            raise LadderError(f"{f.name}: read {ls['labels']} column {ls['column']}")
        want_sha = want_sha or ls["summary_sha256"]
        if ls["summary_sha256"] != want_sha:
            raise LadderError(f"{f.name}: another label source summary ({ls['summary_sha256']})")
        recs.setdefault(r, {})[col] = {"summary": _keys(d["summary"]),
                                       "records": _keys(d["records_readable"])}
    for r in READERS:
        miss = [c for c in (*LADDER, *ARMS, *LEARNED) if c not in recs.get(r, {})]
        if miss:
            raise LadderError(f"{r}: no record for {miss}")
    summ = json.loads((labels / "summary.json").read_text())
    for r in READERS:
        for s, rec in recs[r]["c3"]["records"].items():
            src = summ[f"step{s}"]["columns"]["c3"]
            if (rec["n"], rec["readable"]) != (src["records"], src["readable"]):
                raise LadderError(f"{r} c3 step {s}: readable {rec} against the source's {src}")
    return {"recs": recs, "summary": summ, "summary_sha256": want_sha}


def primary(recs: Dict) -> Dict[int, str]:
    return {s: ("c3" if 2 * r["readable"] >= r["n"] else "c2") for s, r in recs["tc"]["c3"]["records"].items()}


# ---------------------------------------------------------------------------
# One row's cells: {(quantity, layer): {step: (value, label)}} per column
# ---------------------------------------------------------------------------

def tc_cells(col_rec: Dict) -> Dict:
    out: Dict = {}
    for s, c in col_rec["summary"]["by_step"].items():
        u = c["unique"]
        out.setdefault(("freq", "all"), {})[s] = (u["freq_contrast"], sign(u["freq_contrast"], CONTRAST_FLOOR))
        out.setdefault(("class", "all"), {})[s] = (u["class_contrast"], sign(u["class_contrast"], CONTRAST_FLOOR))
        out.setdefault(("verdict", "all"), {})[s] = (None, u["trash_collection"])
    return out


def _lift(summary: Dict, s: int, L, p: str, reader: str) -> Optional[float]:
    by = (summary["all"] if reader == "cm" else summary).get(s, {})
    x = by.get(L, {}).get(p)
    if x is None:
        return None
    return x["lift"] if reader == "cm" else x


def delta_cells(reader: str, col_rec: Dict, base_rec: Dict) -> Dict:
    props = CM_READ if reader == "cm" else LC_DELTA
    summ, base = col_rec["summary"], base_rec["summary"]
    steps = summ["all"] if reader == "cm" else summ
    out: Dict = {}
    for p in props:
        for L in LAYERS:
            b = _lift(base, 0, L, p, reader)
            for s in steps:
                a = _lift(summ, s, L, p, reader)
                d = None if a is None or b is None else a - b
                out.setdefault((p, L), {})[s] = (d, delta_word(d))
    if reader == "lc":
        for s, by in summ.items():
            for L in LAYERS:
                x = by.get(L, {})
                gap = (x.get("carry_gap") or {}).get("self_pct")
                out.setdefault(("carry_gap", L), {})[s] = (gap, sign(gap, LEVEL_FLOOR))
                o, k = x.get("class_given_emb_40"), x.get("class_given_emb_40_knn")
                ex = None if o is None or k is None else o - k
                out.setdefault(("cge40_over_knn", L), {})[s] = (ex, sign(ex, LEVEL_FLOOR))
    return out


def row_cells(data: Dict, reader: str) -> Dict[str, Dict]:
    recs = data["recs"][reader]
    cols = {}
    for col in (*LADDER, *ARMS, *LEARNED):
        if reader == "tc":
            cols[col] = tc_cells(recs[col])
        else:
            cols[col] = delta_cells(reader, recs[col], recs["c2" if col in ON_C2_BASE else col])
    return cols


def holds(cols: Dict, prim: Dict[int, str], reader: str) -> Dict:
    """Per (quantity, layer, step): primary vs c0, and the first column that changed it."""
    agree, cells, changed, settled = 0, [], Counter(), Counter()
    for q, by_step in cols["c0"].items():
        for s, (_, lab0) in by_step.items():
            if reader != "tc" and s == 0:
                continue
            pc = prim.get(s, "c3")
            lab = cols[pc].get(q, {}).get(s, (None, "absent"))[1]
            if lab == lab0:
                agree += 1
                continue
            labs = [cols[c].get(q, {}).get(s, (None, "absent"))[1] for c in LADDER]
            first = LADDER[next(i for i, x in enumerate(labs) if x != lab0)]
            # c0's label differs from the primary's, so some column at or before the primary does
            stay = LADDER[1 + max(i for i in range(LADDER.index(pc) + 1) if labs[i] != lab)]
            changed[first] += 1
            settled[stay] += 1
            cells.append({"quantity": q[0], "layer": q[1], "step": s, "primary": pc, "c0": lab0,
                          "label": lab, "first_changed_at": first, "settled_at": stay,
                          "ladder": dict(zip(LADDER, labs))})
    return {"n": agree + len(cells), "agree": agree, "first_changed_at": dict(changed),
            "settled_at": dict(settled), "differ": cells}


def arms_differ(cols: Dict, prim: Dict[int, str]) -> Dict:
    """Cells where an arm's label differs from c3's, at steps read on c3 (design: quoted only there)."""
    out = {}
    for arm in ARMS:
        n, diff = 0, []
        for q, by_step in cols["c3"].items():
            for s, (_, lab) in by_step.items():
                if prim.get(s) != "c3":
                    continue
                n += 1
                alab = cols[arm].get(q, {}).get(s, (None, "absent"))[1]
                if alab != lab:
                    diff.append({"quantity": q[0], "layer": q[1], "step": s, "c3": lab, arm: alab})
        out[arm] = {"n": n, "differ": diff}
    return out


def primary_labels(cols: Dict, prim: Dict[int, str]) -> Dict:
    return {(q, s): cols[prim.get(s, "c3")].get(q, {}).get(s, (None, "absent"))[1]
            for q, by in cols["c0"].items() for s in by}


def against(data: Dict, other: Dict) -> Dict:
    """Per row, the primary labels that differ between two R1 runs on the same source
    (e.g. another BLAS thread count): the size of that run-to-run noise at the reading."""
    prim = primary(data["recs"])
    out = {}
    for reader, row in READERS.items():
        a = primary_labels(row_cells(data, reader), prim)
        b = primary_labels(row_cells(other, reader), primary(other["recs"]))
        diff = sorted(f"{q[0]}|{q[1]}|{st}" for (q, st) in a if a[(q, st)] != b.get((q, st)))
        out[row] = {"n": len(a), "differ": diff}
    return out


def _fmt(v) -> str:
    return "—" if v is None else f"{v:+.2f}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", type=Path, required=True, help="the R1 records ({tc,cm,lc}_<column>.json)")
    ap.add_argument("--labels", type=Path, required=True, help="the R0 label source they were read on")
    ap.add_argument("--out", type=Path, default=None, help="default <dir>/ladder.json")
    ap.add_argument("--against", type=Path, default=None,
                    help="another R1 run's records on the same source: count primary labels that differ")
    args = ap.parse_args(argv)
    data = load(args.dir, args.labels)
    prim = primary(data["recs"])
    floor = data["summary"]["step0"]["columns"]["c3"]
    res = {"label_source": str(args.labels), "summary_sha256": data["summary_sha256"],
           "primary": prim, "floor": {"c3_group_layer_records_step0": floor["groups"],
                                      "c3_readable_step0": [floor["readable"], floor["records"]]},
           "rows": {}}
    for reader, row in READERS.items():
        cols = row_cells(data, reader)
        h = holds(cols, prim, reader)
        res["rows"][row] = {"holds": h, "arms": arms_differ(cols, prim),
                            "columns": {c: {f"{q[0]}|{q[1]}": {s: list(v) for s, v in by.items()}
                                            for q, by in cc.items()} for c, cc in cols.items()}}
        print(f"\n{row}: {h['agree']} of {h['n']} labels agree, primary vs c0; the rest first changed at "
              f"{h['first_changed_at'] or 'none'}, settled at {h['settled_at'] or 'none'}")
        for a, x in res["rows"][row]["arms"].items():
            print(f"  arm {a}: differs from c3 in {len(x['differ'])} of {x['n']}")
    if args.against is not None:
        res["against"] = {"dir": str(args.against), "rows": against(data, load(args.against, args.labels))}
        for row, x in res["against"]["rows"].items():
            print(f"against {args.against.name}: {row} {len(x['differ'])} of {x['n']} primary labels differ")
    out = args.out or args.dir / "ladder.json"
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"\nprimary: c2 at steps {[s for s, c in prim.items() if c == 'c2']}, c3 elsewhere; floor "
          f"{floor['groups']} c3 records at step 0 ({floor['readable']} of {floor['records']} readable)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
