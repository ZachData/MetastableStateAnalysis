"""
p1d_cluster_ensemble/candidates.py — the candidate definition of the
Blocked 11⁗ programme: unit 1's groups that move ∩ unit 2's learned,
replicating groups, with a per-layer origin from L0 (`design-1d.md` "The
candidates, and where each comes from"; readings fixed there before any run).

Inputs, on **one token set per prompt** (unit 2's trained union,
``<unit2>/step0/token_sets.json``):

- unit 1 re-run on that set (`move_text run --kept-from`), both steps: each
  P = 0 group's class (`move_text.group_classes`);
- unit 2's records (`arch_null`): each seed-0 group's ``learned`` and
  ``replicates`` (recomputed here with `arch_null`'s own functions; at
  step143000 checked against the stored ``trained.json``);
- L0 (`l0`): seed 0's level-set groups on ``hidden_states[0]``, the
  embedding output, plus L1 from the same forward pass as a consistency
  check against unit 1's.

`read` refuses if a record's unit 1 and unit 2 group lists are not the same
member sets, if L1 from `l0` differs from unit 1's, or if any input was run
on another token set. Per seed-0 group it writes the class, the status, bulk,
and the origin: the smallest ℓ₀ with the group present (best Jaccard ≥
``PRESENT_JACCARD``) at every layer from ℓ₀ to its own. **Candidate** =
moves ∧ learned ∧ replicates ∧ not bulk.

`definitions` counts, from `read`'s rows, each group set Phase 10 could be
re-read on (``DEFINITIONS``; Blocked 22 decided (a), ``moves``), per step.

Tier 1: exploratory, unregistered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import arch_null as an
from .move_text import (CLASSES, FIXED_BAR, FLOOR_ZERO, FRAMES, LAYERS, MIN_CLUSTER_SIZES,
                        V1_PASSAGES, band, best_jaccard, forward, group_classes, groups_of)

STEPS = ("step0", an.TRAINED_STEP)
#: "Present at a layer", placed (`design-1d.md`): replication's bar.
PRESENT_JACCARD = an.REPLICATE_JACCARD
#: Fewer distinct centred size-2 candidates than this is "few", placed.
FEW = 20
STATUSES = ("learned+replicates", "learned only", "not learned", "refused")
UNIT1_CLASSES = (*CLASSES, "unstable", FLOOR_ZERO)
ORIGINS = ("carried", "formed at L1", "formed later")


# ---------------------------------------------------------------------------
# Pure pieces
# ---------------------------------------------------------------------------

def origin(members: Sequence[int], by_layer: Dict[int, Sequence[Sequence[int]]], layer: int) -> Dict:
    """
    The group at ``layer`` against every layer's groups in ``by_layer``:
    ``origin`` (smallest ℓ₀ with best Jaccard >= `PRESENT_JACCARD` at every
    layer ℓ₀..``layer``), the same with identical member sets
    (``origin_identical``), the first layer present at all, and the last
    layer of the run forward from ``layer``.
    """
    g = frozenset(int(x) for x in members)
    sets = {L: [frozenset(int(x) for x in h) for h in gs] for L, gs in by_layer.items()}
    near = {L: best_jaccard(g, gs) >= PRESENT_JACCARD for L, gs in sets.items()}
    same = {L: g in gs for L, gs in sets.items()}
    if not (near.get(layer) and same.get(layer)):
        raise ValueError(f"the group is not among layer {layer}'s own groups")

    def back(p: Dict[int, bool]) -> int:
        lo = layer
        while p.get(lo - 1):
            lo -= 1
        return lo

    hi = layer
    while near.get(hi + 1):
        hi += 1
    return {"origin": back(near), "origin_identical": back(same),
            "first_present": min(L for L, v in near.items() if v), "last_forward": hi}


def origin_class(l0: int) -> str:
    return "carried" if l0 == 0 else "formed at L1" if l0 == 1 else "formed later"


def status(learned: Optional[bool], replicates: bool) -> str:
    if learned is None:
        return "refused"
    return "learned+replicates" if learned and replicates else "learned only" if learned else "not learned"


def same_groups(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> bool:
    """Whether two group lists hold the same member sets (order free)."""
    return Counter(frozenset(map(int, g)) for g in a) == Counter(frozenset(map(int, g)) for g in b)


def outcome(distinct: Sequence[Dict]) -> str:
    """`design-1d.md`'s outcome row for the distinct centred size-2 candidates."""
    if len(distinct) < FEW:
        return "few candidates"
    c = Counter(origin_class(d["origin"]) for d in distinct)
    for k, name in (("carried", "most carried"), ("formed at L1", "most formed at L1"),
                    ("formed later", "most formed later")):
        if c[k] > len(distinct) / 2:
            return name
    return "mixed origins"


# ---------------------------------------------------------------------------
# L0: seed 0's groups on the embedding output (and L1, the consistency check)
# ---------------------------------------------------------------------------

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def l0_cmd(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="candidates l0")
    ap.add_argument("--token-sets", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--torch-threads", type=int, default=None)
    args = ap.parse_args(argv)
    import torch
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    sets = json.loads(args.token_sets.read_text())["sets"]
    tok = an._tok()
    ids = an.prompt_ids(tok)
    args.out.mkdir(parents=True, exist_ok=True)
    for step in STEPS:
        f = args.out / f"l0_{step}.json"
        if f.exists():
            print(f"already {f}", flush=True)
            continue
        model = an.load("init:0", step)
        res = {}
        for k in V1_PASSAGES:
            H, _ = forward(model, ids[k])
            kept = np.asarray(sets[k]["kept"])
            res[k] = {f"{fr}|{m}|{L}": [kept[g].tolist() for g in groups_of(H[L][kept], fr, m)[1]]
                      for fr in FRAMES for m in MIN_CLUSTER_SIZES for L in (0, 1)}
        del model
        f.write_text(json.dumps({"git": an._git_head(), "step": step, "model": "init:0",
                                 "token_sets_sha256": _sha(args.token_sets), "inputs": an.input_names(ids),
                                 "groups": res}) + "\n")
        print(f"done {f}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------

def _refuse(msg: str) -> int:
    print(f"refusing: {msg}", file=sys.stderr)
    return 1


def unit2_status(unit2: Path, step: str) -> Tuple[Dict[Tuple, Dict], int]:
    """Seed 0's groups at ``step`` by (prompt, layer, frame, size, members): learned, replicates."""
    inits, reinits = an.model_ids("init"), an.model_ids("reinit")
    seeds = [int(m.split(":")[1]) for m in inits]
    recs0 = an.load_records(unit2, "step0", inits + reinits, "trained")
    if len(recs0) != len(V1_PASSAGES) * (len(inits) + len(reinits)):
        raise SystemExit(f"refusing: {len(recs0)} step-0 records")
    fc = json.loads((unit2 / "first_check.json").read_text())
    if fc.get("union") != "trained":
        raise SystemExit("refusing: unit 2's first check is not on the trained union")
    failed = an.failed_cells(fc["rows"])
    bars = an.group_bars(recs0, reinits)
    recs = (an.load_records(unit2, step, inits, "trained") if step != "step0"
            else [r for r in recs0 if r["model"] in inits])
    if len(recs) != len(V1_PASSAGES) * len(inits):
        raise SystemExit(f"refusing: {len(recs)} {step} records")
    rows = an.group_rules(recs, bars, failed)
    rep = an.replication(rows, seeds)
    repl = {(r["prompt"], r["layer"], r["frame"], r["size"], frozenset(r["members"])): r["replicates"]
            for r in rep}
    out = {}
    for r in rows:
        if r["seed"] != 0:
            continue
        key = (r["prompt"], r["layer"], r["frame"], r["size"], frozenset(r["members"]))
        out[key] = {"learned": r["learned"], "replicates": bool(repl.get(key, False)), "s": r["s"],
                    "bar": r["bar"]}
    return out, sum(r["replicates"] for r in rep)


def distinct_groups(rows) -> List[Dict]:
    """One entry per member set per (prompt, frame, size), at its smallest origin."""
    distinct: Dict[Tuple, Dict] = {}
    for r in rows:
        key = (r["prompt"], r["frame"], r["size"], tuple(sorted(r["members"])))
        d = distinct.setdefault(key, {"prompt": r["prompt"], "frame": r["frame"], "size": r["size"],
                                      "members": sorted(r["members"]), "tokens": r["tokens"],
                                      "layers": [], "origin": r["origin"]})
        d["layers"].append(r["layer"])
        d["origin"] = min(d["origin"], r["origin"])
    dist = list(distinct.values())
    for d in dist:
        d["origin_class"] = origin_class(d["origin"])
    return dist


#: The group sets Phase 10 could be re-read on (`STATE.md` Blocked 22), as filters
#: on a non-bulk seed-0 row. (a) is the decided one (`design-1d.md` "The working
#: definition"); the others are beside it.
DEFINITIONS = {
    "moves": lambda r: r["class"] == "moves",
    "moves+learned": lambda r: r["class"] == "moves" and r["status"] in ("learned+replicates", "learned only"),
    "candidate": lambda r: r["class"] == "moves" and r["status"] == "learned+replicates",
    "all": lambda r: True,
}


def definition_counts(rows: Sequence[Dict]) -> Dict:
    """Per definition and (frame, size): group-layer records by band, distinct groups by origin."""
    out: Dict[str, Dict] = {}
    for name, keep in DEFINITIONS.items():
        for cell in sorted({(r["frame"], r["size"]) for r in rows}):
            kept = [r for r in rows if (r["frame"], r["size"]) == cell and not r["bulk"] and keep(r)]
            out.setdefault(name, {})[f"{cell[0]}/{cell[1]}"] = {
                "records": len(kept),
                "records_by_band": dict(sorted(Counter(band(r["layer"]) for r in kept).items())),
                "distinct": dict(Counter(d["origin_class"] for d in distinct_groups(kept))),
                "n_distinct": len(distinct_groups(kept))}
    return out


def definitions_cmd(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="candidates definitions")
    ap.add_argument("--rows", type=Path, required=True, help="`read`'s candidate_rows.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    rows = json.loads(args.rows.read_text())
    if sorted(rows) != sorted(STEPS) or not all(rows.values()):
        return _refuse(f"{args.rows} does not hold rows for both steps {STEPS}")
    res = {"git": an._git_head(), "rows": {"path": str(args.rows), "sha256": _sha(args.rows)},
           "steps": {s: definition_counts(rows[s]) for s in STEPS}}
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    for s, v in res["steps"].items():
        for name, cells in v.items():
            for cell, c in cells.items():
                print(f"{s:10s} {name:14s} {cell:10s} records {c['records']:5d} "
                      f"{json.dumps(c['records_by_band'])} distinct {c['n_distinct']:5d} {json.dumps(c['distinct'])}")
    return 0


def tables(rows: Sequence[Dict]) -> Dict:
    """Cross table, origin tables and distinct candidates, per (frame, size, band)."""
    cross, bulk, orig = {}, {}, {}
    for r in rows:
        cell = f"{r['frame']}/{r['size']}/{band(r['layer'])}"
        tgt = bulk if r["bulk"] else cross
        tgt.setdefault(cell, Counter())[f"{r['class']} | {r['status']}"] += 1
        if r["bulk"]:
            continue
        o = orig.setdefault(cell, {"candidates": Counter(), "learned+replicates": Counter(), "all": Counter(),
                                   "candidate_origin_hist": Counter()})
        o["all"][r["origin_class"]] += 1
        if r["status"] == "learned+replicates":
            o["learned+replicates"][r["origin_class"]] += 1
        if r["candidate"]:
            o["candidates"][r["origin_class"]] += 1
            o["candidate_origin_hist"][r["origin"]] += 1
    dist = distinct_groups(r for r in rows if r["candidate"])
    by_cell: Dict[str, Counter] = {}
    for d in dist:
        by_cell.setdefault(f"{d['frame']}/{d['size']}", Counter())[d["origin_class"]] += 1
    return {"cross": {k: dict(v) for k, v in sorted(cross.items())},
            "cross_bulk": {k: dict(v) for k, v in sorted(bulk.items())},
            "origin": {k: {x: dict(y) for x, y in v.items()} for k, v in sorted(orig.items())},
            "distinct_by_frame_size": {k: dict(v) for k, v in sorted(by_cell.items())},
            "distinct": dist}


def read_step(step: str, unit1: Path, unit2: Path, l0dir: Path, sets: Dict, ts_sha: str,
              tokens: Dict[str, List[str]]) -> Dict:
    u1 = [json.loads((unit1 / step / f"{k}.json").read_text()) for k in V1_PASSAGES
          if (unit1 / step / f"{k}.json").exists()]
    if len(u1) != len(V1_PASSAGES):
        raise SystemExit(f"refusing: {len(u1)} unit 1 records at {step}")
    for r in u1:
        kf = r["meta"].get("kept_from") or {}
        if kf.get("sha256") != ts_sha or r["kept_offsets"] != sets[r["passage"]]["kept"]:
            raise SystemExit(f"refusing: unit 1 {step}/{r['passage']} is not on unit 2's token set")
    l0 = json.loads((l0dir / f"l0_{step}.json").read_text())
    if l0["token_sets_sha256"] != ts_sha:
        raise SystemExit(f"refusing: l0_{step}.json is on another token set")
    u2, n_repl = unit2_status(unit2, step)
    if step == an.TRAINED_STEP:
        stored = json.loads((unit2 / "trained.json").read_text())
        n_stored = sum(r["replicates"] for r in stored["seed0_learned_groups"])
        if n_stored != n_repl:
            raise SystemExit(f"refusing: {n_repl} replicating groups here, {n_stored} in trained.json")
    u2_lists: Dict[Tuple, List] = {}
    for (p, L, f, m, g) in u2:
        u2_lists.setdefault((p, L, f, m), []).append(g)
    rows, mismatch, l1_mismatch = [], [], []
    for r in u1:
        p, nk = r["passage"], len(r["kept_offsets"])
        by: Dict[Tuple, Dict[int, List]] = {}
        for lay in r["layers"]:
            for m, mm in lay["mcs"].items():
                by.setdefault((lay["frame"], int(m)), {})[lay["layer"]] = [g["offsets"] for g in mm["groups"]]
        for (f, m), d in by.items():
            d[0] = l0["groups"][p][f"{f}|{m}|0"]
            if not same_groups(d[1], l0["groups"][p][f"{f}|{m}|1"]):
                l1_mismatch.append((p, f, m))
        for lay in r["layers"]:
            L, f = lay["layer"], lay["frame"]
            for m, mm in lay["mcs"].items():
                m = int(m)
                gs = [g["offsets"] for g in mm["groups"]]
                if not same_groups(gs, u2_lists.get((p, L, f, m), [])):
                    mismatch.append((p, L, f, m))
                    continue
                cls, cls_bar, cls_nl2 = (group_classes(mm), group_classes(mm, bar=FIXED_BAR),
                                         group_classes(mm, "nl2"))
                for i, g in enumerate(mm["groups"]):
                    st = u2[(p, L, f, m, frozenset(g["offsets"]))]
                    o = origin(g["offsets"], by[(f, m)], L)
                    s = status(st["learned"], st["replicates"])
                    blk = len(g["offsets"]) >= an.BULK_SHARE * nk
                    rows.append({"prompt": p, "layer": L, "frame": f, "size": m, "members": g["offsets"],
                                 "tokens": [tokens[p][x] for x in g["offsets"]], "class": cls[i],
                                 "class_bar": cls_bar[i], "class_nl2": cls_nl2[i], "status": s,
                                 "s": st["s"], "bar": st["bar"], "bulk": blk, **o,
                                 "origin_class": origin_class(o["origin"]),
                                 "candidate": cls[i] == "moves" and s == "learned+replicates" and not blk})
    if mismatch or l1_mismatch:
        raise SystemExit(f"refusing: {step}: unit 1 and unit 2 groups differ in {len(mismatch)} records "
                         f"{mismatch[:3]}; L1 from l0 differs from unit 1's in {len(l1_mismatch)} {l1_mismatch[:3]}")
    return {"step": step, "unit1_git": sorted({r["meta"]["git"] for r in u1}), "l0_git": l0["git"],
            "n_replicating_seed0": n_repl, "rows": rows, **tables(rows)}


def read_cmd(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="candidates read")
    ap.add_argument("--unit1", type=Path, required=True, help="move_text output re-run with --kept-from")
    ap.add_argument("--unit2", type=Path, required=True, help="arch_null trained output")
    ap.add_argument("--l0", type=Path, required=True, help="`l0`'s output directory")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    fc = args.unit1 / "first_checks.json"
    if not fc.exists() or not all(v == "pass" for v in json.loads(fc.read_text())["verdicts"].values()):
        return _refuse(f"unit 1's first checks have not passed on the re-run ({fc})")
    tsf = args.unit2 / "step0" / "token_sets.json"
    sets, ts_sha = json.loads(tsf.read_text())["sets"], _sha(tsf)
    tok = an._tok()
    ids = an.prompt_ids(tok)
    tokens = {k: tok.convert_ids_to_tokens(v) for k, v in ids.items()}
    res = {"git": an._git_head(), "token_sets": {"path": str(tsf), "sha256": ts_sha},
           "inputs": an.input_names(ids), "present_jaccard": PRESENT_JACCARD, "few": FEW,
           "bulk_share": an.BULK_SHARE, "steps": {}}
    for step in (an.TRAINED_STEP, "step0"):
        res["steps"][step] = read_step(step, args.unit1, args.unit2, args.l0, sets, ts_sha, tokens)
    t = res["steps"][an.TRAINED_STEP]
    dist = [d for d in t["distinct"] if d["frame"] == "centred" and d["size"] == 2]
    res["outcome"] = {"n_distinct_centred_2": len(dist),
                      "by_origin": dict(Counter(d["origin_class"] for d in dist)), "reading": outcome(dist)}
    args.out.mkdir(parents=True, exist_ok=True)
    rows = {s: v.pop("rows") for s, v in res["steps"].items()}
    (args.out / "candidates.json").write_text(json.dumps(res, indent=1) + "\n")
    (args.out / "candidate_rows.json").write_text(json.dumps(rows) + "\n")
    for s, v in res["steps"].items():
        print(f"== {s} (replicating seed-0 groups {v['n_replicating_seed0']})")
        for cell, c in v["cross"].items():
            print(f"  {cell:18s} " + json.dumps(dict(sorted(c.items()))))
        for cell, c in v["origin"].items():
            print(f"  origin {cell:18s} " + json.dumps(c))
        print("  distinct candidates by origin: " + json.dumps(v["distinct_by_frame_size"]))
        print("  bulk: " + json.dumps(v["cross_bulk"]))
    print("outcome (step143000, distinct centred size-2 candidates): " + json.dumps(res["outcome"]))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmds = {"l0": l0_cmd, "read": read_cmd, "definitions": definitions_cmd}
    if not argv or argv[0] not in cmds:
        print(f"usage: candidates {{{','.join(cmds)}}} ...", file=sys.stderr)
        return 2
    return cmds[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
