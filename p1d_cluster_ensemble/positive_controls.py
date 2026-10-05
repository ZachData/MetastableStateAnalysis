"""
p1d_cluster_ensemble/positive_controls.py — unit 4 on the designed prompts
(`design-1d.md` "Unit 4 on the designed prompts"; Blocked 21, option (a);
every rule fixed there before this file or any record of units 2–3 on them).

**Group route (units 1–2), the verdict** (`groups`): at ``step143000``, seed 0,
centred, size 2, L1–24, a unit-2 level-set group is a **candidate** if it is a
content group (`designed_prompts` rule 1 on its kept members), **learned**
(`arch_null.group_rules`) and **moves** (its best-Jaccard group in unit 1's
P = 0 record at the same layer, frame and size has Jaccard >= ``MOVE_JACCARD``
and class ``moves`` by `move_text.group_classes`, its own floor). Per prompt:
**pass** if (i) >= 1 candidate and (ii) seed 0's learned-content count over
L1–24 exceeds the maximum of the 10 real inits' counts at step 0 (same bars).
Verdict: **>= ``PASS_PROMPTS`` of 3** prompts pass. Beside: every (frame, size),
seeds 1–9's counts, seed-0 candidates' replication (unit 2's rule), per label.

**Reader (unit 3), beside: its measured sensitivity** (`reader`): on a
`scale_real` step-0 batch and its `trained` reading on these prompts, per
plateau the first cut's clusters of the arm's size, each tested by rule 1
(**content cluster**), and the ARI of the cut against the labels over the
labelled tokens it clusters. Per band and arm: trained clouds with >= 1 plateau
holding a content cluster; the same for the step-0 real inits ranked among the
40 re-inits and each re-init among the other 39 (`scale_real.plateau_table`).
No pass: a measurement.

Tier 1: exploratory, unregistered.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import designed_prompts as dp
from .arch_null import (BANDS, FRAMES, MIN_CLUSTER_SIZES, TRAINED_STEP, _git_head, _tok, failed_cells,
                        first_check, group_bars, group_rules, jaccard, load_records, model_ids,
                        prompt_keys, replication)
from .move_text import LAYERS, band, content_label, group_classes, passage_inputs

#: A unit-2 group moves if its best unit-1 match is at least this (unit 2's replication Jaccard).
MOVE_JACCARD = 0.5
#: Verdict: this many of the 3 designed prompts pass (placed; `design-1d.md` "Unit 4").
PASS_PROMPTS = 2
#: The verdict's cell: seed 0, centred, size 2.
PRIMARY = ("centred", 2)
PROMPTS = "designed"


def labels_of(tokenizer) -> Dict[str, List[Optional[str]]]:
    """Per designed prompt, each position's label (None if unlabelled)."""
    return {k: v["labels"] for k, v in passage_inputs(PROMPTS, tokenizer).items()}


def group_content(members: Sequence[int], labels: Sequence[Optional[str]]) -> Optional[str]:
    """`designed_prompts` rule 1 on a group's kept positions."""
    return content_label([labels[i] for i in members])


# ---------------------------------------------------------------------------
# Group route: unit 2's learned groups joined to unit 1's moves
# ---------------------------------------------------------------------------

def unit1_groups(move_dir: Path) -> Dict[Tuple[str, int, str, int], List[Tuple[List[int], str]]]:
    """
    ``{(prompt, layer, frame, size): [(offsets, class), ...]}`` from unit 1's
    ``step143000`` P = 0 records of the designed prompts; class recomputed by
    `group_classes` (own floor), never read from the record. At P = 0 a
    designed prompt's passage offset is its position.
    """
    out = {}
    for k in dp.KEYS:
        f = move_dir / TRAINED_STEP / f"{k}.json"
        if not f.exists():
            raise SystemExit(f"refusing: unit 1's record {f} is missing")
        r = json.loads(f.read_text())
        if r["meta"].get("designed_hash") != dp.designed_hash():
            raise SystemExit(f"refusing: {f} was run on designed prompts {r['meta'].get('designed_hash')}, "
                             f"not {dp.designed_hash()}")
        for lay in r["layers"]:
            for m, rec in lay["mcs"].items():
                out[(k, lay["layer"], lay["frame"], int(m))] = [
                    (g["offsets"], c) for g, c in zip(rec["groups"], group_classes(rec))]
    return out


def best_unit1(members: Sequence[int], cands: Sequence[Tuple[List[int], str]]) -> Tuple[float, Optional[str]]:
    """Best Jaccard against unit 1's groups and that group's class (ties: the first)."""
    best, cls = 0.0, None
    for off, c in cands:
        j = jaccard(members, off)
        if j > best:
            best, cls = j, c
    return best, cls


def annotate(rows: Sequence[Dict], labels: Dict[str, Sequence], u1: Optional[Dict] = None) -> None:
    """In place: each group row's ``content`` label; with ``u1``, its best unit-1 match, class and ``moves``."""
    for r in rows:
        r["content"] = group_content(r["members"], labels[r["prompt"]])
        if u1 is not None:
            j, c = best_unit1(r["members"], u1.get((r["prompt"], r["layer"], r["frame"], r["size"]), []))
            r["unit1_jaccard"], r["unit1_class"] = round(j, 3), c
            r["moves"] = bool(j >= MOVE_JACCARD and c == "moves")


def learned_content_counts(rows: Sequence[Dict], frame: str, size: int) -> Dict[Tuple[int, str], int]:
    """``{(seed, prompt): learned content groups over L1–24}`` at one (frame, size)."""
    out: Dict[Tuple[int, str], int] = {}
    for r in rows:
        if r["frame"] == frame and r["size"] == size:
            key = (r["seed"], r["prompt"])
            out[key] = out.get(key, 0) + bool(r["learned"] and r["content"])
    return out


def prompt_verdicts(trained: Sequence[Dict], step0: Sequence[Dict], seeds0: Sequence[int],
                    frame: str = PRIMARY[0], size: int = PRIMARY[1]) -> Dict[str, Dict]:
    """
    Per prompt at (``frame``, ``size``): seed 0's candidates and learned-content
    count, the step-0 inits' counts, and pass ((i) and (ii)). ``learned`` None
    (a refused cell) counts as not learned, and the cell is flagged.
    """
    lt = learned_content_counts(trained, frame, size)
    l0 = learned_content_counts(step0, frame, size)
    out = {}
    for k in dp.KEYS:
        rs = [r for r in trained if r["seed"] == 0 and r["prompt"] == k and r["frame"] == frame and r["size"] == size]
        cand = [r for r in rs if r["content"] and r["learned"] and r.get("moves")]
        base = [l0.get((sd, k), 0) for sd in seeds0]
        n0 = lt.get((0, k), 0)
        out[k] = {"groups": len(rs), "content": sum(bool(r["content"]) for r in rs),
                  "content_learned": n0, "content_moves": sum(bool(r["content"] and r.get("moves")) for r in rs),
                  "candidates": len(cand), "step0_content_learned": base, "step0_max": max(base) if base else None,
                  "refused": any(r["learned"] is None for r in rs),
                  "by_seed": {sd: v for (sd, kk), v in sorted(lt.items()) if kk == k},
                  "candidates_by_label": _by_label(cand),
                  "pass": bool(cand) and bool(base) and n0 > max(base)}
    return out


def _by_label(rows: Sequence[Dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        out[r["content"]] = out.get(r["content"], 0) + 1
    return dict(sorted(out.items()))


def _share(rows: Sequence[Dict]) -> Optional[float]:
    return round(float(np.mean([bool(r["content"]) for r in rows])), 3) if rows else None


def filter_ablation(trained: Sequence[Dict], step0: Sequence[Dict], frame: str, size: int) -> Dict[str, Dict]:
    """
    Beside the verdict (added after `/challenge-pr` on #142, finding 1): per prompt,
    the content share of seed 0's groups by filter (all, learned, not learned, moves,
    not moves), over all 10 seeds for learned, and of the step-0 inits' groups; and
    seed 0's counts of content ∧ moves, content ∧ learned, and both. A filter that
    selects content has a share above its complement's.
    """
    out = {}
    for k in dp.KEYS:
        cell = [r for r in trained if r["prompt"] == k and r["frame"] == frame and r["size"] == size]
        s0 = [r for r in cell if r["seed"] == 0]
        z0 = [r for r in step0 if r["prompt"] == k and r["frame"] == frame and r["size"] == size]
        out[k] = {"share_all_seeds": {"all": _share(cell), "learned": _share([r for r in cell if r["learned"]]),
                                      "not_learned": _share([r for r in cell if r["learned"] is False])},
                  "share_seed0": {"all": _share(s0), "learned": _share([r for r in s0 if r["learned"]]),
                                  "not_learned": _share([r for r in s0 if r["learned"] is False]),
                                  "moves": _share([r for r in s0 if r.get("moves")]),
                                  "not_moves": _share([r for r in s0 if not r.get("moves")])},
                  "share_step0_inits": _share(z0),
                  "seed0_counts": {"content_moves": sum(bool(r["content"] and r.get("moves")) for r in s0),
                                   "content_learned": sum(bool(r["content"] and r["learned"]) for r in s0),
                                   "content_both": sum(bool(r["content"] and r["learned"] and r.get("moves")) for r in s0)}}
    return out


def verdict(per_prompt: Dict[str, Dict]) -> str:
    n = sum(v["pass"] for v in per_prompt.values())
    return f"{'pass' if n >= PASS_PROMPTS else 'fail'} ({n} of {len(per_prompt)} prompts)"


def groups_cmd(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="positive_controls groups")
    ap.add_argument("--arch", type=Path, required=True, help="`arch_null --prompts designed` directory")
    ap.add_argument("--move", type=Path, required=True, help="unit 1's `move_text` directory")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    keys = prompt_keys(PROMPTS)
    fc_file = args.arch / "first_check.json"
    fc = json.loads(fc_file.read_text()) if fc_file.exists() else {}
    if fc.get("union") != "trained" or fc.get("prompts") != PROMPTS:
        print(f"refusing: {fc_file} is not `check --union trained --prompts designed`", file=sys.stderr)
        return 1
    inits, reinits = model_ids("init"), model_ids("reinit")
    recs0 = load_records(args.arch, "step0", inits + reinits, "trained", keys)
    rect = load_records(args.arch, TRAINED_STEP, inits, "trained", keys)
    if len(recs0) != len(keys) * (len(inits) + len(reinits)) or len(rect) != len(keys) * len(inits):
        print(f"refusing: {len(recs0)} step-0 and {len(rect)} trained records", file=sys.stderr)
        return 1
    check = first_check(recs0, reinits, inits, keys)
    if {(r["stat"], r["frame"], r["band"]): r["verdict"] for r in fc["rows"]} != \
            {(r["stat"], r["frame"], r["band"]): r["verdict"] for r in check}:
        print("refusing: the stored first check does not match these records", file=sys.stderr)
        return 1
    failed = failed_cells(check)
    bars = group_bars(recs0, reinits, strict=False)
    nonfinite = sorted(k for k, v in bars.items() if v is None)
    trained = group_rules(rect, bars, failed)
    step0 = group_rules([r for r in recs0 if r["model"] in inits], bars, failed)
    labels = labels_of(_tok())
    annotate(trained, labels, unit1_groups(args.move))
    annotate(step0, labels)
    seeds = [int(m.split(":")[1]) for m in inits]
    cells = {f"{f}/{m}": prompt_verdicts(trained, step0, seeds, f, m) for f in FRAMES for m in MIN_CLUSTER_SIZES}
    primary = cells[f"{PRIMARY[0]}/{PRIMARY[1]}"]
    ablation = {f"{f}/{m}": filter_ablation(trained, step0, f, m) for f in FRAMES for m in MIN_CLUSTER_SIZES}
    rep = replication(trained, seeds)
    cand = {(r["prompt"], r["layer"], tuple(r["members"])) for r in trained
            if r["seed"] == 0 and (r["frame"], r["size"]) == PRIMARY and r["content"] and r["learned"] and r["moves"]}
    rep_c = [r for r in rep if (r["frame"], r["size"]) == PRIMARY and (r["prompt"], r["layer"], tuple(r["members"])) in cand]
    per_band = {b: {k: sum(1 for r in trained if r["seed"] == 0 and (r["frame"], r["size"]) == PRIMARY
                           and r["prompt"] == k and band(r["layer"]) == b and r["content"] and r["learned"] and r["moves"])
                    for k in keys} for b in BANDS}
    res = {"git": _git_head(), "records_git": sorted({r["meta"]["git"] for r in recs0 + rect}),
           "designed_hash": dp.designed_hash(), "arch": str(args.arch), "move": str(args.move),
           "failed_cells": sorted(failed), "nonfinite_bars": nonfinite, "move_jaccard": MOVE_JACCARD, "pass_prompts": PASS_PROMPTS,
           "verdict": verdict(primary), "primary": primary, "cells": cells, "filter_ablation": ablation, "candidates_per_band": per_band,
           "candidate_replication": {"n": len(rep_c), "replicating": sum(r["replicates"] for r in rep_c),
                                     "rows": rep_c},
           "candidates": [{k: r[k] for k in ("prompt", "layer", "members", "s", "bar", "content", "unit1_jaccard")}
                          for r in trained if r["seed"] == 0 and (r["frame"], r["size"]) == PRIMARY
                          and r["content"] and r["learned"] and r["moves"]]}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "groups.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"failed first-check cells: {sorted(failed) or 'none'}; non-finite bars (refused): {nonfinite or 'none'}")
    print(f"group route, seed 0, centred, size 2 (candidate = content ∧ learned ∧ moves): {res['verdict']}")
    for k, v in primary.items():
        print(f"  {k:24s} groups {v['groups']:3d} content {v['content']:3d} content∧learned {v['content_learned']:3d} "
              f"content∧moves {v['content_moves']:3d} candidates {v['candidates']:3d} {v['candidates_by_label']} | "
              f"step-0 inits {v['step0_content_learned']} (max {v['step0_max']}) | seeds {list(v['by_seed'].values())}"
              f"{' REFUSED cell' if v['refused'] else ''} | {'PASS' if v['pass'] else 'fail'}")
    print("beside, every (frame, size): pass per prompt")
    for c, t in cells.items():
        print(f"  {c:11s} " + " ".join(f"{k.split('_', 1)[1]} {v['content_learned']}/{v['step0_max']}"
                                       f"{'+' if v['pass'] else '-'}" for k, v in t.items()))
    print("beside: content share by filter (a filter selects content if above its complement)")
    for c, t in ablation.items():
        for k, v in t.items():
            a, z = v["share_all_seeds"], v["share_seed0"]
            print(f"  {c:11s} {k.split('_', 1)[1]:13s} all seeds: all {a['all']} learned {a['learned']} not {a['not_learned']}"
                  f" | seed 0: moves {z['moves']} not {z['not_moves']} | step-0 inits {v['share_step0_inits']} | "
                  f"seed 0 content∧moves / ∧learned / both {list(v['seed0_counts'].values())}")
    print(f"seed-0 candidates replicating (unit 2's rule): {res['candidate_replication']['replicating']} "
          f"of {res['candidate_replication']['n']}; per band {per_band}")
    return 0


# ---------------------------------------------------------------------------
# Reader: unit 3's plateaus scored against the labels
# ---------------------------------------------------------------------------

def score_partition(clusters: Sequence[Sequence[int]], labels: Sequence[Optional[str]],
                    n_labelled_kept: Optional[int] = None) -> Dict:
    """
    Rule 1 on each cluster (positions); the ARI of cluster against label over the
    labelled tokens clustered (purity, not recovery); and ``recovery``, the share
    of the prompt's labelled kept tokens that the cut clusters (added after
    `/challenge-pr` on #142, finding 2).
    """
    from sklearn.metrics import adjusted_rand_score
    content = [group_content(c, labels) for c in clusters]
    cid, lab = [], []
    for i, c in enumerate(clusters):
        for p in c:
            if labels[p] is not None:
                cid.append(i)
                lab.append(labels[p])
    ari = float(adjusted_rand_score(lab, cid)) if len(set(lab)) > 1 and len(cid) >= 2 else None
    return {"content": [c for c in content if c], "n_content": sum(bool(c) for c in content),
            "ari": None if ari is None else round(ari, 3), "n_labelled": len(cid),
            "recovery": round(len(cid) / n_labelled_kept, 3) if n_labelled_kept else None}


def _clusters(lab: np.ndarray, size: int, kept: Sequence[int]) -> List[List[int]]:
    ids, counts = np.unique(lab, return_counts=True)
    return [[int(kept[i]) for i in np.flatnonzero(lab == c)] for c in ids[counts >= size]]


def reader_cmd(argv: Optional[Sequence[str]] = None) -> int:
    from .scale_real import ARMS, MIN_RUN, load_run, load_sets, plateau_table
    ap = argparse.ArgumentParser(prog="positive_controls reader")
    ap.add_argument("--step0", type=Path, required=True, help="`scale_real run --prompts designed` step-0 batch")
    ap.add_argument("--trained", type=Path, required=True, help="its `scale_real trained --prompts designed` directory")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    keys = prompt_keys(PROMPTS)
    tj = json.loads((args.trained / "trained.json").read_text())
    if Path(tj["ref"]).resolve() != args.step0.resolve():
        print(f"refusing: {args.trained} was read against {tj['ref']}, not {args.step0}", file=sys.stderr)
        return 1
    if tj["records"].get("prompts") != PROMPTS:
        print(f"refusing: {args.trained} is not a designed-prompt reading", file=sys.stderr)
        return 1
    kept = load_sets(Path(tj["records"]["sets"]), PROMPTS)
    labels = labels_of(_tok())
    n_lab = {k: sum(labels[k][p] is not None for p in kept[k]) for k in keys}
    inits, reinits = model_ids("init"), model_ids("reinit")
    hits: Dict[str, Dict] = {}
    plats: List[Dict] = []
    for c in tj["cells"]:
        for a in ARMS:
            pl = c["plateaus"][a]["conjunction"]
            sc = [score_partition([d["positions"] for d in p["partition"]], labels[c["prompt"]], n_lab[c["prompt"]])
                  for p in pl]
            key = ("trained", a, c["band"])
            h = hits.setdefault(str(key), {"kind": "trained", "arm": a, "band": c["band"], "clouds": 0,
                                           "plateau": 0, "content": 0, "readable": 0, "per_prompt": {}})
            h["clouds"] += 1
            h["readable"] += bool(c["window"][a]["readable"])
            h["plateau"] += bool(pl)
            got = any(s["n_content"] for s in sc)
            h["content"] += got
            h["per_prompt"][c["prompt"]] = h["per_prompt"].get(c["prompt"], 0) + got
            plats += [{"kind": "trained", "arm": a, "model": c["model"], "prompt": c["prompt"], "layer": c["layer"],
                       "r_lo": p["r_lo"], "r_hi": p["r_hi"], **s} for p, s in zip(pl, sc)]
    recs, labs = load_run(args.step0, prompts=keys)
    for a, size in ARMS.items():
        tab = plateau_table(recs, labs, a, (MIN_RUN,))
        for kind in ("real_init", "reinit"):
            for (mid, k, L), pl in tab[kind][MIN_RUN].items():
                sc = [score_partition(_clusters(labs[mid][k][L][p["start"]], size, kept[k]), labels[k], n_lab[k])
                      for p in pl]
                key = (kind, a, band(L))
                h = hits.setdefault(str(key), {"kind": f"step0_{kind}", "arm": a, "band": band(L), "clouds": 0,
                                               "plateau": 0, "content": 0, "per_prompt": {}})
                h["clouds"] += 1
                h["plateau"] += bool(pl)
                got = any(s["n_content"] for s in sc)
                h["content"] += got
                h["per_prompt"][k] = h["per_prompt"].get(k, 0) + got
                plats += [{"kind": f"step0_{kind}", "arm": a, "model": mid, "prompt": k, "layer": L,
                           "r_lo": p["r_lo"], "r_hi": p["r_hi"], **s} for p, s in zip(pl, sc)]
    table = sorted(hits.values(), key=lambda h: (h["arm"], h["band"], h["kind"]))
    res = {"git": _git_head(), "designed_hash": dp.designed_hash(), "step0": str(args.step0),
           "trained": str(args.trained), "trained_git": tj["git"], "records": tj["records"],
           "table": table, "plateaus": plats}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "reader.json").write_text(json.dumps(res, indent=1) + "\n")
    print("reader sensitivity: clouds with >= 1 conjunction plateau holding a content cluster (rule 1)")
    for h in table:
        rd = f", readable {h['readable']}" if "readable" in h else ""
        print(f"  {h['arm']:5s} {h['band']:7s} {h['kind']:16s} content {h['content']:4d} / {h['clouds']:4d} "
              f"(any plateau {h['plateau']}{rd}) per prompt {h['per_prompt']}")
    for a in ARMS:
        cp = [p for p in plats if p["kind"] == "trained" and p["arm"] == a and p["n_content"]]
        if cp:
            aris = [p["ari"] for p in cp if p["ari"] is not None]
            print(f"  trained {a}: content plateaus at r {min(p['r_lo'] for p in cp):.2f}-{max(p['r_hi'] for p in cp):.2f}, "
                  f"median ARI vs labels (purity) {np.median(aris) if aris else None}, recovery "
                  f"{min(p['recovery'] for p in cp)}-{max(p['recovery'] for p in cp)}, L17-24 below r 0.6: "
                  f"{sum(p['layer'] >= 17 and p['r_lo'] < 0.6 for p in cp)}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmds = {"groups": groups_cmd, "reader": reader_cmd}
    if not argv or argv[0] not in cmds:
        print(f"usage: positive_controls {{{','.join(cmds)}}} ...", file=sys.stderr)
        return 2
    return cmds[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
