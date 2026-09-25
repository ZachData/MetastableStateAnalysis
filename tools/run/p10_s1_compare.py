#!/usr/bin/env python3
"""
p10_s1_compare.py — Stage 1 steps 1 and 2 on two sweeps, side by side
(`handoff-10.md` §1.3 step 3; numbers: `status-10.md` §1.11 only).

Reads the records `p10_ext_sem_threshold.py` and `p10_token_composition.py`
wrote for Stage 0 (through the index) and for the pilot sweep (`--run-root`),
and prints the columns of `status-10.md` §1.6 and §1.7's first two tables for
each sweep, one row per step, marking steps the other sweep lacks. It computes
nothing new: every number is a field of a record, so the table cannot drift
from its producer.

Refuses two records of one kind whose prompt sets differ: a step mean over
different prompts is not a comparison.

``--raw`` also reads the run dirs the records name (numpy + sklearn, and
`HDD_1TB` mounted). It produces §1.11's input and label numbers: the largest
activation gap, identical label vectors (overall, per layer, per step), ARI,
equal ``max_alive``, for pilot vs Stage 0 on the shared runs, and for Stage 0
vs the WDS backfill (the battery-``1e47918ef77a`` Phase 1 dirs under
``data/phase12``, labels through ``read_labels``).

TIER 1, EXPLORATORY, NOT REGISTERED. Descriptive, no null.

Run:
    python tools/run/p10_s1_compare.py [--raw --v1-only]
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[2])))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
A = DATA / "analysis"
SWEEPS = {"stage0": "", "pilot": "_pilot"}

EXT_COLS = (  # (header, frame, getter)
    ("repeat share", "self", lambda s: s["identical_token_fraction"]),
    ("ext_sem@0.5 self", "self", lambda s: s["ext_semantic_fraction"]["abs_0.5"]),
    ("nr cos>0.2", "frozen", lambda s: s["non_repeat"]["frac_above"]["abs_0.2"]),
    ("nr pct", "frozen", lambda s: s["non_repeat"]["mean_percentile"]),
    ("same-cl repeats", "self", lambda s: s["identical_same_cluster_frac"]),
)
TOK_CELLS = (("unique", "all|all|unique"), ("2 copies", "count|2|first"),
             ("3-5", "count|3-5|repeat"), ("6-20", "count|6-20|repeat"),
             ("u rank<1k", "rank|<1k|unique"), ("u rank>=20k", "rank|>=20k|unique"),
             ("u word_start", "cls|word_start|unique"), ("u punct", "cls|punct|unique"))


def load(kind: str) -> dict:
    recs = {sw: json.loads((A / f"p10_s1_{kind}{suf}.json").read_text())
            for sw, suf in SWEEPS.items()}
    prompts = {sw: tuple(r["prompts"]) for sw, r in recs.items()}
    if len(set(prompts.values())) != 1:
        sys.exit(f"{kind}: prompt sets differ: {prompts}")
    return recs


def fmt(x) -> str:
    return "   —  " if x is None else f"{x:6.3f}"


def rows(recs: dict, cells) -> None:
    steps = sorted({int(s) for r in recs.values() for s in r["steps"]})
    for st in steps:
        line = []
        for sw, r in recs.items():
            have = st in r["steps"]
            line.append(" ".join(fmt(get(r, str(st)) if have else None) for get in cells))
        mark = "" if all(st in r["steps"] for r in recs.values()) else "  (one sweep)"
        print(f"{st:>7} | " + " | ".join(line) + mark)


def main() -> None:
    sys.path.insert(0, str(REPO))
    from core.holdout import add_holdout_args
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", action="store_true", help="also compare labels and activations run by run")
    add_holdout_args(ap)
    args = ap.parse_args()
    ext = load("ext_sem_threshold")
    tok = load("token_composition")
    for sw in SWEEPS:
        print(f"{sw}: ext inputs {ext[sw]['inputs_sha256']} ({ext[sw]['n_runs']} runs), "
              f"tok inputs {tok[sw]['inputs_sha256']} ({tok[sw]['n_runs']} runs), "
              f"battery {ext[sw]['prompt_battery_hash']}")
    print(f"prompts: {', '.join(ext['stage0']['prompts'])}\n")

    print("§1.6 columns, " + " | ".join(SWEEPS) + ": " + ", ".join(h for h, _, _ in EXT_COLS))
    rows(ext, [lambda r, s, f=f, g=g: g(r["summary"]["by_step"][f][s]) for _, f, g in EXT_COLS])
    for sw, r in ext.items():
        print(f"  verdicts {sw}: " + ", ".join(f"{f} {v['verdict']}" for f, v in r["summary"]["verdict"].items()))

    print("\n§1.7 columns, " + " | ".join(SWEEPS) + ": noise, " + ", ".join(h for h, _ in TOK_CELLS))
    rows(tok, [lambda r, s: r["summary"]["by_step"][s]["noise_rate"]]
         + [lambda r, s, c=c: r["summary"]["balanced"][s].get(c, {}).get("rate") for _, c in TOK_CELLS])

    print("\nunique-only contrasts (freq, class) and max_alive without repeated_tokens (L0 peaks), "
          + " | ".join(SWEEPS))
    rows(tok, [lambda r, s: r["summary"]["by_step"][s]["unique"]["freq_contrast"],
               lambda r, s: r["summary"]["by_step"][s]["unique"]["class_contrast"],
               lambda r, s: r["summary"]["cluster_count"][s]["without_repeated_tokens"]["max_alive"],
               lambda r, s: r["summary"]["cluster_count"][s]["without_repeated_tokens"]["max_alive_at_layer0"]])
    if args.raw:
        raw(tok, args)


def _pair_stats(pairs: dict, labels) -> dict:
    """pairs: {(step, key): (dir_a, dir_b)}; labels: dir -> {layer: array}."""
    import numpy as np
    from sklearn.metrics import adjusted_rand_score
    gap, by_step, by_layer, aris, n_same, n, alive_eq = {}, {}, {}, [], 0, 0, 0
    for (st, key), (a, b) in sorted(pairs.items()):
        xa = np.load(a / "activations.npz")["activations"]
        xb = np.load(b / "activations.npz")["activations"]
        gap[st] = max(gap.get(st, 0.0), float(np.abs(xa - xb).max()))
        la, lb = labels(a), labels(b)
        count = lambda l: max(len(set(v.tolist()) - {-1}) for v in l.values())
        alive_eq += count(la) == count(lb)
        for L in la:
            same = bool(np.array_equal(la[L], lb[L]))
            n += 1
            n_same += same
            by_step[st] = by_step.get(st, 0) + (not same)
            by_layer[L] = by_layer.get(L, 0) + same
            aris.append(adjusted_rand_score(la[L], lb[L]))
    return {"n_runs": len(pairs), "n": n, "identical": n_same, "differ_by_step": by_step,
            "identical_by_layer": by_layer, "gap_by_step": gap, "alive_eq": alive_eq,
            "ari_p5": float(np.percentile(aris, 5)), "ari_min": float(min(aris))}


def raw(tok: dict, args) -> None:
    from core.holdout import refuse_held_out
    from tools.run.backfill_hdbscan import read_labels
    runs = {sw: {(int(k.split("|")[0]), k.split("|")[1]): Path(p) for k, p in r["inputs"]}
            for sw, r in tok.items()}
    s0 = runs["stage0"]
    wds = {}
    for m in sorted((DATA / "phase12").glob("2026-*/pythia-410m-step*/manifest.json")):
        man = json.loads(m.read_text())
        if m.parent not in s0.values() and man.get("prompt_battery_hash") == "1e47918ef77a":
            wds[(int(man["checkpoint_step"]), man["prompt_key"])] = m.parent
    kept, _ = refuse_held_out(sorted(wds.values()), allow=args.allow_holdout, drop=args.v1_only, context="p10_s1_compare")
    wds = {k: p for k, p in wds.items() if p in set(kept)}
    for name, other in (("pilot vs stage0", runs["pilot"]), ("wds backfill vs stage0", wds)):
        pairs = {k: (other[k], s0[k]) for k in sorted(set(other) & set(s0))}
        st = _pair_stats(pairs, read_labels)
        print(f"\n{name}: {st['n_runs']} runs, {st['identical']}/{st['n']} layer-records identical, "
              f"ARI p5 {st['ari_p5']:.3f} min {st['ari_min']:.3f}, equal max_alive {st['alive_eq']}/{st['n_runs']}")
        print("  identical by layer: " + " ".join(str(st["identical_by_layer"][L])
                                                  for L in sorted(st["identical_by_layer"])))
        print("  per step: max |Δact| / differing layer-records")
        for s in sorted(st["gap_by_step"]):
            print(f"    {s:>7}: {st['gap_by_step'][s]:.1e} / {st['differ_by_step'].get(s, 0)}")


if __name__ == "__main__":
    main()
