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

``--per-layer`` does the same for §1.9 (`p10_comembership.py`) and §1.10
(`p10_lexical_carry.py`), whose claims are per layer: their table columns for
both sweeps, then, over the shared steps, how many per-layer "above / below /
as step 0" readings agree and the largest gap in the value read, with each
disagreement listed.

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
    python tools/run/p10_s1_compare.py [--raw --v1-only] [--per-layer]
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
    for r in recs.values():  # lexical_carry names neither prompts nor steps; its inputs do
        r.setdefault("prompts", sorted({k.split("|")[1] for k, _ in r["inputs"]}))
        r.setdefault("steps", sorted({int(k.split("|")[0]) for k, _ in r["inputs"]}))
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
    ap.add_argument("--per-layer", action="store_true", help="§1.9-§1.10: co-membership and lexical carry")
    add_holdout_args(ap)
    args = ap.parse_args()
    if args.per_layer:
        per_layer()
        return
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


COMEM_PROPS = ("same_class", "copy_share", "no_copy", "adjacent", "emb_pct_own")
LEX_COLS = (("cge40", "class_given_emb_40"), ("knn40", "class_given_emb_40_knn"),
            ("only40", "class_given_emb_40_classonly"), ("egc", "emb_given_class"))


def _agreement(recs: dict, cells) -> None:
    """cells: [(name, reading-getter, value-getter)], each (record, step, layer) -> x.
    Over shared steps and every layer both records hold."""
    a, b = recs.values()
    shared = sorted(set(map(str, a["steps"])) & set(map(str, b["steps"])) - {"0"}, key=int)
    for name, word, val in cells:
        n = agree = 0
        gap, off = 0.0, []
        for st in shared:
            for L in a["reading_layers"](st):
                wa, wb = word(a, st, L), word(b, st, L)
                if wa is None or wb is None:
                    continue
                n += 1
                agree += wa == wb
                va, vb = val(a, st, L), val(b, st, L)
                gap = max(gap, abs(va - vb))
                if wa != wb:
                    off.append(f"{st}/L{L} {va:+.3f}|{vb:+.3f}")
        print(f"  {name:<22} {agree}/{n} agree, max |Δ value| {gap:.3f}"
              + (f"; differ: {', '.join(off)}" if off else ""))


def per_layer() -> None:
    com = load("comembership")
    lex = load("lexical_carry")
    for sw in SWEEPS:
        print(f"{sw}: comembership inputs {com[sw]['inputs_sha256']} ({com[sw]['n_runs']} runs), "
              f"lexical_carry inputs {lex[sw]['inputs_sha256']} ({lex[sw]['n_runs']} runs)")
    L_ = ("0", "12", "24", "mean")
    print("\n§1.9 lift, all-positions draw, L0 L12 L24 mean, " + " | ".join(SWEEPS))
    for prop in COMEM_PROPS:
        print(f" {prop}")
        rows(com, [lambda r, s, L=L, p=prop: r["summary"]["all"][s][L][p]["lift"] for L in L_])
    print("\n§1.10 class_given_emb at 40 bins: observed, kNN control, class-only; emb_given_class (10 bins)")
    for L in ("12", "24", "mean"):
        print(f" L{L}")
        rows(lex, [lambda r, s, L=L, k=k: r["summary"][s][L][k] for _, k in LEX_COLS])
    print("\n§1.10 carry self_pct, focal / unclustered, L12 L24")
    rows(lex, [lambda r, s, L=L, g=g: r["summary"][s][L][g]["self_pct"]
               for L in ("12", "24") for g in ("focal", "unclustered")])

    print("\nper-layer readings vs step 0 (±0.05), pilot vs stage0, shared steps > 0")
    # the record's "reading" holds L0/L12/L24/mean only; the per-layer claim is every
    # layer, so the delta is re-derived from the lifts with the reader's own floor
    from tools.run.p10_comembership import DELTA_FLOOR
    print(f" §1.9, all 25 layers and the mean, all-positions draw / clustered draw (floor {DELTA_FLOOR})")
    delta = lambda r, s, L, p, d: (r["summary"]["all"][s][L][p]["lift" + d]
                                   - r["summary"]["all"]["0"][L][p]["lift" + d])
    for r in com.values():
        r["reading_layers"] = lambda st, r=r: list(r["summary"]["all"][st])
    _agreement(com, [(f"{p}{d}",
                      lambda r, s, L, p=p, d=d: (delta(r, s, L, p, d) > DELTA_FLOOR)
                      - (delta(r, s, L, p, d) < -DELTA_FLOOR),
                      lambda r, s, L, p=p, d=d: delta(r, s, L, p, d))
                     for p in COMEM_PROPS for d in ("", "_cl")])
    print(" §1.10, L0 L12 L24 mean")
    for r in lex.values():
        r["reading_layers"] = lambda st, r=r: list(r["reading"][st])
    names = [k for _, k in LEX_COLS] + ["class_given_emb", "emb_same", "emb_cross", "carry_gap.self_pct"]
    _agreement(lex, [(k, lambda r, s, L, k=k: r["reading"][s][L][k]["reading"],
                      lambda r, s, L, k=k: r["reading"][s][L][k]["delta"]) for k in names])


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
