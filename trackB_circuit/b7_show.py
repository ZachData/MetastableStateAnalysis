"""B7 unit 2's tables as markdown, from the read's `step<S>.json` (`b7_activity.py read`). Only
formats; every number is the record's. Tier 1: exploratory, unregistered.
    python -m trackB_circuit.b7_show <dir> [picture|cells|step <S>]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CODE = {"not read": "·", "positional, not separable": "X", "rerouting": "R",
        "depth-local persistence": "L", "frozen across depth": "F",
        "persistent, less than at distant layers": "W"}


def load(d: str):
    recs = [json.loads(p.read_text()) for p in Path(d).glob("step*.json")]
    return sorted(recs, key=lambda r: r["step"])


def fmt(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def letter(c) -> str:
    return CODE[c["description"]] + ("²" if c.get("two_cycle") else "")


def grid(recs, f, title: str) -> None:
    steps = [r["step"] for r in recs]
    print(f"\n**{title}**\n")
    print("| l → l + 1 | " + " | ".join(str(s) for s in steps) + " |")
    print("|---|" + "---|" * len(steps))
    for l in range(len(recs[0]["cells"])):
        print(f"| {l} | " + " | ".join(f(r["cells"][l]) for r in recs) + " |")


def picture(recs) -> None:
    print("Codes: · not read, X positional not separable, R rerouting, L depth-local persistence, "
          "F frozen across depth, W persistent less than at distant layers; ² a two-cycle candidate.")
    print(f"\ncode `{recs[0]['code']}`, prompts `{recs[0]['prompts_hash']}`, gate `{recs[0]['gate']}`")
    grid(recs, letter, "Descriptions (23 layer pairs × 18 steps)")
    grid(recs, lambda c: fmt(c.get("P")), "P (per-head persistence, the cell's value)")
    grid(recs, lambda c: fmt(c.get("P", np.nan) - c["N1"]) if "P" in c and c.get("N1") is not None else "—",
         "P − N1")
    grid(recs, lambda c: fmt(c.get("N0")), "N0")
    grid(recs, lambda c: str(c["n_enter"]), "entering heads")
    print("\n**By step** (cells read = not \"not read\"; share = median over layers of the condensed "
          "share; thin = read cells whose thinnest entering head has < 50 leave-one-out rows; "
          "rule = cells where unit 1b's entry count differs)\n")
    print("| step | X | R | L | F | W | · | ² | share | thin | rule | D_1 median | α median |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in recs:
        cs = r["cells"]
        n = {k: sum(c["description"] == d for c in cs) for d, k in CODE.items()}
        two = sum(bool(c.get("two_cycle")) for c in cs)
        thin = sum(c["description"] != "not read" and (c.get("loo_rows_min") or 0) < 50 for c in cs)
        rule = sum(c.get("n_enter_loo_rule") != c["n_enter"] for c in cs)
        share = np.nanmedian(np.asarray(r["share"]["layer"], float))
        d1 = np.nanmedian(np.asarray(r["D"]["D1"], float))
        al = np.nanmedian(np.asarray(r["alpha"]["layer_median"], float))
        print(f"| {r['step']} | {n['X']} | {n['R']} | {n['L']} | {n['F']} | {n['W']} | {n['·']} | {two} | "
              f"{fmt(share)} | {thin} | {rule} | {fmt(d1, 3)} | {fmt(al)} |")


def cells(recs, step=None) -> None:
    print("| step | l | description | heads (rule 1b) | joint rows | loo min / median | P [CI] | N0 | N1 | "
          "P − N0 CI | P − N1 CI | P_2 | soft | pooled P | nb P / N1 | share |")
    print("|" + "---|" * 16)
    ci = lambda v: "—" if not v or v[0] is None else f"{v[0]:.3f}, {v[1]:.3f}"
    for r in recs:
        if step is not None and r["step"] != step:
            continue
        for c in r["cells"]:
            if c["description"] == "not read":
                print(f"| {r['step']} | {c['l']} | not read: {c['not_read_because']} |" + " |" * 13)
                continue
            print(f"| {r['step']} | {c['l']} | {c['description']} | {c['n_enter']} ({c['n_enter_loo_rule']}) | "
                  f"{c['joint_rows']:.0f} | {c['loo_rows_min']:.0f} / {c['loo_rows_median']:.0f} | "
                  f"{fmt(c['P'], 3)} [{ci(c['P_ci'])}] | {fmt(c['N0'], 3)} | {fmt(c['N1'], 3)} | "
                  f"{ci(c['P_minus_N0_ci'])} | {ci(c['P_minus_N1_ci'])} | {fmt(c.get('P2'), 3)} | "
                  f"{fmt(c.get('soft'), 3)} | {fmt(c.get('pooled_P'), 3)} | "
                  f"{fmt(c.get('nb_P'), 3)} / {fmt(c.get('nb_N1'), 3)} | {fmt(r['share']['layer'][c['l']], 3)} |")


def main(argv) -> None:
    recs = load(argv[0])
    what = argv[1] if len(argv) > 1 else "picture"
    if what == "picture":
        picture(recs)
    else:
        cells(recs, int(argv[2]) if what == "step" else None)


if __name__ == "__main__":
    main(sys.argv[1:])
