"""Pass 2 of the catalogue: is the redundancy set ONE set, or several?

WHY THIS EXISTS
---------------
`PROJECT.md` §3.14.4-B: the members are **not** known to be independent. Exactly
one pair has ever had its interaction measured -- `L5H2` x `L7H8` at **+4.151**,
joint 2.2x the parts-sum (§3.12-S) -- and the other 44 cells of the top-10
matrix are empty. "They are all so independent" is an impression from reading
the single-head column of §3.12-T, which by construction says nothing about
pairs. This runner fills the matrix.

WHAT THE ANSWER LOOKS LIKE. A block of mutually super-additive heads is ONE
redundancy set holding one regime. Several blocks that are super-additive within
and ~0 across are DISJOINT sets, and the phase has been describing a list as if
it were an object. Sub-additive cells are the serial signature -- and see the
ceiling warning below before believing any of them.

THE CEILING, WHICH IS WHY THIS RUNS LATE AND NOT EARLY (§3.14.4-D). `dNLL` on
the second copy is bounded by uniform prediction, `ln 50304 = 10.83`. A joint arm
near that bound is reporting a floor, not a measurement, and the compression
biases interactions toward apparent **sub**-additivity (§3.12-M5) -- the serial
signature, manufactured. Step 16000 leaves ~2.8 nats of headroom; steps 1000-3000
do not. Every cell therefore carries its own `joint_nll` and `headroom`, and any
cell under `--headroom-warn` is flagged in the output rather than left for a
reader to notice.

THE PROBE IS THE OTHER HANDLE ON THAT CEILING, and at a small rung it is the
binding one. `--probe freq` draws the repeated tokens from `[1000, 5000)`
instead of `wide`'s `[1000, 40000)`, which lowers the baseline NLL and so buys
headroom: at pythia-70m it cuts above-ceiling positions from 42 % to 8 %
(`p8_scale_ladder/probe_distribution.py`). That matters here because 70m's
`mean`-ablation matrix came back with **7 of 15 cells censored**, all five of
them involving `L2H1` -- including `L2H1` x `L3H6`, the direct analogue of
410m's headline pair (`status-8.md`, the matched cross-rung read). `wide` stays
the default: every existing 7d/7e/8 number is on it, and a `freq` run is an
ADDED arm, never a replacement.

THE GEOMETRIC MATRIX IS FREE. The singles are run with hidden states anyway, so
the residual-delta cosine between every pair costs no additional forward pass.
It is reported beside the causal cell because §3.12-S found the two dissociate --
chance-level weight overlap, 87 %-aligned effect -- and because, unlike `dNLL`,
it has no ceiling. Where the two disagree, the cosine is the one to trust.

Reported BOTH ways per §3.13: raw `dNLL`, and relative to the checkpoint's own
baseline. Neither view is chosen after seeing the data.

COST: one model load per step, then `1 + n` single arms (hidden states) and
`n(n-1)/2` joint arms (logits only). n=10 is 56 arms.

NO p-value; nothing registered. pythia-410m is spent under `check_registry`
rule 3 -- exploratory, and not registrable on this data.
"""
import argparse
import gc
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", "/run/media/system/WDS_500/Mets"))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))
_want = str(REPO / ".venv")
if not sys.prefix.startswith(_want):
    raise SystemExit(f"wrong interpreter: {sys.prefix!r}, need {_want!r}")

import numpy as np

from core.lm_loading import load_causal_lm
from tools.run.induction_rank_sweep import (
    EVAL_SEED, PROBE_ARMS, ablated, arch_dims, head_means,
)

from p7d_redundancy.member_formation_curves import batch, catalog_path_for, members, probe

OUT = DATA / "analysis" / "pairwise_interaction_matrix.json"
DEFAULT_MODEL = "pythia-410m"

#: Uniform prediction over pythia's vocabulary. The readout cannot report a
#: larger NLL than this, so an arm that approaches it is censored.
CEILING = float(np.log(50304))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="registry family prefix, e.g. pythia-70m, pythia-410m "
                         "(p8_scale_ladder/design-8.md rung policy: 1b/1.4b "
                         "are reserved -- do not pass those without a "
                         "registered prediction)")
    ap.add_argument("--steps", default="16000",
                    help="16000 or later -- see the ceiling note in __doc__; "
                         "earlier steps bias interactions sub-additive")
    ap.add_argument("--top", type=int, default=10,
                    help="catalogue members to include; 4 gives 6 cells, "
                         "10 gives 45")
    ap.add_argument("--heads", default="", help="explicit 'L5H2,L7H8' override")
    ap.add_argument("--seqs", type=int, default=16,
                    help="16 matches §3.12-S, so the L5H2 x L7H8 cell is a "
                         "reproduction check rather than a new number")
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--probe", default="wide", choices=tuple(PROBE_ARMS),
                    help="token range the repeated sequences are drawn from. "
                         "'wide' is every existing number; 'freq' lowers the "
                         "baseline NLL and so buys headroom against the "
                         "uniform ceiling -- the handle on the censored cells "
                         "at a small rung. See PROBE_ARMS and "
                         "p8_scale_ladder/probe_distribution.py")
    ap.add_argument("--ablation", default="ov", choices=("ov", "zero", "mean"),
                    help="'mean' replaces each head's output with its clean-run "
                         "mean -- the control for zero-ablation's "
                         "off-distribution bias (status-8.md's A/B)")
    ap.add_argument("--headroom-warn", type=float, default=2.0,
                    help="flag cells whose joint arm lands within this many "
                         "nats of the uniform ceiling")
    ap.add_argument("--out", default="",
                    help="a non-default --model gets its own default filename "
                         "instead of the 410m one")
    args = ap.parse_args()
    _abl = "" if args.ablation == "ov" else f"_{args.ablation}"
    _abl += "" if args.probe == "wide" else f"_{args.probe}"
    if args.out:
        out = Path(args.out)
    elif args.model == DEFAULT_MODEL and not _abl:
        out = OUT
    else:
        out = (DATA / "analysis" /
               f"pairwise_interaction_matrix_{args.model}{_abl}.json")

    def parse(spec):
        return [(int(h[1:h.index("H")]), int(h[h.index("H") + 1:]))
                for h in spec.split(",") if h]

    if args.heads:
        heads, source = parse(args.heads), "--heads"
    else:
        heads, source = members(args.top, catalog_path_for(args.model))
    steps = [int(x) for x in args.steps.split(",") if x]
    names = {k: f"L{k[0]}H{k[1]}" for k in heads}
    cells = list(itertools.combinations(heads, 2))

    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    res = {"_what_this_is": __doc__, "git_sha": git_sha, "model": args.model,
           "ablation": args.ablation, "probe": args.probe,
           "probe_vocab_range": list(PROBE_ARMS[args.probe]),
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
           "membership_source": source, "uniform_ceiling_nll": CEILING,
           "eval": {"n_seqs": args.seqs, "seed": EVAL_SEED,
                    "vocab_range": list(PROBE_ARMS[args.probe]),
                    "headroom_warn": args.headroom_warn},
           "heads": [names[k] for k in heads], "steps": steps, "per_step": {}}

    print(f"model:   {args.model}   probe: {args.probe} "
          f"{PROBE_ARMS[args.probe]}   ablation: {args.ablation}")
    print(f"members: {', '.join(names[k] for k in heads)}   ({source})")
    print(f"grid:    {len(steps)} steps, {len(cells)} pairs, {args.seqs} "
          f"sequences, {1 + len(heads) + len(cells)} arms per step")
    print(f"ceiling: uniform NLL {CEILING:.3f}; cells within "
          f"{args.headroom_warn} nats are flagged, not trusted\n")

    for s in steps:
        model, _ = load_causal_lm(f"{args.model}-step{s}")
        model.eval()
        d_model, d_head, n_heads = arch_dims(model)
        ids = batch(np.random.default_rng(EVAL_SEED), args.seqs,
                    *PROBE_ARMS[args.probe])
        means = (head_means(model, ids, heads, args.chunk)
                 if args.ablation == "mean" else None)

        def run(ablate, want_vec):
            with ablated(model, ablate, args.ablation, means):
                return probe(model, ids, args.chunk, want_vec)

        nll0, vec0 = probe(model, ids, args.chunk, True)
        print(f"step {s}: baseline NLL {nll0:.4f}, headroom to ceiling "
              f"{CEILING - nll0:.2f} nats", flush=True)

        single, delta = {}, {}
        for k in heads:
            nll, vec = run([k], True)
            single[k] = nll - nll0
            delta[k] = vec - vec0
            print(f"    {names[k]:>7}  dNLL {single[k]:>+8.4f}", flush=True)

        print(f"\n{'pair':>17} {'joint':>9} {'Σparts':>9} {'INTER':>9} "
              f"{'ratio':>7} {'cos':>7} {'joint NLL':>10}")
        pairs, flagged = {}, []
        for a, b in cells:
            nll_j, _ = run([a, b], False)
            dJ = nll_j - nll0
            parts = single[a] + single[b]
            inter = dJ - parts
            da, db = delta[a], delta[b]
            cos = float(np.dot(da, db) /
                        (np.linalg.norm(da) * np.linalg.norm(db)))
            head = CEILING - nll_j
            key = f"{names[a]}x{names[b]}"
            pairs[key] = {"a": names[a], "b": names[b],
                          "d_a": single[a], "d_b": single[b],
                          "joint": dJ, "parts_sum": parts, "interaction": inter,
                          "ratio_joint_to_parts": dJ / parts if parts else None,
                          "delta_cosine": cos,
                          "joint_nll": nll_j, "headroom": head,
                          # §3.13's second view: the same cell against the
                          # checkpoint's own baseline, which falls ~20x across
                          # the grid.
                          "joint_rel": dJ / nll0, "interaction_rel": inter / nll0,
                          "ceiling_contaminated": bool(head < args.headroom_warn)}
            if head < args.headroom_warn:
                flagged.append(key)
            print(f"{key:>17} {dJ:>+9.4f} {parts:>+9.4f} {inter:>+9.4f} "
                  f"{(dJ/parts if parts else float('nan')):>7.2f} {cos:>+7.3f} "
                  f"{nll_j:>10.4f}{'  <- CEILING' if head < args.headroom_warn else ''}",
                  flush=True)

        nll_chk, _ = probe(model, ids, args.chunk, False)
        res["per_step"][str(s)] = {
            "baseline_nll": nll0, "restore_abs_diff": abs(nll_chk - nll0),
            "dnll": {names[k]: single[k] for k in heads},
            "dnll_rel": {names[k]: single[k] / nll0 for k in heads},
            "delta_norm": {names[k]: float(np.linalg.norm(delta[k]))
                           for k in heads},
            "pairs": pairs, "ceiling_contaminated_cells": flagged}

        # Written per step: a killed run must not lose the steps it finished.
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(res, fh, indent=2)

        v = np.array([c["interaction"] for c in pairs.values()])
        print(f"\n  restore {abs(nll_chk - nll0):.1e}   "
              f"super-additive (>+0.02): {(v > 0.02).sum()}/{len(v)}   "
              f"sub (<-0.02): {(v < -0.02).sum()}   ~0: {(np.abs(v) <= 0.02).sum()}")
        if flagged:
            print(f"  {len(flagged)} cell(s) against the ceiling and NOT "
                  f"interpretable at raw dNLL: {', '.join(flagged)}")
        print()
        del model, ids, single, delta, vec0
        gc.collect()

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
