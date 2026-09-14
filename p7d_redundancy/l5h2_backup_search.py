"""Who backs `L7H8` up when `L5H2` is ablated alone? A full-model search.

WHY THIS EXISTS
---------------
`PROJECT.md` §3.20 / `status-7d.md` ("The upstream-relay check") closed the
mechanism half of §3.12-U's `L5H2` puzzle: ablating `L5H2` demonstrably
degrades `L7H8`'s own induction-attention score (60-600x any generic control,
0.0000 for a magnitude-matched one), so `L5H2` genuinely feeds `L7H8`'s
matching mechanism. That leaves the half it does NOT close: §3.12-S found the
JOINT ablation of `L5H2` + `L7H8` is *super*-additive (2.2x the sum of the
parts), which is the opposite of what a dependency this direct predicts (a
serial circuit predicts *sub*-additivity -- remove the input, the matcher has
less to match on, so removing both should cost less extra than removing
`L7H8` alone). §3.16 already invokes Hydra-effect self-repair (2307.15771) for
44/45 pairwise cells in the interaction matrix; the standing, UNTESTED
hypothesis is that self-repair is what reconciles the two findings here too --
something elsewhere in the 384 heads partly restores `L7H8`'s effect on the
LOSS when `L5H2` alone is gone, without restoring `L7H8`'s own ATTENTION
PATTERN (which is exactly why the upstream-relay check, reading attention
rather than NLL, could still see the dependency `L5H2` x `L7H8`'s joint ΔNLL
number cannot).

THE SEARCH. Self-repair here would show up as some OTHER head's own
induction-attention score RISING when `L5H2` is ablated -- a head partly
taking over the matching job `L7H8` is doing less of. This is cheap to look
for directly rather than guessing: `induction_scores` already batches over
every head in one attention-output forward pass, so scoring all 384 heads
under baseline and under `L5H2`-ablation costs exactly the two forward passes
this file's own `upstream_relay_check.py` was already paying for one head.

WHAT THIS DOES NOT DO. A rising induction-attention score is evidence a head
could be compensating; it is not itself a causal test of whether that head's
OV ablation newly costs more once `L5H2` is gone (the ΔNLL-marginal-effect
version of the same idea, which is the harder and more expensive causal
follow-up if this narrows the search). No p-value; nothing registered;
pythia-410m spent under `check_registry` rule 3.
"""
import argparse
import gc
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
from tools.run.induction_rank_sweep import PROBE_ARMS, EVAL_SEED, ablated, arch_dims
from p7d_redundancy.member_formation_curves import batch
from p7d_redundancy.fv_score import induction_scores

OUT = DATA / "analysis" / "l5h2_backup_search.json"
DEFAULT_MODEL = "pythia-410m"


def parse_head(s):
    return (int(s[1:s.index("H")]), int(s[s.index("H") + 1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--steps", default="4000,16000,143000",
                     help="L7H8's own formation window plus the trained "
                          "endpoint -- the two points where the "
                          "upstream-relay check's delta was largest and "
                          "where it stabilised")
    ap.add_argument("--source", default="L5H2", help="head to ablate")
    ap.add_argument("--seqs", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--probe", default="wide", choices=tuple(PROBE_ARMS))
    ap.add_argument("--top-n", type=int, default=15,
                     help="how many risers/fallers to print and keep")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    source = parse_head(args.source)
    out = Path(args.out) if args.out else (
        OUT if args.model == DEFAULT_MODEL else
        DATA / "analysis" / f"l5h2_backup_search_{args.model}.json")

    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    steps = [int(x) for x in args.steps.split(",") if x]

    res = {"_what_this_is": __doc__, "git_sha": git_sha, "model": args.model,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
           "source": args.source, "probe": args.probe, "n_seqs": args.seqs,
           "steps": steps, "per_step": {}}

    print(f"model: {args.model}   ablated: {args.source}   "
          f"scoring all heads' own induction-attention\n")

    for s in steps:
        model, _ = load_causal_lm(f"{args.model}-step{s}")
        model.eval()
        _, _, n_heads = arch_dims(model)
        n_layers = model.config.num_hidden_layers
        all_heads = [(L, H) for L in range(n_layers) for H in range(n_heads)]

        ids = batch(np.random.default_rng(EVAL_SEED), args.seqs,
                    *PROBE_ARMS[args.probe])

        base = induction_scores(model, ids, all_heads, args.chunk)
        with ablated(model, [source], mode="ov"):
            abl = induction_scores(model, ids, all_heads, args.chunk)
        chk = induction_scores(model, ids, [source], args.chunk)

        rows = []
        for h in all_heads:
            if h == source:
                continue
            d = abl[h] - base[h]
            rows.append({"head": f"L{h[0]}H{h[1]}", "layer": h[0],
                         "head_idx": h[1], "baseline": base[h],
                         "ablated": abl[h], "delta": d})
        rows.sort(key=lambda r: -r["delta"])
        risers = rows[:args.top_n]
        fallers = sorted(rows, key=lambda r: r["delta"])[:args.top_n]

        restore_diff = abs(chk[source] - base[source])
        res["per_step"][str(s)] = {
            "restore_check_abs_diff": restore_diff,
            "risers": risers, "fallers": fallers,
            "n_heads_scored": len(rows)}

        print(f"step {s:>6}  restore check abs diff {restore_diff:.2e}")
        print(f"  top risers (own induction score UP when {args.source} "
              f"ablated -- backup candidates):")
        for r in risers[:8]:
            print(f"    {r['head']:>7}  {r['baseline']:.4f} -> "
                  f"{r['ablated']:.4f}  delta {r['delta']:+.4f}")
        print(f"  top fallers (own induction score DOWN -- also fed by "
              f"{args.source}):")
        for r in fallers[:8]:
            print(f"    {r['head']:>7}  {r['baseline']:.4f} -> "
                  f"{r['ablated']:.4f}  delta {r['delta']:+.4f}")
        print()

        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(res, fh, indent=2)

        del model
        gc.collect()

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
