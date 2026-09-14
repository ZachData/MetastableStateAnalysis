"""Do the attention-score risers actually pick up causal slack when `L5H2` is
ablated? The ΔNLL-marginal-effect version of `l5h2_backup_search.py`'s finding.

WHY THIS EXISTS
---------------
`l5h2_backup_search.py` scored all 384 heads' own induction-attention under
baseline and under `L5H2`-ablation and found consistent RISERS: `L10H7`
(dominant at the trained endpoint, delta +0.136), `L10H15` (dominant
mid-training, +0.048 to +0.118), and three already-known redundancy-set
members -- `L11H14`, `L8H6`, `L8H9` -- rising modestly and consistently from
step 1000 onward. That is evidence a head COULD be compensating; it is not a
causal test of whether it does, and PROJECT.md §3.20 named exactly this as
the next step: search for what backs `L7H8` up when `L5H2` alone is ablated,
in the sense that matters for §3.12-S's puzzle -- the joint-ablation ΔNLL of
`L5H2` + `L7H8` is SUPER-additive (2.2x the sum of parts) when a direct
dependency predicts sub-additive, and self-repair elsewhere in the network is
the standing, untested explanation.

THE TEST. For each candidate head C, four arms: baseline NLL, `L5H2` alone,
C alone, and `L5H2` + C jointly. C's MARGINAL cost once `L5H2` is already gone
is `dnll(L5H2+C) - dnll(L5H2 alone)`; C's OWN solo cost is `dnll(C alone)`. If
C is genuinely picking up slack, its marginal cost when `L5H2` is already
missing should exceed its solo cost -- the same super-additive signature
§3.12-S found for `L7H8` itself, now asked of the candidates the attention
search surfaced instead of assumed.

CANDIDATES, fixed in advance from the attention search rather than chosen
after seeing this run's numbers: `L10H7`, `L10H15` (the two large,
non-member risers) and `L11H14`, `L8H6`, `L8H9` (known members that also
rose). `L7H8` itself is included as the value this is calibrated against --
§3.12-S already measured it at step 16000 (+4.15 interaction) and it should
reproduce here as a control on the method.

NO p-value; nothing registered; pythia-410m spent under `check_registry`
rule 3. A positive result here is suggestive of a mechanism, not proof of
THE mechanism -- other heads, or MLPs (never checked for this), could also
contribute.
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
from p7d_redundancy.redundancy_catalog import nll

OUT = DATA / "analysis" / "l5h2_backup_causal_check.json"
DEFAULT_MODEL = "pythia-410m"
DEFAULT_CANDIDATES = "L10H7,L10H15,L11H14,L8H6,L8H9,L7H8"


def parse_head(s):
    return (int(s[1:s.index("H")]), int(s[s.index("H") + 1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--steps", default="4000,16000,143000")
    ap.add_argument("--source", default="L5H2")
    ap.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    ap.add_argument("--seqs", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--probe", default="wide", choices=tuple(PROBE_ARMS))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    source = parse_head(args.source)
    candidates = [parse_head(c) for c in args.candidates.split(",") if c]
    out = Path(args.out) if args.out else (
        OUT if args.model == DEFAULT_MODEL else
        DATA / "analysis" / f"l5h2_backup_causal_check_{args.model}.json")

    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    steps = [int(x) for x in args.steps.split(",") if x]

    res = {"_what_this_is": __doc__, "git_sha": git_sha, "model": args.model,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
           "source": args.source,
           "candidates": [f"L{L}H{H}" for L, H in candidates],
           "probe": args.probe, "n_seqs": args.seqs,
           "steps": steps, "per_step": {}}

    print(f"model: {args.model}   source: {args.source}   "
          f"candidates: {', '.join(f'L{L}H{H}' for L, H in candidates)}\n")

    for s in steps:
        model, _ = load_causal_lm(f"{args.model}-step{s}")
        model.eval()

        ids = batch(np.random.default_rng(EVAL_SEED), args.seqs,
                    *PROBE_ARMS[args.probe])
        base = nll(model, ids, args.chunk)
        with ablated(model, [source], mode="ov"):
            nll_source_alone = nll(model, ids, args.chunk)
        d_source = nll_source_alone - base

        rows = {}
        print(f"step {s:>6}   baseline NLL {base:.4f}   "
              f"{args.source} alone dNLL {d_source:+.4f}")
        for (L, H) in candidates:
            tag = f"L{L}H{H}"
            with ablated(model, [(L, H)], mode="ov"):
                nll_c_alone = nll(model, ids, args.chunk)
            d_c_solo = nll_c_alone - base

            with ablated(model, [source, (L, H)], mode="ov"):
                nll_joint = nll(model, ids, args.chunk)
            d_joint = nll_joint - base
            d_c_marginal = d_joint - d_source
            interaction = d_joint - (d_source + d_c_solo)

            rows[tag] = {
                "dnll_solo": d_c_solo, "dnll_marginal_given_source": d_c_marginal,
                "dnll_joint": d_joint, "interaction": interaction,
                "marginal_exceeds_solo": d_c_marginal > d_c_solo}
            flag = "  <-- marginal > solo" if d_c_marginal > d_c_solo else ""
            print(f"  {tag:>7}  solo {d_c_solo:+.4f}   "
                  f"marginal(given {args.source}) {d_c_marginal:+.4f}   "
                  f"interaction {interaction:+.4f}{flag}")

        res["per_step"][str(s)] = {
            "baseline_nll": base, "dnll_source_alone": d_source,
            "candidates": rows}
        print()

        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(res, fh, indent=2)

        del model
        gc.collect()

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
