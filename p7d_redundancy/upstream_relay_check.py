"""Does ablating `L5H2` actually degrade `L7H8`'s own induction attention?

WHY THIS EXISTS
---------------
`PROJECT.md` §3.12-U/§3.16/§3.18: `L5H2` has the largest single-head causal
effect on the second-copy readout of any redundancy-set member, yet scores
near zero on both instruments that have been pointed at it -- the QK-based
induction-attention score (§3.12-U, `fv_score.py`) and the FV score (§3.18).
Filed as a puzzle because neither instrument measures what a head's *causal*
role via composition into a downstream head would look like.

Two facts already on disk, never connected:

- **`L5H2` is confirmed to be a previous-token head on its own attention
  pattern** (Stage 0, PROJECT.md line ~1403: mean attention at offset -1 is
  0.895 against a 384-head median of 0.018, ~49x). It does not attend to the
  induction position -- so a near-zero induction-attention score is not a
  puzzle about `L5H2`'s own behaviour, it is the correct reading of a head
  that is not doing that job itself.
- **`L5H2`'s OV output composes strongly into `L7H8`'s attention read-space**
  (`induction_composition_whitening.py`, H1-REVISED): from rank 83/112 of the
  model's own composition distribution to rank 0 (z ~ +6), onset between step
  512 and 1000 -- exactly `L5H2`'s own formation window (§3.12-U) -- and
  sustained through training. That is a WEIGHTS-ONLY quantity: it says the
  composition pathway exists, not that it does anything.

**What has never been measured is the functional half.** If `L5H2` is
supplying `L7H8`'s matching signal through that composition, ablating `L5H2`
should measurably degrade `L7H8`'s OWN induction-attention score (the same
QK-based quantity `induction_scores` reads for `L5H2` itself) -- not just the
downstream NLL, which §3.12-S already showed is confounded by super-additive
self-repair from elsewhere in the 384-head population (the Hydra-effect
signature, §3.16). Reading attention directly bypasses that confound: a
network-wide backup can restore the NLL without restoring THIS head's own
attention pattern, so a drop here is closer to the composition claim itself
than the ablation-NLL numbers already on disk.

**The control this needs, per §3.12-V3's measured-null discipline.** Ablating
any head perturbs the residual stream by some amount, so a drop in `L7H8`'s
induction score after ablating `L5H2` is only informative against what
ablating a GENERIC head does. `--controls` draws heads uniformly from outside
the catalogue's own top members (`p7d_redundancy.member_formation_curves`'s
own pool convention, reused rather than reinvented) and scores them
identically.

`--background` (added 2026-09-13) ablates a fixed set in EVERY arm including
the baseline, which is what lets the same instrument ask the conditional
question §3.21's sweep raised: does a candidate backup restore the target's
attention **once the relay is already gone**? On the clean model a backup is
by definition redundant, so the unconditional arm reads ~0 for it and says
nothing either way.

WHAT THIS DOES NOT SETTLE. A drop confirms the composition is functional but
not that it is THE explanation for `L5H2`'s causal effect -- §3.12-S's
residual-delta-cosine reading (structurally distinct operators converging on
the same effect through the network, not by sharing a pathway) is a live
alternative and would predict little or no change here. No p-value; nothing
registered; pythia-410m is spent under `check_registry` rule 3.
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
# Script-time only: modules are importable by tests on runners that are not
# this machine\'s .venv; a real run still refuses the wrong interpreter.
if __name__ == "__main__" and not sys.prefix.startswith(_want):
    raise SystemExit(f"wrong interpreter: {sys.prefix!r}, need {_want!r}")

import numpy as np

from core.lm_loading import load_causal_lm
from tools.run.induction_rank_sweep import PROBE_ARMS, EVAL_SEED, ablated, arch_dims
from p7d_redundancy.member_formation_curves import batch, catalog_path_for, members
from p7d_redundancy.fv_score import induction_scores

OUT = DATA / "analysis" / "upstream_relay_check.json"
DEFAULT_MODEL = "pythia-410m"


def parse_head(s):
    return (int(s[1:s.index("H")]), int(s[s.index("H") + 1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--steps", default="1000,2000,4000,8000,16000,143000",
                     help="§3.12-U's window plus the endpoint, matching "
                          "fv_score.py's own default range")
    ap.add_argument("--source", default="L5H2", help="head to ablate")
    ap.add_argument("--target", default="L7H8",
                     help="head whose own induction-attention score is read")
    ap.add_argument("--controls", type=int, default=4,
                     help="generic heads ablated identically, as the "
                          "measured null (§3.12-V3)")
    ap.add_argument("--background", default="",
                     help="heads ablated in EVERY arm, the baseline reference "
                          "included, so each delta is measured against the "
                          "background-ablated state rather than the clean "
                          "model. `--background L5H2 --source L4H9` asks "
                          "whether a candidate backup restores the target's "
                          "attention once the relay is already gone -- a "
                          "question the clean-model arm cannot pose, because "
                          "on the clean model the backup is redundant "
                          "(§3.21).")
    ap.add_argument("--top", type=int, default=6,
                     help="catalogue members excluded from the control pool")
    ap.add_argument("--seqs", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--probe", default="wide", choices=tuple(PROBE_ARMS))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    source, target = parse_head(args.source), parse_head(args.target)
    background = [parse_head(h) for h in args.background.split(",") if h]
    out = Path(args.out) if args.out else (
        OUT if args.model == DEFAULT_MODEL else
        DATA / "analysis" / f"upstream_relay_check_{args.model}.json")

    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    steps = [int(x) for x in args.steps.split(",") if x]
    catalog_members, msource = members(args.top, catalog_path_for(args.model))

    res = {"_what_this_is": __doc__, "git_sha": git_sha, "model": args.model,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
           "source": args.source, "target": args.target,
           "background": [f"L{L}H{H}" for L, H in background],
           "membership_source": msource, "probe": args.probe,
           "n_seqs": args.seqs, "n_controls": args.controls,
           "steps": steps, "per_step": {}}

    print(f"model: {args.model}   source (ablated): {args.source}   "
          f"target (read): {args.target}"
          + (f"   background: {args.background}" if background else "") + "\n")

    for s in steps:
        model, _ = load_causal_lm(f"{args.model}-step{s}")
        model.eval()
        _, _, n_heads = arch_dims(model)
        n_layers = model.config.num_hidden_layers

        ids = batch(np.random.default_rng(EVAL_SEED), args.seqs,
                    *PROBE_ARMS[args.probe])

        rng_c = np.random.default_rng(EVAL_SEED)
        pool = [(L, H) for L in range(n_layers) for H in range(n_heads)
                if (L, H) not in set(catalog_members) | {source, target}]
        ctrl = [pool[i] for i in
                rng_c.choice(len(pool), size=args.controls, replace=False)]

        def measure(extra, want=(source, target)):
            """Scores with `background + extra` ablated, in ONE arm.

            One `ablated` call rather than nested ones: nesting two `ov`-mode
            contexts over overlapping heads would have the inner restore write
            back weights the outer had already zeroed.
            """
            heads = list(dict.fromkeys(list(background) + list(extra)))
            if not heads:
                return induction_scores(model, ids, list(want),
                                        max(1, args.chunk // 2))
            with ablated(model, heads, mode="ov"):
                return induction_scores(model, ids, list(want),
                                        max(1, args.chunk // 2))

        if source in background:
            raise SystemExit(
                f"--source {args.source} is also in --background; the source "
                f"must be ablated by the arm, not by the background")

        base = measure([])
        row = {"baseline_source": base[source], "baseline_target": base[target],
               "controls": [f"L{L}H{H}" for L, H in ctrl],
               "background": [f"L{L}H{H}" for L, H in background],
               "ablations": {}}

        for (L, H) in [source] + ctrl:
            abl = measure([(L, H)], want=(target,))
            d = abl[target] - base[target]
            row["ablations"][f"L{L}H{H}"] = {
                "target_induction_ablated": abl[target], "delta": d}
            tag = "SOURCE" if (L, H) == source else "control"
            print(f"  step {s:>6}  ablate L{L}H{H:<2} ({tag:>7})  "
                  f"target induction {base[target]:.4f} -> {abl[target]:.4f}  "
                  f"delta {d:+.4f}", flush=True)

        chk = measure([])
        restore_diff = max(abs(chk[source] - base[source]),
                            abs(chk[target] - base[target]))
        row["restore_check_abs_diff"] = restore_diff
        res["per_step"][str(s)] = row
        print(f"  step {s:>6}  restore check abs diff {restore_diff:.2e}\n")

        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(res, fh, indent=2)

        del model
        gc.collect()

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
