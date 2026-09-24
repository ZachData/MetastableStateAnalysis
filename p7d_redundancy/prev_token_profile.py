"""Prev-token attention for every head: is a stand-in for `L5H2` a head that
can supply what `L5H2` supplied?

WHY THIS EXISTS
---------------
`PROJECT.md` §3.21 left one question open by name: **why is `L11H14` the
strongest stand-in for `L5H2`?** Its solo causal effect is only +0.19 (fifth
in the set) yet its super-additive interaction with `L5H2` is +1.53 to +2.18,
and reading the on-disk 45-cell matrix against its own magnitude regression
(2026-09-13) sharpens that: `L5H2`x`L11H14` is the **largest positive residual
of all 45 cells at both checkpoints** (+1.37 at 16000, +0.97 at 143000). So it
is not explained by the rule that explains the other 44 cells -- §3.12-V's
"74-81 % of the interaction is the product of the two heads' own magnitudes".

THE HYPOTHESIS THIS TESTS. `L5H2` is a previous-token head (Stage 0: mean
attention at offset -1 = 0.895 against a 384-head median of 0.018) and §3.20
showed its causal role is to feed downstream matchers. The obvious mechanism
for standing in for it is **being able to supply the same signal**: a head with
real prev-token attention of its own can partly cover the lost offset -1
write, and a head with none cannot. If that is right, prev-token score should
predict which heads are stand-ins.

THE CONTROL IS BUILT IN, and it is what makes this a test rather than a
just-so story. This scores **all** heads, so it also finds heads with high
prev-token attention that are NOT stand-ins. If prev-token capacity is common
and stand-in behaviour is rare, the hypothesis is wrong -- capacity would not
be what selects them. Read against `backup_sweep_full.py`'s interaction
column (all heads, same step) this becomes a correlation over the whole
population rather than a story about one head.

CONVENTION, stated because this repo has two. `prev_token` here is mean
post-softmax attention from query `i` to key `i-1`, averaged over every query
position with a predecessor -- the quantity Stage 0 used. It is reported both
over all positions and restricted to the second-copy region (`i >= N_REP`),
where the induction readout lives, because a head could be a prev-token head
only in one regime. `induction` is the repo's own induction-attention score
(query `N_REP+j` -> key `j`), carried beside it unchanged so the two can be
compared per head; `what_l7h8_writes.py` PROBE 1 records the open question
about whether `j` or `j+1` is the right offset for that one, and nothing here
depends on its resolution.

COST: one attention-output forward pass per checkpoint -- every head is read
off the same pass. No ablation, no p-value, nothing registered.
"""
import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[1])))  # this checkout, not a hard-coded main tree
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))
_want = str(REPO / ".venv")
# Script-time only: modules are importable by tests on runners that are not
# this machine\'s .venv; a real run still refuses the wrong interpreter.
if __name__ == "__main__" and not sys.prefix.startswith(_want):
    raise SystemExit(f"wrong interpreter: {sys.prefix!r}, need {_want!r}")

import numpy as np
import torch

from core.lm_loading import load_causal_lm
from tools.run.induction_rank_sweep import N_REP, PROBE_ARMS, EVAL_SEED, arch_dims
from p7d_redundancy.member_formation_curves import batch
from p7d_redundancy.fv_score import induction_scores

OUT = DATA / "analysis" / "prev_token_profile.json"
DEFAULT_MODEL = "pythia-410m"


@torch.no_grad()
def prev_token_scores(model, ids, heads, chunk=2):
    """`{(layer, head): (prev_all, prev_second_copy)}` -- mean attention from
    query `i` to key `i-1`, over all `i >= 1` and over `i >= N_REP`."""
    want = {}
    for (L, H) in heads:
        want.setdefault(L, []).append(H)
    tot_all = {h: 0.0 for h in heads}
    tot_sec = {h: 0.0 for h in heads}
    n = 0
    T = ids.shape[1]
    i_all = torch.arange(1, T)
    i_sec = torch.arange(N_REP, T)
    for i in range(0, len(ids), chunk):
        out = model(ids[i:i + chunk], output_attentions=True)
        b = ids[i:i + chunk].shape[0]
        for L, Hs in want.items():
            att = out.attentions[L].float()
            for H in Hs:
                tot_all[(L, H)] += float(att[:, H, i_all, i_all - 1].mean()) * b
                tot_sec[(L, H)] += float(att[:, H, i_sec, i_sec - 1].mean()) * b
        n += b
        del out
    return {h: (tot_all[h] / n, tot_sec[h] / n) for h in heads}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--steps", default="4000,16000,143000")
    ap.add_argument("--seqs", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--probe", default="wide", choices=tuple(PROBE_ARMS))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out = Path(args.out) if args.out else (
        OUT if args.model == DEFAULT_MODEL else
        DATA / "analysis" / f"prev_token_profile_{args.model}.json")
    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    steps = [int(x) for x in args.steps.split(",") if x]

    res = {"_what_this_is": __doc__, "git_sha": git_sha, "model": args.model,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
           "probe": args.probe, "n_seqs": args.seqs, "n_rep": N_REP,
           "steps": steps, "per_step": {}}

    for s in steps:
        model, _ = load_causal_lm(f"{args.model}-step{s}")
        model.eval()
        _, _, n_heads = arch_dims(model)
        n_layers = model.config.num_hidden_layers
        all_heads = [(L, H) for L in range(n_layers) for H in range(n_heads)]

        ids = batch(np.random.default_rng(EVAL_SEED), args.seqs,
                    *PROBE_ARMS[args.probe])
        prev = prev_token_scores(model, ids, all_heads, args.chunk)
        ind = induction_scores(model, ids, all_heads, args.chunk)

        rows = {}
        for h in all_heads:
            pa, ps = prev[h]
            rows[f"L{h[0]}H{h[1]}"] = {
                "layer": h[0], "head_idx": h[1],
                "prev_token": pa, "prev_token_second_copy": ps,
                "induction": ind[h]}
        pv = np.array([r["prev_token"] for r in rows.values()])
        res["per_step"][str(s)] = {
            "prev_token_median": float(np.median(pv)),
            "prev_token_p99": float(np.percentile(pv, 99)),
            "heads": rows}

        top = sorted(rows.items(), key=lambda kv: -kv[1]["prev_token"])[:12]
        print(f"step {s:>6}   prev-token median {np.median(pv):.4f}   "
              f"p99 {np.percentile(pv, 99):.4f}")
        print(f"  {'head':>8} {'prev_tok':>9} {'prev(2nd)':>10} {'induction':>10}")
        for k, r in top:
            print(f"  {k:>8} {r['prev_token']:>9.4f} "
                  f"{r['prev_token_second_copy']:>10.4f} {r['induction']:>10.4f}")
        print()

        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(res, fh, indent=2)
        del model
        gc.collect()

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
