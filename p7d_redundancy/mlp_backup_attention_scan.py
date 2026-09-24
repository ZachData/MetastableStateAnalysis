"""Which MLP backs up the relay, measured on ATTENTION rather than on loss.

WHY THIS EXISTS
---------------
`PROJECT.md` §3.22-§3.25 built the self-repair chain on pythia-410m and it is
`n = 1`. Porting it to the 70m rung (the other exploration rung --
`design-8.md`'s rung policy; 1b/1.4b stay reserved) runs straight into the
blocker §3.17 recorded: **70m's relay `L2H1` costs +6.88 on a 2.67 baseline, so
it censors its own cells against `ln 50304` even on the `freq` probe.** Every
ΔNLL interaction in §3.22/§3.23 is therefore unavailable at that rung, and
`mlp_backup_check.py` cannot be pointed at it honestly.

Attention is immune to that ceiling: a distribution over keys is well defined
however badly the model is doing. §3.25 already used total-variation distance
for exactly this reason, and this runner turns it into the MLP search itself.

WHAT IT MEASURES, per MLP layer:

    unconditional_tv = TV( attention | MLP ablated ,  attention )
    conditional_tv   = TV( attention | source + MLP ablated ,  attention | source )

over every head DOWNSTREAM of that MLP (heads at or below it cannot be
affected, and are reported as the zero check). The backup is the MLP whose
conditional disruption most exceeds its unconditional one -- the same
conditional-vs-unconditional logic §3.22 used on the loss, now on a readout
that survives the ceiling.

**BOTH THE MEAN AND THE MAX, per §3.13, and here the max is the instrument that
works.** The first version of this runner reported only the mean over
downstream heads and its 410m positive control FAILED: it nominated MLP 5 and
put MLP 6 at -0.0160, because MLP 6's effect is enormous on ONE head (`L7H8`,
0.751 -> 0.045) out of 272 and a mean over 272 averages it away. That is
§3.12-G6's finding one axis over -- a mean over a population where one unit
carries the signal destroys it -- and it is exactly what §3.13 exists to
prevent. The max over downstream heads recovers MLP 6; both are reported and
neither was chosen after seeing this rung's data.

WHY TV OVER ALL DOWNSTREAM HEADS rather than one target head's induction score:
at 70m there is **no causally-important head that also does induction
attention** (measured 2026-09-13: the top induction-attention head is `L0H3` at
0.906, which sits in layer 0 with a catalogue ΔNLL of **−0.839**, while every
head above +1.0 in the catalogue scores under 0.024). A target-head readout
would have nothing to point at. TV presumes no role.

POSITIVE CONTROL FIRST. Run at 410m before trusting it at 70m: the instrument
must recover MLP 6, which §3.23 established by four other routes. A new
instrument that cannot reproduce a known answer is not evidence about a new
rung. `--model pythia-410m --source L5H2` is that control.

No p-value; nothing registered; both rungs here are exploration and spent under
`check_registry` rule 3.
"""
import argparse
import contextlib
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
from tools.run.induction_rank_sweep import N_REP, PROBE_ARMS, EVAL_SEED, ablated
from p7d_redundancy.member_formation_curves import batch
from p7d_redundancy.mlp_backup_check import ablated_mlp, mlp_output_means

OUT = DATA / "analysis" / "mlp_backup_attention_scan.json"


def parse_head(s):
    return (int(s[1:s.index("H")]), int(s[s.index("H") + 1:]))


@torch.no_grad()
def tv_per_head(model, ids, ctx_a, ctx_b, n_layers, n_heads, chunk=2):
    """Mean TV distance per head between two intervened states, same inputs.

    Restricted to SECOND-COPY query positions (`i >= N_REP`). Averaging over
    all positions instead halves any second-copy-only effect against the first
    copy, where there is no match to make -- which is what made the 410m
    positive control put `L7H8` below a layer-23 head: its induction score
    falls 0.751 -> 0.045 but half the queries it was averaged over never
    moved. This scope matches `induction_scores`.
    """
    tot = np.zeros((n_layers, n_heads))
    n = 0
    for i in range(0, len(ids), chunk):
        x = ids[i:i + chunk]
        with ctx_a():
            a = [t.detach().float()
                 for t in model(x, output_attentions=True).attentions]
        with ctx_b():
            b = [t.detach().float()
                 for t in model(x, output_attentions=True).attentions]
        for L in range(n_layers):
            tv = 0.5 * (a[L][:, :, N_REP:, :] - b[L][:, :, N_REP:, :]).abs().sum(-1)
            tot[L] += tv.mean(dim=(0, 2)).double().numpy() * x.shape[0]
        n += x.shape[0]
        del a, b
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-410m")
    ap.add_argument("--steps", default="16000,143000")
    ap.add_argument("--source", default="L5H2",
                     help="the relay: L5H2 at 410m, L2H1 at 70m")
    ap.add_argument("--mode", default="mean", choices=("mean", "zero"),
                     help="MLP ablation mode; `mean` is §3.15's control for "
                          "zero's off-distribution bias and is the default")
    ap.add_argument("--seqs", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--probe", default="wide", choices=tuple(PROBE_ARMS))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    source = parse_head(args.source)
    out = Path(args.out) if args.out else (
        OUT if args.model == "pythia-410m" else
        DATA / "analysis" / f"mlp_backup_attention_scan_{args.model}.json")
    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    steps = [int(x) for x in args.steps.split(",") if x]

    res = {"_what_this_is": __doc__, "git_sha": git_sha, "model": args.model,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
           "source": args.source, "mode": args.mode, "probe": args.probe,
           "n_seqs": args.seqs, "steps": steps, "per_step": {}}

    for s in steps:
        model, _ = load_causal_lm(f"{args.model}-step{s}")
        model.eval()
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
        ids = batch(np.random.default_rng(EVAL_SEED), args.seqs,
                    *PROBE_ARMS[args.probe])

        means_clean = mlp_output_means(model, ids, args.chunk)
        with ablated(model, [source], mode="ov"):
            means_cond = mlp_output_means(model, ids, args.chunk)

        @contextlib.contextmanager
        def nothing():
            yield

        @contextlib.contextmanager
        def src_only():
            with ablated(model, [source], mode="ov"):
                yield

        rows = {}
        print(f"== {args.model} step {s}   source {args.source}   "
              f"mode {args.mode}   probe {args.probe}")
        print(f"  {'MLP':>4} {'mean_u':>8} {'mean_c':>8} {'d_mean':>8} "
              f"{'max_u':>8} {'max_c':>8} {'d_max':>9} {'argmax head':>12} "
              f"{'zero':>7}")
        for L in range(n_layers):
            @contextlib.contextmanager
            def mlp_only(L=L):
                with ablated_mlp(model, L, mode=args.mode, means=means_clean):
                    yield

            @contextlib.contextmanager
            def src_and_mlp(L=L):
                with ablated(model, [source], mode="ov"):
                    with ablated_mlp(model, L, mode=args.mode, means=means_cond):
                        yield

            tv_u = tv_per_head(model, ids, nothing, mlp_only,
                               n_layers, n_heads, args.chunk)
            tv_c = tv_per_head(model, ids, src_only, src_and_mlp,
                               n_layers, n_heads, args.chunk)
            down = [(l, h) for l in range(L + 1, n_layers) for h in range(n_heads)]
            zc = max([tv_c[l, h] for l in range(L + 1) for h in range(n_heads)]
                     + [0.0])
            if not down:
                rows[str(L)] = {"layer": L, "n_downstream": 0,
                                "delta_mean": None, "delta_max": None,
                                "zero_check_max_tv": float(zc)}
                print(f"  {L:>4}   (no downstream heads)")
                continue
            vu = np.array([tv_u[l, h] for l, h in down])
            vc = np.array([tv_c[l, h] for l, h in down])
            u, c = float(vu.mean()), float(vc.mean())
            mu_, mc = float(vu.max()), float(vc.max())
            amax = down[int(vc.argmax())]
            rows[str(L)] = {
                "layer": L, "n_downstream": len(down),
                "unconditional_tv_mean": u, "conditional_tv_mean": c,
                "delta_mean": c - u,
                "unconditional_tv_max": mu_, "conditional_tv_max": mc,
                "delta_max": mc - mu_,
                "argmax_head": f"L{amax[0]}H{amax[1]}",
                "zero_check_max_tv": float(zc)}
            print(f"  {L:>4} {u:>8.4f} {c:>8.4f} {c - u:>+8.4f} "
                  f"{mu_:>8.4f} {mc:>8.4f} {mc - mu_:>+9.4f} "
                  f"{f'L{amax[0]}H{amax[1]}':>12} {zc:>7.0e}", flush=True)

        scored = [r for r in rows.values() if r["delta_max"] is not None]
        b_mean = max(scored, key=lambda r: r["delta_mean"])
        b_max = max(scored, key=lambda r: r["delta_max"])
        res["per_step"][str(s)] = {
            "mlps": rows,
            "argmax_delta_mean_layer": b_mean["layer"],
            "argmax_delta_max_layer": b_max["layer"]}
        print(f"  -> by MAX (the instrument that passes its control): MLP "
              f"{b_max['layer']}  d_max {b_max['delta_max']:+.4f} on "
              f"{b_max['argmax_head']}")
        print(f"  -> by MEAN (diluted, recorded per §3.13): MLP "
              f"{b_mean['layer']}  d_mean {b_mean['delta_mean']:+.4f}\n")

        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(res, fh, indent=2)
        del model

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
