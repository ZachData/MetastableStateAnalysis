"""Do the MLPs stand in for `L5H2`? The untested half of §3.21's self-repair.

WHY THIS EXISTS
---------------
`PROJECT.md` §3.21 found that three redundancy-set members reproduce `L7H8`'s
super-additive interaction with `L5H2`, and listed one gap by name: **the MLPs
are untested.** Every composition and interaction measurement in §3.12-§3.21 is
head->head, and §3.12-Q6 is the only one that looked at an MLP at all (`L7H8`'s
OV into each downstream MLP's input projection: best rank 44 of 159, z +0.57 --
no elevated pathway). That was a *weights-only* composition score on a
different question. Whether an MLP picks up causal slack when `L5H2` is ablated
has never been measured, and the MLPs are the majority of the parameters.

WHAT IT MEASURES, per layer, the same four-arm structure §3.21 used for heads:

    solo        = dNLL(MLP ablated)
    joint       = dNLL(L5H2 + MLP ablated)
    marginal    = joint - dNLL(L5H2 alone)
    interaction = joint - dNLL(L5H2 alone) - solo

BOTH ABLATION MODES, because §3.15 exists for this reason and an MLP is a
much larger object than a head. `zero` removes the sublayer's contribution
outright; `mean` replaces its output with that output's own mean over (batch,
position), taken on a CLEAN pass, which is the control for zero-ablation's
off-distribution bias. Neither is privileged here and both are reported.
`L5H2` itself is always ablated the same way as everywhere else in this thread
(`ov`, via `write_ov`), so its arm stays comparable with §3.20/§3.21.

THE CEILING IS THE REAL HAZARD HERE and it is why this runner flags rather
than hides it. `dNLL` is bounded by `ln 50304 = 10.826` (§3.14.4-D); `L5H2`
alone already costs ~+2.0 on a ~0.58 baseline, and removing a whole MLP on top
of that can land within a nat of the bound. §3.12-M5: the compressive readout
biases interactions toward apparent SUB-additivity, so for a contaminated arm
a positive interaction is conservative while a negative one is not evidence.
`headroom` and `ceiling_contaminated` are stored per arm and printed.

COST: one model load, one clean pass for the means, then 2 + 2*n_layers NLL
evaluations per mode. No p-value; nothing registered; pythia-410m spent under
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
from tools.run.induction_rank_sweep import PROBE_ARMS, EVAL_SEED, ablated
from p7d_redundancy.member_formation_curves import batch
from p7d_redundancy.redundancy_catalog import nll

OUT = DATA / "analysis" / "mlp_backup_check.json"
DEFAULT_MODEL = "pythia-410m"
LN_VOCAB = float(np.log(50304))


def parse_head(s):
    return (int(s[1:s.index("H")]), int(s[s.index("H") + 1:]))


@torch.no_grad()
def mlp_output_means(model, ids, chunk=8):
    """`{layer: mean MLP output vector}` from a clean pass.

    Precomputed on the clean run for the same reason `head_means` is: taken
    in-pass it would be chunk-dependent, and for any arm with an upstream
    intervention it would be measuring the intervened distribution.
    """
    acc = {}

    def make_hook(L):
        def hook(mod, args, out):
            v = out.reshape(-1, out.shape[-1])
            s, n = acc.get(L, (0.0, 0))
            acc[L] = (s + v.double().sum(0), n + v.shape[0])
            return out
        return hook

    handles = [model.gpt_neox.layers[L].mlp.register_forward_hook(make_hook(L))
               for L in range(model.config.num_hidden_layers)]
    try:
        for i in range(0, len(ids), chunk):
            model(ids[i:i + chunk])
    finally:
        for h in handles:
            h.remove()
    return {L: (s / n) for L, (s, n) in acc.items()}


@contextlib.contextmanager
def ablated_mlp(model, layer, mode="mean", means=None):
    """Replace one MLP sublayer's output with zeros or with its clean mean."""
    if mode not in ("zero", "mean"):
        raise ValueError(f"mode must be 'zero' or 'mean', got {mode!r}")

    def hook(mod, args, out):
        if mode == "zero":
            return torch.zeros_like(out)
        return means[layer].to(out.dtype).expand_as(out).clone()

    h = model.gpt_neox.layers[layer].mlp.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--step", type=int, default=16000)
    ap.add_argument("--source", default="L5H2")
    ap.add_argument("--modes", default="mean,zero")
    ap.add_argument("--seqs", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--probe", default="wide", choices=tuple(PROBE_ARMS))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    source = parse_head(args.source)
    out = Path(args.out) if args.out else (
        OUT if args.model == DEFAULT_MODEL else
        DATA / "analysis" / f"mlp_backup_check_{args.model}.json")
    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    modes = [m for m in args.modes.split(",") if m]

    model, _ = load_causal_lm(f"{args.model}-step{args.step}")
    model.eval()
    n_layers = model.config.num_hidden_layers

    ids = batch(np.random.default_rng(EVAL_SEED), args.seqs,
                *PROBE_ARMS[args.probe])
    base = nll(model, ids, args.chunk)
    with ablated(model, [source], mode="ov"):
        nll_source = nll(model, ids, args.chunk)
    d_source = nll_source - base
    means = mlp_output_means(model, ids, args.chunk)
    print(f"baseline NLL {base:.4f}   {args.source} alone dNLL {d_source:+.4f} "
          f"(absolute {nll_source:.4f}; ln V = {LN_VOCAB:.3f})\n", flush=True)

    res = {"_what_this_is": __doc__, "git_sha": git_sha, "model": args.model,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
           "step": args.step, "source": args.source, "probe": args.probe,
           "n_seqs": args.seqs, "ln_vocab": LN_VOCAB,
           "baseline_nll": base, "dnll_source_alone": d_source,
           "source_ablation_mode": "ov", "per_mode": {}}

    for mode in modes:
        rows = {}
        print(f"-- MLP ablation mode: {mode} --")
        print(f"  {'layer':>5} {'solo':>9} {'marginal':>10} "
              f"{'interaction':>12} {'headroom':>9}")
        for L in range(n_layers):
            with ablated_mlp(model, L, mode=mode, means=means):
                solo = nll(model, ids, args.chunk) - base
            with ablated(model, [source], mode="ov"):
                with ablated_mlp(model, L, mode=mode, means=means):
                    nll_joint = nll(model, ids, args.chunk)
            joint = nll_joint - base
            marginal = joint - d_source
            interaction = joint - d_source - solo
            headroom = LN_VOCAB - nll_joint
            rows[str(L)] = {
                "layer": L, "dnll_solo": solo, "dnll_joint": joint,
                "dnll_marginal_given_source": marginal,
                "interaction": interaction,
                "nll_joint_absolute": nll_joint, "headroom": headroom,
                "ceiling_contaminated": headroom < 1.5}
            print(f"  {L:>5} {solo:>+9.4f} {marginal:>+10.4f} "
                  f"{interaction:>+12.4f} {headroom:>9.3f}"
                  f"{'  [CEILING]' if headroom < 1.5 else ''}", flush=True)
        inter = np.array([r["interaction"] for r in rows.values()])
        res["per_mode"][mode] = {
            "layers": rows,
            "interaction_median": float(np.median(inter)),
            "interaction_max": float(inter.max()),
            "interaction_min": float(inter.min()),
            "argmax_layer": int(np.argmax(inter)),
            "n_ceiling_contaminated": int(
                sum(r["ceiling_contaminated"] for r in rows.values()))}
        print(f"  median interaction {np.median(inter):+.4f}   "
              f"max {inter.max():+.4f} at layer {int(np.argmax(inter))}   "
              f"({res['per_mode'][mode]['n_ceiling_contaminated']} of "
              f"{n_layers} arms ceiling-contaminated)\n")

        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(res, fh, indent=2)

    chk = abs(nll(model, ids, args.chunk) - base)
    res["restore_abs_diff"] = float(chk)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"restore check abs diff {chk:.2e}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
