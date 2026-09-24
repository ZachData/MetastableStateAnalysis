"""Does MLP 6 RESPOND to `L5H2`'s ablation, or was it always doing this?

WHY THIS EXISTS
---------------
`mlp6_content_vs_scale.py` (2026-09-13) showed that `L7H8`'s matching, once
`L5H2` is gone, depends on a specific largely position-independent direction in
MLP 6's output: replacing MLP 6's whole output with its own mean `mu` keeps the
attention (0.64 at step 16000, 0.57 at 143000) while a norm-matched RANDOM
constant destroys it exactly as zeroing does (0.03-0.05 / 0.011-0.014), with
both ablations leaving the residual norm entering layer 7 essentially equal.

That leaves the question this thread has been circling since §3.21 without ever
stating it precisely. §3.16 imported "self-repair" from the Hydra-effect
literature, which is a claim that components **change their behaviour** to
compensate. But the measurements here are all ablation interactions, and a
large interaction is equally consistent with a component that does **exactly
the same thing** and merely becomes load-bearing once its partner is gone.
Those are different mechanisms and nothing measured so far separates them:

  ACTIVE self-repair    -- MLP 6's output MOVES when `L5H2` is ablated
  PRE-EXISTING redundancy -- MLP 6's output is unchanged; only its necessity changes

WHAT IT MEASURES, all off the same two forward passes:

1. `cos(mu_clean, mu_cond)` and `||mu_cond|| / ||mu_clean||` for each MLP --
   does the mean output move when the relay is removed? Under pre-existing
   redundancy this is ~1.0 and ~1.0; under active self-repair MLP 6 should
   move more than the MLPs that are not carrying the backup.
2. The same for the per-position output, as RMS `||out_cond - out_clean||`
   relative to RMS `||out_clean||`, so a change concentrated in the varying
   part is not hidden by averaging.
3. `cos(mu_MLP6, d_L5H2)` where `d_L5H2` is the residual contribution `L5H2`
   itself makes (mean of clean-minus-ablated residual entering layer 7). If
   MLP 6 is standing in for `L5H2` by writing where `L5H2` wrote, these should
   align; if it enables the matcher some other way, they need not.

The MLPs other than 6 are the built-in control: whatever generic drift
ablating `L5H2` causes downstream shows up in all of them.

No p-value; nothing registered; pythia-410m spent under `check_registry`
rule 3.
"""
import argparse
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

OUT = DATA / "analysis" / "mlp6_response.json"


def parse_head(s):
    return (int(s[1:s.index("H")]), int(s[s.index("H") + 1:]))


@torch.no_grad()
def mlp_outputs(model, ids, chunk=4):
    """`{layer: (n, d) float64 tensor}` of every MLP's output, all positions."""
    n_layers = model.config.num_hidden_layers
    acc = {L: [] for L in range(n_layers)}

    def make(L):
        def hook(mod, args, out):
            acc[L].append(out.detach().reshape(-1, out.shape[-1]).double())
            return out
        return hook

    handles = [model.gpt_neox.layers[L].mlp.register_forward_hook(make(L))
               for L in range(n_layers)]
    try:
        for i in range(0, len(ids), chunk):
            model(ids[i:i + chunk])
    finally:
        for h in handles:
            h.remove()
    return {L: torch.cat(v, 0) for L, v in acc.items()}


@torch.no_grad()
def resid_at(model, ids, layer, chunk=4):
    """Residual entering `layer`, all positions, float64."""
    got = []

    def hook(mod, args):
        got.append(args[0].detach().reshape(-1, args[0].shape[-1]).double())

    h = model.gpt_neox.layers[layer].register_forward_pre_hook(hook)
    try:
        for i in range(0, len(ids), chunk):
            model(ids[i:i + chunk])
    finally:
        h.remove()
    return torch.cat(got, 0)


def cos(a, b):
    return float(torch.dot(a, b) / max(float(a.norm() * b.norm()), 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-410m")
    ap.add_argument("--steps", default="16000,143000")
    ap.add_argument("--source", default="L5H2")
    ap.add_argument("--target", default="L7H8")
    ap.add_argument("--seqs", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--probe", default="wide", choices=tuple(PROBE_ARMS))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    source, target = parse_head(args.source), parse_head(args.target)
    out = Path(args.out) if args.out else OUT
    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    steps = [int(x) for x in args.steps.split(",") if x]

    res = {"_what_this_is": __doc__, "git_sha": git_sha, "model": args.model,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
           "source": args.source, "target": args.target, "probe": args.probe,
           "n_seqs": args.seqs, "steps": steps, "per_step": {}}

    for s in steps:
        model, _ = load_causal_lm(f"{args.model}-step{s}")
        model.eval()
        ids = batch(np.random.default_rng(EVAL_SEED), args.seqs,
                    *PROBE_ARMS[args.probe])
        tgt_layer = target[0]

        clean_out = mlp_outputs(model, ids, args.chunk)
        clean_resid = resid_at(model, ids, tgt_layer, args.chunk)
        with ablated(model, [source], mode="ov"):
            cond_out = mlp_outputs(model, ids, args.chunk)
            cond_resid = resid_at(model, ids, tgt_layer, args.chunk)

        # What L5H2 itself contributes to the residual layer 7 reads.
        d_source = (clean_resid - cond_resid).mean(0)

        rows = {}
        print(f"== step {s}   ||d_{args.source}|| = {float(d_source.norm()):.3f}")
        print(f"  {'MLP':>4} {'cos(mu_clean,mu_cond)':>22} {'||mu_c||/||mu_0||':>18} "
              f"{'RMS Δout / RMS out':>20} {'cos(mu_cond, d_src)':>20}")
        for L in sorted(clean_out):
            a, b = clean_out[L], cond_out[L]
            mu_a, mu_b = a.mean(0), b.mean(0)
            rel = float((b - a).norm(dim=-1).pow(2).mean().sqrt()
                        / max(float(a.norm(dim=-1).pow(2).mean().sqrt()), 1e-30))
            rows[str(L)] = {
                "layer": L,
                "cos_mu_clean_cond": cos(mu_a, mu_b),
                "mu_norm_ratio": float(mu_b.norm() / max(float(mu_a.norm()), 1e-30)),
                "rel_output_change": rel,
                "cos_mu_cond_with_source_delta": cos(mu_b, d_source),
                "mu_norm_clean": float(mu_a.norm()),
                "mu_norm_cond": float(mu_b.norm())}
            r = rows[str(L)]
            mark = "  <-- MLP 6" if L == 6 else ""
            print(f"  {L:>4} {r['cos_mu_clean_cond']:>22.4f} "
                  f"{r['mu_norm_ratio']:>18.4f} {rel:>20.4f} "
                  f"{r['cos_mu_cond_with_source_delta']:>20.4f}{mark}")

        res["per_step"][str(s)] = {
            "source_delta_norm": float(d_source.norm()), "mlps": rows}
        print()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(res, fh, indent=2)
        del model

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
