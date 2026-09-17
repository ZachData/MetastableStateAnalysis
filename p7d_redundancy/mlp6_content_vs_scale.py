"""Is MLP 6's conditional role content or scale? The zero-vs-mean gap, opened.

WHY THIS EXISTS
---------------
`mlp_relay_role.py` (2026-09-13) placed MLP 6 as an upstream supplier for
`L7H8`'s matching, but the two ablation modes disagree by 6.5x on how big the
effect is. With `L5H2` already ablated (`L7H8` attention 0.7513):

    zero-ablate MLP 6  ->  attention 0.0450   (delta -0.706, the matcher blinds)
    mean-ablate MLP 6  ->  attention 0.6425   (delta -0.109)

The difference between the two modes is **exactly the mean (constant) component
of MLP 6's output**: `zero` removes the constant and the variation, `mean`
removes only the variation. So the gap says MLP 6's load-bearing contribution
to the matcher is carried by its CONSTANT component, not by what it computes
per token -- or else `zero` is simply off-distribution in the way §3.15 warns
about and the `mean` number is the honest one. Those are different claims and
this runner separates them.

WHAT IT MEASURES

1. **How constant is the output.** `||mu|| / RMS(||out||)` and the variation
   norm, per layer. If MLP 6's output is dominated by a position-independent
   vector, `zero` and `mean` remove very different amounts of stuff and the
   gap is expected rather than anomalous.
2. **What each ablation does to the residual entering layer 7** -- the thing
   `L7H8` actually reads. If `zero` drives the residual far outside anything
   the model sees, the attention collapse is a distribution artifact and
   `mean` is the number to quote. If both modes leave the residual in a
   comparable regime and only `zero` blinds the matcher, the constant
   component is carrying real signal.
3. **A direction control.** Replacing MLP 6's output with a norm-matched
   RANDOM constant instead of its own mean: if what matters is merely having
   *something* of that size in the residual, the random constant recovers the
   attention; if the specific direction matters, it does not. That separates
   "scale" from "content" more sharply than the mode comparison can, and it is
   the arm neither §3.15 nor this thread has had.

§3.15's rule is that `mean` is the control for `zero`'s off-distribution bias,
so the burden here is on showing `zero` measures something real -- not the
other way round.

No p-value; nothing registered; pythia-410m spent under `check_registry`
rule 3.
"""
import argparse
import contextlib
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
import torch

from core.lm_loading import load_causal_lm
from tools.run.induction_rank_sweep import PROBE_ARMS, EVAL_SEED, ablated
from p7d_redundancy.member_formation_curves import batch
from p7d_redundancy.fv_score import induction_scores
from p7d_redundancy.mlp_backup_check import ablated_mlp, mlp_output_means
from p7d_redundancy.redundancy_catalog import nll

OUT = DATA / "analysis" / "mlp6_content_vs_scale.json"


def parse_head(s):
    return (int(s[1:s.index("H")]), int(s[s.index("H") + 1:]))


@contextlib.contextmanager
def constant_mlp(model, layer, vec):
    """Replace one MLP's output with a fixed vector at every position."""
    def hook(mod, args, out):
        return vec.to(out.dtype).expand_as(out).clone()
    h = model.gpt_neox.layers[layer].mlp.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


@torch.no_grad()
def output_stats(model, ids, layer, chunk=8):
    """(||mu||, RMS ||out||, RMS ||out - mu||) for one MLP's output."""
    outs = []

    def hook(mod, args, out):
        outs.append(out.detach().reshape(-1, out.shape[-1]).double())
        return out

    h = model.gpt_neox.layers[layer].mlp.register_forward_hook(hook)
    try:
        for i in range(0, len(ids), chunk):
            model(ids[i:i + chunk])
    finally:
        h.remove()
    V = torch.cat(outs, 0)
    mu = V.mean(0)
    return (float(mu.norm()),
            float(V.norm(dim=-1).pow(2).mean().sqrt()),
            float((V - mu).norm(dim=-1).pow(2).mean().sqrt()))


@torch.no_grad()
def resid_rms(model, ids, layer, chunk=8):
    """RMS norm of the residual ENTERING `layer` (what its attention reads)."""
    got = []

    def hook(mod, args):
        got.append(args[0].detach().reshape(-1, args[0].shape[-1]).double())

    h = model.gpt_neox.layers[layer].register_forward_pre_hook(hook)
    try:
        for i in range(0, len(ids), chunk):
            model(ids[i:i + chunk])
    finally:
        h.remove()
    V = torch.cat(got, 0)
    return float(V.norm(dim=-1).pow(2).mean().sqrt())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-410m")
    ap.add_argument("--step", type=int, default=16000)
    ap.add_argument("--source", default="L5H2")
    ap.add_argument("--target", default="L7H8")
    ap.add_argument("--mlp", type=int, default=6)
    ap.add_argument("--seqs", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--controls", type=int, default=3,
                     help="norm-matched random constant directions")
    ap.add_argument("--probe", default="wide", choices=tuple(PROBE_ARMS))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    source, target = parse_head(args.source), parse_head(args.target)
    out = Path(args.out) if args.out else OUT
    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()

    model, _ = load_causal_lm(f"{args.model}-step{args.step}")
    model.eval()
    tgt_layer = target[0]
    ids = batch(np.random.default_rng(EVAL_SEED), args.seqs,
                *PROBE_ARMS[args.probe])
    ck = max(1, args.chunk // 2)

    def attn():
        return induction_scores(model, ids, [target], ck)[target]

    clean = attn()
    nll_clean = nll(model, ids, args.chunk)
    means_clean = mlp_output_means(model, ids, args.chunk)
    with ablated(model, [source], mode="ov"):
        cond_base = attn()
        nll_cond_base = nll(model, ids, args.chunk)
        means_cond = mlp_output_means(model, ids, args.chunk)
        mu_norm, rms_out, rms_var = output_stats(model, ids, args.mlp, args.chunk)
        resid_base = resid_rms(model, ids, tgt_layer, args.chunk)
        with ablated_mlp(model, args.mlp, mode="zero"):
            a_zero = attn()
            n_zero = nll(model, ids, args.chunk)
            resid_zero = resid_rms(model, ids, tgt_layer, args.chunk)
        with ablated_mlp(model, args.mlp, mode="mean", means=means_cond):
            a_mean = attn()
            n_mean = nll(model, ids, args.chunk)
            resid_mean = resid_rms(model, ids, tgt_layer, args.chunk)
        # The CLEAN-state mean, held constant in the ablated state. `mlp6_response.py`
        # found MLP 6's mean rotates (cos 0.837) and grows 21 % when the relay goes,
        # so this arm asks whether that RESPONSE is load-bearing or whether the
        # direction MLP 6 already had would have done: same-vs-different against
        # the `mean` arm above is the whole question.
        with constant_mlp(model, args.mlp, means_clean[args.mlp]):
            a_mean_clean = attn()
            n_mean_clean = nll(model, ids, args.chunk)
            resid_mean_clean = resid_rms(model, ids, tgt_layer, args.chunk)

        mu = means_cond[args.mlp]
        rng = torch.Generator().manual_seed(EVAL_SEED)
        rand_rows = []
        for c in range(args.controls):
            r = torch.randn(mu.shape, generator=rng, dtype=torch.float64)
            r = r / r.norm() * mu.norm()
            with constant_mlp(model, args.mlp, r):
                a_r = attn()
                n_r = nll(model, ids, args.chunk)
                res_r = resid_rms(model, ids, tgt_layer, args.chunk)
            rand_rows.append({"attn": a_r, "delta": a_r - cond_base,
                              "nll": n_r, "resid_rms": res_r})
            print(f"  random constant {c}: attention {a_r:.4f} "
                  f"(delta {a_r - cond_base:+.4f}, resid RMS {res_r:.2f})",
                  flush=True)

    res = {
        "_what_this_is": __doc__, "git_sha": git_sha, "model": args.model,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "step": args.step, "source": args.source, "target": args.target,
        "mlp": args.mlp, "probe": args.probe, "n_seqs": args.seqs,
        "clean_attn": clean, "conditional_baseline_attn": cond_base,
        "nll_clean": nll_clean, "nll_conditional_baseline": nll_cond_base,
        "output_stats_in_conditional_state": {
            "mu_norm": mu_norm, "rms_out_norm": rms_out,
            "rms_variation_norm": rms_var,
            "constant_share": mu_norm / max(rms_out, 1e-30)},
        "arms": {
            "zero": {"attn": a_zero, "delta": a_zero - cond_base,
                     "nll": n_zero, "resid_rms": resid_zero},
            "mean": {"attn": a_mean, "delta": a_mean - cond_base,
                     "nll": n_mean, "resid_rms": resid_mean},
            "mean_from_clean_state": {
                "attn": a_mean_clean, "delta": a_mean_clean - cond_base,
                "nll": n_mean_clean, "resid_rms": resid_mean_clean,
                "cos_with_conditional_mean": float(
                    torch.dot(means_clean[args.mlp], means_cond[args.mlp])
                    / max(float(means_clean[args.mlp].norm()
                                * means_cond[args.mlp].norm()), 1e-30))},
            "random_constant": rand_rows},
        "resid_rms_conditional_baseline": resid_base,
        "restore_check_abs_diff": abs(attn() - clean)}

    print(f"\n{args.target} attention: clean {clean:.4f}   "
          f"{args.source}-ablated {cond_base:.4f}")
    print(f"MLP {args.mlp} output in that state: ||mu|| {mu_norm:.3f}   "
          f"RMS||out|| {rms_out:.3f}   RMS||out-mu|| {rms_var:.3f}   "
          f"constant share {mu_norm / max(rms_out, 1e-30):.3f}")
    print(f"residual RMS entering layer {tgt_layer}: "
          f"baseline {resid_base:.2f}   zero {resid_zero:.2f}   "
          f"mean {resid_mean:.2f}")
    print(f"attention: zero {a_zero:.4f}   mean(cond) {a_mean:.4f}   "
          f"mean(clean) {a_mean_clean:.4f}   "
          f"random-constant {[round(r['attn'], 4) for r in rand_rows]}")
    print(f"NLL: clean {nll_clean:.4f}   {args.source}-ablated "
          f"{nll_cond_base:.4f}   then zero {n_zero:.4f}   "
          f"mean(cond) {n_mean:.4f}   mean(clean) {n_mean_clean:.4f}   "
          f"random {[round(r['nll'], 3) for r in rand_rows]}")
    print(f"restore check abs diff {res['restore_check_abs_diff']:.2e}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
