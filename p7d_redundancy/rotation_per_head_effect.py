"""Does MLP 6's rotation actually move the heads its geometry points at?

WHY THIS EXISTS
---------------
`PROJECT.md` §3.24 showed MLP 6's repair direction puts 7-8x its chance share
into the key read-spaces of the redundancy set -- six of the top nine of 272
downstream heads are catalogue members, on a flat per-layer profile -- and
flagged the gap honestly: **that is a weight-space measure**, and §3.12-S is
this project's own finding that weight-space overlap is not function-space
overlap. The loss arm in §3.24 confirms the rotation matters functionally, but
it is aggregate: it cannot say the rotation acts *on those particular members*.

THE MINIMAL PAIR. Two states differing by exactly the rotation and by nothing
else, both with a CONSTANT in MLP 6's slot so nothing else varies:

    REF  = `L5H2` ablated, MLP 6's output := mu_cond   (rotation present)
    TEST = `L5H2` ablated, MLP 6's output := mu_clean  (rotation absent)

§3.24 measured these at NLL 2.859 and 8.955. That gap is why the readout here
is **attention, not loss**: at 8.955 the loss is a nat off `ln 50304` and any
per-head marginal computed there is ceiling-contaminated (§3.14.4-D), whereas
an attention distribution is well defined however badly the model is doing.

WHAT IT MEASURES, for every head in the model: the **total-variation distance**
between that head's attention distribution in REF and in TEST, averaged over
query positions and sequences. TV is in [0, 1], is role-agnostic -- it does not
presume the head is an induction head, which matters because §3.18 found the
FV-positive members never exceed an induction score of 0.015 -- and needs no
scale calibration.

THE TEST IS THE CORRELATION, NOT THE MEANS. If the geometry predicts the
function, a head's TV should track its `frac_K` from
`mlp6_decode_direction.json`. Two confounds are handled rather than hoped away:

  - **Layer accumulation.** Later layers see more upstream change regardless of
    what the rotation points at, so the per-layer profile is printed and the
    member/non-member comparison is also reported WITHIN each layer, which is
    what killed the proximity confound in §3.24.
  - **Which heads can be affected at all.** Only layers above MLP 6 can be, and
    the runner reports the layers at or below it as the built-in zero check.

No p-value is claimed as an adjudication: heads within a layer are not
independent (§3.12-G6), and the correlation below is descriptive. Nothing
registered; pythia-410m spent under `check_registry` rule 3.
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
from p7d_redundancy.mlp_backup_check import mlp_output_means
from p7d_redundancy.mlp6_content_vs_scale import constant_mlp

OUT = DATA / "analysis" / "rotation_per_head_effect.json"
DECODE = DATA / "analysis" / "mlp6_decode_direction.json"
#: §3.14.2's Q1 answer, the top-10 causal catalogue at step 16000.
MEMBERS = {"L5H2", "L7H8", "L12H5", "L8H6", "L11H14", "L8H9",
           "L15H14", "L7H1", "L9H13", "L10H9"}


def parse_head(s):
    return (int(s[1:s.index("H")]), int(s[s.index("H") + 1:]))


@torch.no_grad()
def attention_tv(model, ids, make_ref, make_test, n_layers, n_heads, chunk=2):
    """Mean total-variation distance per head between two intervened states.

    Both forwards run on the same chunk before either is discarded, so the two
    attention tensors are always compared on identical inputs.
    """
    tot = np.zeros((n_layers, n_heads))
    n = 0
    for i in range(0, len(ids), chunk):
        x = ids[i:i + chunk]
        with make_ref():
            a = model(x, output_attentions=True).attentions
            a = [t.detach().float() for t in a]
        with make_test():
            b = model(x, output_attentions=True).attentions
            b = [t.detach().float() for t in b]
        for L in range(n_layers):
            # (B, H, T, T) -> TV per (batch, head, query), then mean over
            # batch and query. Row sums are 1 by construction, so TV is
            # half the L1 distance.
            tv = 0.5 * (a[L] - b[L]).abs().sum(-1)
            tot[L] += tv.mean(dim=(0, 2)).double().numpy() * x.shape[0]
        n += x.shape[0]
        del a, b
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-410m")
    ap.add_argument("--steps", default="16000,143000")
    ap.add_argument("--source", default="L5H2")
    ap.add_argument("--mlp", type=int, default=6)
    ap.add_argument("--seqs", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--probe", default="wide", choices=tuple(PROBE_ARMS))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    source = parse_head(args.source)
    out = Path(args.out) if args.out else OUT
    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    steps = [int(x) for x in args.steps.split(",") if x]
    decode = json.load(open(DECODE)) if DECODE.exists() else None
    if decode is None:
        print(f"note: {DECODE} absent -- TV only, no geometry join")

    res = {"_what_this_is": __doc__, "git_sha": git_sha, "model": args.model,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
           "source": args.source, "mlp": args.mlp, "probe": args.probe,
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
        mu_cond, mu_clean = means_cond[args.mlp], means_clean[args.mlp]

        import contextlib

        @contextlib.contextmanager
        def state(vec):
            with ablated(model, [source], mode="ov"):
                with constant_mlp(model, args.mlp, vec):
                    yield

        tv = attention_tv(model, ids, lambda: state(mu_cond),
                          lambda: state(mu_clean), n_layers, n_heads,
                          args.chunk)

        frac = {}
        if decode and str(s) in decode["per_step"]:
            frac = {r["head"]: r["frac_K"]
                    for r in decode["per_step"][str(s)]["specificity"]["all_heads"]}

        rows = {}
        for L in range(n_layers):
            for h in range(n_heads):
                tag = f"L{L}H{h}"
                rows[tag] = {"layer": L, "head": h, "tv": float(tv[L, h]),
                             "member": tag in MEMBERS,
                             "frac_K": frac.get(tag)}

        above = [r for r in rows.values() if r["layer"] > args.mlp]
        at_or_below = [r for r in rows.values() if r["layer"] <= args.mlp]
        tv_above = np.array([r["tv"] for r in above])
        mem = np.array([r["tv"] for r in above if r["member"]])
        non = np.array([r["tv"] for r in above if not r["member"]])

        print(f"== step {s}")
        print(f"  zero check, layers 0..{args.mlp}: max TV "
              f"{max(r['tv'] for r in at_or_below):.2e}")
        print(f"  layers >{args.mlp}: n={len(above)}  median TV "
              f"{np.median(tv_above):.4f}")
        print(f"    members    n={len(mem)}  median {np.median(mem):.4f}")
        print(f"    non-members n={len(non)}  median {np.median(non):.4f}")

        prof = {}
        for L in range(args.mlp + 1, n_layers):
            v = [r["tv"] for r in rows.values() if r["layer"] == L]
            prof[L] = float(np.median(v))
        print("  per-layer median TV: " + "  ".join(
            f"L{L} {m:.3f}" for L, m in list(prof.items())[:10]))

        within = {}
        for L in range(args.mlp + 1, n_layers):
            m = [r["tv"] for r in rows.values()
                 if r["layer"] == L and r["member"]]
            nm = [r["tv"] for r in rows.values()
                  if r["layer"] == L and not r["member"]]
            if m:
                within[L] = {"members": [f"L{L}H{r['head']}"
                                         for r in rows.values()
                                         if r["layer"] == L and r["member"]],
                             "member_median": float(np.median(m)),
                             "nonmember_median": float(np.median(nm)),
                             "nonmember_max": float(np.max(nm))}
                w = within[L]
                print(f"    within L{L}: members {w['members']} median "
                      f"{w['member_median']:.4f}  |  non-members median "
                      f"{w['nonmember_median']:.4f}, max {w['nonmember_max']:.4f}")

        corr = None
        pairs = [(r["frac_K"], r["tv"]) for r in above if r["frac_K"] is not None]
        if pairs:
            from scipy import stats
            x = np.array([p[0] for p in pairs])
            y = np.array([p[1] for p in pairs])
            sp = stats.spearmanr(x, y)
            pe = stats.pearsonr(x, y)
            corr = {"n": len(pairs), "spearman_rho": float(sp[0]),
                    "spearman_p": float(sp[1]), "pearson_r": float(pe[0]),
                    "pearson_p": float(pe[1])}
            print(f"  TV vs frac_K over {len(pairs)} heads: "
                  f"spearman {sp[0]:+.3f} (p={sp[1]:.2e})   "
                  f"pearson {pe[0]:+.3f} (p={pe[1]:.2e})")

        top = sorted(above, key=lambda r: -r["tv"])[:10]
        print("  top TV: " + ", ".join(
            f"L{r['layer']}H{r['head']}{'*' if r['member'] else ''} "
            f"{r['tv']:.3f}" for r in top))
        print()

        res["per_step"][str(s)] = {
            "heads": rows, "per_layer_median_tv": prof,
            "within_layer": within, "tv_vs_frac_K": corr,
            "member_median_tv": float(np.median(mem)),
            "nonmember_median_tv": float(np.median(non)),
            "zero_check_max_tv_at_or_below_mlp": float(
                max(r["tv"] for r in at_or_below))}

        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(res, fh, indent=2)
        del model

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
