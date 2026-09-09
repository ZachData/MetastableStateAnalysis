"""The position axis: is the copying effect concentrated, and is `r*` read off a
diluted curve?

WHY THIS EXISTS
---------------
`PROJECT.md` §3.13.3. Every readout in §3.11-§3.12 is a **mean over token
positions**, and that axis has never been looked at. In particular
`second_copy_nll` -- the readout the entire OV rank sweep, the **82 %**, and
every `dOV_nll` in this project rest on -- is

    second = tok_lp[:, N_REP - 1:]  ;  second_copy_nll = -second.mean()

a mean over `N_REP = 96` second-copy positions. §3.12-G6 showed that a mean over
a population that is mostly noise can average away a signal living in a few
members. If the copying effect is concentrated in particular positions then
`r*` is being read off a diluted curve, and that is the same error one axis
over, underneath the measurement everything else depends on.

**Concentration is the expected shape, not an exotic one.** Position `j = 0` of
the second copy predicts `ids[N_REP]` from the first copy alone -- no earlier
occurrence of the current token exists yet, so induction *cannot* fire there.
From `j = 1` on it can. So a step at `j = 0 -> 1` is predicted by the mechanism,
and anything beyond that is the empirical question.

WHAT IT EMITS, per §3.13's own rule that exploratory work reports the mean view
AND the extremum view and never chooses between them after the fact:

  * the per-position baseline NLL, full-OV-ablation NLL, and their delta;
  * the per-position rank-`r` recovered fraction, computed as a RATIO OF SUMS
    over a position set rather than a mean of per-position ratios -- the
    denominator `nll_0[j] - nll_base[j]` is near zero at positions where the
    head does nothing, and a mean of ratios there is dominated by noise;
  * concentration: the share of total `dNLL` mass in the top-k positions, and
    the position at which it peaks;
  * `r*` recomputed on the CONCENTRATED subset and on the DILUTE one, which is
    the actual question -- does the 82 % move when the mean stops hiding the
    profile?
  * per-position induction attention, the same mean the behavioural score takes.

COST: one model load, then one forward pass per rank. `--seqs` sets the
per-position sample count (8 in the original readout, which is 8 samples per
position; the default here is 64).
"""
import argparse
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
import torch

from core.lm_loading import load_causal_lm
from tools.run.induction_rank_sweep import (
    D_HEAD, D_MODEL, N_HEADS, N_REP, VOCAB_LO, VOCAB_HI, EVAL_SEED,
    ov_factors, write_ov, truncate,
)

OUT = DATA / "analysis" / "induction_position_profile.json"


def batch(rng, n_seqs):
    seqs = []
    for _ in range(n_seqs):
        s = rng.integers(VOCAB_LO, VOCAB_HI, size=N_REP)
        seqs.append(np.concatenate([s, s]))
    return torch.tensor(np.stack(seqs), dtype=torch.long)


@torch.no_grad()
def per_position_nll(model, ids, chunk=16):
    """(N_REP,) mean NLL at each second-copy position. Same slice convention as
    induction_rank_sweep.measure: tok_lp index t predicts ids[t+1], so the
    second copy starts at t = N_REP - 1."""
    acc = []
    for i in range(0, len(ids), chunk):
        out = model(ids[i:i + chunk])
        logits = out.logits[:, :-1, :].float()
        lp = torch.log_softmax(logits, dim=-1)
        tok_lp = lp.gather(-1, ids[i:i + chunk, 1:].unsqueeze(-1)).squeeze(-1)
        acc.append((-tok_lp[:, N_REP - 1:]).numpy().astype(np.float64))
        del out, logits, lp, tok_lp
    return np.concatenate(acc, 0).mean(0)          # (N_REP,)


@torch.no_grad()
def per_position_attn(model, ids, layer, head, chunk=8):
    """(N_REP,) mean post-softmax attention on the repo's induction pairs:
    query N_REP + t, key t. The behavioural score is this vector's mean."""
    acc = []
    q = torch.arange(N_REP, 2 * N_REP)
    k = torch.arange(0, N_REP)
    for i in range(0, len(ids), chunk):
        out = model(ids[i:i + chunk], output_attentions=True)
        att = out.attentions[layer][:, head, :, :].float()
        acc.append(att[:, q, k].numpy().astype(np.float64))
        del out, att
    return np.concatenate(acc, 0).mean(0)


def frac_recovered(nll_r, nll_0, nll_b, mask):
    """Ratio of SUMS over `mask`, not a mean of ratios: the per-position
    denominator is near zero where the head does nothing."""
    num = float(np.sum(nll_0[mask] - nll_r[mask]))
    den = float(np.sum(nll_0[mask] - nll_b[mask]))
    return num / den if abs(den) > 1e-12 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=4000)
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--head", type=int, default=8)
    ap.add_argument("--seqs", type=int, default=64)
    ap.add_argument("--ranks", default="0,1,2,4,8,16,64")
    args = ap.parse_args()
    ranks = [int(x) for x in args.ranks.split(",")]

    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    model, _ = load_causal_lm(f"pythia-410m-step{args.step}")
    model.eval()
    ids = batch(np.random.default_rng(EVAL_SEED), args.seqs)

    A0, B0 = ov_factors(model, args.layer, args.head)
    nll_b = per_position_nll(model, ids)
    attn = per_position_attn(model, ids, args.layer, args.head)
    print(f"  baseline mean NLL {nll_b.mean():.4f}   "
          f"induction attn mean {attn.mean():.4f}", flush=True)

    curves = {}
    for r in ranks:
        if r == 0:
            write_ov(model, args.layer, args.head,
                     np.zeros((D_MODEL, 1)), np.zeros((1, D_MODEL)))
        else:
            A, B = truncate(A0, B0, r, "svd")
            write_ov(model, args.layer, args.head, A, B)
        curves[r] = per_position_nll(model, ids)
        print(f"  rank {r:>3} (svd): mean NLL {curves[r].mean():.4f}", flush=True)
    write_ov(model, args.layer, args.head, A0, B0)
    chk = float(np.max(np.abs(per_position_nll(model, ids) - nll_b)))
    print(f"  restore check (max |dNLL|): {chk:.3e}", flush=True)

    nll_0 = curves[0]
    d = nll_0 - nll_b                              # per-position ablation effect
    order = np.argsort(-d)
    tot = float(d.sum())
    share = np.cumsum(d[order]) / max(tot, 1e-12)

    # concentration: how few positions carry half / ninety percent of the mass
    n_half = int(np.searchsorted(share, 0.5) + 1)
    n_90 = int(np.searchsorted(share, 0.9) + 1)

    top_mask = np.zeros(N_REP, bool); top_mask[order[:n_half]] = True
    bot_mask = ~top_mask
    all_mask = np.ones(N_REP, bool)

    res = {
        "_what_this_is": __doc__, "git_sha": git_sha,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "step": args.step, "layer": args.layer, "head": args.head,
        "n_seqs": args.seqs, "restore_check_max_abs": chk,
        "per_position": {
            "baseline_nll": nll_b.tolist(),
            "full_ablation_nll": nll_0.tolist(),
            "delta": d.tolist(),
            "induction_attn": attn.tolist(),
        },
        "summary": {
            "delta_mean": float(d.mean()), "delta_max": float(d.max()),
            "delta_min": float(d.min()), "delta_at_j0": float(d[0]),
            "delta_median": float(np.median(d)),
            "argmax_position": int(order[0]),
            "n_positions_for_half_the_mass": n_half,
            "n_positions_for_90pct_mass": n_90,
            "gini_like_top10pct_share": float(
                d[order[:max(N_REP // 10, 1)]].sum() / max(tot, 1e-12)),
            "attn_mean": float(attn.mean()), "attn_max": float(attn.max()),
            "attn_at_j0": float(attn[0]),
        },
        "frac_recovered": {},
    }
    for r in ranks:
        if r == 0:
            continue
        res["frac_recovered"][str(r)] = {
            "all_positions": frac_recovered(curves[r], nll_0, nll_b, all_mask),
            "concentrated_half": frac_recovered(curves[r], nll_0, nll_b, top_mask),
            "dilute_half": frac_recovered(curves[r], nll_0, nll_b, bot_mask),
            "excluding_j0": frac_recovered(
                curves[r], nll_0, nll_b,
                np.r_[False, np.ones(N_REP - 1, bool)]),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {OUT}")

    s = res["summary"]
    print(f"\n=== per-position OV ablation effect, L{args.layer}H{args.head} "
          f"step {args.step} ({args.seqs} seqs) ===")
    print(f"  dNLL  mean {s['delta_mean']:+.4f}   median {s['delta_median']:+.4f}"
          f"   max {s['delta_max']:+.4f} (at j={s['argmax_position']})"
          f"   min {s['delta_min']:+.4f}")
    print(f"  dNLL at j=0 (induction CANNOT fire): {s['delta_at_j0']:+.4f}")
    print(f"  positions carrying half the mass: {s['n_positions_for_half_the_mass']}"
          f" of {N_REP}   ninety percent: {s['n_positions_for_90pct_mass']}")
    print(f"  top-10% of positions carry {s['gini_like_top10pct_share']:.3f} "
          f"of the total")
    print(f"  induction attn  mean {s['attn_mean']:.4f}  max {s['attn_max']:.4f}"
          f"  at j=0 {s['attn_at_j0']:.4f}")

    print(f"\n  profile (every 8th position): j, dNLL, attn")
    for j in range(0, N_REP, 8):
        print(f"    j={j:>3}  dNLL {d[j]:+.4f}   attn {attn[j]:.4f}")

    print(f"\n=== does r* move when the mean stops hiding the profile? ===")
    print(f"{'rank':>5} {'all pos':>9} {'concentrated':>13} {'dilute':>9} "
          f"{'excl j=0':>9}")
    for r in ranks:
        if r == 0:
            continue
        f = res["frac_recovered"][str(r)]
        print(f"{r:>5} {f['all_positions']:>9.3f} {f['concentrated_half']:>13.3f} "
              f"{f['dilute_half']:>9.3f} {f['excluding_j0']:>9.3f}")


if __name__ == "__main__":
    main()
