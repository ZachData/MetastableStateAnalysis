"""Sub-phase 7d, step 1: how many heads are in the redundancy set at all?

WHY THIS EXISTS
---------------
`PROJECT.md` §3.12-S found `L5H2` and `L7H8` are **functionally redundant and
structurally distinct** -- interaction +4.15, joint ablation 2.2x the sum of
parts, residual-delta cosine 0.868 against a chance-level weight overlap. The
immediate question is whether they are a pair or the visible members of a set,
and that cannot be answered from the 38-head sample §3.12-R happened to draw.

**PASS 1 (this runner): every head, one at a time.** Single-head OV ablation
`ΔNLL` for all 384, so membership is measured rather than assumed. §3.12-Q
already showed the distribution is extremely heavy-tailed -- ordinary heads move
the readout by under 0.01 while these two move it by 1-2 -- so the screen's job
is to find where the tail ends.

WHY A FULL SWEEP RATHER THAN A PROXY. §3.12-R ruled out `‖OV‖_F` as a predictor
(r² = 0.001, and the relation runs backwards), §3.12-G6 ruled out every spectral
field, and §3.12-S showed weight-space overlap and function-space overlap come
apart -- so **no weights-only quantity found so far predicts causal effect**, and
a proxy screen would inherit exactly that failure. The ablation is the ground
truth and it is the only thing that defines membership.

WHAT PASS 2 WILL NEED (not run here): the pairwise interaction within whatever
set this finds, which is `n(n-1)/2` further arms, and the per-checkpoint
formation curves that answer §3.14.2's timing questions.

COST: one model load, then 1 + 384 forward passes at `--seqs` sequences. Logits
only -- no hidden states -- because the screen needs `ΔNLL` and nothing else.
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
    D_MODEL, N_HEADS, N_REP, VOCAB_LO, VOCAB_HI, EVAL_SEED,
    ov_factors, write_ov,
)

OUT = DATA / "analysis" / "redundancy_catalog.json"


def batch(rng, n):
    return torch.tensor(np.stack([
        np.concatenate([s, s]) for s in
        (rng.integers(VOCAB_LO, VOCAB_HI, size=N_REP) for _ in range(n))
    ]), dtype=torch.long)


@torch.no_grad()
def nll(model, ids, chunk=8):
    acc = []
    for i in range(0, len(ids), chunk):
        out = model(ids[i:i + chunk])
        lg = out.logits[:, :-1, :].float()
        lp = torch.log_softmax(lg, dim=-1)
        tok = lp.gather(-1, ids[i:i + chunk, 1:].unsqueeze(-1)).squeeze(-1)
        acc.append((-tok[:, N_REP - 1:]).numpy().astype(np.float64))
        del out, lg, lp, tok
    return float(np.concatenate(acc).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=16000)
    ap.add_argument("--seqs", type=int, default=8)
    args = ap.parse_args()

    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    model, _ = load_causal_lm(f"pythia-410m-step{args.step}")
    model.eval()
    ids = batch(np.random.default_rng(EVAL_SEED), args.seqs)
    base = nll(model, ids)
    print(f"baseline second-copy NLL {base:.4f}  ({args.seqs} seqs, "
          f"step {args.step})", flush=True)

    Z = (np.zeros((D_MODEL, 1)), np.zeros((1, D_MODEL)))
    n_layers = len(model.gpt_neox.layers)
    rows, t0 = [], time.time()
    for L in range(n_layers):
        for H in range(N_HEADS):
            a, b = ov_factors(model, L, H)
            write_ov(model, L, H, *Z)
            d = nll(model, ids) - base
            write_ov(model, L, H, a, b)
            rows.append({"head": f"L{L}H{H}", "layer": L, "head_idx": H,
                         "dnll": d})
            if abs(d) > 0.05:
                print(f"    L{L}H{H:<2}  dNLL {d:>+8.4f}", flush=True)
        el = time.time() - t0
        print(f"  layer {L:>2}/{n_layers - 1} done  ({el/60:.1f} min elapsed)",
              flush=True)

    chk = abs(nll(model, ids) - base)
    d = np.array([r["dnll"] for r in rows])
    res = {"_what_this_is": __doc__, "git_sha": git_sha, "step": args.step,
           "n_seqs": args.seqs, "baseline_nll": base,
           "restore_abs_diff": float(chk),
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
           "median": float(np.median(d)), "p99": float(np.percentile(d, 99)),
           "max": float(d.max()), "min": float(d.min()),
           "n_above_0.05": int((d > 0.05).sum()),
           "n_above_0.2": int((d > 0.2).sum()),
           "n_above_1.0": int((d > 1.0).sum()),
           "n_negative_0.05": int((d < -0.05).sum()),
           "heads": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {OUT}  (restore {chk:.1e})")

    order = np.argsort(-d)
    print(f"\n=== where does the tail end? (384 heads, step {args.step}) ===")
    print(f"  median {res['median']:+.5f}   p99 {res['p99']:+.4f}   "
          f"max {res['max']:+.4f}   min {res['min']:+.4f}")
    print(f"  above +0.05: {res['n_above_0.05']}   above +0.2: "
          f"{res['n_above_0.2']}   above +1.0: {res['n_above_1.0']}")
    print(f"  below -0.05 (anti-copiers): {res['n_negative_0.05']}")
    print(f"\n=== top 20 by dNLL ===")
    for j in order[:20]:
        print(f"    {rows[j]['head']:>7}  {rows[j]['dnll']:>+8.4f}")
    print(f"\n=== most negative 8 ===")
    for j in order[-8:][::-1]:
        print(f"    {rows[j]['head']:>7}  {rows[j]['dnll']:>+8.4f}")


if __name__ == "__main__":
    main()
