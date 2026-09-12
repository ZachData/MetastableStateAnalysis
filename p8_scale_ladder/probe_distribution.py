"""Is 70m's late-checkpoint induction collapse the PRIOR interfering, or real?

The probe samples token ids uniformly from [1000, 40000). Those are arbitrary
mid-frequency BPE tokens, so a well-trained LM assigns them very little mass --
first-copy median rank of the true token is ~23000 of 50304 at every rung. If
the collapse is the language prior overriding a working copy mechanism, then
making the tokens EASIER for the prior should recover it. If induction itself
degraded, changing the token distribution should not help.

Three arms, same sequence length, same repetition structure:

  wide    ids ~ U[1000, 40000)  -- the current probe
  freq    ids ~ U[1000,  5000)  -- lower BPE ids, much more frequent tokens
  text    a real paragraph, repeated -- copying and the prior AGREE here
                                        instead of competing

`text` is the arm `ambient_budget.py`'s docstring promises as `--text` and that
was never implemented. It is the one that matters: if 70m's induction is intact
on repeated natural text at step 143000, then the whole late-checkpoint 70m
column is measuring prior-vs-probe conflict, not induction.
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
from tools.run.induction_rank_sweep import N_REP, EVAL_SEED

CEIL = float(np.log(50304))

PARAS = [
    "The city council met on Tuesday to discuss the new transport plan, which "
    "would add three bus routes and extend the tram line toward the harbour.",
    "She had been reading the same paragraph for several minutes without taking "
    "any of it in, and finally set the book down on the table beside her chair.",
    "Researchers reported that the material becomes superconducting below a "
    "critical temperature, though the mechanism responsible remains disputed.",
    "He walked down to the river every evening after supper, following the path "
    "past the mill until the light failed and the far bank went dark.",
]


def rand_batch(rng, lo, hi, n=8):
    return torch.tensor(np.stack([
        np.concatenate([s, s]) for s in
        (rng.integers(lo, hi, size=N_REP) for _ in range(n))]), dtype=torch.long)


def text_batch(tok, n=8):
    rows = []
    for i in range(n):
        t = PARAS[i % len(PARAS)] + " " + PARAS[(i + 1) % len(PARAS)]
        ids = tok(t)["input_ids"]
        while len(ids) < N_REP:            # pad by repeating the paragraph
            ids = ids + tok(" " + t)["input_ids"]
        s = np.array(ids[:N_REP])
        rows.append(np.concatenate([s, s]))
    return torch.tensor(np.stack(rows), dtype=torch.long)


@torch.no_grad()
def induction_stats(model, ids):
    out = model(ids)
    lg = out.logits[:, :-1, :].float()
    tgt = ids[:, 1:]
    lp = torch.log_softmax(lg, dim=-1)
    nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    top1 = (lg.argmax(-1) == tgt).float()
    rank = (lg > lg.gather(-1, tgt.unsqueeze(-1))).sum(-1) + 1
    f = nll[:, :N_REP - 1].reshape(-1).numpy()     # first copy: prior only
    d = nll[:, N_REP:].reshape(-1).numpy()         # induction window
    return {"first_med": float(np.median(f)),
            "ind_mean": float(d.mean()), "ind_med": float(np.median(d)),
            "ind_top1": float(top1[:, N_REP:].mean()),
            "ind_rank": float(rank[:, N_REP:].reshape(-1).median()),
            "icl_med": float(np.median(f) - np.median(d)),
            "above_ceil": float((d > CEIL).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="pythia-70m-step16000,"
                                        "pythia-70m-step143000,"
                                        "pythia-410m-step143000")
    ap.add_argument("--out", default=str(DATA / "analysis" / "probe_distribution.json"))
    args = ap.parse_args()
    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    res = {"_what_this_is": __doc__, "git_sha": git_sha,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
           "ceiling_nll": CEIL, "per_model": {}}
    print(f"ceiling ln(50304) = {CEIL:.3f};  induction window only\n")
    hdr = (f"{'model/step':>22} {'arm':>5} {'1st med':>8} {'ind mean':>9} "
           f"{'ind med':>8} {'top1':>6} {'medrank':>8} {'ICL med':>8} {'>ceil':>6}")
    print(hdr); print("-" * len(hdr))
    for name in [m for m in args.models.split(",") if m]:
        model, tk = load_causal_lm(name)
        model.eval()
        arms = {"wide": rand_batch(np.random.default_rng(EVAL_SEED), 1000, 40000),
                "freq": rand_batch(np.random.default_rng(EVAL_SEED), 1000, 5000),
                "text": text_batch(tk)}
        res["per_model"][name] = {}
        for arm, ids in arms.items():
            s = induction_stats(model, ids)
            res["per_model"][name][arm] = s
            print(f"{name:>22} {arm:>5} {s['first_med']:>8.2f} "
                  f"{s['ind_mean']:>9.3f} {s['ind_med']:>8.3f} "
                  f"{s['ind_top1']:>6.3f} {s['ind_rank']:>8.0f} "
                  f"{s['icl_med']:>8.2f} {s['above_ceil']:>6.2f}")
        del model
        print()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
