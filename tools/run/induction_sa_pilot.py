"""Stage 3 PILOT: is the S/A factorial viable at all?

WHY THIS EXISTS
---------------
`PROJECT.md` §3.12-L and `MATH_SPECTRAL_OT.md` §2.4. Before a differential
prediction is registered on the `{M, M^T, -M, -M^T}` factorial, three things
have to be checked, and all three can only be checked by running it.

**1. IS `-M` OFF-SCALE?** `-M` reverses the head's *write*: where the copier
raised the correct token's logit it now lowers it. So `dNLL(-M)` should EXCEED
full ablation, which removes the contribution rather than reversing it. If it
does, `-M` is not a gentle arm, and the balanced main effects -- which average
`-M` with `-M^T` -- are dominated by it. Factorial main effects assume the
response is approximately ADDITIVE in the factors; a saturating or catastrophic
arm breaks that assumption rather than merely enlarging an error bar.

**2. WHAT IS THE COMMON CURRENCY?** `dNLL` has no natural scale. A graded
SCALING `lambda * M` IS writable (rank is unchanged) and gives a response curve
in magnitude, so every corner's effect can be reported as the `lambda` that
would produce it -- "this intervention is worth scaling the head to 0.6" is
interpretable where a raw `dNLL` is not.

**3. AND THE ONE THAT DECIDES THE DESIGN.** The graded S/A path
`M(e) = c(e)[(1-2e)S + A]`, which would have tested additivity directly rather
than assuming it, is **NOT IMPLEMENTABLE**. Measured on `L7H8`:

    rank(M) = rank(M^T) = rank(-M^T) = 64        (= d_head)
    rank(S) = rank(A)   = 128                    (outside the head's budget)
    rank((1-2e)S + A)   = 128  at e = 0.25, 0.5, 0.75

**A head cannot write its own symmetric part.** `S` and `A` are rank-`2*d_head`
objects; the head's operator lives on a rank-`d_head` manifold. The
interpolation leaves that manifold everywhere except at its two endpoints, which
are exactly `M` (e=0) and `-M^T` (e=1). So the Klein four-group is not merely an
elegant choice -- it is **the complete set of S/A sign interventions that stay
inside the architecture**, and no graded version of this experiment exists.

The consequence for the registration is a LIMITATION to be stated, not
engineered around: with four points and two factors the interaction term *is*
the additivity check, and §2.4.3 shows it is aliased with the read/write swap.
**Additivity cannot be separated from the role swap in this design.** That has
to go in the entry as a stated limitation.

ARMS, all rank <= d_head and so all writable:
    M        (A0,      B0)      identity
    ablation (0,       0)       the reference scale
    lambda*M (l*A0,    B0)      magnitude response, the common currency
    M^T      (B0^T,    A0^T)    role swap, S preserved, A flipped
    -M       (-A0,     B0)      write reversal, both flipped
    -M^T     (-B0^T,   A0^T)    role swap, S flipped, A preserved

Every corner is an ISOMETRY of the operator -- same singular values, same
Frobenius norm, same eigenvalue moduli -- which is also what §3.12-L requires
after finding the existing rank sweep's arms are energy-mismatched by up to 16x.

READOUT: `second_copy_nll` and KL from the unablated model, the same pair
§3.11's Stages 1-2 use. Weights saved and restored around every measurement,
with an end-of-run exactness check.
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

from core.intervention import next_token_kl
from core.lm_loading import load_causal_lm
from tools.run.induction_rank_sweep import (
    D_MODEL, N_REP, VOCAB_LO, VOCAB_HI, EVAL_SEED,
    ov_factors, write_ov, measure,
)

OUT = DATA / "analysis" / "induction_sa_pilot.json"
HEADS = [(7, 8, "L7H8_copier"), (9, 9, "L9H9"), (2, 10, "L2H10_anticopier")]


def batch(rng, n_seqs):
    seqs = []
    for _ in range(n_seqs):
        s = rng.integers(VOCAB_LO, VOCAB_HI, size=N_REP)
        seqs.append(np.concatenate([s, s]))
    return torch.tensor(np.stack(seqs), dtype=torch.long)


def arms(A0, B0, lambdas):
    """Every writable arm, as (name, A, B). All rank <= d_head."""
    out = [("ablation", np.zeros((D_MODEL, 1)), np.zeros((1, D_MODEL)))]
    for l in lambdas:
        out.append((f"scale_{l:g}", l * A0, B0))
    out += [
        ("M_transpose", B0.T.copy(), A0.T.copy()),      # S kept, A flipped
        ("minus_M", -A0, B0),                           # both flipped
        ("minus_M_transpose", -B0.T.copy(), A0.T.copy()),  # S flipped, A kept
    ]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=4000)
    ap.add_argument("--seqs", type=int, default=32)
    ap.add_argument("--lambdas", default="0.25,0.5,0.75")
    args = ap.parse_args()
    lambdas = [float(x) for x in args.lambdas.split(",")]

    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    model, _ = load_causal_lm(f"pythia-410m-step{args.step}")
    model.eval()
    ids = batch(np.random.default_rng(EVAL_SEED), args.seqs)

    base = measure(model, ids)
    base_lp = base["logprobs"]
    print(f"  baseline second-copy NLL {base['second_copy_nll']:.4f}", flush=True)

    res = {"_what_this_is": __doc__, "git_sha": git_sha,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
           "step": args.step, "n_seqs": args.seqs,
           "baseline_second_copy_nll": base["second_copy_nll"], "heads": {}}

    for layer, head, label in HEADS:
        A0, B0 = ov_factors(model, layer, head)
        M = A0 @ B0
        fro = float(np.linalg.norm(M))
        symS = float(np.sum(((M + M.T) / 2) ** 2) / np.sum(M * M))
        rows = {}
        for name, A, B in arms(A0, B0, lambdas):
            write_ov(model, layer, head, A, B)
            m = measure(model, ids)
            kl = float(np.mean([
                next_token_kl(base_lp[i].numpy(), m["logprobs"][i].numpy(), p)
                for i in range(len(ids)) for p in (-1,)]))
            Mi = A @ B
            rows[name] = {
                "second_copy_nll": m["second_copy_nll"],
                "delta_nll": m["second_copy_nll"] - base["second_copy_nll"],
                "kl": kl,
                "frobenius_ratio": float(np.linalg.norm(Mi) / max(fro, 1e-300)),
            }
            print(f"    {label:>18} {name:>18}: dNLL "
                  f"{rows[name]['delta_nll']:+.4f}  KL {kl:.4f}  "
                  f"||.||/||M|| {rows[name]['frobenius_ratio']:.3f}", flush=True)
        write_ov(model, layer, head, A0, B0)
        chk = abs(measure(model, ids)["second_copy_nll"] - base["second_copy_nll"])
        rows["_restore_abs_diff"] = float(chk)

        abl = rows["ablation"]["delta_nll"]
        # express each corner in the common currency: the lambda whose scaling
        # would produce the same dNLL (monotone interpolation, clipped)
        lam_x = [0.0] + lambdas + [1.0]
        lam_y = [rows["ablation"]["delta_nll"]] + \
                [rows[f"scale_{l:g}"]["delta_nll"] for l in lambdas] + [0.0]
        order = np.argsort(lam_y)
        for name in ("M_transpose", "minus_M", "minus_M_transpose"):
            d = rows[name]["delta_nll"]
            rows[name]["equivalent_lambda"] = float(
                np.interp(d, np.asarray(lam_y)[order], np.asarray(lam_x)[order]))
            rows[name]["dnll_over_ablation"] = float(d / abl) if abs(abl) > 1e-9 else None
        res["heads"][label] = {
            "layer": layer, "head": head, "frobenius": fro,
            "ov_symmetric_fraction": symS, "arms": rows}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {OUT}")

    print(f"\n=== Stage 3 pilot, step {args.step} ({args.seqs} seqs) ===")
    for label, h in res["heads"].items():
        r = h["arms"]
        print(f"\n  {label}  (||M||_F {h['frobenius']:.3f}, "
              f"OV sym frac {h['ov_symmetric_fraction']:.3f}, "
              f"restore {r['_restore_abs_diff']:.1e})")
        print(f"    {'arm':>20} {'dNLL':>9} {'KL':>8} {'/ablation':>10} {'~lambda':>9}")
        for name in ["ablation"] + [f"scale_{l:g}" for l in lambdas] + \
                    ["M_transpose", "minus_M", "minus_M_transpose"]:
            e = r[name]
            ratio = e.get("dnll_over_ablation")
            lam = e.get("equivalent_lambda")
            print(f"    {name:>20} {e['delta_nll']:>+9.4f} {e['kl']:>8.4f} "
                  f"{('-' if ratio is None else f'{ratio:.2f}x'):>10} "
                  f"{('-' if lam is None else f'{lam:.3f}'):>9}")

    print("\n  QUESTION 1 -- is -M off-scale?  dNLL(-M) vs dNLL(ablation):")
    for label, h in res["heads"].items():
        a = h["arms"]["ablation"]["delta_nll"]
        m = h["arms"]["minus_M"]["delta_nll"]
        verdict = "OFF-SCALE" if m > a * 1.25 else ("comparable" if m > a * 0.75 else "smaller")
        print(f"    {label:>20}: ablation {a:+.4f}  -M {m:+.4f}  -> {verdict}")

    print("\n  QUESTION 2 -- the informative contrast, M vs M^T "
          "(alignment at fixed gain):")
    for label, h in res["heads"].items():
        t = h["arms"]["M_transpose"]["delta_nll"]
        a = h["arms"]["ablation"]["delta_nll"]
        print(f"    {label:>20}: dNLL(M^T) {t:+.4f} = "
              f"{(t / a if abs(a) > 1e-9 else float('nan')):.2f}x ablation")


if __name__ == "__main__":
    main()
