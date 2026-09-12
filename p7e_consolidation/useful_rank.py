"""How many of a head's 64 OV directions actually carry its causal effect?

WHY THIS EXISTS
---------------
Every rank in 7d is a **variance** rank -- participation ratio, r90, CKA -- and
none of them says which directions carry *function*. `design-7e.md`'s capacity
question turns on exactly that gap: `ambient_budget.py` found only **73-78 %**
of the set's joint effect energy inside one head's rank-64 budget, which fails
the energy criterion, but a tail spread over hundreds of near-unused directions
may carry almost no function. **Energy is not usefulness**, and this runner
measures usefulness.

THE CURVE. Truncate one head's OV to rank `r` (Eckart-Young optimal, the `svd`
basis of §3.11 decision 1), measure second-copy `dNLL`, and sweep `r`. At
`r = 0` the head is fully ablated and `dNLL` is its whole causal effect; at
`r = D_HEAD = 64` the OV is untouched and `dNLL` must return to **~0**, which is
a free correctness check on every row. It is `~1.7e-4` rather than exactly zero,
and the reason is worth knowing: at `r = 64` the SVD reproduces `OV` to machine
precision, but it hands back a **different factorisation** of the same product,
and `write_ov` stores those factors in float32. The residue is that rounding,
not lost structure -- it survived moving the decomposition off the full
`(1024, 1024)` product, which is how it was identified. The exact check is the
separate `restore_abs_diff` at the end, which puts the ORIGINAL factors back and
reads `0.0e+00`.
`recovery(r) = 1 - dNLL(r)/dNLL(0)` runs 0 -> 1, and `r*` is the smallest `r`
recovering `--recover` of the effect.

THE CONTROL, AND WHY IT IS NOT `induction_rank_sweep.truncate`'s. `r*` is only
meaningful against a control -- a head whose function needed all 64 directions
would show a straight line. The natural control is a **random** rank-`r`
truncation of the same operator at the **same Frobenius norm**, so only the
DIRECTIONS are structureless.

`induction_rank_sweep.truncate`'s `random` branch claims to be that and is not:
it applies a random `r`-dim projector in the head core, whose norm is
`~sqrt(r/64)` of the full operator, while the `svd` branch keeps the *largest*
`r` singular values. On a concentrated spectrum those differ by a large factor
-- **`PROJECT.md` §3.14.3 defect 1 records it as off 16x at `r = 1`**. Since
`r*` is *defined* by where the real curve leaves the control band, an unmatched
band moves `r*`. This file therefore builds its own control, rescaled so
`||OV_control||_F == ||OV_svd,r||_F` exactly, and **does not modify the shared
runner**, which other results depend on.

(`ov_factors` returns the transpose of the residual operator -- §3.14.3 defect
3. Harmless here: rank and Frobenius norm are both transpose-invariant, and
nothing in this file reads a direction.)

WHAT IT DECIDES. Small `r*` across members means each head's function is
low-rank and several could in principle share one 64-dim budget -- the
consolidation of `design-7e.md` is viable. Large `r*` means the capacity
objection stands causally as well as geometrically and the phase stops.

Read this **per member**, not on the joint arm: the joint ablation lands 0.93
nats from the uniform ceiling (`design-7e.md`), where `dNLL` is a floor rather
than a measurement, while single-head arms have headroom.

NO p-value; nothing registered. pythia-410m is spent under `check_registry`
rule 3.
"""
import argparse
import gc
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

from core.lm_loading import load_causal_lm
from tools.run.induction_rank_sweep import (
    EVAL_SEED, ablated, arch_dims, head_means, ov_factors, write_ov,
)

from p7d_redundancy.member_formation_curves import batch, catalog_path_for, members, probe

OUT = DATA / "analysis" / "useful_rank.json"
DEFAULT_MODEL = "pythia-410m"

#: `0` and `D_HEAD` are not optional: `0` sets the scale every recovery is
#: measured against, and `D_HEAD` must return to ~0 (see __doc__) or the
#: truncation path is wrong.
RANKS = "0,1,2,3,4,6,8,12,16,24,32,48,64"


def head_svd(A, B):
    """SVD of the head's OV, computed through its 64-dim core.

    `OV = A B` with `A` (d_model, 64) and `B` (64, d_model), so the product has
    rank <= 64 and its SVD can be read off a 64x64 problem:
    `A = Qa Ra`, `B^T = Qb Rb`, then `OV = Qa (Ra Rb^T) Qb^T` and the SVD of the
    small middle factor lifts straight back.

    The first version formed the full `(1024, 1024)` product and decomposed
    that, **once per rank per head**, and was killed for memory twice. This path
    allocates nothing larger than `(1024, 64)` and runs once per head.

    It does NOT change the `r = 64` residue (~1.7e-4), which was the other
    suspected cause -- that is float32 refactorisation in `write_ov`, and ruling
    the SVD out is how it was pinned down. See __doc__.
    """
    Qa, Ra = np.linalg.qr(A)
    Qb, Rb = np.linalg.qr(B.T)
    Us, s, Vst = np.linalg.svd(Ra @ Rb.T, full_matrices=False)
    return Qa @ Us, s, (Qb @ Vst.T).T


def svd_rank(U, s, Vt, r, d_model, bottom=False):
    """Rank-`r` factors, and the norm a control should be matched to.

    `bottom=False` keeps the LARGEST `r` singular directions -- the Eckart-Young
    optimal rank-`r` approximation, and the default everywhere.

    `bottom=True` keeps the SMALLEST `r` instead, which is the worst possible
    rank-`r` approximation by gain and is here precisely for that reason. If a
    head's causal effect is carried by its top directions the bottom curve stays
    flat; if the two curves cross, gain ordering and causal ordering disagree.
    `L11H14` motivated it: its top-`r` curve is non-monotonic (recovery is
    **negative** at `r = 1`) and drops only between `r = 48` and `r = 64`, which
    says the last sixteen directions -- its smallest singular values -- do the
    work.
    """
    if r == 0:
        return np.zeros((d_model, 1)), np.zeros((1, d_model)), 0.0
    if bottom:
        return U[:, -r:] * s[-r:], Vt[-r:], float(np.sqrt((s[-r:] ** 2).sum()))
    return U[:, :r] * s[:r], Vt[:r], float(np.sqrt((s[:r] ** 2).sum()))


def random_rank(A, B, r, target_norm, rng):
    """Random rank-`r` truncation rescaled to `target_norm` exactly.

    The matched-norm control `truncate`'s `random` branch was supposed to be
    (§3.14.3 defect 1). Structure is destroyed, scale is preserved, so a
    difference between this and the `svd` curve is about DIRECTIONS only.
    """
    if r == 0:
        return np.zeros((A.shape[0], 1)), np.zeros((1, B.shape[1]))
    Qr, _ = np.linalg.qr(rng.normal(size=(B.shape[0], r)))
    Ar = A @ (Qr @ Qr.T)
    n = float(np.linalg.norm(Ar @ B))
    if n > 0:
        Ar = Ar * (target_norm / n)
    return Ar, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="registry family prefix, e.g. pythia-70m, pythia-410m "
                         "(p8_scale_ladder/design-8.md rung policy: 1b/1.4b "
                         "are reserved -- do not pass those without a "
                         "registered prediction). The default --ranks tops "
                         "out at 64 = d_head for both 410m and 70m; a "
                         "different d_head rung needs its own --ranks")
    ap.add_argument("--step", type=int, default=16000)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--heads", default="")
    ap.add_argument("--seqs", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--ranks", default=RANKS)
    ap.add_argument("--controls", type=int, default=3,
                    help="matched-norm random truncations per rank")
    ap.add_argument("--bottom", action="store_true",
                    help="also keep the SMALLEST r directions per rank -- the "
                         "worst rank-r approximation by gain, which is the "
                         "direct test of whether gain ordering matches causal "
                         "ordering (see svd_rank)")
    ap.add_argument("--recover", type=float, default=0.90,
                    help="recovery fraction defining r*")
    ap.add_argument("--ablation", default="ov", choices=("ov", "mean"),
                    help="which reference the r=0 row uses -- i.e. the "
                         "DENOMINATOR of every recovery fraction, and so what "
                         "r* is measured against. 'ov' keeps the rank family's "
                         "own bottom (a rank-0 OV is the zero matrix). 'mean' "
                         "uses the head's clean-run mean instead, which is the "
                         "better estimate of the head's true causal effect and "
                         "is NOT a member of the rank family -- an inflated "
                         "denominator deflates recovery and inflates r*")
    ap.add_argument("--out", default="",
                    help="a non-default --model gets its own default filename "
                         "instead of the 410m one")
    args = ap.parse_args()
    if args.out:
        out_path = Path(args.out)
    elif args.model == DEFAULT_MODEL and args.ablation == "ov":
        out_path = OUT
    else:
        _abl = "" if args.ablation == "ov" else f"_{args.ablation}"
        out_path = DATA / "analysis" / f"useful_rank_{args.model}{_abl}.json"

    def parse(spec):
        return [(int(h[1:h.index("H")]), int(h[h.index("H") + 1:]))
                for h in spec.split(",") if h]

    if args.heads:
        heads, source = parse(args.heads), "--heads"
    else:
        heads, source = members(args.top, catalog_path_for(args.model))
    ranks = [int(x) for x in args.ranks.split(",") if x]
    names = {k: f"L{k[0]}H{k[1]}" for k in heads}

    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    res = {"_what_this_is": __doc__, "git_sha": git_sha, "model": args.model,
           "ablation": args.ablation,
           "step": args.step, "membership_source": source,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
           "eval": {"n_seqs": args.seqs, "seed": EVAL_SEED,
                    "recover_frac": args.recover, "n_controls": args.controls},
           "ranks": ranks, "heads": [names[k] for k in heads], "per_head": {}}

    model, _ = load_causal_lm(f"{args.model}-step{args.step}")
    model.eval()
    d_model, d_head, n_heads = arch_dims(model)
    res["d_head"] = d_head
    ids = batch(np.random.default_rng(EVAL_SEED), args.seqs)
    nll0, _ = probe(model, ids, args.chunk, False)
    rng = np.random.default_rng(EVAL_SEED)

    print(f"model:   {args.model}")
    print(f"members: {', '.join(names[k] for k in heads)}   ({source})")
    print(f"step {args.step}, baseline NLL {nll0:.4f}, {args.seqs} seqs, "
          f"{len(ranks)} ranks x (1 + {args.controls}) arms per head\n")

    for k in heads:
        A, B = ov_factors(model, *k)
        U, s_ov, Vt = head_svd(A, B)          # once per head, not once per rank
        row, rec_real, rec_ctrl = {}, {}, {}
        print(f"{names[k]}:")
        print(f"{'r':>4} {'dNLL':>9} {'recovery':>9} {'ctrl dNLL':>10} "
              f"{'ctrl recov':>11} {'bot recov':>9}")
        d0 = None
        for r in ranks:
            Ar, Br, tn = svd_rank(U, s_ov, Vt, r, d_model)
            write_ov(model, *k, Ar, Br)
            d = probe(model, ids, args.chunk, False)[0] - nll0
            db = None
            if args.bottom and 0 < r < d_head:
                Ab, Bb, _ = svd_rank(U, s_ov, Vt, r, d_model, bottom=True)
                write_ov(model, *k, Ab, Bb)
                db = probe(model, ids, args.chunk, False)[0] - nll0
            cd = []
            for _ in range(args.controls if 0 < r < d_head else 0):
                Ac, Bc = random_rank(A, B, r, tn, rng)
                write_ov(model, *k, Ac, Bc)
                cd.append(probe(model, ids, args.chunk, False)[0] - nll0)
            write_ov(model, *k, A, B)
            if d0 is None:
                # r = 0: the head's whole causal effect. Under --ablation mean
                # the reference is the mean-ablation effect instead of the
                # rank-family's zero matrix, so `recovery` is measured against
                # what the head actually contributes rather than against the
                # off-distribution response to deleting it.
                if args.ablation == "mean":
                    with ablated(model, [k], "mean",
                                 head_means(model, ids, [k], args.chunk)):
                        d = probe(model, ids, args.chunk, False)[0] - nll0
                d0 = d
            rv = 1.0 - d / d0 if d0 else float("nan")
            cv = [1.0 - c / d0 for c in cd] if d0 else []
            row[str(r)] = {"dnll": d, "recovery": rv,
                           "control_dnll": cd,
                           "control_recovery": cv,
                           "bottom_dnll": db,
                           "bottom_recovery": (1.0 - db / d0) if (db is not None
                                                                 and d0) else None,
                           "matched_norm": tn}
            rec_real[r] = rv
            rec_ctrl[r] = float(np.mean(cv)) if cv else None
            cds = f"{np.mean(cd):>10.4f}" if cd else f"{'-':>10}"
            cvs = f"{np.mean(cv):>11.3f}" if cv else f"{'-':>11}"
            bvs = (f"{1.0 - db / d0:>9.3f}" if (db is not None and d0)
                   else f"{'-':>9}")
            print(f"{r:>4} {d:>+9.4f} {rv:>9.3f} {cds} {cvs} {bvs}", flush=True)

        hit = [r for r in ranks if rec_real[r] >= args.recover]
        r_star = min(hit) if hit else None
        hitc = [r for r in ranks if rec_ctrl.get(r) is not None
                and rec_ctrl[r] >= args.recover]
        r_star_ctrl = min(hitc) if hitc else None
        res["per_head"][names[k]] = {
            "full_effect_dnll": d0, "by_rank": row,
            "r_star": r_star, "r_star_control": r_star_ctrl,
            "identity_check_dnll_at_full_rank": row[str(d_head)]["dnll"]}
        print(f"  r* (>= {args.recover:.0%} recovery) = {r_star}   "
              f"control r* = {r_star_ctrl}   "
              f"full-rank dNLL {row[str(d_head)]['dnll']:+.2e} (must be ~0)\n",
              flush=True)
        del U, s_ov, Vt, A, B
        gc.collect()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(res, fh, indent=2)

    nll_chk, _ = probe(model, ids, args.chunk, False)
    res["restore_abs_diff"] = abs(nll_chk - nll0)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=2)

    print(f"=== r* summary (step {args.step}) ===")
    for k in heads:
        h = res["per_head"][names[k]]
        print(f"  {names[k]:>7}  effect {h['full_effect_dnll']:>+7.3f}   "
              f"r* {str(h['r_star']):>4} of {d_head}   "
              f"control r* {str(h['r_star_control']):>4}")
    print(f"\nrestore {abs(nll_chk - nll0):.1e}   wrote {out_path}")


if __name__ == "__main__":
    main()
