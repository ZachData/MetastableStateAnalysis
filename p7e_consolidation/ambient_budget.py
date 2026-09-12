"""Does the set's joint effect fit inside one head's 64 dimensions?

WHY THIS EXISTS
---------------
`design-7e.md` proposes collapsing the redundancy set into a single head. The
obvious objection is arithmetic: one head's OV is **rank <= 64** (`D_HEAD`),
while 7d measured the members' individual effect subspaces at **150-300**
dimensions and their union larger still. If the space that must be spanned is
really that big, consolidation is impossible and the phase should not run.

7d's ambient measurement says the objection is probably wrong. The baseline
residual stream's own participation ratio on this probe is **20.3 of 1024** at
step 16000 and **8.4 at 143000**, and the members put 59 % of their effect
energy in the ambient top-50. The 150-300 dimensions are mostly spread over
directions the residual stream barely uses; what has to be spanned is the
ambient part.

THIS RUNNER SETTLES IT. Ablate all members jointly, take the residual delta, and
project it onto the baseline residual's leading ambient directions. Report the
smallest `k` recovering 90 % of the joint effect's energy. **`k <= 64` and the
phase proceeds; `k >> 64` and the capacity objection stands**, and the design
needs a multi-head target instead of a single one.

WHY THE JOINT ARM AND NOT THE SUM OF SINGLES. The quantity a survivor would have
to reproduce is what the set does *together*, which 7d showed is not the sum of
what its members do apart -- 44/45 pairwise cells are super-additive. Six
single-head deltas would answer a question nobody is asking.

BOTH READOUTS, per `design-7e.md`. The induction probe is where consolidation is
easiest and where a false success would appear first, so the natural-text arm is
reported beside it and neither is quoted alone. `--text` is required for the
second; without it the run is explicitly labelled probe-only.

NO p-value; nothing registered. pythia-410m is spent under `check_registry`
rule 3, and this phase additionally measures modified models.
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

from core.lm_loading import load_causal_lm
from tools.run.induction_rank_sweep import (
    EVAL_SEED, ablated, arch_dims, head_means,
)

from p7d_redundancy.member_formation_curves import batch, catalog_path_for, members
from p7d_redundancy.member_subspace_geometry import (
    participation_ratio, resid_matrix,
)

OUT = DATA / "analysis" / "ambient_budget.json"
DEFAULT_MODEL = "pythia-410m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="registry family prefix, e.g. pythia-70m, pythia-410m "
                         "(p8_scale_ladder/design-8.md rung policy: 1b/1.4b "
                         "are reserved -- do not pass those without a "
                         "registered prediction)")
    ap.add_argument("--steps", default="16000,143000")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--heads", default="")
    ap.add_argument("--seqs", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=2,
                    help="2, not 4 -- the 7d geometry runner was killed for "
                         "memory at 4 with 16 sequences")
    ap.add_argument("--energy", type=float, default=0.90)
    ap.add_argument("--ablation", default="ov", choices=("ov", "zero", "mean"),
                    help="'mean' replaces each head's output with its clean-run "
                         "mean -- the control for zero-ablation's "
                         "off-distribution bias (status-8.md's A/B)")
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
        out_path = DATA / "analysis" / f"ambient_budget_{args.model}{_abl}.json"

    def parse(spec):
        return [(int(h[1:h.index("H")]), int(h[h.index("H") + 1:]))
                for h in spec.split(",") if h]

    if args.heads:
        heads, source = parse(args.heads), "--heads"
    else:
        heads, source = members(args.top, catalog_path_for(args.model))
    steps = [int(x) for x in args.steps.split(",") if x]
    names = [f"L{L}H{H}" for L, H in heads]

    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    res = {"_what_this_is": __doc__, "git_sha": git_sha, "model": args.model,
           "ablation": args.ablation,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
           "membership_source": source, "heads": names, "steps": steps,
           "readout": "induction probe only -- natural-text arm NOT run here",
           "per_step": {}}

    print(f"model:   {args.model}")
    print(f"members: {', '.join(names)}   ({source})")

    for s in steps:
        model, _ = load_causal_lm(f"{args.model}-step{s}")
        model.eval()
        d_model, d_head, n_heads = arch_dims(model)
        res["d_head"] = d_head
        print(f"budget:  one head's OV is rank <= {d_head}; the question is "
              f"whether {args.energy:.0%} of the JOINT effect fits in that "
              f"many ambient directions\n")
        ids = batch(np.random.default_rng(EVAL_SEED), args.seqs)
        means = (head_means(model, ids, heads, args.chunk)
                 if args.ablation == "mean" else None)

        nll0, R0 = resid_matrix(model, ids, args.chunk)
        _, s_amb, Vt_amb = np.linalg.svd(R0, full_matrices=False)

        with ablated(model, heads, args.ablation, means):
            nll_j, Rj = resid_matrix(model, ids, args.chunk)
        nll_chk, _ = resid_matrix(model, ids, args.chunk)

        Dj = Rj - R0
        # Energy of the joint effect along the residual stream's OWN ordering.
        per = ((Dj @ Vt_amb.T) ** 2).sum(0)
        cum = np.cumsum(per) / per.sum()
        k_needed = int(np.searchsorted(cum, args.energy) + 1)
        # And the same question asked of the effect's own basis, which is the
        # quantity the capacity objection was originally stated in.
        sv = np.linalg.svd(Dj, compute_uv=False)
        e = sv ** 2
        r_own = int(np.searchsorted(np.cumsum(e) / e.sum(), args.energy) + 1)

        rec = {"baseline_nll": nll0, "joint_ablated_nll": nll_j,
               "joint_dnll": nll_j - nll0,
               "restore_abs_diff": abs(nll_chk - nll0),
               "ambient_participation_ratio": participation_ratio(s_amb),
               "joint_effect_participation_ratio": participation_ratio(sv),
               "k_ambient_for_energy": k_needed,
               "r_own_basis_for_energy": r_own,
               "energy_in_ambient_top_64": float(cum[d_head - 1]),
               "fits_in_one_head": bool(k_needed <= d_head)}
        res["per_step"][str(s)] = rec

        print(f"step {s}  baseline NLL {nll0:.4f}  joint dNLL "
              f"{nll_j - nll0:+.4f}  restore {abs(nll_chk - nll0):.1e}")
        print(f"    ambient participation ratio      {participation_ratio(s_amb):>8.1f} "
              f"of {d_model}")
        print(f"    joint effect, its OWN basis      {r_own:>8d} dims for "
              f"{args.energy:.0%}")
        print(f"    joint effect, AMBIENT ordering   {k_needed:>8d} dims for "
              f"{args.energy:.0%}")
        print(f"    energy inside ambient top-{d_head}      "
              f"{cum[d_head - 1]:>8.3f}")
        print(f"    -> {'FITS in one head' if k_needed <= d_head else 'DOES NOT FIT -- capacity objection stands'}"
              f" (budget {d_head})\n", flush=True)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(res, fh, indent=2)
        del model, ids, R0, Rj, Dj

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
