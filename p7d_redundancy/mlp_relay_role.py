"""Is MLP 6 an upstream supplier or an output-side compensator? The conditional
attention arm, for MLPs.

WHY THIS EXISTS
---------------
`PROJECT.md` §3.22 found that MLP 6's interaction with `L5H2` is **+6.26** --
above `L7H8`'s +4.07 and above every head in the model -- on a solo effect of
+0.14, and that it is specific to `L5H2` (+0.125 against `L7H8`, +0.127
against `L12H5`). §3.22 also established that `L5H2`'s stand-ins fall into two
classes, separated by position relative to the matcher:

  upstream   -- partly restore `L7H8`'s own matching attention once the relay
                is gone (`L5H9` -0.087, 23x the conditional null)
  downstream -- carry large ΔNLL interactions while moving `L7H8`'s attention
                by EXACTLY 0.0000 (`L11H14`, `L9H5`); they compensate at the
                readout instead

MLP 6 has not been placed in that taxonomy, and §3.22 closed with "nothing here
opens the MLP itself".

THE ARCHITECTURE MAKES A PREDICTION, and it is worth stating before measuring.
pythia-410m has `use_parallel_residual = True` (verified on the loaded config,
not assumed): at every layer the attention and the MLP read the *same*
layernormed residual and both write into the next one. So **MLP 5 cannot see
`L5H2`'s output at all** -- they are parallel -- and **MLP 6 is the first
sublayer in the network that can read `L5H2`'s write**, while writing into the
residual `L7H8` then reads. MLP 6 is uniquely interposed between the relay and
the matcher, which predicts it should behave as an UPSTREAM supplier. If
instead its conditional effect on `L7H8`'s attention is ~0, then an object
sitting directly in the path is compensating without touching the matching
pathway, and the two-class picture needs a third entry.

WHAT IT MEASURES, for every MLP layer, so the profile carries its own null:

    unconditional = attn(L7H8 | MLP L ablated)      - attn(L7H8)
    conditional   = attn(L7H8 | L5H2 + MLP L abl.)  - attn(L7H8 | L5H2 abl.)

BOTH ABLATION MODES, and the `mean` arm recomputes its means INSIDE each
background condition rather than reusing the clean-model means. That is the
one methodological choice here that could silently go wrong: the clean mean
injected into an `L5H2`-ablated forward pass is an off-distribution constant,
and the arm would then differ from its baseline in two ways at once. `zero` is
carried as the mode with no mean-estimation question in it at all, and §3.22
already showed the ΔNLL result holds in both (+6.26 `mean` / +5.25 `zero`).

Attention, not NLL, on purpose: §3.20's whole point is that a network-wide
backup can restore the loss without restoring THIS head's attention pattern,
so the attention readout is what separates the two classes.

COST: one model load, then 2 + 2*n_layers attention passes per mode. No
p-value; nothing registered; pythia-410m spent under `check_registry` rule 3.
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

from core.lm_loading import load_causal_lm
from tools.run.induction_rank_sweep import PROBE_ARMS, EVAL_SEED, ablated
from p7d_redundancy.member_formation_curves import batch
from p7d_redundancy.fv_score import induction_scores
from p7d_redundancy.mlp_backup_check import ablated_mlp, mlp_output_means

OUT = DATA / "analysis" / "mlp_relay_role.json"
DEFAULT_MODEL = "pythia-410m"


def parse_head(s):
    return (int(s[1:s.index("H")]), int(s[s.index("H") + 1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--steps", default="16000,143000")
    ap.add_argument("--source", default="L5H2", help="the relay, ablated to "
                                                      "create the conditional state")
    ap.add_argument("--target", default="L7H8", help="head whose own attention is read")
    ap.add_argument("--modes", default="zero,mean")
    ap.add_argument("--seqs", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--probe", default="wide", choices=tuple(PROBE_ARMS))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    source, target = parse_head(args.source), parse_head(args.target)
    out = Path(args.out) if args.out else (
        OUT if args.model == DEFAULT_MODEL else
        DATA / "analysis" / f"mlp_relay_role_{args.model}.json")
    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    steps = [int(x) for x in args.steps.split(",") if x]
    modes = [m for m in args.modes.split(",") if m]

    res = {"_what_this_is": __doc__, "git_sha": git_sha, "model": args.model,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
           "source": args.source, "target": args.target, "probe": args.probe,
           "n_seqs": args.seqs, "steps": steps, "per_step": {}}

    for s in steps:
        model, _ = load_causal_lm(f"{args.model}-step{s}")
        model.eval()
        n_layers = model.config.num_hidden_layers
        parallel = bool(getattr(model.config, "use_parallel_residual", False))
        ids = batch(np.random.default_rng(EVAL_SEED), args.seqs,
                    *PROBE_ARMS[args.probe])
        ck = max(1, args.chunk // 2)

        def attn(want=(target,)):
            return induction_scores(model, ids, list(want), ck)[target]

        clean = attn()
        with ablated(model, [source], mode="ov"):
            cond_base = attn()
        # Means for each background condition, taken IN that condition.
        means_clean = mlp_output_means(model, ids, args.chunk)
        with ablated(model, [source], mode="ov"):
            means_cond = mlp_output_means(model, ids, args.chunk)

        print(f"== step {s}   parallel_residual={parallel}   "
              f"{args.target} attention: clean {clean:.4f}   "
              f"{args.source}-ablated {cond_base:.4f}")

        step_rows = {}
        for mode in modes:
            rows = {}
            print(f"  -- mode {mode} --")
            print(f"  {'MLP':>4} {'uncond Δ':>10} {'cond Δ':>10} "
                  f"{'cond attn':>10}")
            for L in range(n_layers):
                with ablated_mlp(model, L, mode=mode, means=means_clean):
                    a_uncond = attn()
                with ablated(model, [source], mode="ov"):
                    with ablated_mlp(model, L, mode=mode, means=means_cond):
                        a_cond = attn()
                rows[str(L)] = {
                    "layer": L,
                    "unconditional_delta": a_uncond - clean,
                    "conditional_delta": a_cond - cond_base,
                    "conditional_attn": a_cond,
                    "unconditional_attn": a_uncond}
                print(f"  {L:>4} {a_uncond - clean:>+10.4f} "
                      f"{a_cond - cond_base:>+10.4f} {a_cond:>10.4f}",
                      flush=True)
            cd = np.array([r["conditional_delta"] for r in rows.values()])
            step_rows[mode] = {
                "layers": rows,
                "conditional_median": float(np.median(cd)),
                "conditional_min": float(cd.min()),
                "argmin_layer": int(np.argmin(cd))}
            print(f"   median conditional Δ {np.median(cd):+.4f}   "
                  f"most negative {cd.min():+.4f} at MLP "
                  f"{int(np.argmin(cd))}\n")

        chk = attn()
        res["per_step"][str(s)] = {
            "use_parallel_residual": parallel,
            "clean_attn": clean, "conditional_baseline_attn": cond_base,
            "restore_check_abs_diff": abs(chk - clean),
            "per_mode": step_rows}
        print(f"   restore check abs diff {abs(chk - clean):.2e}\n")

        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(res, fh, indent=2)
        del model

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
