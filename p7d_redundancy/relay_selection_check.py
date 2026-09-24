"""What selects `L5H2` as the relay: prev-token attention, or composition into
the matcher's read-space? Per-head, weights-only, both quantities side by side.

WHY THIS EXISTS
---------------
`PROJECT.md` §3.21 asked why `L11H14` is `L5H2`'s strongest stand-in.
`prev_token_profile.py` (2026-09-13) answered the obvious hypothesis with a
clean NO: the three stand-ins rank **372nd, 377th and 380th of 384** on
prev-token attention, *below* the population median, while **13 heads sit above
0.3** and none of them is a stand-in. So standing in for `L5H2` is not about
supplying its prev-token signal.

That refutation raises a sharper question about `L5H2` itself. If half a dozen
heads carry substantial prev-token attention (`L4H9` 0.81, `L3H1` 0.71,
`L5H9` 0.62 at step 16000, against `L5H2`'s 0.97), **why is only `L5H2` the
relay?** §3.12-H1-REVISED measured `L5H2`'s composition into `L7H8`'s read-space
at rank 0 of 112 with z ~ +6, but it stored only the population SUMMARY
(median, max, rank, z) -- not which heads the other 111 were. So the natural
comparison, "do the strong prev-token heads also compose into `L7H8`?", was
not answerable from the artifact.

WHAT THIS MEASURES. For every head in layers `0..DST_LAYER-1`, the Elhage
composition score into `L7H8`'s K, Q and V read paths, joined with that head's
prev-token and induction attention from `prev_token_profile.json`. If
composition is what selects the relay, `L5H2` should top the composition
ranking while `L4H9`/`L3H1`/`L5H9` sit in the ordinary population despite
their prev-token attention -- prev-token capacity common, composition rare.

IMPLEMENTATION REUSES `tools/run/induction_composition_whitening.py`'s own
`comp_score`, `qkv_head` and `_load_ov` rather than reimplementing them,
because the transpose convention here is a live defect elsewhere in the repo
(§3.14.3 item 3: `ov_factors` returns the TRANSPOSE of the residual operator
with no warning) and the population numbers must stay comparable with
H1-REVISED's.

410m-ONLY, deliberately. The imported module takes `D_MODEL/D_HEAD/N_HEADS`
from `induction_rank_sweep`'s module-level 410m constants rather than from
`arch_dims`, and the cached `ov_weights_*.npz` this reads are 410m artifacts.
Do not point it at another rung without fixing both.

Weights only -- no forward pass, no p-value, nothing registered.
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
from tools.run.induction_composition_whitening import (
    DST_HEAD, DST_LAYER, SRC_HEAD, SRC_LAYER, _load_ov, comp_score, qkv_head,
)
from tools.run.induction_rank_sweep import N_HEADS

OUT = DATA / "analysis" / "relay_selection_check.json"
PREV = DATA / "analysis" / "prev_token_profile.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="4000,16000,143000")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out = Path(args.out) if args.out else OUT
    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    steps = [int(x) for x in args.steps.split(",") if x]

    prev = json.load(open(PREV)) if PREV.exists() else None
    if prev is None:
        print(f"note: {PREV} not on disk -- composition only, no join")

    res = {"_what_this_is": __doc__, "git_sha": git_sha, "model": "pythia-410m",
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
           "src": f"L{SRC_LAYER}H{SRC_HEAD}", "dst": f"L{DST_LAYER}H{DST_HEAD}",
           "prev_token_source": str(PREV) if prev else None,
           "steps": steps, "per_step": {}}

    for s in steps:
        model, _ = load_causal_lm(f"pythia-410m-step{s}")
        model.eval()
        W_Q, W_K, W_V = qkv_head(model, DST_LAYER, DST_HEAD)
        pv = (prev["per_step"].get(str(s), {}).get("heads", {})
              if prev else {})

        rows = []
        for L in range(DST_LAYER):
            for h in range(N_HEADS):
                ov = _load_ov(s, L, h)
                tag = f"L{L}H{h}"
                r = {"head": tag, "layer": L, "head_idx": h,
                     "K": comp_score(W_K, ov), "Q": comp_score(W_Q, ov),
                     "V": comp_score(W_V, ov)}
                if tag in pv:
                    r["prev_token"] = pv[tag]["prev_token"]
                    r["induction"] = pv[tag]["induction"]
                rows.append(r)

        K = np.array([r["K"] for r in rows])
        order = np.argsort(-K)
        for rank, i in enumerate(order):
            rows[i]["K_rank"] = rank
        rows_by_tag = {r["head"]: r for r in rows}
        zK = (K - K.mean()) / max(K.std(), 1e-300)
        for i, r in enumerate(rows):
            r["K_z"] = float(zK[i])

        res["per_step"][str(s)] = {
            "K_median": float(np.median(K)), "K_max": float(K.max()),
            "n_scored": len(rows), "heads": rows_by_tag}

        src = rows_by_tag[f"L{SRC_LAYER}H{SRC_HEAD}"]
        print(f"== step {s}   composition into L{DST_LAYER}H{DST_HEAD}, "
              f"{len(rows)} heads in layers 0..{DST_LAYER - 1}")
        print(f"   L{SRC_LAYER}H{SRC_HEAD}: K {src['K']:.4f}  "
              f"rank {src['K_rank']} / {len(rows)}  z {src['K_z']:+.2f}"
              + (f"  prev_token {src['prev_token']:.4f}"
                 if "prev_token" in src else ""))
        print(f"   {'head':>7} {'K':>8} {'K_rank':>7} {'K_z':>7} "
              f"{'prev_tok':>9} {'pt_rank*':>9}")
        top = [rows[i] for i in order[:10]]
        # the strong prev-token heads in range, whatever their composition
        if pv:
            in_range = [r for r in rows if "prev_token" in r]
            strong = sorted(in_range, key=lambda r: -r["prev_token"])[:8]
        else:
            strong = []
        seen = set()
        for label, group in (("top composition", top),
                             ("top prev-token", strong)):
            print(f"   -- {label} --")
            for r in group:
                if (label, r["head"]) in seen:
                    continue
                seen.add((label, r["head"]))
                pt = f"{r['prev_token']:.4f}" if "prev_token" in r else "-"
                print(f"   {r['head']:>7} {r['K']:>8.4f} {r['K_rank']:>7} "
                      f"{r['K_z']:>+7.2f} {pt:>9}")
        print()

        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(res, fh, indent=2)
        del model

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
