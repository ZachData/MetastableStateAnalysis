"""Every head's marginal causal cost once `L5H2` is already ablated -- the
recall check on `l5h2_backup_search.py`'s attention-based candidate list.

WHY THIS EXISTS
---------------
`PROJECT.md` §3.21 established that three redundancy-set members (`L11H14`,
`L8H6`, `L8H9`) reproduce `L7H8`'s own super-additive interaction with `L5H2`,
so §3.12-S's super-additivity is a set-wide property rather than one pair's
quirk. But the candidates tested there came from an ATTENTION search -- heads
whose own induction-attention rose when `L5H2` was ablated -- and that search
has an obvious recall hole: **a head can pick up causal slack without its
attention pattern moving at all.** Its OV write can become more load-bearing
because what reaches it changed, while it goes on attending exactly where it
always did. §3.21 named this gap explicitly ("the search only read attention,
not every head's own marginal ΔNLL") and this runner closes it by brute force.

WHAT IT MEASURES. For all `n_layers x n_heads` heads, both arms in one
process off the same weights and the same batch:

    solo      = dNLL(head ablated)                       -- the catalogue's own quantity
    joint     = dNLL(L5H2 + head ablated)
    marginal  = joint - dNLL(L5H2 alone)                 -- cost of removing the head
                                                            once L5H2 is already gone
    interaction = joint - dNLL(L5H2 alone) - solo        -- §3.12-S's quantity

A stand-in for `L5H2` is a head with a large positive `interaction`. Ranking
every head by it says whether §3.21's three are the whole story or the visible
part of a longer tail.

BOTH ARMS IN ONE PROCESS, not one arm against `redundancy_catalog.json`. The
solo column is the catalogue's own measurement and at matched settings
(step 16000, 8 seqs, `ov`, `EVAL_SEED`) it should reproduce -- which is a free
validation and is reported as `catalog_agreement` -- but the COMPARISON is
computed in-process so it cannot rest on an uncontrolled instrument
difference (`fv_score.py`'s own rule, applied here).

THE CEILING IS RECORDED PER ARM, because this is exactly the regime §3.14.4-D
warns about: `dNLL` is bounded by `ln 50304 = 10.826`, `L5H2` alone already
costs +2.2 on a ~0.53 baseline, and a joint arm with a large head lands
several nats up. `headroom = ln V - nll_joint` is stored for every head and
`ceiling_contaminated` flags `headroom < 1.5`. §3.12-M5: the compressive
readout biases interactions toward apparent SUB-additivity, so a contaminated
cell's positive interaction is conservative while a contaminated cell's
negative interaction is not evidence.

COST: one model load, then 2 + 2*n_heads NLL evaluations (~770 at 410m,
~30 min at 8 seqs on this box). Rows are written per layer, so a killed run
keeps what it finished. No p-value; nothing registered; pythia-410m spent
under `check_registry` rule 3.
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
from tools.run.induction_rank_sweep import PROBE_ARMS, EVAL_SEED, ablated, arch_dims
from p7d_redundancy.member_formation_curves import batch, catalog_path_for
from p7d_redundancy.redundancy_catalog import nll

OUT = DATA / "analysis" / "backup_sweep_full.json"
DEFAULT_MODEL = "pythia-410m"
LN_VOCAB = float(np.log(50304))


def parse_head(s):
    return (int(s[1:s.index("H")]), int(s[s.index("H") + 1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--step", type=int, default=16000,
                     help="the catalogue's own step, so the solo column can be "
                          "cross-validated against redundancy_catalog.json")
    ap.add_argument("--source", default="L5H2")
    ap.add_argument("--seqs", type=int, default=8,
                     help="8 matches redundancy_catalog.py's default exactly")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--probe", default="wide", choices=tuple(PROBE_ARMS))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    source = parse_head(args.source)
    out = Path(args.out) if args.out else (
        OUT if args.model == DEFAULT_MODEL else
        DATA / "analysis" / f"backup_sweep_full_{args.model}.json")

    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()

    model, _ = load_causal_lm(f"{args.model}-step{args.step}")
    model.eval()
    _, _, n_heads = arch_dims(model)
    n_layers = model.config.num_hidden_layers

    ids = batch(np.random.default_rng(EVAL_SEED), args.seqs,
                *PROBE_ARMS[args.probe])
    base = nll(model, ids, args.chunk)
    with ablated(model, [source], mode="ov"):
        nll_source = nll(model, ids, args.chunk)
    d_source = nll_source - base
    print(f"baseline NLL {base:.4f}   {args.source} alone dNLL {d_source:+.4f} "
          f"(absolute {nll_source:.4f}, ln V = {LN_VOCAB:.3f})", flush=True)

    rows, t0 = [], time.time()
    for L in range(n_layers):
        for H in range(n_heads):
            if (L, H) == source:
                continue
            with ablated(model, [(L, H)], mode="ov"):
                solo = nll(model, ids, args.chunk) - base
            with ablated(model, [source, (L, H)], mode="ov"):
                nll_joint = nll(model, ids, args.chunk)
            joint = nll_joint - base
            marginal = joint - d_source
            interaction = joint - d_source - solo
            headroom = LN_VOCAB - nll_joint
            rows.append({
                "head": f"L{L}H{H}", "layer": L, "head_idx": H,
                "dnll_solo": solo, "dnll_joint": joint,
                "dnll_marginal_given_source": marginal,
                "interaction": interaction,
                "nll_joint_absolute": nll_joint, "headroom": headroom,
                "ceiling_contaminated": headroom < 1.5})
            if interaction > 0.1 or interaction < -0.1:
                print(f"    L{L}H{H:<2}  solo {solo:>+7.4f}  "
                      f"marginal {marginal:>+8.4f}  "
                      f"interaction {interaction:>+8.4f}"
                      f"{'  [CEILING]' if headroom < 1.5 else ''}", flush=True)
        print(f"  layer {L:>2}/{n_layers - 1} done "
              f"({(time.time() - t0) / 60:.1f} min elapsed)", flush=True)

        # Per-layer write: a killed run must not lose the layers it finished.
        res = _payload(args, git_sha, base, d_source, nll_source, rows,
                       n_layers, n_heads)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(res, fh, indent=2)

    chk = abs(nll(model, ids, args.chunk) - base)
    res = _payload(args, git_sha, base, d_source, nll_source, rows,
                   n_layers, n_heads)
    res["restore_abs_diff"] = float(chk)

    # Free validation: the solo column against the catalogue, if it is on disk
    # at matched settings.
    cat = catalog_path_for(args.model)
    if cat.exists() and args.step == 16000 and args.probe == "wide":
        c = json.load(open(cat))
        # A catalogue written before `--ablation` existed has no such key, and
        # `ov` is the historical path it used (redundancy_catalog.py's own
        # docstring). Treat missing/None as `ov` rather than skipping the
        # check, which is what silently suppressed it the first time.
        if (c.get("n_seqs") == args.seqs
                and (c.get("ablation") or "ov") == "ov"):
            on_disk = {r["head"]: r["dnll"] for r in c["heads"]}
            diffs = [abs(r["dnll_solo"] - on_disk[r["head"]])
                     for r in rows if r["head"] in on_disk]
            res["catalog_agreement"] = {
                "catalog": str(cat), "n_compared": len(diffs),
                "max_abs_diff": float(max(diffs)) if diffs else None,
                "mean_abs_diff": float(np.mean(diffs)) if diffs else None}
            print(f"\ncatalogue agreement on the solo column: "
                  f"max |diff| {max(diffs):.2e} over {len(diffs)} heads")

    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)

    top = sorted(rows, key=lambda r: -r["interaction"])[:20]
    print(f"\ntop 20 by interaction with {args.source}:")
    for r in top:
        print(f"  {r['head']:>7}  solo {r['dnll_solo']:>+7.4f}  "
              f"interaction {r['interaction']:>+8.4f}"
              f"{'  [CEILING]' if r['ceiling_contaminated'] else ''}")
    print(f"\nrestore check abs diff {chk:.2e}")
    print(f"wrote {out}")


def _payload(args, git_sha, base, d_source, nll_source, rows, n_layers, n_heads):
    inter = np.array([r["interaction"] for r in rows]) if rows else np.array([0.0])
    return {
        "_what_this_is": __doc__, "git_sha": git_sha, "model": args.model,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "step": args.step, "source": args.source, "probe": args.probe,
        "n_seqs": args.seqs, "ablation": "ov",
        "n_layers": n_layers, "n_heads_per_layer": n_heads,
        "ln_vocab": LN_VOCAB,
        "baseline_nll": base, "dnll_source_alone": d_source,
        "nll_source_alone_absolute": nll_source,
        "interaction_median": float(np.median(inter)),
        "interaction_p99": float(np.percentile(inter, 99)),
        "n_interaction_above_0.1": int((inter > 0.1).sum()),
        "n_interaction_below_-0.1": int((inter < -0.1).sum()),
        "n_ceiling_contaminated": int(sum(r["ceiling_contaminated"] for r in rows)),
        "heads": rows}


if __name__ == "__main__":
    main()
