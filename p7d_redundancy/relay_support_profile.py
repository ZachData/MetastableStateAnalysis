"""Which heads does the relay actually support? The cross-rung, role-agnostic
version of §3.20, on a readout both rungs can carry.

WHY THIS EXISTS
---------------
`PROJECT.md` §3.26 ported the self-repair chain to 70m and found the relay
replicates (`L2H1`: prev-token 0.950 and the largest catalogue effect, +6.17 --
the same double signature as `L5H2`) while the circuit around it does not: 70m's
only strong same-token matcher sits in **layer 0**, upstream of the relay, so
the 410m ordering cannot exist. That section closed by admitting the
replication question as posed "does not quite have a subject" at 70m.

This is the better-posed question. `L2H1` moves the readout by **+6.17 on a
5.73 baseline**, so *something* depends on it, and naming that something is the
70m analogue of §3.20 -- which found `L5H2` feeds `L7H8`'s matching attention
(-0.19) and `L6H0`'s harder still (-0.44).

WHAT IT MEASURES. Per head, the total-variation distance between its attention
distribution with the relay intact and with the relay OV-ablated, over
second-copy query positions. TV presumes no role, which is required here: at
70m no causally-important head does induction attention at all (§3.26), so
`induction_scores` has nothing to point at, while at 410m it does. One readout
that works at both rungs is the whole point -- §3.26's cross-rung table had to
be built from two different instruments and this one does not.

WHAT TO COMPARE. Not the head names, which never transfer (`design-8.md`:
"nothing transfers by head name"). The **shape**: how concentrated the relay's
support is, whether it lands on the causally-defined catalogue, and how far
above the model's own population it sits. Each rung is read against its own
median, never a placed threshold.

SCOPE. Only heads DOWNSTREAM of the relay can be affected; heads at or below it
are the built-in zero check and must return exactly 0.

THE RAW TV FAILED ITS POSITIVE CONTROL, AND THAT IS WHY THE NULL IS HERE.
First version reported raw TV and put `L7H8` at **rank 191 of 288**, with TV
0.1887 *below* the population median of 0.2400 -- even though §3.20 measured
`L5H2`'s ablation dropping `L7H8`'s induction attention 0.938 -> 0.751. The two
agree numerically (TV 0.189 vs the 0.190 drop); what fails is the comparison,
because ablating the relay moves EVERY downstream head's distribution by ~0.24
and a specific loss of 0.19 does not stand out against that. Raw TV detects a
near-total collapse (which is why `mlp_backup_attention_scan.py` passed its
control -- there `L7H8` goes to 0.045) and misses a moderate targeted shift.

So every head is calibrated against **its own sensitivity to generic
ablation**: `--controls K` ablates K heads drawn uniformly from outside the
catalogue and takes each head's mean TV under those, giving a per-head null.
The reported statistic is the ratio `tv_relay / tv_control`, which asks whether
this head moved MORE than it moves when something arbitrary is removed. That is
§3.12-V3's measured-null discipline applied to a readout that needed it.

410m IS THE POSITIVE CONTROL: this must recover `L7H8`, which §3.20 established
on a different instrument. A readout that cannot reproduce a known answer is
not evidence about another rung.

No p-value; nothing registered; both rungs are exploration and spent under
`check_registry` rule 3.
"""
import argparse
import contextlib
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
from tools.run.induction_rank_sweep import PROBE_ARMS, EVAL_SEED, ablated
from p7d_redundancy.member_formation_curves import batch, catalog_path_for
from p7d_redundancy.mlp_backup_attention_scan import tv_per_head

OUT = DATA / "analysis" / "relay_support_profile.json"


def parse_head(s):
    return (int(s[1:s.index("H")]), int(s[s.index("H") + 1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-410m")
    ap.add_argument("--steps", default="16000")
    ap.add_argument("--source", default="L5H2",
                     help="the relay: L5H2 at 410m, L2H1 at 70m")
    ap.add_argument("--seqs", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--probe", default="wide", choices=tuple(PROBE_ARMS))
    ap.add_argument("--top-n", type=int, default=12)
    ap.add_argument("--controls", type=int, default=4,
                     help="generic heads whose ablation gives each head its "
                          "own sensitivity null (§3.12-V3)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    source = parse_head(args.source)
    out = Path(args.out) if args.out else (
        OUT if args.model == "pythia-410m" else
        DATA / "analysis" / f"relay_support_profile_{args.model}.json")
    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    steps = [int(x) for x in args.steps.split(",") if x]

    cat_path = catalog_path_for(args.model)
    cat = {}
    if cat_path.exists():
        cat = {r["head"]: r["dnll"] for r in json.load(open(cat_path))["heads"]}

    res = {"_what_this_is": __doc__, "git_sha": git_sha, "model": args.model,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
           "source": args.source, "probe": args.probe, "n_seqs": args.seqs,
           "catalogue": str(cat_path) if cat else None,
           "steps": steps, "per_step": {}}

    for s in steps:
        model, _ = load_causal_lm(f"{args.model}-step{s}")
        model.eval()
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
        ids = batch(np.random.default_rng(EVAL_SEED), args.seqs,
                    *PROBE_ARMS[args.probe])

        @contextlib.contextmanager
        def nothing():
            yield

        @contextlib.contextmanager
        def relay_ablated():
            with ablated(model, [source], mode="ov"):
                yield

        tv = tv_per_head(model, ids, nothing, relay_ablated,
                         n_layers, n_heads, args.chunk)

        # Per-head sensitivity null: what this head's attention does when an
        # ARBITRARY head is removed. Drawn from outside the catalogue so a
        # control is never itself a member.
        rng_c = np.random.default_rng(EVAL_SEED)
        # The catalogue scores EVERY head, so "in cat" is not membership --
        # members are its top few by dNLL. Excluding the whole catalogue here
        # emptied the pool outright, which is how this was caught.
        members = {h for h, _ in sorted(cat.items(), key=lambda kv: -kv[1])[:10]}
        pool = [(L, h) for L in range(n_layers) for h in range(n_heads)
                if (L, h) != source and f"L{L}H{h}" not in members]
        ctrl = [pool[i] for i in
                rng_c.choice(len(pool), size=args.controls, replace=False)]
        tv_ctrl = np.zeros_like(tv)
        for (cl, ch) in ctrl:
            @contextlib.contextmanager
            def ctrl_ablated(cl=cl, ch=ch):
                with ablated(model, [(cl, ch)], mode="ov"):
                    yield
            tv_ctrl += tv_per_head(model, ids, nothing, ctrl_ablated,
                                   n_layers, n_heads, args.chunk)
            print(f"  control L{cl}H{ch} done", flush=True)
        tv_ctrl /= max(len(ctrl), 1)

        down, at_or_below = [], []
        for L in range(n_layers):
            for h in range(n_heads):
                tag = f"L{L}H{h}"
                base = float(tv_ctrl[L, h])
                row = {"head": tag, "layer": L, "tv": float(tv[L, h]),
                       "tv_control": base,
                       "ratio": float(tv[L, h]) / max(base, 1e-6),
                       "catalogue_dnll": cat.get(tag)}
                (down if L > source[0] else at_or_below).append(row)

        v = np.array([r["ratio"] for r in down])
        med = float(np.median(v))
        order = sorted(down, key=lambda r: -r["ratio"])
        # Concentration: how many heads carry half the total departure from
        # the population median -- the §3.13 "is it a few units or all of
        # them" question, asked of the relay's support.
        excess = np.maximum(v - med, 0.0)
        srt = np.sort(excess)[::-1]
        half = float(srt.sum()) / 2.0
        n_half = int(np.searchsorted(np.cumsum(srt), half) + 1) if srt.sum() > 0 else 0

        zc = max([r["tv"] for r in at_or_below] + [0.0])
        print(f"== {args.model} step {s}   relay {args.source}   "
              f"probe {args.probe}")
        print(f"  zero check (layers <= {source[0]}): max TV {zc:.1e}")
        print(f"  downstream n={len(down)}   median ratio {med:.3f}   "
              f"max {v.max():.3f}   heads carrying half the above-median "
              f"mass: **{n_half} of {len(down)}**")
        print(f"  {'head':>8} {'tv_relay':>9} {'tv_ctrl':>8} {'ratio':>7} "
              f"{'x median':>9} {'catalogue':>10}")
        for r in order[:args.top_n]:
            c = (f"{r['catalogue_dnll']:+.3f}"
                 if r["catalogue_dnll"] is not None else "-")
            print(f"  {r['head']:>8} {r['tv']:>9.4f} {r['tv_control']:>8.4f} "
                  f"{r['ratio']:>7.2f} {r['ratio'] / max(med, 1e-9):>9.1f} "
                  f"{c:>10}")

        # Where the catalogue's own top heads land in this ranking.
        rank = {r["head"]: i for i, r in enumerate(order)}
        top_cat = sorted((h for h in cat if h in rank),
                         key=lambda h: -cat[h])[:8]
        if top_cat:
            print("  catalogue top-8 downstream of the relay, by their TV rank:")
            for h in top_cat:
                d = dict((r["head"], r) for r in down)[h]
                print(f"     {h:>8} catalogue {cat[h]:+.3f}   ratio "
                      f"{d['ratio']:.2f}   rank {rank[h]} of {len(order)}")
        print()

        res["per_step"][str(s)] = {
            "zero_check_max_tv": float(zc), "n_downstream": len(down),
            "median_ratio": med, "max_ratio": float(v.max()),
            "controls": [f"L{a}H{b}" for a, b in ctrl],
            "n_heads_half_above_median_mass": n_half,
            "ranked": order, "catalogue_ranks": {
                h: {"catalogue_dnll": cat[h], "tv_rank": rank[h]}
                for h in cat if h in rank}}

        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(res, fh, indent=2)
        del model

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
