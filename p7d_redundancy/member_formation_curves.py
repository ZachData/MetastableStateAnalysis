"""Formation curves for the redundancy set, and for the redundancy itself.

WHY THIS EXISTS
---------------
`PROJECT.md` §3.14.2 asks when the members of the redundancy set formed (Q2) and
whether they formed at the earliest point the network could (Q3). The instrument
is the causal one -- OV-ablation `dNLL` per checkpoint -- because §3.12-R and
§3.12-G6 between them ruled out every weights-only predictor of causal effect,
so a structural proxy would inherit exactly that failure.

**`L7H8` is the only member with such a curve** (§3.11-A: ~0 through step 2000,
then +0.24 -> +0.73 -> +1.02 -> +1.18 at 143000), and it was read off rank 0 of
`induction_rank_sweep` at 8 sequences, one checkpoint per invocation. `L5H2` has
none despite **twice** the effect at step 16000, and `L12H5` and `L8H6` were not
known to be members until the catalogue sweep. This runs the whole set on one
footing, at the 16 sequences §3.12-S used, so the curves are comparable with
each other and with the pair's interaction rather than only with themselves.

THE ARM NO SINGLE-HEAD CURVE CAN PROVIDE. `dNLL(L5H2 + L7H8 ablated jointly)`
minus the two singles is the interaction, +4.151 at step 16000 (§3.12-S: joint
2.2x the parts-sum, redundant rather than serial). Per checkpoint it dates the
**redundancy** rather than the heads -- two heads can each be present long
before they become substitutes for one another, and only this arm can tell those
apart.

AND THE ALIGNMENT, WHICH IS NEARLY FREE. §3.12-S found the pair's residual
effects 87 %-aligned at step 16000 out of chance-level weight overlap. The
cosine between the two ablation deltas costs one mean vector per arm, and dates
that alignment on the same grid.

Reported BOTH ways per §3.13: raw `dNLL`, and `dNLL` relative to the checkpoint's
own baseline NLL, which falls by an order of magnitude across the grid. Neither
view is chosen after seeing the data.

NO p-value; nothing registered. Everything here is 410m and therefore spent
under `check_registry` rule 3 -- exploratory, and not registrable on this data.
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
import torch

from core.lm_loading import load_causal_lm
from tools.run.induction_rank_sweep import (
    D_MODEL, N_REP, VOCAB_LO, VOCAB_HI, EVAL_SEED, ov_factors, write_ov,
)

OUT = DATA / "analysis" / "member_formation_curves.json"
CATALOG = DATA / "analysis" / "redundancy_catalog.json"

#: The registered 19-step grid (`core.changepoint_colocation`), imported rather
#: than restated so this cannot drift from the grid the gate reads.
from core.changepoint_colocation import REGISTERED_P_I1_SWEEP
ALL_STEPS = list(REGISTERED_P_I1_SWEEP)

#: The pair whose interaction §3.12-S measured, and `--pair`'s default. Its
#: two singles are always run, so `--heads` cannot leave the joint arm without
#: the parts it must be differenced against.
PAIR = "L5H2,L7H8"

#: Fallback membership if the git-ignored catalogue is not on disk: §3.14.2's
#: Q1 answer, every head above +0.1 at step 16000, 8 sequences.
FALLBACK = [(5, 2), (7, 8), (12, 5), (8, 6), (11, 14), (8, 9)]


def members(n):
    """Top-`n` catalogue members, from the sweep if it is on disk."""
    if CATALOG.exists():
        rows = json.load(open(CATALOG))["heads"]
        rows.sort(key=lambda r: -r["dnll"])
        return [(r["layer"], r["head_idx"]) for r in rows[:n]], str(CATALOG)
    return FALLBACK[:n], "fallback list (catalogue not on disk)"


def batch(rng, n):
    return torch.tensor(np.stack([
        np.concatenate([s, s]) for s in
        (rng.integers(VOCAB_LO, VOCAB_HI, size=N_REP) for _ in range(n))
    ]), dtype=torch.long)


@torch.no_grad()
def probe(model, ids, chunk=4, want_vec=True):
    """(second-copy NLL, mean final-layer residual vector over copied positions).

    Same estimator as `two_big_heads.probe` and as rank 0 of
    `induction_rank_sweep`: the loss is scored only from position `N_REP - 1`,
    where the second copy of the sequence begins.

    `want_vec` is off for the arms whose delta nobody reads. The hidden states
    are 25 x (chunk, 192, 1024) and the logits (chunk, 192, 50304); the first
    version asked for both on all eight arms of all nineteen steps and the run
    was killed for memory at step 4000.
    """
    lc, lz, vecs = [], [], []
    for i in range(0, len(ids), chunk):
        out = model(ids[i:i + chunk], output_hidden_states=want_vec)
        lg = out.logits[:, :-1, :].float()
        t = ids[i:i + chunk, 1:]
        z = torch.logsumexp(lg, dim=-1)
        c = lg.gather(-1, t.unsqueeze(-1)).squeeze(-1)
        lc.append(c[:, N_REP - 1:].numpy().astype(np.float64))
        lz.append(z[:, N_REP - 1:].numpy().astype(np.float64))
        if want_vec:
            h = out.hidden_states[-1][:, N_REP - 1:, :].float()
            vecs.append(h.reshape(-1, D_MODEL).mean(0).numpy().astype(np.float64))
            del h
        del out, lg, z, c
    return (float(-(np.concatenate(lc).mean() - np.concatenate(lz).mean())),
            np.mean(vecs, axis=0) if want_vec else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default=",".join(str(s) for s in ALL_STEPS))
    ap.add_argument("--seqs", type=int, default=16,
                    help="16 matches §3.12-S; the catalogue screen used 8 and "
                         "reads ~10 %% low throughout")
    ap.add_argument("--top", type=int, default=6,
                    help="how many catalogue members to curve")
    ap.add_argument("--heads", default="",
                    help="explicit 'L5H2,L7H8' override of --top")
    ap.add_argument("--pair", default=PAIR,
                    help="the two heads the joint arm ablates together; the "
                         "default is §3.12-S's pair, so the interaction stays "
                         "comparable with it")
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--append", action="store_true",
                    help="merge into an existing output rather than replacing it")
    ap.add_argument("--out", default=str(OUT),
                    help="a run over a different --pair or --heads belongs in "
                         "its own file; the default is the main six-member "
                         "curve and a partial run WILL replace it")
    args = ap.parse_args()
    out = Path(args.out)

    def parse(spec):
        return [(int(h[1:h.index("H")]), int(h[h.index("H") + 1:]))
                for h in spec.split(",") if h]

    if args.heads:
        heads, source = parse(args.heads), "--heads"
    else:
        heads, source = members(args.top)
    pair = parse(args.pair)
    if len(pair) != 2:
        raise SystemExit(f"--pair needs exactly two heads, got {args.pair!r}")
    for k in pair:                       # the joint arm needs both singles
        if k not in heads:
            heads.append(k)
    steps = [int(x) for x in args.steps.split(",") if x]
    names = {k: f"L{k[0]}H{k[1]}" for k in heads}

    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    res = {"_what_this_is": __doc__, "git_sha": git_sha,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
           "membership_source": source,
           "eval": {"n_seqs": args.seqs, "n_rep": N_REP, "seed": EVAL_SEED,
                    "vocab_range": [VOCAB_LO, VOCAB_HI],
                    "scored_from_position": N_REP - 1},
           "heads": [names[k] for k in heads], "steps": steps,
           "pair": [names[k] for k in pair], "per_step": {}}
    if args.append and out.exists():
        prev = json.load(open(out))
        res["per_step"] = prev["per_step"]
        res["steps"] = sorted(set(prev["steps"]) | set(steps))

    print(f"members: {', '.join(names[k] for k in heads)}   ({source})")
    print(f"grid:    {len(steps)} steps, {args.seqs} sequences, "
          f"{len(heads) + 2} forward arms per step\n")
    print(f"{'step':>7} {'baseline':>9} " +
          " ".join(f"{names[k]:>8}" for k in heads) +
          f" {'joint':>8} {'INTER':>8} {'cos':>7} {'restore':>8}")

    Z = (np.zeros((D_MODEL, 1)), np.zeros((1, D_MODEL)))
    for s in steps:
        model, _ = load_causal_lm(f"pythia-410m-step{s}")
        model.eval()
        ids = batch(np.random.default_rng(EVAL_SEED), args.seqs)

        def run(ablate, want_vec):
            saved = {k: ov_factors(model, *k) for k in ablate}
            for k in ablate:
                write_ov(model, *k, *Z)
            r = probe(model, ids, args.chunk, want_vec)
            for k, (a, b) in saved.items():
                write_ov(model, *k, a, b)
            return r

        nll0, vec0 = probe(model, ids, args.chunk, True)
        single = {}
        for k in heads:
            nll, vec = run([k], True)
            single[k] = (nll - nll0, vec - vec0)
        nll_j, _ = run(pair, False)
        nll_chk, _ = probe(model, ids, args.chunk, False)

        dA, dB = single[pair[0]][0], single[pair[1]][0]
        dJ = nll_j - nll0
        inter = dJ - dA - dB
        da, db = single[pair[0]][1], single[pair[1]][1]
        cos = float(np.dot(da, db) / (np.linalg.norm(da) * np.linalg.norm(db)))

        rec = {"baseline_nll": nll0, "joint_pair": dJ, "interaction": inter,
               "pair_delta_cosine": cos,
               "restore_abs_diff": abs(nll_chk - nll0),
               "dnll": {names[k]: single[k][0] for k in heads},
               # §3.13: the same numbers against the checkpoint's own baseline,
               # which falls ~20x across the grid. Both views, neither chosen
               # after the fact.
               "dnll_rel": {names[k]: single[k][0] / nll0 for k in heads},
               "joint_pair_rel": dJ / nll0, "interaction_rel": inter / nll0,
               "delta_norm": {names[k]: float(np.linalg.norm(single[k][1]))
                              for k in heads}}
        res["per_step"][str(s)] = rec

        print(f"{s:>7} {nll0:>9.4f} " +
              " ".join(f"{single[k][0]:>+8.4f}" for k in heads) +
              f" {dJ:>+8.4f} {inter:>+8.4f} {cos:>+7.3f} "
              f"{abs(nll_chk - nll0):>8.1e}", flush=True)
        del model, ids, single, vec0
        gc.collect()
        # Written every step, not at the end: the first run of this was killed
        # for memory at step 4000 and the thirteen steps it had finished were
        # only in the terminal.
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(res, fh, indent=2)

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
