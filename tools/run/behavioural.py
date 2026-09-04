"""P-I1's behavioural arm over the registered 19-step grid.

The sibling of `curve.py`. `curve.py` turns the 19 interaction tables into the
relay (A) series; this turns the 19 checkpoints' Phase 1 attention tensors into
the behavioural (B) series P-I1 pairs it against. Until now every B-side number
in the repository was synthetic -- `claims/audits/p_i1_attainable_floor.json`
carries `b_side_is_synthetic: True`, and PROJECT.md §3.5 names running this arm
over the sweep as the one piece of P-I1 that was runnable without an author
decision. It is now run.

WHAT THE BEHAVIOURAL SCORE IS
----------------------------
`p7_motifs/formation_curve.behavioural_induction_score`: mean post-softmax
attention on a prompt's induction pairs, per (layer, head). The pairs are
`core.battery_structure.induction_candidates` on the prompt's token ids -- the
same call `run_7.py` makes to type the interaction table's induction rows, so
the two arms see the same pair set. The score is read from `attentions.npz`
rather than from the table's `weight` column: the table is thinned by a
top-k-by-force cutoff and averaging over it would select on force magnitude, the
quantity the A side is built from, and the pairing null could not then separate
the arms (`formation_curve.py`'s module docstring).

THE CROSS-PROMPT CONVENTION -- registered by the author 2026-09-03
----------------------------------------------------------------
"Mirror the relay side." `curve.py` pools relay counts across the seven
non-`repeated_tokens` prompts into one per-head number per step, and carries a
`repeated_tokens`-included series beside it, reported and never scored
(PROJECT.md §5.1 decision 4). This does the same on the B side: all induction
pairs from the seven prompts are pooled and the per-head score is the mean
post-softmax attention over that pooled set -- pair-count weighted, exactly as a
pooled count is. `series_incl_repeated` carries the eight-prompt version.

Because `induction_candidates` is token-identity based, a prompt's pair set does
NOT depend on the checkpoint: `n_pairs_excl` and `n_pairs_incl` are asserted
constant across the 19 steps, and a step that breaks that assertion has a
tokenisation that drifted from `tokens.txt`.

WHAT THIS DOES NOT DO
--------------------
It does not produce P-I1's p-value. That is blocked on two author decisions
PROJECT.md §3.1 and §3.4 record -- the scored head axis (all 384 heads vs the
forming subset) and the relay-count null's shape -- and neither is startable
from the code. This writes the real B series into `data/analysis/
behavioural_series.json`, beside `formation_series.json` and NOT into
`curve.json` (§7.1: `curve.json` is the file that gets diffed, so a new key does
not go in it).

PATHS ARE DERIVED, as in `curve.py`: `METS_REPO` and `METS_DATA` override, and
nothing is inferred from a Phase 1 directory's name -- the (step, prompt) ->
directory map is built by globbing for the one Phase 1 run that wrote
`pythia-410m-step<s>_<prompt>/attentions.npz`, with the `p2_eigenspectra_` runs
excluded, and a step/prompt that does not resolve to exactly one is fatal.
"""
import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", "/run/media/system/WDS_500/Mets"))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))

import numpy as np

from core.battery_structure import induction_candidates, tokenize_prompt
from core.changepoint_colocation import REGISTERED_P_I1_SWEEP
from core.config import PROMPTS

STEPS = list(REGISTERED_P_I1_SWEEP)

#: The one prompt carried beside the scored series and never scored, on the
#: B side for the same reason as the A side. PROJECT.md §5.1 decision 4.
DOMINANT_PROMPT = "repeated_tokens"

#: pythia-410m's tokenizer does not change across training revisions, so one
#: tokenizer covers all 19 checkpoints. Re-tokenising and truncating to the
#: attention width reproduces every prompt's `tokens.txt` exactly (verified
#: per step per prompt below), which is what keeps the pair indices aligned
#: with the attention matrix they index.
TOKENIZER_ID = "EleutherAI/pythia-410m"

N_LAYERS, N_HEADS = 24, 16
HEAD_KEYS = [(l, h) for l in range(N_LAYERS) for h in range(N_HEADS)]

OUT = Path(os.environ.get("METS_SCRATCH", str(DATA / "analysis"))) / "behavioural_series.json"


class BehaviouralArmRefused(RuntimeError):
    """An input this arm will not score."""


# --------------------------------------------------------------------------
# Phase 1 run discovery
# --------------------------------------------------------------------------

def run_dir_for(step: int, prompt: str) -> Path:
    """The single Phase 1 run that wrote this (step, prompt)'s attention tensor.

    Nothing is read from a directory name; the match is on the file the run
    wrote. `p2_eigenspectra_*` are Phase 2 runs under the same root and are
    excluded.
    """
    hits = [
        p.parent
        for p in DATA.glob(f"phase12/*/pythia-410m-step{step}_{prompt}/attentions.npz")
        if not p.parent.parent.name.startswith("p2_eigenspectra_")
    ]
    if len(hits) != 1:
        raise BehaviouralArmRefused(
            f"step {step} prompt {prompt!r}: {len(hits)} Phase 1 runs with an "
            f"attentions.npz, need exactly 1 -- {[str(h) for h in hits]}"
        )
    return hits[0]


def battery_prompts() -> list:
    """The battery keys with a Phase 1 attention tensor at every one of the 19
    steps, sorted, `repeated_tokens` guaranteed present."""
    present = None
    for s in STEPS:
        here = {
            p.parent.name.split("_", 1)[1]
            for p in DATA.glob(f"phase12/*/pythia-410m-step{s}_*/attentions.npz")
            if not p.parent.parent.name.startswith("p2_eigenspectra_")
        }
        present = here if present is None else (present & here)
    prompts = sorted(present)
    unknown = [p for p in prompts if p not in PROMPTS]
    if unknown:
        raise BehaviouralArmRefused(f"not battery keys: {unknown}")
    if DOMINANT_PROMPT not in prompts:
        raise BehaviouralArmRefused(
            f"{DOMINANT_PROMPT!r} has no attention tensor across all 19 steps; "
            "the carried-beside series cannot be built"
        )
    return prompts


# --------------------------------------------------------------------------
# Tokenisation, verified against the stored token list
# --------------------------------------------------------------------------

def _tokens_txt(run_dir: Path) -> list:
    """`tokens.txt` parsed back to token strings.

    The writer is `p1_io.py`'s `f.write(f"{i:3d}  {tok}\\n")` -- a 3-wide
    right-justified index, two spaces, then the token verbatim. Indices are
    <= 511 so the field never overflows and `line[5:]` is the token.
    """
    out = []
    for line in (run_dir / "tokens.txt").read_text().splitlines():
        out.append(line[5:])
    return out


def prompt_ids(tokenizer, prompt: str, n_tokens: int, run_dir: Path) -> list:
    """Token ids for `prompt`, truncated to the attention width and verified
    to reproduce `tokens.txt` token for token."""
    ids = tokenize_prompt(tokenizer, PROMPTS[prompt])["ids"]
    if len(ids) < n_tokens:
        raise BehaviouralArmRefused(
            f"{prompt!r}: tokeniser produced {len(ids)} ids, attention is "
            f"{n_tokens} wide -- the prompt text and the Phase 1 run disagree"
        )
    ids = ids[:n_tokens]
    got = list(tokenizer.convert_ids_to_tokens(ids))
    want = _tokens_txt(run_dir)
    if got != want:
        first = next(i for i in range(min(len(got), len(want)))
                     if got[i] != want[i])
        raise BehaviouralArmRefused(
            f"{prompt!r}: re-tokenisation does not match tokens.txt at index "
            f"{first}: {got[first]!r} vs {want[first]!r}. The induction pairs "
            "would index a different tokenisation than the attention tensor."
        )
    return ids


# --------------------------------------------------------------------------
# The arm
# --------------------------------------------------------------------------

def prompt_pair_sums(attentions: np.ndarray, pairs) -> tuple:
    """(sum of post-softmax attention over `pairs`, per (layer, head)); n_pairs.

    Kept as a sum rather than a mean so the cross-prompt pool is pair-count
    weighted when the per-prompt sums are added and divided by the total
    count -- which is what "pool all the pairs" means and what matches a
    pooled relay count.
    """
    a = np.asarray(attentions)
    if a.ndim != 4 or a.shape[:2] != (N_LAYERS, N_HEADS):
        raise BehaviouralArmRefused(
            f"attentions must be ({N_LAYERS}, {N_HEADS}, n, n); got {a.shape}")
    pairs = list(pairs)
    if not pairs:
        return np.zeros((N_LAYERS, N_HEADS), dtype=np.float64), 0
    q = np.asarray([p[0] for p in pairs], dtype=int)
    k = np.asarray([p[1] for p in pairs], dtype=int)
    n = a.shape[-1]
    if q.max() >= n or k.max() >= n:
        raise BehaviouralArmRefused(
            f"induction pair index {max(int(q.max()), int(k.max()))} outside an "
            f"attention matrix of {n} tokens")
    vals = a[:, :, q, k].astype(np.float64)          # (n_layers, n_heads, n_pairs)
    return vals.sum(axis=-1), len(pairs)


def build(verbose: bool = True) -> dict:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:                          # pragma: no cover
        raise BehaviouralArmRefused(f"transformers unavailable: {exc}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    prompts = battery_prompts()
    scored_prompts = [p for p in prompts if p != DOMINANT_PROMPT]

    # induction_candidates is token-identity based, so a prompt's pair count is
    # a property of the prompt and not the checkpoint. Computed once, then
    # asserted equal at every step as a tokenisation-drift tripwire.
    ref_pairs = {}
    per_step = {}
    series_excl = {k: [0.0] * len(STEPS) for k in HEAD_KEYS}
    series_incl = {k: [0.0] * len(STEPS) for k in HEAD_KEYS}
    n_pairs_excl = n_pairs_incl = None

    for i, s in enumerate(STEPS):
        S_excl = np.zeros((N_LAYERS, N_HEADS), dtype=np.float64)
        S_incl = np.zeros((N_LAYERS, N_HEADS), dtype=np.float64)
        c_excl = c_incl = 0
        rec = {"run_dirs": {}, "per_prompt_n_pairs": {}}

        for p in prompts:
            rd = run_dir_for(s, p)
            rec["run_dirs"][p] = str(rd.relative_to(REPO)) if rd.is_relative_to(REPO) else str(rd)
            with np.load(rd / "attentions.npz", allow_pickle=False) as z:
                attn = z["attentions"]
                n = attn.shape[-1]
                ids = prompt_ids(tokenizer, p, n, rd)
                pairs = induction_candidates(ids)
                psum, npair = prompt_pair_sums(attn, pairs)

            if p in ref_pairs and ref_pairs[p] != npair:
                raise BehaviouralArmRefused(
                    f"{p!r}: {npair} induction pairs at step {s}, "
                    f"{ref_pairs[p]} at an earlier step -- the tokenisation "
                    "is not stable across checkpoints"
                )
            ref_pairs.setdefault(p, npair)
            rec["per_prompt_n_pairs"][p] = npair

            S_incl += psum
            c_incl += npair
            if p in scored_prompts:
                S_excl += psum
                c_excl += npair

        if c_excl == 0:
            raise BehaviouralArmRefused(
                f"step {s}: no induction pairs across {scored_prompts} -- "
                "the scored B series would be all zeros")

        if n_pairs_excl is None:
            n_pairs_excl, n_pairs_incl = c_excl, c_incl
        elif (c_excl, c_incl) != (n_pairs_excl, n_pairs_incl):
            raise BehaviouralArmRefused(
                f"step {s}: pooled pair counts ({c_excl}, {c_incl}) differ from "
                f"({n_pairs_excl}, {n_pairs_incl}) -- tokenisation drift")

        score_excl = S_excl / c_excl
        score_incl = S_incl / c_incl
        for (l, h) in HEAD_KEYS:
            series_excl[(l, h)][i] = float(score_excl[l, h])
            series_incl[(l, h)][i] = float(score_incl[l, h])

        rec["n_pairs_excl"] = c_excl
        rec["n_pairs_incl"] = c_incl
        per_step[str(s)] = rec

        if verbose:
            fl = int(np.argmax(score_excl))
            l, h = divmod(fl, N_HEADS)
            print(
                f"step{s:<7d} pairs(excl)={c_excl:>6d}  "
                f"mean={score_excl.mean():.5f}  max={score_excl.max():.5f} "
                f"@L{l}H{h}",
                flush=True,
            )

    payload = {
        "_what_this_is": "P-I1's behavioural (B) arm over REGISTERED_P_I1_SWEEP: "
                         "pooled mean post-softmax attention on induction pairs, "
                         "per (layer, head), per checkpoint.",
        "_cross_prompt_convention": "mirror the relay side (PROJECT.md §5.1 "
                                    "decision 4, registered 2026-09-03): pool "
                                    "induction pairs across the seven "
                                    "non-repeated_tokens prompts; carry the "
                                    "eight-prompt series beside it, never scored.",
        "_not_a_p_value": "P-I1's p-value is blocked on PROJECT.md §3.1 (the "
                          "scored head axis) and §3.4 (the relay-count null's "
                          "shape); this is the B input those decisions need.",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "tokenizer": TOKENIZER_ID,
        "steps": STEPS,
        "scored_prompts": scored_prompts,
        "carried_beside_prompt": DOMINANT_PROMPT,
        "n_pairs_excl_repeated": n_pairs_excl,
        "n_pairs_incl_repeated": n_pairs_incl,
        "n_heads": len(HEAD_KEYS),
        "per_step": per_step,
        "series_excl_repeated": {f"{l},{h}": series_excl[(l, h)] for (l, h) in HEAD_KEYS},
        "series_incl_repeated": {f"{l},{h}": series_incl[(l, h)] for (l, h) in HEAD_KEYS},
    }
    return payload


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------

def check(payload: dict) -> list:
    """Structural checks on a written payload. Returns a list of problems."""
    problems = []
    if payload.get("steps") != STEPS:
        problems.append(f"steps != REGISTERED_P_I1_SWEEP")
    for field in ("series_excl_repeated", "series_incl_repeated"):
        ser = payload.get(field, {})
        if len(ser) != len(HEAD_KEYS):
            problems.append(f"{field}: {len(ser)} heads, expected {len(HEAD_KEYS)}")
        for (l, h) in HEAD_KEYS:
            v = ser.get(f"{l},{h}")
            if v is None:
                problems.append(f"{field}: L{l}H{h} missing")
                continue
            if len(v) != len(STEPS):
                problems.append(f"{field}: L{l}H{h} has {len(v)} points")
            if any((x != x) or x < 0 for x in v):    # NaN or negative attention
                problems.append(f"{field}: L{l}H{h} has a NaN or negative value")
    ps = payload.get("per_step", {})
    counts = {(r["n_pairs_excl"], r["n_pairs_incl"]) for r in ps.values()}
    if len(counts) > 1:
        problems.append(f"per-step pooled pair counts are not constant: {counts}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--write", action="store_true", help="build and write the series")
    ap.add_argument("--check", action="store_true",
                    help="structural checks on the written series")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    if args.write:
        payload = build(verbose=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"\nWROTE {args.out}")
        problems = check(payload)
        if problems:
            print("CHECK FAILED:")
            for p in problems:
                print(f"  - {p}")
            return 1
        print(f"check OK: {len(HEAD_KEYS)} heads x {len(STEPS)} steps, "
              f"{payload['n_pairs_excl_repeated']} pooled pairs "
              f"({payload['n_pairs_incl_repeated']} with {DOMINANT_PROMPT})")
        return 0

    if args.check:
        if not args.out.exists():
            print(f"{args.out} is missing; run --write")
            return 1
        payload = json.loads(args.out.read_text())
        problems = check(payload)
        if problems:
            print("CHECK FAILED:")
            for p in problems:
                print(f"  - {p}")
            return 1
        print("check OK")
        return 0

    ap.error("nothing to do: pass --write or --check")


if __name__ == "__main__":
    raise SystemExit(main())
