"""Stage 1, §1.9's parked check: is the class effect of clustered unique tokens lexical carry or contextual?

`status-10.md` §1.9 found that trained, clustered unique tokens cluster with
their own class (``same_class`` Δ +0.20 over step 0) and, a little, with tokens
their own layer 0 calls similar (``emb_pct_own`` Δ +0.12). Same-class tokens are
embedding-similar in the trained model, so the class effect may be the residual
stream carrying each token's layer-0 vector (lexical), not context. Parked in
`handoff-10.md` as two cheap checks on the same inputs; this reader runs both.

UNITS AND INPUTS
----------------
As `p10_comembership`: focal token = a clustered unique token at position > 0,
at one layer of one run; its co-members; the draw from all other positions.
Same index, holdout screen, native labels and refusals as
`p10_comembership.measure_run`. The own
layer 0 is the run's own ``activations[0]`` (L2-normed, as stored).

CHECK A — CARRY: does the layer-L state still point at its own layer-0 vector?
-----------------------------------------------------------------------------
For a position i at layer L: ``self_cos`` = cos(x_L[i], x_0[i]); ``self_pct`` =
mid-rank percentile of that within cos(x_L[i], x_0[j]) over all j (i
included). ``self_pct`` cancels the common direction that inflates raw cosines at
depth; a random vector scores 0.5, and 1.0 means x_L[i] is nearer its own
layer-0 vector than any other token's. Recorded for focal tokens and for
**unclustered** unique tokens (position > 0, label -1), and ``self_top1`` (share
with self first).

CHECK B — SPLIT: class and embedding similarity, each given the other
---------------------------------------------------------------------
Per focal token, with p_j = mid-rank percentile of cos_own(f, j) over the pool
(as ``emb_pct_own``) and s_j = [class(j) == class(f)]:

- ``emb_given_class``: observed mean p over co-members minus its expectation
  if each co-member were redrawn from the pool members with its own s (the
  class composition held). Embedding similarity beyond class.
- ``class_given_emb``: observed same-class share minus its expectation if each
  co-member were redrawn from the pool members in its own p-decile
  (``N_BINS`` = 10; the embedding similarity held). Class beyond embedding
  similarity, to within a decile.
- ``emb_same`` / ``emb_cross``: mean p over same-class / cross-class
  co-members minus the mean p over same-class / cross-class pool members (the
  split the handoff names). Absent for a focal token with no such co-member.

Each is a lift (0 under a random draw); per run-layer the mean over focal
tokens, then prompt-balanced as `p10_comembership.summarise`, then
Δ = lift(step) - lift(step 0).

WHAT IS READ, FIXED HERE BEFORE ANY TABLE IS SEEN (2026-09-24)
--------------------------------------------------------------
At step 143000, layers 12, 24 and the layer mean, Δ against step 0, floor
±0.05 (placed, as §1.9's; no null, no e-value):

1. ``class_given_emb`` Δ > +0.05: the class effect is **not** the own
   embedding's similarity; it survives holding it. ``class_given_emb`` Δ within
   ±0.05 while §1.9's ``same_class`` Δ > +0.05: the class effect **is**
   embedding similarity (lexical), to within a decile. Between (Δ positive but
   well under ``same_class``'s): part of each; report the fraction
   Δ(class_given_emb) / Δ(same_class).
2. ``emb_given_class`` Δ > +0.05: within class, clusters still prefer the
   token's embedding neighbours (lexical inside class). Within ±0.05: §1.9's
   ``emb_pct_own`` Δ is class composition.
3. Carry: focal ``self_pct`` minus unclustered ``self_pct`` at the same
   step-layer. > +0.05 at trained depth: clustered unique tokens keep more of
   their own layer-0 vector than unclustered ones, which the lexical reading
   predicts; ≤ +0.05: clustering does not go with keeping it. Its Δ against
   step 0 is also reported. The level of ``self_pct`` alone is not read as
   lexical: a residual stream carries its input at every layer by construction.

At step 0 the embedding is random, so class and own-embedding similarity are
unrelated there: step 0 is the baseline for Δ, as in §1.9, not a finding.

LIMITS. ``class_given_emb`` holds similarity to within a decile; class
information finer than that is counted as "beyond embedding". "Contextual"
here means only "not explained by the token's own layer-0 vector"; positional
and attention-mediated sources are not separated. All-positions draw only.

CHANGED AFTER THE FIRST RUN (post hoc, 2026-09-24; the reading above is kept)
------------------------------------------------------------------------------
Reading 1 had no positive control. At trained **layer 0** the partition is
made from the own embedding, so its class effect is lexical by construction,
yet ``class_given_emb`` there was +0.18 against +0.15 at L12 / L24: decile
matching leaves class information that is purely embedding. So a positive
``class_given_emb`` does not by itself say "beyond embedding". Added: the same
lift at 20 and 40 bins (``class_given_emb_20`` / ``_40``; each co-member's own
s is in its bin's share, which shrinks the lift by ~1/bin size, 10 at 40
bins, and keeps a random draw's expectation at exactly 0), and the comparison
depth vs trained L0 at the same bin count, L0 being the lexical reference.

After `/challenge-pr` on #93: ``*_knn`` scores the same CGE on a purely lexical
cluster of the same size at the same layer (the focal token's k nearest in its
own layer 0), a matched lexical reference in place of L0; ``*_classonly`` scores
it on a class-only cluster (same-class tokens, the embedding ignored, in
expectation): a reference for what pure class grouping scores, not a maximum.
(Named ``*_ceil`` in schema 3, where it was None for a class with < k members,
which averaged it over fewer tokens than the observed value; #94's review.) The carry groups are averaged over runs that have both
groups (``n_runs_carry``); the first version mixed 7 and 8 prompts.

TIER 1, EXPLORATORY, NOT REGISTERED. Nothing here touches `claims/registry.json`.

Run:
    python tools/run/p10_lexical_carry.py --v1-only
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[2])))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))

import numpy as np

from core.holdout import add_holdout_args, refuse_held_out
from tools.run.backfill_hdbscan import labels_provenance, read_labels
from tools.run.p10_comembership import DELTA_FLOOR, PICK_LAYERS, _mean, midrank_pct
from tools.run.p10_ext_sem_threshold import load_index, read_tokens
from tools.run.p10_token_composition import (
    CompositionError, find_tokenizer, load_vocab, token_features)

N_BINS = 10
FINER_BINS = (20, 40)   # post hoc sensitivity (docstring, "CHANGED AFTER THE FIRST RUN")
CGE = ("class_given_emb", *(f"class_given_emb_{b}" for b in FINER_BINS))
# after /challenge-pr on #93: CGE on a purely lexical cluster (the focal token's k nearest in
# its own layer 0) and on a class-only one (same-class tokens, the embedding ignored): the matched
# lexical reference and a class-only reference (an expectation, not a maximum)
CONTROLS = tuple(f"{c}_{kind}" for kind in ("knn", "classonly") for c in CGE)
SPLIT = ("emb_given_class", "class_given_emb", "emb_same", "emb_cross",
         *CGE[1:], *CONTROLS)
CARRY = ("self_cos", "self_pct", "self_top1")
GROUPS = ("focal", "unclustered")


def split_stats(f: int, co: np.ndarray, pool: np.ndarray, own_row: np.ndarray,
                cls: np.ndarray) -> dict:
    """Check B's four lifts for one focal token f (None where undefined)."""
    if f in pool or not np.isin(co, pool).all():
        raise CompositionError(f"position {f}: pool must hold every co-member and not f")
    p_pool = midrank_pct(own_row[pool], own_row[pool])
    p_of = dict(zip(pool.tolist(), p_pool))
    s_pool = cls[pool] == cls[f]
    p_co = np.array([p_of[c] for c in co])
    s_co = cls[co] == cls[f]
    mean_p = {True: p_pool[s_pool].mean() if s_pool.any() else None,
              False: p_pool[~s_pool].mean() if (~s_pool).any() else None}

    def class_given(nb):
        bins = np.minimum((p_pool * nb).astype(int), nb - 1)
        bin_share = {b: s_pool[bins == b].mean() for b in np.unique(bins)}
        co_bins = np.minimum((p_co * nb).astype(int), nb - 1)
        return float(s_co.mean() - np.mean([bin_share[b] for b in co_bins]))

    out = {
        "emb_given_class": float(p_co.mean() - np.mean([mean_p[bool(s)] for s in s_co])),
        **{c: class_given(nb) for c, nb in zip(CGE, (N_BINS, *FINER_BINS))},
        "emb_same": float(p_co[s_co].mean() - mean_p[True]) if s_co.any() else None,
        "emb_cross": float(p_co[~s_co].mean() - mean_p[False]) if (~s_co).any() else None,
    }
    return out


def control_stats(f: int, k: int, pool: np.ndarray, own_row: np.ndarray, cls: np.ndarray) -> dict:
    """CGE for a purely lexical k-cluster (f's k nearest in its own layer 0) and for a
    class-only one (the expectation over same-class pool members; it does not depend on k)."""
    near = pool[np.argsort(-own_row[pool], kind="stable")[:k]]
    knn = split_stats(f, near, pool, own_row, cls)
    out = {f"{c}_knn": knn[c] for c in CGE}
    # the lift is linear in the co-members, so the mean over all k-subsets of them is
    # the mean over single same-class members: 1 - the same-class share of each one's bin
    p_pool = midrank_pct(own_row[pool], own_row[pool])
    s_pool = cls[pool] == cls[f]
    ref = {}
    for c, nb in zip(CGE, (N_BINS, *FINER_BINS)):
        bins = np.minimum((p_pool * nb).astype(int), nb - 1)
        share = np.bincount(bins, weights=s_pool, minlength=nb) / np.maximum(np.bincount(bins, minlength=nb), 1)
        ref[f"{c}_classonly"] = None if not s_pool.any() else float(1.0 - share[bins[s_pool]].mean())
    return out | ref


def carry_stats(acts: np.ndarray, layer: int, positions: list) -> dict:
    """Check A for `positions` at `layer`: mean self_cos, self_pct, self_top1 (None if empty)."""
    if not positions:
        return {c: None for c in CARRY} | {"n": 0}
    x0 = acts[0]
    xl = acts[layer][positions]
    cos = xl @ x0.T                                   # (m, n); rows are L2-normed
    own = cos[np.arange(len(positions)), positions]
    pct = [midrank_pct(cos[r], own[r:r + 1])[0] for r in range(len(positions))]
    top1 = (cos.argmax(axis=1) == np.array(positions))
    return {"self_cos": float(own.mean()), "self_pct": float(np.mean(pct)),
            "self_top1": float(top1.mean()), "n": len(positions)}


def measure_run(run_dir: Path, vocab: dict, added: set, acts: np.ndarray = None) -> dict:
    """{layer: {'focal': carry, 'unclustered': carry, split prop: lift, 'n_focal'}}."""
    prov = labels_provenance(run_dir)
    if prov != "native":
        raise CompositionError(f"{run_dir.name}: labels are {prov}, not native")
    labels = read_labels(run_dir)
    tokens = read_tokens(run_dir)
    if not labels:
        raise CompositionError(f"{run_dir.name}: no labels")
    if all((lab == -1).all() for lab in labels.values()):
        raise CompositionError(f"{run_dir.name}: noise at every layer (an empty partition, not a result)")
    if acts is None:
        acts = np.load(run_dir / "activations.npz")["activations"].astype(np.float32)
    if acts.shape[1] != len(tokens):
        raise CompositionError(f"{run_dir.name}: activations hold {acts.shape[1]} tokens, {len(tokens)} in tokens.txt")
    own_gram = acts[0] @ acts[0].T
    feats = token_features(tokens, vocab, added)
    is_copy = np.array([f["copies"] != "unique" for f in feats])
    cls = np.array([f["cls"] for f in feats], dtype=object)
    unique_pos = [p for p in range(1, len(tokens)) if not is_copy[p]]
    everyone = np.arange(len(tokens))
    out = {}
    for layer, lab in sorted(labels.items()):
        if len(lab) != len(tokens):
            raise CompositionError(f"{run_dir.name} layer {layer}: {len(lab)} labels, {len(tokens)} tokens")
        if layer >= acts.shape[0]:
            raise CompositionError(f"{run_dir.name}: layer {layer} has labels but no activations")
        focal = [f for f in unique_pos if lab[f] != -1]
        rec = {"n_focal": len(focal),
               "focal": carry_stats(acts, layer, focal),
               "unclustered": carry_stats(acts, layer, [f for f in unique_pos if lab[f] == -1])}
        rows = []
        for f in focal:
            members = np.flatnonzero(lab == lab[f])
            co, pool = members[members != f], np.delete(everyone, f)
            rows.append(split_stats(f, co, pool, own_gram[f], cls)
                        | control_stats(f, len(co), pool, own_gram[f], cls))
        for p in SPLIT:
            rec[p] = _mean(r[p] for r in rows) if rows else None
        out[layer] = rec
    return out


def summarise(results: dict) -> dict:
    """{step: {layer|'mean': {split prop: lift, 'focal'|'unclustered': {carry}, 'carry_gap': {..}}}}.

    Prompt-balanced: per-run values, then the mean over runs that have one.
    """
    out = {}
    for s in sorted({st for st, _ in results}):
        runs = [r for (st, _), r in results.items() if st == s]
        layers = sorted({L for r in runs for L in r})
        by = {}
        for L in layers:
            rs = [r[L] for r in runs if L in r]
            by[L] = {p: _mean(r[p] for r in rs) for p in SPLIT}
            # carry groups over the same runs: only those with both groups (#93 review:
            # repeated_tokens has unclustered unique tokens but never a focal one)
            both = [r for r in rs if all(r[g]["n"] for g in GROUPS)]
            by[L]["n_runs_carry"] = len(both)
            for g in GROUPS:
                by[L][g] = {c: _mean(r[g][c] for r in both) for c in CARRY}
            by[L]["carry_gap"] = {c: _mean(r["focal"][c] - r["unclustered"][c] for r in rs
                                           if r["focal"][c] is not None and r["unclustered"][c] is not None)
                                  for c in CARRY}
        by["mean"] = {p: _mean(by[L][p] for L in layers) for p in SPLIT}
        for g in (*GROUPS, "carry_gap"):
            by["mean"][g] = {c: _mean(by[L][g][c] for L in layers) for c in CARRY}
        out[s] = by
    return out


def deltas(summary: dict) -> dict:
    """lift(step) - lift(step 0) for the split props and the carry gap, with the pre-stated word."""
    if 0 not in summary:
        raise CompositionError("no step 0, the baseline every delta is taken against")
    base = summary[0]

    def word(d):
        return ("unavailable" if d is None else "above step 0" if d > DELTA_FLOOR
                else "below step 0" if d < -DELTA_FLOOR else "as step 0")

    out = {}
    for s, by in summary.items():
        out[s] = {}
        for L in [*PICK_LAYERS, "mean"]:
            if L not in by or L not in base:
                continue
            row = {}
            vals = {p: (by[L][p], base[L][p]) for p in SPLIT}
            vals |= {f"carry_gap.{c}": (by[L]["carry_gap"][c], base[L]["carry_gap"][c]) for c in CARRY}
            for name, (a, b) in vals.items():
                d = None if a is None or b is None else a - b
                row[name] = {"delta": d, "reading": word(d)}
            out[s][L] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--index", default=str(DATA / "phase12" / "stage0_logs" / "stage0_index.json"))
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", str(DATA / "hf")))
    ap.add_argument("--out", default=str(DATA / "analysis" / "p10_s1_lexical_carry.json"))
    add_holdout_args(ap)
    args = ap.parse_args()

    tok_path = find_tokenizer(Path(args.hf_home))
    vocab, added = load_vocab(tok_path)
    idx = load_index(Path(args.index))
    kept, holdout = refuse_held_out(
        sorted(idx["runs"].values()), allow=args.allow_holdout, drop=args.v1_only,
        context="p10_lexical_carry")
    kept = set(kept)
    runs = {k: v for k, v in idx["runs"].items() if v in kept}
    print(f"{len(runs)} runs, {len({k for _, k in runs})} prompts, steps {sorted({s for s, _ in runs})}")

    t0 = time.time()
    results = {}
    for n, (step, key) in enumerate(sorted(runs), 1):
        results[(step, key)] = measure_run(runs[(step, key)], vocab, added)
        if n == 1:   # refuse rather than degrade: the first output must be populated
            rec = results[(step, key)]
            if not any(r["n_focal"] and r["class_given_emb"] is not None for r in rec.values()):
                raise CompositionError(f"first run {step}|{key}: no layer has a focal token with a lift")
        if n % 40 == 0 or n == len(runs):
            print(f"  [{n}/{len(runs)}] step{step} {key}  ({time.time() - t0:.0f}s)", flush=True)

    inputs = sorted((f"{s}|{k}", str(p)) for (s, k), p in runs.items())
    summary = summarise(results)
    record = {
        "schema": "p10_s1_lexical_carry/4",
        "row": "Stage 1, §1.9 parked: class effect lexical carry or contextual",
        "tier": "1 (exploratory, unregistered, descriptive; no null)",
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "index": str(args.index),
        "index_pin": idx["pin"],
        "prompt_battery_hash": idx["prompt_battery_hash"],
        "inputs_sha256": hashlib.sha256(json.dumps(inputs).encode()).hexdigest()[:12],
        "tokenizer_sha256": hashlib.sha256(tok_path.read_bytes()).hexdigest()[:12],
        "n_bins": N_BINS,
        "finer_bins": list(FINER_BINS),
        "n_runs": len(runs),
        "holdout": holdout,
        "summary": summary,
        "reading": deltas(summary),
        "inputs": inputs,
        "runs": {f"{s}|{k}": r for (s, k), r in sorted(results.items())},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    print(f"\nwrote {out}  (inputs {record['inputs_sha256']})")
    for s, by in summary.items():
        for L in [*PICK_LAYERS, "mean"]:
            x = by[L]
            f = lambda v: "  n/a" if v is None else f"{v:+.2f}"
            g = lambda v: " n/a" if v is None else f"{v:.2f}"
            print(f"  step{s:>6} L{L!s:>4} " + " ".join(f"{p.replace('class_given_emb', 'cge')} {f(x[p])}" for p in SPLIT)
                  + "".join(f"  {c} foc/unc {g(x['focal'][c])}/{g(x['unclustered'][c])}" for c in CARRY))


if __name__ == "__main__":
    main()
