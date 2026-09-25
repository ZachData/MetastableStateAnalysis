"""Stage 1, after step 2 of `p10_cluster_function/handoff-10.md`: what clustered unique tokens cluster *with*.

Step 2 (`status-10.md` §1.7) found that clustered vs noise is mostly copy
count, and §1.8 found that at step 0 layer 0 the partition is what HDBSCAN
makes of noise with duplicates: a clustered unique token at init is a noise
point glued to a copy group. The residual to explain is the clustered
**unique** tokens, and the quantity is trained minus step 0, not the trained
rate. This reader asks what those tokens' co-members are.

THE UNIT
--------
A *focal* token: one occurrence of a token id that occurs once in the prompt
(``unique`` in §1.7), at position > 0 (position 0 is the attention sink), with
HDBSCAN label != -1, in one run at one layer. Its *co-members* are the other
members of its cluster (k of them). *Others* are every other position of the
prompt (n - 1, position 0 included).

FIVE PROPERTIES OF THE CO-MEMBERS, EACH AGAINST A RANDOM DRAW
-------------------------------------------------------------
For each focal token the observed value, and its expectation if the k
co-members were a uniform draw without replacement from the others (exact, no
RNG):

- ``copy_share``: share of co-members whose token id occurs >= 2 times in the
  prompt. Expectation: the same share over the others.
- ``no_copy``: 1 if no co-member is a copy (a cluster of unique tokens only).
  Expectation: the hypergeometric probability of drawing no copy.
- ``adjacent``: 1 if some co-member is at position +-1. Expectation: the
  hypergeometric probability of drawing at least one of the 1-2 neighbours.
- ``same_class``: share of co-members with the focal token's class
  (`p10_token_composition.token_class`). Expectation: the same share over the
  others.
- ``emb_pct``: mean percentile of the co-members' cosine to the focal token,
  within the focal token's cosines to all others, in the **frozen** frame
  (layer 0 of step 143000, the trained embedding, as in §1.6). Mid-rank
  percentile, so a random draw has expectation exactly 0.5.

``lift`` = mean observed - mean expected over the focal tokens of a run-layer.
Also recorded: k (cluster size - 1) and the focal count.

WHAT IS READ, FIXED HERE BEFORE ANY TABLE IS SEEN
-------------------------------------------------
Per run-layer, the lifts; per step and layer, their mean over prompts
(prompt-balanced: per-run means, then the mean over runs with >= 1 focal
token), with and without ``repeated_tokens`` (1 repeated type; §1.8 found a
different mechanism there). The reading is **delta = lift(step) -
lift(step 0)** per property, at layers 0, 12, 24 and the layer mean. Step 0's
lifts are the instrument's baseline (§1.8(a)), not a finding. A delta is
"above step 0" iff > +0.05, "below" iff < -0.05, else "as step 0". The 0.05
is a floor under noise from 8 prompts (7 contribute: `repeated_tokens` has
no focal token), not a test: no null, no e-value.

What each reading would mean:
- ``copy_share`` / ``no_copy`` above step 0 at trained deep layers: unique
  tokens are glued to copy groups *more* than chance, so step 0's glue
  persists or grows; ``no_copy`` above: clusters of unique tokens form that
  step 0 does not make.
- ``adjacent`` above step 0: position (local context) is what they cluster
  with. It is the first suspect for `repeated_tokens`' ~50 deep clusters.
- ``emb_pct`` / ``same_class`` above step 0: they cluster with tokens the
  trained embedding calls similar, or of their class. The semantic reading.

``emb_pct`` at layer 0 of step 143000 is circular (that layer's partition was
made from that geometry); read it at depth or at other steps.

CHANGED AFTER `/challenge-pr` ON #92 (post hoc, 2026-09-24; the rule above is kept)
-------------------------------------------------------------------------------
- **The frozen-frame ``emb_pct`` has no valid step-0 baseline.** Step 0's
  clusters come from a random embedding unrelated to the trained one, so their
  lift in the trained frame is ~0 by construction, and its delta mixes "the
  embedding moved toward its final form" with "clusters follow the embedding".
  Its delta is recorded but its reading is ``n/a (frozen frame)``.
  ``emb_pct_own`` is added: the same statistic in each run's **own** layer 0,
  which has a valid step-0 baseline. At layer 0 it is circular at every step.
  At depth, a step-0 lift says clusters follow the token's own (random)
  embedding carried in the residual, which is the lexical reading.
- **A second expectation**: a draw from the prompt's **clustered** positions
  only (``*_cl``), beside the draw from all of them, because §1.7 found class
  predicts clustered vs noise, so the all-positions draw mixes that in. The
  pre-stated reading uses the all-positions draw; both are recorded.
- ``n_clusters``: distinct clusters behind the focal tokens of a run-layer,
  since a cluster of many unique tokens counts once per member.
- ``deltas`` refuses when step 0 is absent rather than using the earliest step.

TIER 1, EXPLORATORY, NOT REGISTERED. Descriptive, no null. Nothing here
touches `claims/registry.json`.

INPUT
-----
Stage 0's runs, selected **only** through `stage0_logs/stage0_index.json`,
screened by `core.holdout`. Labels and tokens as in
`p10_token_composition.measure_run` (native only, lengths checked, all-noise
refused). The frozen frame's run must have the same tokens. The record names
the index pin, battery hash, inputs sha256 and tokenizer sha256.

``--run-root DIR`` reads one flat run root instead (the pilot sweep on
`HDD_1TB`), as `p10_ext_sem_threshold.load_input`; ``--prompts`` keeps named keys.

Run:
    python tools/run/p10_comembership.py --v1-only
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[2])))  # this checkout, not a hard-coded main tree
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))

import numpy as np

from core.holdout import add_holdout_args, refuse_held_out
from tools.run.backfill_hdbscan import labels_provenance, read_labels
from tools.run.p10_ext_sem_threshold import FROZEN_STEP, layer0_gram, add_input_args, load_input, read_tokens
from tools.run.p10_token_composition import (
    CompositionError, find_tokenizer, load_vocab, token_features)

PROPS = ("copy_share", "no_copy", "adjacent", "same_class", "emb_pct", "emb_pct_own")
NOT_READ = {"emb_pct": "n/a (frozen frame)"}   # no valid step-0 baseline (docstring)
POOLS = ("", "_cl")                             # expectation drawn from all others / clustered others
DELTA_FLOOR = 0.05  # placed, not calibrated: a floor under noise from 7 prompts, no null behind it
PICK_LAYERS = (0, 12, 24)


def p_none(n_pool: int, n_good: int, k: int) -> float:
    """P(a uniform k-draw without replacement from n_pool hits none of n_good)."""
    if k > n_pool - n_good:
        return 0.0
    p = 1.0
    for i in range(k):
        p *= (n_pool - n_good - i) / (n_pool - i)
    return p


def midrank_pct(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Percentile of each x within `values` (x drawn from values), ties at mid-rank.

    (#less + (#equal - 1) / 2) / (m - 1): over all of `values` it averages exactly 0.5.
    """
    if len(values) == 1:        # a one-position pool: the draw is forced
        return np.full(len(x), 0.5)
    s = np.sort(values)
    lo = np.searchsorted(s, x, side="left")
    hi = np.searchsorted(s, x, side="right")
    return (lo + (hi - lo - 1) / 2) / (len(values) - 1)


def focal_stats(f: int, members: np.ndarray, pool: np.ndarray, is_copy: np.ndarray,
                cls: np.ndarray, emb_rows: dict) -> dict:
    """Observed and expected values for one focal token f, the draw taken from `pool`.

    `pool`: the positions a random co-member set is drawn from (f excluded, and
    every co-member in it). `emb_rows`: {property: f's row of that frame's Gram}.
    """
    co = members[members != f]
    k = len(co)
    n = len(pool)
    if f in pool or not np.isin(co, pool).all():
        raise CompositionError(f"position {f}: pool must hold every co-member and not f")
    adj = [p for p in (f - 1, f + 1) if p in set(pool.tolist())]
    n_copy = int(is_copy[pool].sum())
    out = {
        "k": k,
        "copy_share": (float(is_copy[co].mean()), n_copy / n),
        "no_copy": (float(not is_copy[co].any()), p_none(n, n_copy, k)),
        "adjacent": (float(np.isin(adj, co).any()), 1.0 - p_none(n, len(adj), k)),
        "same_class": (float((cls[co] == cls[f]).mean()), float((cls[pool] == cls[f]).mean())),
    }
    for name, row in emb_rows.items():
        out[name] = (float(midrank_pct(row[pool], row[co]).mean()), 0.5)
    return out


def measure_run(run_dir: Path, vocab: dict, added: set, frozen_gram: np.ndarray,
                frozen_tokens: np.ndarray, own_gram: np.ndarray = None) -> dict:
    """{layer: {prop: [obs, exp, lift, obs_cl, exp_cl, lift_cl], 'k', 'n_focal', 'n_clusters'}}.

    `own_gram` defaults to the run's own layer-0 Gram (read from activations.npz).
    """
    prov = labels_provenance(run_dir)
    if prov != "native":
        raise CompositionError(f"{run_dir.name}: labels are {prov}, not native")
    labels = read_labels(run_dir)
    tokens = read_tokens(run_dir)
    if not labels:
        raise CompositionError(f"{run_dir.name}: no labels")
    if all((lab == -1).all() for lab in labels.values()):
        raise CompositionError(f"{run_dir.name}: noise at every layer (an empty partition, not a result)")
    if len(frozen_tokens) != len(tokens) or (frozen_tokens != tokens).any():
        raise CompositionError(f"{run_dir.name}: tokens differ from the frozen frame's run")
    if own_gram is None:
        own_gram = layer0_gram(run_dir)
    for name, g in (("frozen", frozen_gram), ("own", own_gram)):
        if g.shape != (len(tokens), len(tokens)):
            raise CompositionError(f"{run_dir.name}: {name} Gram is {g.shape}, {len(tokens)} tokens")
    feats = token_features(tokens, vocab, added)
    is_copy = np.array([f["copies"] != "unique" for f in feats])
    cls = np.array([f["cls"] for f in feats], dtype=object)
    unique_pos = [p for p in range(1, len(tokens)) if not is_copy[p]]
    out = {}
    for layer, lab in sorted(labels.items()):
        if len(lab) != len(tokens):
            raise CompositionError(f"{run_dir.name} layer {layer}: {len(lab)} labels, {len(tokens)} tokens")
        focal = [f for f in unique_pos if lab[f] != -1]
        clustered = np.flatnonzero(lab != -1)
        rows = {pool: [] for pool in POOLS}
        for f in focal:
            members = np.flatnonzero(lab == lab[f])
            emb = {"emb_pct": frozen_gram[f], "emb_pct_own": own_gram[f]}
            rows[""].append(focal_stats(f, members, np.delete(np.arange(len(tokens)), f),
                                        is_copy, cls, emb))
            rows["_cl"].append(focal_stats(f, members, clustered[clustered != f], is_copy, cls, emb))
        rec = {"n_focal": len(focal), "n_unique": len(unique_pos),
               "n_clusters": len({int(lab[f]) for f in focal})}
        if focal:
            rec["k"] = float(np.mean([r["k"] for r in rows[""]]))
            for p in PROPS:
                rec[p] = []
                for pool in POOLS:
                    o = float(np.mean([r[p][0] for r in rows[pool]]))
                    e = float(np.mean([r[p][1] for r in rows[pool]]))
                    rec[p] += [o, e, o - e]
        out[layer] = rec
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


STAT_NAMES = tuple(f"{x}{pool}" for pool in POOLS for x in ("obs", "exp", "lift"))


def summarise(results: dict) -> dict:
    """{subset: {step: {layer|'mean': {prop: {obs, exp, lift, *_cl}, k, n_runs, focal_frac, n_clusters}}}}."""
    out = {}
    for subset, keep in {"all": lambda k: True,
                         "without_repeated_tokens": lambda k: k != "repeated_tokens"}.items():
        out[subset] = {}
        for s in sorted({st for st, _ in results}):
            runs = [r for (st, key), r in results.items() if st == s and keep(key)]
            layers = sorted({L for r in runs for L in r})
            by = {}
            for L in layers:
                rs = [r[L] for r in runs if L in r and r[L]["n_focal"]]
                by[L] = {"n_runs": len(rs),
                         "focal_frac": _mean(r[L]["n_focal"] / r[L]["n_unique"]
                                             for r in runs if L in r and r[L]["n_unique"]),
                         "k": _mean(r["k"] for r in rs),
                         "n_clusters": _mean(r[L]["n_clusters"] for r in runs if L in r and r[L]["n_focal"])}
                for p in PROPS:
                    by[L][p] = {name: _mean(r[p][i] for r in rs) for i, name in enumerate(STAT_NAMES)}
            by["mean"] = {x: _mean(by[L][x] for L in layers) for x in ("focal_frac", "k", "n_clusters")}
            for p in PROPS:
                by["mean"][p] = {name: _mean(by[L][p][name] for L in layers) for name in STAT_NAMES}
            out[subset][s] = by
    return out


def deltas(summary: dict) -> dict:
    """The pre-stated reading: lift(step) - lift(step 0), with its word, for both draws."""
    out = {}
    for subset, steps in summary.items():
        if 0 not in steps:
            raise CompositionError(f"{subset}: no step 0, the baseline every delta is taken against")
        base = steps[0]
        out[subset] = {}
        for s, by in steps.items():
            out[subset][s] = {}
            for L in [*PICK_LAYERS, "mean"]:
                if L not in by or L not in base:
                    continue
                out[subset][s][L] = {}
                for p in PROPS:
                    out[subset][s][L][p] = {}
                    for pool in POOLS:
                        a, b = by[L][p][f"lift{pool}"], base[L][p][f"lift{pool}"]
                        d = None if a is None or b is None else a - b
                        word = ("unavailable" if d is None else NOT_READ.get(p) or
                                ("above step 0" if d > DELTA_FLOOR
                                 else "below step 0" if d < -DELTA_FLOOR else "as step 0"))
                        out[subset][s][L][p][f"delta{pool}"] = d
                        out[subset][s][L][p][f"reading{pool}"] = word
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_input_args(ap, DATA / "phase12" / "stage0_logs" / "stage0_index.json")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", str(DATA / "hf")))
    ap.add_argument("--out", default=str(DATA / "analysis" / "p10_s1_comembership.json"))
    add_holdout_args(ap)
    args = ap.parse_args()

    tok_path = find_tokenizer(Path(args.hf_home))
    vocab, added = load_vocab(tok_path)
    idx = load_input(args)
    kept, holdout = refuse_held_out(
        sorted(idx["runs"].values()), allow=args.allow_holdout, drop=args.v1_only,
        context="p10_comembership")
    kept = set(kept)
    runs = {k: v for k, v in idx["runs"].items() if v in kept}
    prompts = sorted({key for _, key in runs})
    missing = [p for p in prompts if (FROZEN_STEP, p) not in runs]
    if missing:
        raise CompositionError(f"no step-{FROZEN_STEP} run (the frozen frame) for {missing}")
    print(f"{len(runs)} runs, {len(prompts)} prompts, steps {sorted({s for s, _ in runs})}")

    frozen = {p: (layer0_gram(runs[(FROZEN_STEP, p)]), read_tokens(runs[(FROZEN_STEP, p)]))
              for p in prompts}
    t0 = time.time()
    results = {}
    for n, (step, key) in enumerate(sorted(runs), 1):
        results[(step, key)] = measure_run(runs[(step, key)], vocab, added, *frozen[key])
        if n % 40 == 0 or n == len(runs):
            print(f"  [{n}/{len(runs)}] step{step} {key}  ({time.time() - t0:.0f}s)", flush=True)

    inputs = sorted((f"{s}|{k}", str(p)) for (s, k), p in runs.items())
    summary = summarise(results)
    reading = deltas(summary)
    record = {
        "schema": "p10_s1_comembership/2",
        "row": "Stage 1 — what clustered unique tokens cluster with, against step 0",
        "tier": "1 (exploratory, unregistered, descriptive; no null)",
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "index": idx["source"],
        "index_pin": idx["pin"],
        "prompt_battery_hash": idx["prompt_battery_hash"],
        "inputs_sha256": hashlib.sha256(json.dumps(inputs).encode()).hexdigest()[:12],
        "tokenizer": str(tok_path),
        "tokenizer_sha256": hashlib.sha256(tok_path.read_bytes()).hexdigest()[:12],
        "frozen_step": FROZEN_STEP,
        "n_runs": len(runs),
        "prompts": prompts,
        "steps": sorted({s for s, _ in runs}),
        "holdout": holdout,
        "criterion": f"delta = lift(step) - lift(step 0) per property; above/below step 0 iff "
                     f"|delta| > {DELTA_FLOOR}; lift = mean observed - mean expected under a "
                     "uniform draw of the same size from the prompt's other positions (*_cl: from its "
                     "other clustered positions); emb_pct (frozen frame) is not read against step 0",
        "summary": summary,
        "reading": reading,
        "inputs": inputs,
        "runs": {f"{s}|{k}": r for (s, k), r in sorted(results.items())},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    print(f"\nwrote {out}  (inputs {record['inputs_sha256']})")
    for subset in summary:
        print(f"\n{subset}: obs / exp (lift | lift, clustered draw) per property; layers "
              + ", ".join(map(str, PICK_LAYERS)) + ", mean")
        for s, by in summary[subset].items():
            for L in [*PICK_LAYERS, "mean"]:
                if L not in by:
                    continue
                x = by[L]
                print(f"  step{s:>6} L{L!s:>4} focal {x['focal_frac']:.2f} k {x['k']:.1f} "
                      f"clusters {x['n_clusters']:.1f}  " + "  ".join(
                          f"{p} {x[p]['obs']:.2f}/{x[p]['exp']:.2f} ({x[p]['lift']:+.2f}|{x[p]['lift_cl']:+.2f})"
                          for p in PROPS))


if __name__ == "__main__":
    main()
