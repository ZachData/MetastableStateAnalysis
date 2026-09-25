"""Stage 1 step 2 of `p10_cluster_function/handoff-10.md`: the token-composition table.

Which tokens end up in an HDBSCAN cluster, and which end up noise? Per layer
per checkpoint, the clustered rate (label != -1) of each kind of token, with
repeat / non-repeat carried as a column on every table because step 1
(`status-10.md` §1.6) found repeats drive the only semantic number the project
had.

THE FEATURES (one token occurrence of one run at one layer is the unit)
----------------------------------------------------------------------
- ``copies`` (the repeat column): ``repeat`` when the token id has an earlier
  copy in the prompt (§1.3's definition); otherwise ``first`` when it has a
  later one, ``unique`` when it has none. §1.3's non-repeat is ``unique`` +
  ``first``. The three-way split was added after the first run (below).
- ``count``: how many times the token id occurs in the prompt, binned
  ``COUNT_BINS``. Separates "first of many" from "unique".
- ``rank``: the token id, binned ``RANK_BINS``. In the GPT-NeoX tokenizer every
  merged token's id is its BPE merge index + 245 (checked at load), and bytes
  sit below the merges. **Merge order is a frequency proxy, not a frequency:**
  it ranks pairs by count in the tokenizer's training text at the moment of
  merging, so a long common word can have a later id than a rare fragment.
  No corpus unigram count is on disk.
- ``cls``: from the decoded bytes. ``whitespace`` (only whitespace),
  ``punct`` (only Unicode P*/S* after stripping), ``numeric`` (digits),
  ``word_start`` (alphabetic with leading whitespace, or alphabetic after a
  token ending in whitespace, or at position 0), ``continuation`` (alphabetic,
  glued to a previous non-whitespace token), ``byte_fragment`` (not valid
  UTF-8 alone: a split multi-byte character), ``other`` (mixed).
- ``pos``: position, binned ``POS_BINS``. Position 0 is the attention sink;
  a confound the handoff's list leaves out, carried for that reason.

Every cell is (step, layer, feature, level, repeat) with ``n`` and
``n_clustered``, summed over prompts. The summary also averages per-prompt
rates, so a prompt that repeats a lot (``repeated_tokens``) cannot carry a
pooled cell alone.

WHAT WOULD MAKE THE SEMANTIC QUESTION CONCRETE, FIXED BEFORE ANY TABLE IS READ
------------------------------------------------------------------------------
Trash collection (`handoff-10.md` §1.3) predicts clusters dominated by
high-frequency, low-information tokens. Among **non-repeat, non-position-0**
tokens, per step, averaged over prompts x layers (per-prompt rates, then the
mean):

- ``freq_contrast`` = clustered rate at rank < 1000 minus at rank >= 20000.
- ``class_contrast`` = clustered rate of whitespace + punct minus of word_start.

Trash collection is **consistent** at a step iff both are > +0.05, **against**
iff both are < -0.05, otherwise **unclear**. The 0.05 is a floor under
float-noise, not a test: no null, no e-value. Reported per step, not pooled
over steps, because a sign that flips with training is itself the answer.
Repeats are excluded from the contrast because they sit at cosine 1 at
layer 0 (§1.6) and co-cluster for that reason alone.

ADDED AFTER THE FIRST RUN (post hoc, 2026-09-24; the criterion above is unchanged)
---------------------------------------------------------------------------------
A ``first`` occurrence still has cosine-1 twins later in the prompt, and
high-frequency tokens recur more, so the pre-stated non-repeat contrast keeps
part of the confound it meant to drop. The first run showed the clustered rate
is set mostly by copy count: HDBSCAN runs with ``min_cluster_size=2`` (and
``min_samples`` defaulting to it), and at step 0 every token with 3-5 copies is
clustered. The same two contrasts are therefore also computed on ``unique``
tokens only, with their own verdict under the same rule. Both are in the record.

CLUSTER COUNT VS REPEATED TYPES (added 2026-09-24, parked from `/challenge-pr` on #90)
------------------------------------------------------------------------------------
The reviewer's rough check: a run's HDBSCAN cluster count is about the number
of token types occurring >= 2 times. Per run per layer: ``n_clusters``,
``n_repeated_types`` (ids with count >= 2), and ``single_type`` (clusters whose
members are all one token id) and ``holds_repeat`` (clusters holding >= 2
copies of some id). Summarised per step and layer as means over runs of
``n_clusters / n_repeated_types`` and of the two cluster fractions, with and
without ``repeated_tokens`` (the reviewer's exception). Also Phase 1's
``max_alive`` (`cluster_tracking.py`: the most clusters at any one layer) and
the layer where it falls, so the carrying capacity is read off the same
labels (added after `/challenge-pr` on #91). If the ratio is near 1
and most clusters hold a repeat, Phase 1's cluster counts are largely a count
of repeated types. `tools/run/p10_hdbscan_planted.py` is the known answer for
structureless input.

TIER 1, EXPLORATORY, NOT REGISTERED. Descriptive, no null. Nothing here
touches `claims/registry.json`.

INPUT
-----
Stage 0's runs, selected **only** through `stage0_logs/stage0_index.json`,
screened by `core.holdout`. Labels through `backfill_hdbscan.read_labels`;
a run whose labels are not ``native``, do not match `tokens.txt` in length, or
are noise at every layer is refused, not averaged in. The record names the
index pin, battery hash, the sha256 of the sorted (key, path) list read, and
the tokenizer.json sha256.

``--run-root DIR`` reads one flat run root instead (the pilot sweep on
`HDD_1TB`), keyed by each run's `manifest.json` and refused if it mixes
batteries; ``--prompts`` keeps named keys. Comparison: `p10_s1_compare.py`.

Run:
    python tools/run/p10_token_composition.py --v1-only
"""
import argparse
import hashlib
import json
import os
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[2])))  # this checkout, not a hard-coded main tree
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))

import numpy as np

from core.holdout import add_holdout_args, refuse_held_out
from tools.run.backfill_hdbscan import labels_provenance, read_labels
from tools.run.p10_ext_sem_threshold import add_input_args, load_input, read_tokens

RANK_BINS = (1000, 5000, 20000)           # [0,1000) [1000,5000) [5000,20000) [20000,)
COUNT_BINS = (1, 2, 5, 20)                # 1, 2, 3-5, 6-20, 21+
POS_BINS = (1, 16, 64)                    # 0, 1-15, 16-63, 64+
CLASSES = ("whitespace", "punct", "numeric", "word_start", "continuation",
           "byte_fragment", "other")
FEATURES = ("cls", "rank", "count", "pos")
COPIES = ("unique", "first", "repeat")
MERGE_OFFSET = 245
CONTRAST_FLOOR = 0.05
CONTRAST_SUBSETS = {"non_repeat": ("unique", "first"),   # pre-stated
                    "unique": ("unique",)}                # post hoc, see docstring


class CompositionError(RuntimeError):
    """An input that would make a cell wrong rather than empty."""


def _bytes_to_unicode() -> dict:
    """GPT-2's byte-level map (the one GPT-NeoX's ByteLevel pre-tokenizer uses)."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


UNI_TO_BYTE = {u: b for b, u in _bytes_to_unicode().items()}


def find_tokenizer(hf_home: Path) -> Path:
    snaps = sorted((hf_home / "hub" / "models--EleutherAI--pythia-410m" / "snapshots")
                   .glob("*/tokenizer.json"))
    if not snaps:
        raise CompositionError(f"no pythia-410m tokenizer.json under {hf_home}")
    shas = {hashlib.sha256(p.read_bytes()).hexdigest() for p in snaps}
    if len(shas) != 1:
        raise CompositionError(f"{len(shas)} different tokenizer.json files among snapshots")
    return snaps[0]


def load_vocab(path: Path) -> tuple:
    """(token string -> id, ids of added tokens), after checking id = merge index + MERGE_OFFSET.

    The added tokens (ids 50254+) are literal runs of spaces outside the BPE
    vocab. Their ids say nothing about frequency, so they get rank level
    ``added`` and are decoded as the literal string they are.
    """
    t = json.loads(path.read_text())
    vocab, merges = dict(t["model"]["vocab"]), t["model"]["merges"]
    for n, m in enumerate(merges):
        tok = "".join(m.split(" ")) if isinstance(m, str) else "".join(m)
        if vocab[tok] != n + MERGE_OFFSET:
            raise CompositionError(f"merge {n} ({tok!r}) has id {vocab[tok]}: ids are not merge order")
    added = set()
    for a in t["added_tokens"]:
        if a["content"] not in vocab:
            vocab[a["content"]] = a["id"]
            added.add(a["id"])
    return vocab, added


def decode(tok: str):
    """The token's text, or None when its bytes are not valid UTF-8 alone."""
    try:
        return bytes(UNI_TO_BYTE[c] for c in tok).decode("utf-8")
    except (KeyError, UnicodeDecodeError):
        return None


def token_class(text, prev_text, pos: int) -> str:
    if text is None:
        return "byte_fragment"
    core = text.strip()
    if not core:
        return "whitespace"
    if all(unicodedata.category(c)[0] in "PS" for c in core):
        return "punct"
    if core.isdigit():
        return "numeric"
    if core.isalpha():
        if text[0].isspace() or pos == 0 or (prev_text and prev_text[-1].isspace()):
            return "word_start"
        return "continuation"
    return "other"


def _bin(x: int, edges: tuple) -> int:
    return int(np.searchsorted(edges, x, side="right"))


def rank_level(i: int, added: set = frozenset()) -> str:
    if i in added:
        return "added"
    names = ("<1k", "1k-5k", "5k-20k", ">=20k")
    return names[_bin(i, RANK_BINS)]


def count_level(c: int) -> str:
    names = ("1", "2", "3-5", "6-20", "21+")
    return names[int(np.searchsorted(COUNT_BINS, c, side="left"))]


def pos_level(p: int) -> str:
    names = ("0", "1-15", "16-63", "64+")
    return names[_bin(p, POS_BINS)]


def token_features(tokens: np.ndarray, vocab: dict, added: set = frozenset()) -> list:
    """Per position: {feature: level, 'repeat': first|repeat, 'id': int}."""
    missing = sorted({t for t in tokens if t not in vocab})
    if missing:
        raise CompositionError(f"tokens not in the vocab: {missing[:5]}")
    ids = [vocab[t] for t in tokens]
    counts = defaultdict(int)
    for i in ids:
        counts[i] += 1
    seen = set()
    texts = [t if i in added else decode(t) for t, i in zip(tokens, ids)]
    out = []
    for p, (i, text) in enumerate(zip(ids, texts)):
        out.append({
            "id": i,
            "copies": "repeat" if i in seen else ("first" if counts[i] > 1 else "unique"),
            "cls": token_class(text, texts[p - 1] if p else None, p),
            "rank": rank_level(i, added),
            "count": count_level(counts[i]),
            "pos": pos_level(p),
        })
        seen.add(i)
    return out


def measure_run(run_dir: Path, vocab: dict, added: set = frozenset()) -> dict:
    """{layer: {(feature, level, repeat): [n, n_clustered]}} plus provenance."""
    prov = labels_provenance(run_dir)
    if prov != "native":
        raise CompositionError(f"{run_dir.name}: labels are {prov}, not native")
    labels = read_labels(run_dir)
    tokens = read_tokens(run_dir)
    feats = token_features(tokens, vocab, added)
    if not labels:
        raise CompositionError(f"{run_dir.name}: no labels")
    if all((lab == -1).all() for lab in labels.values()):
        raise CompositionError(f"{run_dir.name}: noise at every layer (an empty partition, not a result)")
    out = {}
    for layer, lab in sorted(labels.items()):
        if len(lab) != len(tokens):
            raise CompositionError(f"{run_dir.name} layer {layer}: {len(lab)} labels, {len(tokens)} tokens")
        cells = defaultdict(lambda: [0, 0])
        for f, clustered in zip(feats, lab != -1):
            for feat in FEATURES:
                c = cells[(feat, f[feat], f["copies"])]
                c[0] += 1
                c[1] += int(clustered)
            c = cells[("all", "all", f["copies"])]
            c[0] += 1
            c[1] += int(clustered)
        out[layer] = dict(cells)
    contrast = {}
    for subset, allowed in CONTRAST_SUBSETS.items():
        contrast[subset] = {}
        for layer, lab in sorted(labels.items()):
            keep = [(f, cl) for p, (f, cl) in enumerate(zip(feats, lab != -1))
                    if f["copies"] in allowed and p > 0]
            def rate(pred):
                xs = [cl for f, cl in keep if pred(f)]
                return (float(np.mean(xs)) if xs else None, len(xs))
            contrast[subset][layer] = {
                "hi_freq": rate(lambda f: f["rank"] == "<1k"),
                "lo_freq": rate(lambda f: f["rank"] == ">=20k"),
                "ws_punct": rate(lambda f: f["cls"] in ("whitespace", "punct")),
                "word_start": rate(lambda f: f["cls"] == "word_start"),
            }
    ids = np.array([f["id"] for f in feats])
    uniq, cnt = np.unique(ids, return_counts=True)
    n_rep_types = int((cnt >= 2).sum())
    counts = {"n_repeated_types": n_rep_types, "by_layer": {}}
    for layer, lab in sorted(labels.items()):
        ks = sorted(set(lab.tolist()) - {-1})
        member_ids = [ids[lab == k] for k in ks]
        counts["by_layer"][layer] = {
            "n_clusters": len(ks),
            "single_type": sum(1 for m in member_ids if len(set(m.tolist())) == 1),
            "holds_repeat": sum(1 for m in member_ids if len(m) > len(set(m.tolist()))),
        }
    per_layer = {L: x["n_clusters"] for L, x in counts["by_layer"].items()}
    counts["max_alive"] = max(per_layer.values())
    counts["max_alive_layers"] = [L for L, c in per_layer.items() if c == counts["max_alive"]]
    return {"cells": out, "contrast": contrast, "cluster_count": counts,
            "provenance": prov, "n_tokens": len(tokens),
            "noise_rate": {layer: float((lab == -1).mean()) for layer, lab in labels.items()}}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def contrast_verdict(per_run: list) -> dict:
    """The criterion in the module docstring, over one step's runs x layers."""
    c = {}
    for name, (a, b) in {"freq_contrast": ("hi_freq", "lo_freq"),
                         "class_contrast": ("ws_punct", "word_start")}.items():
        diffs = [L[a][0] - L[b][0] for layers in per_run for L in layers.values()
                 if L[a][0] is not None and L[b][0] is not None]
        c[name] = _mean(diffs)
        c[f"n_{name}"] = len(diffs)
    fc, cc = c["freq_contrast"], c["class_contrast"]
    if fc is None or cc is None:
        v = "unavailable"
    elif fc > CONTRAST_FLOOR and cc > CONTRAST_FLOOR:
        v = "consistent"
    elif fc < -CONTRAST_FLOOR and cc < -CONTRAST_FLOOR:
        v = "against"
    else:
        v = "unclear"
    c["trash_collection"] = v
    return c


def aggregate(results: dict) -> dict:
    """results: {(step, key): measure_run output}."""
    steps = sorted({s for s, _ in results})
    pooled, balanced, contrasts, cluster_count = {}, {}, {}, {}
    for s in steps:
        runs = {k: r for (st, k), r in results.items() if st == s}
        layers = sorted({L for r in runs.values() for L in r["cells"]})
        cell_keys = sorted({c for r in runs.values() for L in r["cells"].values() for c in L})
        pooled[s], balanced[s] = {}, {}
        for L in layers:
            pooled[s][L] = {}
            for ck in cell_keys:
                n = sum(r["cells"][L].get(ck, (0, 0))[0] for r in runs.values())
                nc = sum(r["cells"][L].get(ck, (0, 0))[1] for r in runs.values())
                if n:
                    pooled[s][L]["|".join(ck)] = [n, nc]
        # per-prompt rate, averaged over prompts then layers: no prompt carries a cell alone
        for ck in cell_keys:
            per_layer = []
            for L in layers:
                rates = [r["cells"][L][ck][1] / r["cells"][L][ck][0]
                         for r in runs.values() if r["cells"][L].get(ck, (0, 0))[0]]
                per_layer.append(_mean(rates))
            n_prompts = sum(1 for r in runs.values() if r["cells"][layers[0]].get(ck, (0, 0))[0])
            balanced[s]["|".join(ck)] = {"rate": _mean(per_layer), "n_prompts": n_prompts,
                                         "n_tokens_layer0": sum(r["cells"][layers[0]].get(ck, (0, 0))[0]
                                                                for r in runs.values())}
        c = {subset: contrast_verdict([r["contrast"][subset] for r in runs.values()])
             for subset in CONTRAST_SUBSETS}
        c["noise_rate"] = _mean(x for r in runs.values() for x in r["noise_rate"].values())
        c["n_runs"] = len(runs)
        contrasts[s] = c
        cluster_count[s] = cluster_count_summary(runs)
    return {"by_step": contrasts, "balanced": balanced, "pooled": pooled,
            "cluster_count": cluster_count}


def cluster_count_summary(runs: dict) -> dict:
    """{all|without_repeated_tokens: {layer: means over runs}} for one step; runs: {key: measure_run}."""
    out = {}
    for subset, keep in {"all": lambda k: True,
                         "without_repeated_tokens": lambda k: k != "repeated_tokens"}.items():
        rs = [r["cluster_count"] for k, r in runs.items() if keep(k)]
        layers = sorted({L for r in rs for L in r["by_layer"]})
        out[subset] = {"n_runs": len(rs),
                       "n_repeated_types": _mean(r["n_repeated_types"] for r in rs),
                       "max_alive": _mean(r["max_alive"] for r in rs),
                       "max_alive_at_layer0": sum(1 for r in rs if min(r["max_alive_layers"]) == 0)}
        for L in layers:
            ls = [(r["n_repeated_types"], r["by_layer"][L]) for r in rs if L in r["by_layer"]]
            out[subset][L] = {
                "n_clusters": _mean(x["n_clusters"] for _, x in ls),
                "ratio": _mean(x["n_clusters"] / n for n, x in ls if n),
                "single_type_frac": _mean(x["single_type"] / x["n_clusters"] for _, x in ls if x["n_clusters"]),
                "holds_repeat_frac": _mean(x["holds_repeat"] / x["n_clusters"] for _, x in ls if x["n_clusters"]),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_input_args(ap, DATA / "phase12" / "stage0_logs" / "stage0_index.json")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", str(DATA / "hf")))
    ap.add_argument("--out", default=str(DATA / "analysis" / "p10_s1_token_composition.json"))
    add_holdout_args(ap)
    args = ap.parse_args()

    tok_path = find_tokenizer(Path(args.hf_home))
    vocab, added = load_vocab(tok_path)
    idx = load_input(args)
    kept, holdout = refuse_held_out(
        sorted(idx["runs"].values()), allow=args.allow_holdout, drop=args.v1_only,
        context="p10_token_composition")
    kept = set(kept)
    runs = {k: v for k, v in idx["runs"].items() if v in kept}
    prompts = sorted({key for _, key in runs})
    print(f"{len(runs)} runs, {len(prompts)} prompts, steps {sorted({s for s, _ in runs})}")

    t0 = time.time()
    results = {}
    for n, (step, key) in enumerate(sorted(runs), 1):
        results[(step, key)] = measure_run(runs[(step, key)], vocab, added)
        if n % 40 == 0 or n == len(runs):
            print(f"  [{n}/{len(runs)}] step{step} {key}  ({time.time() - t0:.0f}s)", flush=True)

    inputs = sorted((f"{s}|{k}", str(p)) for (s, k), p in runs.items())
    summary = aggregate(results)
    record = {
        "schema": "p10_s1_token_composition/1",
        "row": "Stage 1 step 2 — which tokens are clustered and which are noise",
        "tier": "1 (exploratory, unregistered, descriptive; no null)",
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "index": idx["source"],
        "index_pin": idx["pin"],
        "prompt_battery_hash": idx["prompt_battery_hash"],
        "inputs_sha256": hashlib.sha256(json.dumps(inputs).encode()).hexdigest()[:12],
        "tokenizer": str(tok_path),
        "tokenizer_sha256": hashlib.sha256(tok_path.read_bytes()).hexdigest()[:12],
        "n_runs": len(runs),
        "prompts": prompts,
        "steps": sorted({s for s, _ in runs}),
        "holdout": holdout,
        "bins": {"rank": list(RANK_BINS), "count": list(COUNT_BINS), "pos": list(POS_BINS)},
        "rank_is": f"token id = BPE merge index + {MERGE_OFFSET}; a frequency proxy, not a count",
        "criterion": "per step, non-repeat (pre-stated) and unique (post hoc) tokens at position > 0, per-prompt-per-layer rates "
                     "averaged: freq_contrast = rate(rank<1k) - rate(rank>=20k), class_contrast = "
                     "rate(whitespace+punct) - rate(word_start); consistent iff both > "
                     f"{CONTRAST_FLOOR}, against iff both < -{CONTRAST_FLOOR}, else unclear",
        "cell_key": "feature|level|copies -> [n, n_clustered] (pooled); rate (balanced)",
        "summary": summary,
        "inputs": inputs,
        "runs": {f"{s}|{k}": {"contrast": r["contrast"], "noise_rate": r["noise_rate"],
                              "cluster_count": r["cluster_count"],
                              "n_tokens": r["n_tokens"],
                              "cells": {L: {"|".join(ck): v for ck, v in cells.items()}
                                        for L, cells in r["cells"].items()}}
                 for (s, k), r in sorted(results.items())},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    print(f"\nwrote {out}  (inputs {record['inputs_sha256']})")
    for s, c in summary["by_step"].items():
        print(f"  step{s:>6}: " + "   ".join(
            f"{sub}: freq {c[sub]['freq_contrast']:+.3f} class {c[sub]['class_contrast']:+.3f} "
            f"-> {c[sub]['trash_collection']}" for sub in CONTRAST_SUBSETS))
    print("\ncluster count vs repeated types (without repeated_tokens; layers 0 / mid / last):")
    for s, cc in summary["cluster_count"].items():
        w = cc["without_repeated_tokens"]
        Ls = sorted(k for k in w if isinstance(k, int))
        pick = [Ls[0], Ls[len(Ls) // 2], Ls[-1]]
        print(f"  step{s:>6}: repeated types {w['n_repeated_types']:.1f}  max_alive {w['max_alive']:.1f} "
              f"(at L0 in {w['max_alive_at_layer0']}/{w['n_runs']})  " + "  ".join(
            f"L{L}: clusters {w[L]['n_clusters']:.1f} ratio {w[L]['ratio']:.2f} "
            f"hold-repeat {w[L]['holds_repeat_frac']:.2f} single-type {w[L]['single_type_frac']:.2f}"
            for L in pick))


if __name__ == "__main__":
    main()
