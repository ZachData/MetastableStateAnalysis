"""Stage 1 step 1 of `p10_cluster_function/handoff-10.md`: the `ext_sem_threshold` sweep.

`pair_hdbscan_agreement` (`p1_mstate_tracking/clustering.py`) tags each mutual
nearest-neighbour pair of a layer as "ext_semantic" when its cosine in the
layer-0 Gram exceeds 0.5. `handoff-10.md` §1.1 read, on the pilot sweep, that
the fraction of such pairs falls with training (0.833 at step 0 to 0.708 at
step 143000) while the same-cluster rate among them stays flat, and §1.3 gates
that reading on this sweep: **if the decline depends on the threshold it is a
scale artifact and the reading is dead.**

WHAT IS RECOMPUTED, AND HOW IT IS CHECKED
------------------------------------------
Only the tag was stored, not the cosine, so the cosine is rebuilt: layer 0 of
`activations.npz` is the same sphere-projected `hidden_states[0]` that
`analysis_p1.py` built `emb_gram` from. The mutual pairs and their
cross-method tags are read from `clustering.json` as stored. The self frame at
0.5 must reproduce every stored `n_ext_semantic` exactly; a run that does not is
refused, not averaged in.

THREE WAYS OF CUTTING, TWO FRAMES
----------------------------------
- **Absolute thresholds** (``ABS_THRESHOLDS``): the stored rule at other cutoffs.
- **Quantile thresholds** (``QUANTILES``): a pair counts when its cosine exceeds
  the q-quantile of all off-diagonal cosines of the same frame and prompt.
  Invariant under any increasing map of the Gram, so a uniform shift or
  stretch of the cosine distribution (anisotropy, norm) cannot move it. This is
  the scale control.
- **Threshold-free:** the mean percentile rank of the mutual pairs' cosines
  within that distribution.

Frames (defect D6 in `pair_hdbscan_agreement`'s docstring: on a checkpoint
sweep the model's own embeddings are a moving reference):

- ``self``: the run's own layer 0, as stored (`emb_gram_source` "unspecified").
- ``frozen``: layer 0 of the step-143000 run of the same prompt. Tokens are
  identical across steps (same tokenizer, same text), so the index is shared.
  A decline in ``self`` that is absent in ``frozen`` is the embedding moving,
  not the neighbourhoods.

WHAT "SURVIVES" MEANS, FIXED HERE BEFORE ANY SWEEP IS READ
-----------------------------------------------------------
Per frame, the decline is ``mean ext_semantic_fraction at step 0`` minus the
same at step 143000. Means are over prompts x layers, as in §1.1's table. The
decline **survives** in a frame iff it is positive at every quantile threshold
*and* at every absolute threshold where both endpoint fractions lie in
[0.05, 0.95] (outside that range a threshold is saturated and says nothing).
It is **dead** iff it is non-positive at some quantile threshold. Anything else
is **mixed**. Reported per frame, no null, no e-value.

WHAT THE FIRST RUN FOUND (post hoc, 2026-09-24; the criterion above is unchanged)
---------------------------------------------------------------------------------
The criterion assumed a continuous cosine. It is not: Pythia adds no position
embedding before layer 0, so two occurrences of one token have cosine exactly 1
in every frame, and early in training every other mutual pair sits far below
0.5. The mutual pairs are a **mixture** of repeats (a pile at 1) and non-repeats
(continuous), and the stored statistic mostly counts the pile. The criterion's
verdicts are computed as written and recorded. Which cut decides each verdict is
in the record (`summary.verdict.*.decline`); read that, not the verdict word.
Two further flaws, both from the pile: a quantile over **all** off-diagonal
pairs lands inside it when a prompt repeats a lot (`repeated_tokens`), and so
does the all-pairs percentile rank. The ``non_repeat`` block is added for this
reason. It takes cuts, quantiles and ranks over non-repeat pairs only, beside
their base rate in the prompt. That block and ``identical_token_fraction``
carry the finding (numbers: `status-10.md` §1.6 only).

TIER 1, EXPLORATORY, NOT REGISTERED. Descriptive, no null, not quotable as an
adjudication. Nothing here touches `claims/registry.json`.

INPUT
-----
Stage 0's runs, selected **only** through `stage0_logs/stage0_index.json`
(never a glob: `handoff-10.md` Parked, the default glob mixes sweeps), screened
by `core.holdout`. The record names the index's pin, battery hash, and the
sha256 of the sorted (key, path) list read.

``--run-root DIR`` reads one flat run root instead (the pilot sweep on
`HDD_1TB`), keyed by each run's `manifest.json` and refused if it mixes
batteries; ``--prompts`` keeps named keys. Comparison: `p10_s1_compare.py`.

Run:
    python tools/run/p10_ext_sem_threshold.py --v1-only
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[2])))  # this checkout, not a hard-coded main tree
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))

import numpy as np

from core.holdout import add_holdout_args, refuse_held_out

STORED_THRESHOLD = 0.5
ABS_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99)
FROZEN_STEP = 143000
SATURATION = (0.05, 0.95)


class ReproductionError(RuntimeError):
    """The rebuilt cosine does not reproduce the stored tags."""


def load_index(path: Path) -> dict:
    """{(step, prompt_key): run_dir} plus the index's own header."""
    d = json.loads(path.read_text())
    runs = {}
    for k, v in d["runs"].items():
        step, key = k.split("|", 1)
        runs[(int(step), key)] = Path(v)
    return {"pin": d.get("pin"), "prompt_battery_hash": d.get("prompt_battery_hash"),
            "runs": runs}


def load_run_root(root: Path, model: str = "pythia-410m") -> dict:
    """The same shape as ``load_index``, for one flat run root such as the pilot
    sweep: every ``<model>-step*`` dir with a ``manifest.json``, keyed by its
    manifest. Refuses a root that mixes batteries or repeats a (step, prompt)."""
    runs, batteries, shas = {}, set(), set()
    for d in sorted(root.glob(f"{model}-step*")):
        m = d / "manifest.json"
        if not m.is_file():
            continue
        man = json.loads(m.read_text())
        k = (int(man["checkpoint_step"]), man["prompt_key"])
        if k in runs:
            sys.exit(f"{root}: two runs for {k}: {runs[k]} and {d}")
        runs[k] = d
        batteries.add(man.get("prompt_battery_hash"))
        shas.add(man.get("git_sha"))
    if len(batteries) != 1:
        sys.exit(f"{root}: {len(batteries)} prompt batteries {sorted(map(str, batteries))}; need one")
    return {"pin": ",".join(sorted(map(str, shas))), "prompt_battery_hash": batteries.pop(),
            "runs": runs}


def add_input_args(ap, default_index: Path) -> None:
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--index", default=str(default_index))
    g.add_argument("--run-root", help="a flat run root (the pilot sweep) instead of the index")
    ap.add_argument("--prompts", help="comma-separated prompt keys to keep (default: all)")


def load_input(args) -> dict:
    """``--run-root`` or ``--index``, then ``--prompts``; names the source."""
    if args.run_root:
        idx = load_run_root(Path(args.run_root))
        idx["source"] = str(args.run_root)
    else:
        idx = load_index(Path(args.index))
        idx["source"] = str(args.index)
    if args.prompts:
        keep = set(args.prompts.split(","))
        unknown = keep - {k for _, k in idx["runs"]}
        if unknown:
            sys.exit(f"--prompts: not in the input: {sorted(unknown)}")
        idx["runs"] = {sk: p for sk, p in idx["runs"].items() if sk[1] in keep}
    return idx


def layer0_gram(run_dir: Path) -> np.ndarray:
    a = np.load(run_dir / "activations.npz")["activations"][0].astype(np.float32)
    return a @ a.T


def offdiag_sorted(gram: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(gram.shape[0], k=1)
    return np.sort(gram[iu])


def mutual_pairs(run_dir: Path) -> list:
    """Per layer: (i, j, same_cluster) arrays and the stored counts."""
    c = json.loads((run_dir / "clustering.json").read_text())
    out = []
    for lr in c["layers"]:
        pa = lr.get("pair_agreement") or {}
        mp = pa.get("mutual_pairs") or []
        if not mp:
            raise ReproductionError(f"{run_dir.name} layer {lr['layer']}: no mutual pairs stored")
        out.append({
            "layer": lr["layer"],
            "i": np.array([p["i"] for p in mp]),
            "j": np.array([p["j"] for p in mp]),
            "same": np.array([p["cross_method_tag"] == "same_cluster" for p in mp]),
            "ident": np.array([p["tok_i"] == p["tok_j"] for p in mp]),
            "stored_n_ext_semantic": pa["n_ext_semantic"],
            "stored_threshold": pa["ext_sem_threshold"],
        })
    return out


def read_tokens(run_dir: Path) -> np.ndarray:
    """tokens.txt is `{index:>3}  {token}` per line (p1_io._save_tokens)."""
    out = []
    for n, line in enumerate((run_dir / "tokens.txt").read_text().split("\n")):
        if not line.strip():
            continue
        idx, _, tok = line.lstrip().partition("  ")
        if int(idx) != len(out):
            raise ReproductionError(f"{run_dir.name}: tokens.txt line {n} is index {idx}")
        out.append(tok)
    return np.array(out, dtype=object)


def non_repeat_offdiag(tokens: np.ndarray) -> tuple:
    """Upper-triangle indices of the pairs whose two tokens differ."""
    iu = np.triu_indices(len(tokens), k=1)
    keep = tokens[iu[0]] != tokens[iu[1]]
    return iu[0][keep], iu[1][keep]


def measure_non_repeat(sims: np.ndarray, sorted_nr: np.ndarray) -> dict:
    """The non-repeat mutual pairs against the non-repeat pairs of the prompt.

    Repeats sit at cosine 1 in every frame, so any cut or rank over the full
    off-diagonal distribution lands in that pile when a prompt repeats a lot
    and splits it by float rounding. Here both sides exclude them.
    """
    if len(sims) == 0:
        return {"n": 0, "frac_above": None, "frac_above_q": None, "mean_percentile": None}
    return {
        "n": int(len(sims)),
        "frac_above": {f"abs_{t:g}": float((sims > t).mean()) for t in ABS_THRESHOLDS},
        "frac_above_q": {f"q_{q:g}": float((sims > np.quantile(sorted_nr, q)).mean())
                         for q in QUANTILES},
        "mean_percentile": float((np.searchsorted(sorted_nr, sims, side="left")
                                  / len(sorted_nr)).mean()),
    }


def measure_layer(sims: np.ndarray, same: np.ndarray, sorted_offdiag: np.ndarray,
                  ident: np.ndarray = None, sorted_nr: np.ndarray = None) -> dict:
    """Fractions for one layer's mutual pairs under one frame.

    `ident` marks pairs of the same token string: at layer 0 these have cosine
    1 in any frame, since Pythia adds no position embedding before layer 0.
    `sorted_nr` is the frame's sorted cosines over non-repeat pairs.
    """
    cuts = {f"abs_{t:g}": t for t in ABS_THRESHOLDS}
    cuts.update({f"q_{q:g}": float(np.quantile(sorted_offdiag, q)) for q in QUANTILES})
    frac, same_frac = {}, {}
    for name, t in cuts.items():
        sem = sims > t
        frac[name] = float(sem.mean())
        same_frac[name] = float(same[sem].mean()) if sem.any() else None
    pct = np.searchsorted(sorted_offdiag, sims, side="left") / len(sorted_offdiag)
    out = {"ext_semantic_fraction": frac, "ext_sem_same_cluster_frac": same_frac,
           "mean_pair_percentile": float(pct.mean()), "n_pairs": int(len(sims))}
    if ident is not None:
        out["identical_token_fraction"] = float(ident.mean())
        out["identical_same_cluster_frac"] = float(same[ident].mean()) if ident.any() else None
        out["non_identical_above_stored"] = float((~ident & (sims > STORED_THRESHOLD)).mean())
        if sorted_nr is not None:
            out["non_repeat"] = measure_non_repeat(sims[~ident], sorted_nr)
    return out


def measure_run(run_dir: Path, frames: dict) -> dict:
    """`frames` maps frame name to (gram, sorted off-diagonal)."""
    layers = mutual_pairs(run_dir)
    self_gram = frames["self"][0]
    for L in layers:
        if L["stored_threshold"] != STORED_THRESHOLD:
            raise ReproductionError(f"{run_dir.name}: stored threshold {L['stored_threshold']}")
        n = int((self_gram[L["i"], L["j"]] > STORED_THRESHOLD).sum())
        if n != L["stored_n_ext_semantic"]:
            raise ReproductionError(
                f"{run_dir.name} layer {L['layer']}: rebuilt {n} ext_semantic pairs, "
                f"stored {L['stored_n_ext_semantic']}")
    tokens = read_tokens(run_dir)
    for L in layers:
        if not np.array_equal(L["ident"], tokens[L["i"]] == tokens[L["j"]]):
            raise ReproductionError(f"{run_dir.name}: tokens.txt disagrees with the stored pairs")
    ri, rj = non_repeat_offdiag(tokens)
    out = {}
    base = {}
    for fname, (gram, srt) in frames.items():
        if gram.shape[0] != len(tokens):
            raise ReproductionError(f"{run_dir.name}: frame {fname} has {gram.shape[0]} tokens")
        nr = np.sort(gram[ri, rj])
        base[fname] = {f"abs_{t:g}": float((nr > t).mean()) for t in ABS_THRESHOLDS}
        out[fname] = [dict(layer=L["layer"],
                           **measure_layer(gram[L["i"], L["j"]], L["same"], srt, L["ident"], nr))
                      for L in layers]
    out["gram_percentile_of_0.5"] = {
        fname: float(np.searchsorted(srt, STORED_THRESHOLD) / len(srt))
        for fname, (_, srt) in frames.items()}
    out["non_repeat_base_above"] = base
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def aggregate(results: dict) -> dict:
    """results: {(step, key): measure_run output}. Means over prompts x layers."""
    frames = sorted({f for r in results.values() for f, v in r.items() if isinstance(v, list)})
    steps = sorted({s for s, _ in results})
    by_step = {}
    for f in frames:
        by_step[f] = {}
        for s in steps:
            recs = [lr for (st, _), r in results.items() if st == s for lr in r[f]]
            cuts = recs[0]["ext_semantic_fraction"].keys()
            nr = [lr["non_repeat"] for lr in recs if lr.get("non_repeat", {}).get("n")]
            bases = [r["non_repeat_base_above"][f] for (st, _), r in results.items()
                     if st == s and "non_repeat_base_above" in r]
            non_repeat = {
                "n_layer_records": len(nr),
                "frac_above": {c: _mean(x["frac_above"][c] for x in nr) for c in cuts if c.startswith("abs_")},
                "frac_above_q": {c: _mean(x["frac_above_q"][c] for x in nr) for c in cuts if c.startswith("q_")},
                "mean_percentile": _mean(x["mean_percentile"] for x in nr),
                "base_above": {c: _mean(b[c] for b in bases) for c in cuts if c.startswith("abs_")},
            } if nr else None
            by_step[f][s] = {
                "n_runs": sum(1 for (st, _) in results if st == s),
                "ext_semantic_fraction": {c: _mean(lr["ext_semantic_fraction"][c] for lr in recs) for c in cuts},
                "ext_sem_same_cluster_frac": {c: _mean(lr["ext_sem_same_cluster_frac"][c] for lr in recs) for c in cuts},
                "mean_pair_percentile": _mean(lr["mean_pair_percentile"] for lr in recs),
                **{k: _mean(lr[k] for lr in recs) for k in
                   ("identical_token_fraction", "identical_same_cluster_frac",
                    "non_identical_above_stored")},
                "gram_percentile_of_0.5": _mean(r["gram_percentile_of_0.5"][f]
                                                for (st, _), r in results.items() if st == s),
                "non_repeat": non_repeat,
            }
    return {"by_step": by_step, "verdict": verdicts(by_step)}


def verdicts(by_step: dict) -> dict:
    """The criterion in the module docstring, per frame."""
    out = {}
    for f, bs in by_step.items():
        if 0 not in bs or FROZEN_STEP not in bs:
            out[f] = {"verdict": "unavailable", "why": "step 0 or 143000 missing"}
            continue
        a, b = bs[0]["ext_semantic_fraction"], bs[FROZEN_STEP]["ext_semantic_fraction"]
        decline = {c: a[c] - b[c] for c in a}
        lo, hi = SATURATION
        judged = [c for c in a if c.startswith("q_")
                  or (lo <= a[c] <= hi and lo <= b[c] <= hi)]
        if any(decline[c] <= 0 for c in a if c.startswith("q_")):
            v = "dead"
        elif all(decline[c] > 0 for c in judged):
            v = "survives"
        else:
            v = "mixed"
        out[f] = {"verdict": v, "decline": decline, "judged_cuts": judged,
                  "percentile_change": bs[0]["mean_pair_percentile"]
                  - bs[FROZEN_STEP]["mean_pair_percentile"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_input_args(ap, DATA / "phase12" / "stage0_logs" / "stage0_index.json")
    ap.add_argument("--out", default=str(DATA / "analysis" / "p10_s1_ext_sem_threshold.json"))
    add_holdout_args(ap)
    args = ap.parse_args()

    idx = load_input(args)
    kept, holdout = refuse_held_out(
        sorted(idx["runs"].values()), allow=args.allow_holdout, drop=args.v1_only,
        context="p10_ext_sem_threshold")
    kept = set(kept)
    runs = {k: v for k, v in idx["runs"].items() if v in kept}
    prompts = sorted({key for _, key in runs})
    missing_frozen = [p for p in prompts if (FROZEN_STEP, p) not in runs]
    if missing_frozen:
        sys.exit(f"no step-{FROZEN_STEP} run for {missing_frozen}: the frozen frame needs one")
    print(f"{len(runs)} runs, {len(prompts)} prompts, steps "
          f"{sorted({s for s, _ in runs})}")

    t0 = time.time()
    frozen = {}
    results = {}
    for n, (step, key) in enumerate(sorted(runs), 1):
        if key not in frozen:
            g = layer0_gram(runs[(FROZEN_STEP, key)])
            frozen[key] = (g, offdiag_sorted(g))
        g = layer0_gram(runs[(step, key)])
        results[(step, key)] = measure_run(
            runs[(step, key)], {"self": (g, offdiag_sorted(g)), "frozen": frozen[key]})
        if n % 20 == 0 or n == len(runs):
            print(f"  [{n}/{len(runs)}] step{step} {key}  ({time.time() - t0:.0f}s)", flush=True)

    inputs = sorted((f"{s}|{k}", str(p)) for (s, k), p in runs.items())
    summary = aggregate(results)
    record = {
        "schema": "p10_s1_ext_sem_threshold/1",
        "row": "Stage 1 step 1 — does the ext_semantic_fraction decline survive the threshold?",
        "tier": "1 (exploratory, unregistered, descriptive; no null)",
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "index": idx["source"],
        "index_pin": idx["pin"],
        "prompt_battery_hash": idx["prompt_battery_hash"],
        "inputs_sha256": hashlib.sha256(json.dumps(inputs).encode()).hexdigest()[:12],
        "n_runs": len(runs),
        "prompts": prompts,
        "steps": sorted({s for s, _ in runs}),
        "holdout": holdout,
        "reproduced_stored_tags_at": STORED_THRESHOLD,
        "abs_thresholds": list(ABS_THRESHOLDS),
        "quantiles": list(QUANTILES),
        "frames": {"self": "the run's own layer 0 (as stored; D6 applies)",
                   "frozen": f"layer 0 of the step-{FROZEN_STEP} run of the same prompt"},
        "criterion": "survives iff step0 - step143000 > 0 at every quantile cut and at every "
                     f"absolute cut with both endpoints in {list(SATURATION)}; dead iff <= 0 at "
                     "some quantile cut; else mixed",
        "summary": summary,
        "inputs": inputs,
        "runs": {f"{s}|{k}": r for (s, k), r in sorted(results.items())},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    print(f"\nwrote {out}  (inputs {record['inputs_sha256']})")
    for f, v in summary["verdict"].items():
        print(f"  {f}: {v['verdict']}")


if __name__ == "__main__":
    main()
