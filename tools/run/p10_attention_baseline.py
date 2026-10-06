"""Row A0 of `p10_cluster_function/attention-10.md`: divide the causal mask's
structural tilt out of the attention flip, and see what is left.

From artifacts ALREADY ON DISK — no forward pass, no model load. Reads
`attentions.npz` (present in 152/152 directories of the 410m sweep, and
`docs/AXES.md` calls it the most under-exploited artifact in the tree) and the
HDBSCAN partition, and reports the enrichment ratio both ways.

THE CLAIM UNDER TEST, AND WHY IT IS NOT YET KNOWN TO BE ONE
-----------------------------------------------------------
This project reports that trained models route ~1.6x layer-average attention to
unclustered tokens and ~0.5x to clustered ones, with the sign flipped under
random weights (`archive/p5c_unclustered/status-5c.md`, cited by
`PREDICTIONS.md` claim (a)). The statistic is

    received[key] = sum over heads, sum over queries of attn[h, query, key]
    reported as   received[population].mean() / received.mean()

with the diagonal zeroed and NOTHING ELSE divided out.

`math-10.md` §1 derives what that reports when the network does nothing. Under
a causal mask token `j` is visible to only `n - j` queries, so content-free

    received(j) = H_n - H_j,   layer mean exactly 1

which at the battery's n = 264 is 6.155x at position 0 and 0.0038x at the last
token — **a ~1 600x tilt before content enters**, crossing 1.6x at position ~53
and 0.5x at ~160. A partition whose unclustered members average early and whose
clustered members average late reproduces the observed flip exactly, with no
learned behaviour anywhere.

So this runner reports three things per layer, and the third is the one that
means something:

  raw_*         the flip as this project has always computed it
  corrected_*   the same enrichment on received / baseline, mean-renormalised
  position_bias mean normalised position of clustered minus noise tokens —
                the confound itself, measured rather than argued about

E-VALUES, AND WHY THE MERGER IS THE AVERAGE
--------------------------------------------
Each (checkpoint, prompt, layer) gets a permutation p-value against the null
"the population labels are independent of where the attention goes", which
holds the received vector fixed and permutes labels — preserving cluster count
and every cluster size exactly.

Those units are NOT independent: they share a model, a text and a forward pass,
and a layer's attention is not conditionally calibrated given the layer below
it. `core.evalues.combine` (the product) is therefore invalid over them, and
using it here would manufacture an enormous E from a sweep — measured in
`tests/test_core_evalues_average.py`, 19.67 % Type-I at 25 dependent units
against a nominal 5 %. `core.evalues.average` is the merger that survives
arbitrary dependence, and it is what this runner reports.

**Tier 1, exploratory, no registration** (`notes-10.md` §11). The e-value is
reported because a number with a null behind it should be calibrated as one,
not because anything here is registered. `claims/registry.json` is untouched.

DIRECTIONS ARE FIXED HERE, IN THE CODE, BEFORE ANY SWEEP IS READ
-----------------------------------------------------------------
`core.nulls.p_from_null` states that choosing `alternative` after seeing the
data is a one-bit selection that voids the guarantee. The noise-enrichment
alternative is "greater", because that is what the existing claim says and it
was said before this runner existed. `position_bias` is two-sided: both signs
are informative and neither was predicted.

THE RE-READ (`p10_cluster_function/design-10.md`, R3; added 2026-10-05)
----------------------------------------------------------------------
``--labels <dir>`` reads columns of the R0 label source (``--columns``, default
all), or the run refuses; ``--old-partition`` reads the stored labels (the
published record). Each (step, prompt) loads its attention once and reads every
column on it:

- **c0, c0f** run `measure_layer` unchanged: every position, the published
  statistic, so c0 is the old reading on the re-read's inputs.
- **c1 onward apply T4** (`design-1d.md` "Token rules"): the columns of the
  prompt's T1–T2 positions (position 0 and the massive tokens of the token set
  the label source names, hash checked) are dropped and rows renormalised
  (`core.parking.t4_attention`), and the mask baseline is the content-free one
  under the same rule (`t4_received_baseline`). The enrichments, the position
  bias and their nulls are read on the column's **domain** (the kept tokens):
  members against the rest, permuted among the domain. T3 duplicates stay in
  the matrix as queries and keys (T4 drops T1–T2 only) and are outside the means.

Each unit draws from its own generator (``unit_rng``, keyed by step, prompt and
layer, the same in every column). The labels (mask share, where the residual
appears, whether it persists) are `p10_r3_ladder.py`'s.

Run:
    python tools/run/p10_attention_baseline.py --old-partition --out data/analysis/p10_row_a0.json
    python tools/run/p10_attention_baseline.py --labels <R0 labels> --out <file> --jobs 14
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

from core.evalues import (
    DEFAULT_ALPHA,
    DEFAULT_KAPPA,
    average_p,
    calibrate,
    max_attainable_average_E,
)
from core.nulls import label_permutation_null, p_from_null_tolerant
from core.holdout import add_holdout_args, refuse_held_out
from core.parking import (
    mask_corrected_received,
    population_enrichment,
    received_attention,
    relative_to_layer_mean,
    t4_attention,
    t4_received_baseline,
)
from tools.run.backfill_hdbscan import labels_provenance, read_labels
from tools.run.p10_anchor import checkpoint_of

#: Permutations per (directory, layer). Set by
#: `core.evalues.max_attainable_average_E`, not by taste: the mean merger
#: cannot exceed its largest input and a Monte-Carlo p cannot go below
#: 1/(n+1), so the largest merged e-value this design can EVER produce is
#: `calibrate(1/(n+1))`. At 400 draws that is 10.01 against a rejection
#: threshold of 20 -- a design that could not have rejected if every unit in
#: the sweep had come back maximally extreme, and the first full run of this
#: row reported `reject: False` at all 19 checkpoints from exactly that.
#: 1 599 is the minimum that clears it; 2 000 leaves margin.
#:
#: One lucky layer still cannot carry the verdict -- that is the merger's job,
#: not the draw count's, since the mean of 3 646 units holding one at the
#: ceiling is the ceiling over 3 646.
N_PERMUTATIONS = 2000


def noise_enrichment(values: np.ndarray, labels: np.ndarray) -> float:
    """`population_enrichment` on the unclustered population, in the
    ``(values, labels)`` shape `label_permutation_null` calls."""
    return population_enrichment(values, np.asarray(labels) == -1)


def clustered_enrichment(values: np.ndarray, labels: np.ndarray) -> float:
    return population_enrichment(values, np.asarray(labels) != -1)


def _load_attentions(run_dir: Path):
    p = run_dir / "attentions.npz"
    if not p.exists():
        return None
    z = np.load(p)
    key = "attentions" if "attentions" in z.files else z.files[0]
    return z[key]


def measure_layer(attn_layer, labels, rng) -> dict:
    """Raw and corrected enrichments for one layer, each with its own
    permutation p-value.

    Returns ``None`` when the layer has no usable split (all noise or no
    noise), which is a real outcome and must not be counted as a measured 1.0.
    """
    labels = np.asarray(labels)
    noise = labels == -1
    if not noise.any() or noise.all():
        return None

    n = attn_layer.shape[-1]
    if labels.size != n:
        raise ValueError(f"labels ({labels.size}) and attention ({n}) disagree")

    raw = relative_to_layer_mean(received_attention(attn_layer, zero_diagonal=True))
    corrected = mask_corrected_received(attn_layer)
    positions = np.arange(n, dtype=np.float64)
    return score_populations(raw, corrected, positions, labels, n - 1, rng)


def score_populations(raw, corrected, positions, labels, scale: float, rng) -> dict:
    """The enrichments, the position bias and their three p-values on one set of
    tokens (every position for the published reading; a column's domain for the
    re-read). ``positions`` are the tokens' positions and ``scale`` the length
    they are normalised by (n − 1 of the whole prompt), so a domain's bias stays
    in the published units. Draw order is the published one."""
    labels = np.asarray(labels)
    noise = labels == -1
    n = labels.size

    def bias_of(pos, lab):
        # `clustered_position_bias`'s formula, with the prompt's own scale
        pos, nz = np.asarray(pos, dtype=np.float64), np.asarray(lab) == -1
        return float(pos[~nz].mean() / scale - pos[nz].mean() / scale)

    # Held UNROUNDED, and the rounding happens only on the way into the record.
    # Testing a statistic rounded to 4 dp against unrounded null draws puts a
    # 5e-05 error on the comparison -- about fifty thousand times the tie
    # tolerance -- so a draw within that of the true value lands on whichever
    # side the rounding sent it. Small, and wrong for no reason.
    raw_noise = noise_enrichment(raw, labels)
    corrected_noise = noise_enrichment(corrected, labels)
    bias = bias_of(positions, labels)

    out = {
        "n_tokens": int(n),
        "noise_fraction": round(float(noise.mean()), 4),
        "raw_noise": round(float(raw_noise), 4),
        "raw_clustered": round(float(clustered_enrichment(raw, labels)), 4),
        "corrected_noise": round(float(corrected_noise), 4),
        "corrected_clustered": round(float(clustered_enrichment(corrected, labels)), 4),
        "position_bias": round(float(bias), 4),
    }

    # The flip's own direction, on the corrected statistic. Fixed above.
    draws = label_permutation_null(corrected, labels, noise_enrichment,
                                   n_permutations=N_PERMUTATIONS, rng=rng)
    # TIE-TOLERANT, and it is not a nicety here. Where the mask correction
    # explains a layer completely the corrected value is the same at every
    # token, so no permutation can move the enrichment and the honest p is 1.
    # The exact comparison returned the RESOLUTION FLOOR on that case -- a
    # perfectly explained layer reading as the strongest possible evidence --
    # because ~1.0 differs from ~1.0 in the sixteenth digit. See
    # `core.nulls.p_from_null_tolerant`.
    res = p_from_null_tolerant(corrected_noise, draws, alternative="greater")
    out["corrected_p"] = float(res["p_value"])
    out["corrected_at_floor"] = bool(res["at_resolution_floor"])
    out["corrected_degenerate"] = bool(res["degenerate_null"])

    # And the same for the raw statistic, so the two are comparable on one axis.
    raw_draws = label_permutation_null(raw, labels, noise_enrichment,
                                       n_permutations=N_PERMUTATIONS, rng=rng)
    out["raw_p"] = float(p_from_null_tolerant(raw_noise, raw_draws,
                                              alternative="greater")["p_value"])

    # The confound, two-sided.
    bias_draws = label_permutation_null(positions, labels, bias_of,
                                        n_permutations=N_PERMUTATIONS, rng=rng)
    out["position_bias_p"] = float(p_from_null_tolerant(
        bias, bias_draws, alternative="two-sided")["p_value"])
    return out


def measure_directory(run_dir: Path, rng) -> dict:
    labels_by_layer = read_labels(run_dir)
    if not labels_by_layer:
        return {"run_dir": run_dir.name, "skipped": "no HDBSCAN partition"}
    attn = _load_attentions(run_dir)
    if attn is None:
        return {"run_dir": run_dir.name, "skipped": "no attentions.npz"}

    layers = []
    for layer in sorted(labels_by_layer):
        # PAIRING, and it matters. `activations.npz` has 25 rows (embedding +
        # 24 blocks) while `attentions.npz` has 24 (one per block), so a
        # partition layer and an attention index are not the same number and
        # there are two defensible pairings. This takes attn[l] — the
        # attention that READS the layer-l state — because that is the
        # direction the question runs (the state exists, then attention routes
        # over it) and because it is exactly what `noise_importance_proxy`
        # does. Row A0's whole job is to be comparable with the flip as
        # reported, so it must not quietly re-pair the axes. The last
        # partition layer has no block above it and is dropped.
        if layer >= attn.shape[0]:
            continue
        rec = measure_layer(attn[layer], labels_by_layer[layer], rng)
        if rec is not None:
            rec["layer"] = int(layer)
            layers.append(rec)

    return {
        "run_dir": run_dir.name,
        "timestamp": run_dir.parent.name,
        "checkpoint": checkpoint_of(run_dir.name),
        "labels_from": labels_provenance(run_dir),
        "n_heads": int(attn.shape[1]),
        "layers": layers,
    }


def aggregate(dirs: list) -> dict:
    """Sweep-level readout. Averages the enrichments, and merges the p-values
    with the dependence-robust merger.

    The per-checkpoint split is not decoration. Averaging over training reads a
    developmental effect as no effect, and "when" is the axis this project
    exists for.
    """
    rows = [l for d in dirs for l in d.get("layers", [])]
    if not rows:
        return {"n_units": 0}

    def m(key):
        v = np.array([r[key] for r in rows if np.isfinite(r[key])], dtype=np.float64)
        return round(float(v.mean()), 4) if v.size else None

    out = {
        "n_units": len(rows),
        "n_directories": sum(1 for d in dirs if d.get("layers")),
        "mean_raw_noise": m("raw_noise"),
        "mean_raw_clustered": m("raw_clustered"),
        "mean_corrected_noise": m("corrected_noise"),
        "mean_corrected_clustered": m("corrected_clustered"),
        "mean_position_bias": m("position_bias"),
        "mean_noise_fraction": m("noise_fraction"),
    }
    for name, key in (("raw", "raw_p"), ("corrected", "corrected_p"),
                      ("position_bias", "position_bias_p")):
        ps = [r[key] for r in rows if np.isfinite(r[key])]
        E, reject = average_p(ps)
        out[f"{name}_E"] = round(float(E), 4)
        out[f"{name}_reject"] = bool(reject)
        out[f"{name}_median_p"] = round(float(np.median(ps)), 4)
        out[f"{name}_frac_p_below_05"] = round(float(np.mean(np.array(ps) < 0.05)), 4)

    by_ckpt = defaultdict(list)
    for d in dirs:
        if d.get("checkpoint") is not None:
            by_ckpt[d["checkpoint"]].extend(d.get("layers", []))
    out["by_checkpoint"] = {}
    for step, ls in sorted(by_ckpt.items()):
        if not ls:
            continue
        def mm(key, ls=ls):
            v = np.array([l[key] for l in ls if np.isfinite(l[key])])
            return round(float(v.mean()), 4) if v.size else None
        raw_gap = mm("raw_noise") - mm("raw_clustered")
        corr_gap = mm("corrected_noise") - mm("corrected_clustered")
        E, reject = average_p([l["corrected_p"] for l in ls
                               if np.isfinite(l["corrected_p"])])
        out["by_checkpoint"][str(step)] = {
            "n_units": len(ls),
            "raw_noise": mm("raw_noise"),
            "raw_clustered": mm("raw_clustered"),
            "corrected_noise": mm("corrected_noise"),
            "corrected_clustered": mm("corrected_clustered"),
            "raw_gap": round(float(raw_gap), 4),
            "corrected_gap": round(float(corr_gap), 4),
            "gap_surviving_correction": round(float(corr_gap / raw_gap), 4)
            if abs(raw_gap) > 1e-9 else None,
            "position_bias": mm("position_bias"),
            "corrected_E": round(float(E), 4),
            "corrected_reject": bool(reject),
        }
    return out


# ---------------------------------------------------------------------------
# The re-read (R3): columns of the R0 label source (docstring, "THE RE-READ")
# ---------------------------------------------------------------------------

#: Columns read by the published reader, unchanged: `p10_label_source.ALL_POSITIONS` (not imported
#: here, which would pull sklearn into the pure tier; `test_old_reader_columns_are_the_sources`).
OLD_READER = ("c0", "c0f")


def t4_vectors(attn_layer, dropped) -> tuple:
    """Raw received (diagonal zeroed, as published) and mask-corrected received
    (diagonal kept, over `t4_received_baseline`) after T4; ``nan`` at ``dropped``."""
    a4 = t4_attention(attn_layer, dropped)
    n_heads, n = a4.shape[0], a4.shape[-1]
    raw = received_attention(a4, zero_diagonal=True)
    raw[list(dropped)] = np.nan
    corrected = received_attention(a4, zero_diagonal=False) / t4_received_baseline(n, dropped, n_heads)
    return raw, corrected


def score_domain(raw, corrected, labels, rng) -> dict:
    """`score_populations` on a column's domain (``OUTSIDE`` dropped); refuses a
    domain that holds a T4-dropped position."""
    from tools.run.p10_token_composition import OUTSIDE  # the label source's; this import stays pure
    lab = np.asarray(labels)
    dom = lab != OUTSIDE
    if not (np.isfinite(raw[dom]).all() and np.isfinite(corrected[dom]).all()):
        raise ValueError("a T1–T2 position is inside the column's domain")
    out = score_populations(raw[dom], corrected[dom], np.flatnonzero(dom), lab[dom], lab.size - 1, rng)
    out["n_positions"] = int(lab.size)
    return out


def reread_run(run_dir, labels_by_col: dict, dropped, step: int, prompt: str, seed: int) -> dict:
    """One (step, prompt): ``{column: [unit, ...]}``, every column read on one load of
    the attention. Layer ``l`` reads ``attn[l]`` as published (L24 has none)."""
    from tools.run.p10_partition_function import unit_rng
    attn = _load_attentions(Path(run_dir))
    if attn is None:
        raise FileNotFoundError(f"{run_dir}: no attentions.npz")
    out = {c: [] for c in labels_by_col}
    for layer in sorted({L for by in labels_by_col.values() for L in by}):
        if layer >= attn.shape[0]:
            continue
        A = attn[layer]
        t4 = None
        for col, by in labels_by_col.items():
            if layer not in by:
                continue
            lab = np.asarray(by[layer])
            if lab.size != A.shape[-1]:
                raise ValueError(f"{run_dir}: layer {layer} has {lab.size} labels, {A.shape[-1]} tokens")
            rng = unit_rng(seed, step, prompt, layer)
            if col in OLD_READER:
                rec = measure_layer(A, lab, rng)
            else:
                t4 = t4 if t4 is not None else t4_vectors(A, dropped)
                rec = score_domain(*t4, lab, rng)
            if rec is not None:
                rec["layer"] = int(layer)
                out[col].append(rec)
    return out


def t1_t2_positions(labels_dir: Path) -> tuple:
    """Per prompt, the positions T4 drops: 0 and the token set's massive tokens. The token
    set is the one every step file of the label source names (path and hash checked), and
    its kept positions must be the source's at every step."""
    from tools.run.p10_label_source import MODELS
    ts, kept = None, {}
    for s in MODELS:
        f = Path(labels_dir) / f"{s}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        t = d["meta"]["token_sets"]
        if ts is not None and t != ts:
            raise SystemExit(f"refusing: {f.name} names another token set ({t}) than {ts}")
        ts = t
        for k, p in d["prompts"].items():
            kept.setdefault(k, set()).add(tuple(p["kept"]))
    if ts is None:
        raise SystemExit(f"refusing: no label source step files in {labels_dir}")
    raw = Path(ts["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest()[:16] != ts["sha256"]:
        raise SystemExit(f"refusing: {ts['path']} does not match the label source's hash {ts['sha256']}")
    sets = json.loads(raw)["sets"]
    dropped = {}
    for k, ks in kept.items():
        if ks != {tuple(sets[k]["kept"])}:
            raise SystemExit(f"refusing: {k}'s kept positions differ from the token set's")
        d = sorted({0} | {int(m["position"]) for m in sets[k]["massive"]})
        if set(d) & set(sets[k]["kept"]):
            raise SystemExit(f"refusing: {k}: a T1–T2 position {d} is kept")
        dropped[k] = d
    return dropped, ts


def by_step(runs: dict) -> dict:
    """`aggregate` over one column's units, per step and over the sweep."""
    dirs = [{"checkpoint": int(k.split("|")[0]), "layers": rows} for k, rows in runs.items()]
    return aggregate(dirs)


def reread(args) -> dict:
    from concurrent.futures import ProcessPoolExecutor
    from tools.run.p10_label_source import LEARNED_SPLIT, LEARNED_STEP, MODELS, reader_input
    dropped, ts = t1_t2_positions(args.labels)
    srcs, records = {}, {}
    for col in args.columns:
        src = reader_input(args.labels, col)
        want = {LEARNED_STEP} if col in LEARNED_SPLIT else set(MODELS)
        got = {f"step{s}" for s in src["records"]}
        if got != want:
            raise SystemExit(f"refusing: {args.labels} column {col} has steps {sorted(got ^ want)} "
                             f"missing or extra; the re-read reads all {len(want)}")
        srcs[col] = src
        records[col] = {s: {"n": r[0], "readable": r[1]} for s, r in src["records"].items()}
    runs = {k: p for src in srcs.values() for k, p in src["runs"].items()}
    for col, src in srcs.items():
        if any(runs[k] != p for k, p in src["runs"].items()):
            raise SystemExit(f"refusing: column {col} names another run dir for a (step, prompt)")
    refuse_held_out(sorted(set(runs.values())), context="p10_attention_baseline")
    keys = sorted(k for k in runs if any(srcs[c]["labels"].get(k) for c in srcs))
    jobs = [(runs[k], {c: srcs[c]["labels"][k] for c in srcs if srcs[c]["labels"].get(k)},
             dropped[k[1]], k[0], k[1], args.seed) for k in keys]
    print(f"{len(jobs)} (step, prompt) runs, columns {list(srcs)}", flush=True)
    t0 = time.time()
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            out = list(ex.map(reread_run, *zip(*jobs)))
    else:
        out = [reread_run(*j) for j in jobs]
    print(f"read in {time.time() - t0:.0f}s", flush=True)
    per_col = {c: {f"{s}|{p}": res[c] for (s, p), res in zip(keys, out) if c in res} for c in srcs}
    inputs = sorted((f"{s}|{k}", str(p)) for (s, k), p in runs.items())
    first = next(iter(srcs.values()))["meta"]
    return {
        "schema": "p10_row_a0_reread/1",
        "row": "A0 re-read on R0's label source: c0, c0f as published; c1 onward under T4",
        "tier": "1 (exploratory, unregistered)",
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label_source": {"labels": first["labels"], "summary_sha256": first["summary_sha256"],
                         "label_source_git": first["label_source_git"], "readable_min": first["readable_min"]},
        "token_sets": ts, "t4_dropped": dropped,
        "inputs": inputs, "inputs_sha256": hashlib.sha256(json.dumps(inputs).encode()).hexdigest()[:12],
        "seed": args.seed, "n_permutations": N_PERMUTATIONS,
        "alternatives": {"corrected_p": "greater", "raw_p": "greater", "position_bias_p": "two-sided"},
        "columns": {c: {"records_readable": records[c], "summary": by_step(per_col[c]), "runs": per_col[c]}
                    for c in srcs},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(DATA / "phase12"))
    ap.add_argument("--pattern", default="pythia-410m-*")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DATA / "analysis" / "p10_row_a0.json"))
    ap.add_argument("--old-partition", action="store_true",
                    help="read the stored HDBSCAN labels (the published record), not the re-read")
    ap.add_argument("--labels", type=Path, default=None,
                    help="the R0 label source (`p10_label_source build`'s --out): the re-read's only input")
    ap.add_argument("--columns", nargs="+", default=None,
                    help="re-read only: the ladder columns to read (default all)")
    ap.add_argument("--jobs", type=int, default=1, help="re-read only: runs in parallel")
    add_holdout_args(ap)
    args = ap.parse_args()
    if args.old_partition == (args.labels is not None):
        ap.error("refusing: read one label-source --labels <dir> (p10_cluster_function/design-10.md), "
                 "or --old-partition for the stored labels")
    if args.old_partition and (args.columns or args.jobs != 1):
        ap.error("--columns and --jobs are the re-read's; drop them with --old-partition")
    if args.labels is not None:
        from tools.run.p10_label_source import READER_COLUMNS
        args.columns = args.columns or list(READER_COLUMNS)
        bad = set(args.columns) - set(READER_COLUMNS)
        if bad:
            ap.error(f"unknown columns {sorted(bad)}; one of {READER_COLUMNS}")
        if args.root != ap.get_default("root") or args.pattern != ap.get_default("pattern") or args.limit \
                or args.allow_holdout or args.v1_only or args.out == ap.get_default("out"):
            ap.error("--labels fixes the input set (7 v1 passages, 18 steps): no --root, --pattern, --limit, "
                     "--allow-holdout or --v1-only; --out must name a new file")
        record = reread(args)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=1))
        print(f"wrote {out}")
        return

    root = Path(args.root)
    candidates, holdout = refuse_held_out(
        (d for ts in sorted(root.glob("*")) if ts.is_dir()
         for d in sorted(ts.glob(args.pattern)) if d.is_dir()),
        allow=args.allow_holdout, drop=args.v1_only, context="p10_attention_baseline")
    targets = sorted(
        d for d in candidates
        if (d / "attentions.npz").exists() and read_labels(d)
    )
    if args.limit:
        targets = targets[: args.limit]
    print(f"{len(targets)} directories with both attention and a partition")

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    dirs = []
    for i, d in enumerate(targets, 1):
        dirs.append(measure_directory(d, rng))
        if i % 10 == 0 or i == len(targets):
            print(f"  [{i}/{len(targets)}] {d.name}  ({time.time() - t0:.0f}s)", flush=True)

    record = {
        "schema": "p10_row_a0/1",
        "row": "A0 — the causal-mask baseline, divided out",
        "tier": "1 (exploratory, unregistered)",
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(root),
        "holdout": holdout,
        "n_permutations": N_PERMUTATIONS,
        "seed": args.seed,
        "kappa": DEFAULT_KAPPA,
        "alpha": DEFAULT_ALPHA,
        "merger": "arithmetic mean (core.evalues.average) — the units share a "
                  "model, a text and a forward pass, so the product is invalid",
        "resolution_floor_e": round(calibrate(1.0 / (N_PERMUTATIONS + 1)), 3),
        # "Could this design have rejected at all?" -- a different question
        # from the resolution floor, and one a permutation null merged by the
        # mean can answer exactly. See `core.evalues.max_attainable_average_E`.
        "max_attainable_E": round(max_attainable_average_E(N_PERMUTATIONS)[0], 3),
        "design_can_reject": bool(max_attainable_average_E(N_PERMUTATIONS)[1]),
        "alternatives": {"corrected_p": "greater", "raw_p": "greater",
                         "position_bias_p": "two-sided"},
        "summary": aggregate(dirs),
        "directories": dirs,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    print(f"\nwrote {out}")
    for k, v in record["summary"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
