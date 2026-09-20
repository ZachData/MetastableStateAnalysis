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

Run:
    python tools/run/p10_attention_baseline.py --limit 8
    python tools/run/p10_attention_baseline.py --out data/analysis/p10_row_a0.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", "/run/media/system/WDS_500/Mets"))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))

import numpy as np

from core.evalues import DEFAULT_ALPHA, DEFAULT_KAPPA, average_p, calibrate
from core.nulls import label_permutation_null, p_from_null_tolerant
from core.parking import (
    clustered_position_bias,
    mask_corrected_received,
    population_enrichment,
    received_attention,
    relative_to_layer_mean,
)
from tools.run.backfill_hdbscan import labels_provenance, read_labels

#: Permutations per (directory, layer). 400 gives a resolution floor of
#: 1/401 = 0.0025, which calibrates to an e-value of 9.98 -- below 1/alpha = 20,
#: so NO SINGLE UNIT CAN REJECT ON ITS OWN at this draw count. That is a
#: deliberate property, not an oversight: the question is whether the effect is
#: there across the sweep, and a design where one layer can carry the verdict
#: is a design that rewards finding the one layer.
N_PERMUTATIONS = 400


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

    out = {
        "n_tokens": int(n),
        "noise_fraction": round(float(noise.mean()), 4),
        "raw_noise": round(float(noise_enrichment(raw, labels)), 4),
        "raw_clustered": round(float(clustered_enrichment(raw, labels)), 4),
        "corrected_noise": round(float(corrected_noise := noise_enrichment(corrected, labels)), 4),
        "corrected_clustered": round(float(clustered_enrichment(corrected, labels)), 4),
        "position_bias": round(float(bias := clustered_position_bias(positions, labels)), 4),
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
    out["raw_p"] = float(p_from_null_tolerant(out["raw_noise"], raw_draws,
                                              alternative="greater")["p_value"])

    # The confound, two-sided.
    bias_draws = label_permutation_null(positions, labels, clustered_position_bias,
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
        "labels_from": labels_provenance(run_dir),
        "n_heads": int(attn.shape[1]),
        "layers": layers,
    }


def aggregate(dirs: list) -> dict:
    """Sweep-level readout. Averages the enrichments, and merges the p-values
    with the dependence-robust merger."""
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
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(DATA / "phase12"))
    ap.add_argument("--pattern", default="pythia-410m-*")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DATA / "analysis" / "p10_row_a0.json"))
    args = ap.parse_args()

    root = Path(args.root)
    targets = sorted(
        d for ts in sorted(root.glob("*")) if ts.is_dir()
        for d in sorted(ts.glob(args.pattern)) if d.is_dir()
        and (d / "attentions.npz").exists() and read_labels(d)
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
            print(f"  [{i}/{len(targets)}] {d.name}  ({time.time() - t0:.0f}s)")

    record = {
        "schema": "p10_row_a0/1",
        "row": "A0 — the causal-mask baseline, divided out",
        "tier": "1 (exploratory, unregistered)",
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(root),
        "n_permutations": N_PERMUTATIONS,
        "seed": args.seed,
        "kappa": DEFAULT_KAPPA,
        "alpha": DEFAULT_ALPHA,
        "merger": "arithmetic mean (core.evalues.average) — the units share a "
                  "model, a text and a forward pass, so the product is invalid",
        "resolution_floor_e": round(calibrate(1.0 / (N_PERMUTATIONS + 1)), 3),
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
