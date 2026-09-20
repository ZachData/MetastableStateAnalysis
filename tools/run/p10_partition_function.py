"""F12 of `p10_cluster_function/notes-10.md` §8: `Z_{beta,i}` per token, and the
distinction between a particle that is PARKED and one that is PINNED.

From artifacts ALREADY ON DISK — `activations.npz`'s sphere-projected states,
no forward pass and no model load.

THE DISTINCTION, AND WHY DISPLACEMENT ALONE CANNOT MAKE IT
------------------------------------------------------------
`notes-10.md` §3.1's kinematic signature is "its displacement per layer is
small". Two very different things produce that:

  **parked**  nothing is pushing the particle. It is still because the forces
              on it are weak.
  **pinned**  a great deal is pushing it, and the metric makes it expensive to
              move. It is still because the forces on it are strong and
              balanced.

H-PARK is a claim about the first. `attention-10.md` §5 names the distinction;
nothing in this project has measured it, because the quantity that separates
them — the trained per-token metric — has never been examined.

`math-1.md` §1A.6 supplies the reading:

> the partition function is not noise to be normalized away — it is a metric
> ... a high-`Z` token (a sink) is one the metric makes expensive to move.

WHAT `math-10.md` §2 CORRECTED, AND WHY THIS ROW EXISTS AT ALL
----------------------------------------------------------------
That identification is an **unmasked-model statement**, and Pythia is masked.
In the concentration regime:

    unmasked   Z_i = n e^{beta gamma}          — position-independent, so any
                                                 spread in Z is content
    masked     Z_i = (i+1) e^{beta gamma}      — LINEAR IN POSITION, and
                                                 position 0 is the MINIMUM

Meanwhile received attention DECREASES in position, so under a causal mask the
sink at position 0 is simultaneously the largest received attention and the
smallest `Z`. On the metric reading that makes it the *cheapest* token to move,
not the most expensive — the two structural baselines are anti-aligned.

So the instruction is: **measure `Z_i/(i+1)`, not `Z_i`.** What is left after
dividing out the visible-token count is the per-visible-token mean of
`exp(beta <x_i, x_j>)`, which is the quantity the metric reading was about.

WHAT THIS RUNNER REPORTS
-------------------------
Per (directory, layer, beta):

  position_r2_raw        R^2 of log Z against log(i+1). `math-10.md` §2 says
                         the raw partition function is mostly position; this
                         measures how much.
  position_r2_corrected  the same after the correction. Near 0 is the
                         correction working.
  sink_percentile_raw    where position 0 sits in the raw distribution, and
  sink_percentile_corr   where it sits after correction — the concrete form of
                         "the sink is the minimum, not the maximum".
  clustered_minus_noise  standardised mean difference in corrected log Z
                         between the clustered and unclustered populations.
                         **A DIFFERENCE, not a ratio**: log Z is signed, and an
                         enrichment ratio on a signed quantity is meaningless.

The population comparison gets a label-permutation p-value and the sweep is
merged with `core.evalues.average` — the units share a model, a text and a
forward pass, so the product is invalid over them.

BETA IS UNDECIDED AND THAT IS HANDLED, NOT IGNORED
----------------------------------------------------
`docs/AXES.md` §4 records that beta's unit convention is an open decision worth
a factor of 8, and no `beta_eff` exists in any run directory. This runner
therefore sweeps a fixed grid of betas and reports every one separately rather
than picking a number. The three headline quantities above are read per beta,
and a conclusion that holds only at one beta should be read as a conclusion
about that beta.

**Tier 1, exploratory, unregistered.** `claims/registry.json` is untouched.

Run:
    python tools/run/p10_partition_function.py --out data/analysis/p10_f12_z.json
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", "/run/media/system/WDS_500/Mets"))
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
from core.parking import (
    log_partition_function,
    log_position_corrected_partition_function,
)
from tools.run.backfill_hdbscan import labels_provenance, read_labels
from tools.run.p10_anchor import checkpoint_of

N_PERMUTATIONS = 2000

#: Fixed in the code, before any sweep is read. `docs/AXES.md` §4: beta's unit
#: convention is undecided and worth a factor of 8, so a grid is reported
#: rather than a choice made. Overridable for sensitivity work only.
BETAS = [float(b) for b in os.environ.get("METS_P10_BETAS", "1.0,2.0,4.0").split(",")]


def standardised_difference(values: np.ndarray, labels: np.ndarray) -> float:
    """
    ``(mean over clustered - mean over noise) / std over all`` on a signed
    quantity.

    A DIFFERENCE rather than a ratio, deliberately: `log Z` takes both signs,
    and `population_enrichment`'s ratio is meaningless on a quantity whose mean
    can pass through zero. Standardising by the layer's own spread makes the
    number comparable across layers and betas, which a raw difference in nats
    is not.

    ``(values, labels)`` order so it drops into `label_permutation_null`.
    Returns ``nan`` when either population is empty or the spread is zero.
    """
    v = np.asarray(values, dtype=np.float64)
    lab = np.asarray(labels)
    noise = lab == -1
    if not noise.any() or noise.all():
        return float("nan")
    sd = float(v.std())
    if not np.isfinite(sd) or sd <= 1e-12:
        return float("nan")
    return float((v[~noise].mean() - v[noise].mean()) / sd)


def position_r2(values: np.ndarray) -> float:
    """
    Fraction of the variance of `values` explained by ``log(i + 1)``.

    The masked baseline is ``Z_i = (i+1) e^{beta gamma}``, i.e. ``log Z`` is
    ``log(i+1)`` plus a constant, so this IS the "how much of it is position"
    number `math-10.md` §2 asks for. Returns ``nan`` on a degenerate input
    rather than a spurious 0 or 1.
    """
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    if n < 3:
        return float("nan")
    x = np.log(np.arange(1, n + 1, dtype=np.float64))
    vx, vv = x - x.mean(), v - v.mean()
    denom = float(np.sqrt((vx ** 2).sum() * (vv ** 2).sum()))
    if not np.isfinite(denom) or denom <= 1e-12:
        return float("nan")
    return float((float((vx * vv).sum()) / denom) ** 2)


def percentile_of(values: np.ndarray, index: int) -> float:
    """Where `values[index]` sits in `values`, in [0, 1]. 0 is the minimum."""
    v = np.asarray(values, dtype=np.float64)
    if v.size < 2 or not np.isfinite(v[index]):
        return float("nan")
    return float(np.mean(v < v[index]) + 0.5 * np.mean(v == v[index]))


def measure_layer(X: np.ndarray, labels, beta: float, rng) -> dict:
    """One layer at one beta. ``None`` when the layer cannot be scored."""
    lab = np.asarray(labels)
    n = X.shape[0]
    if lab.size != n:
        raise ValueError(f"labels ({lab.size}) and activations ({n}) disagree")
    if n < 4:
        return None

    logZ = log_partition_function(X, beta, causal=True)
    corrected = log_position_corrected_partition_function(logZ)

    out = {
        "beta": float(beta),
        "n_tokens": int(n),
        "position_r2_raw": round(float(position_r2(logZ)), 4),
        "position_r2_corrected": round(float(position_r2(corrected)), 4),
        "sink_percentile_raw": round(float(percentile_of(logZ, 0)), 4),
        "sink_percentile_corrected": round(float(percentile_of(corrected, 0)), 4),
    }

    diff = standardised_difference(corrected, lab)
    if np.isfinite(diff):
        draws = label_permutation_null(corrected, lab, standardised_difference,
                                       n_permutations=N_PERMUTATIONS, rng=rng)
        # TWO-SIDED, and fixed here: H-PARK predicts clustered particles are
        # cheap to move (lower corrected Z) while H-CAT's "quiet but
        # load-bearing" reading predicts the opposite. Both signs are
        # informative and neither was predicted, so neither may be chosen now.
        res = p_from_null_tolerant(diff, draws, alternative="two-sided")
        out["clustered_minus_noise"] = round(float(diff), 4)
        out["clustered_minus_noise_p"] = float(res["p_value"])
        out["degenerate"] = bool(res["degenerate_null"])
    else:
        out["clustered_minus_noise"] = None
        out["clustered_minus_noise_p"] = None
        out["degenerate"] = None
    return out


def measure_directory(run_dir: Path, rng) -> dict:
    labels_by_layer = read_labels(run_dir)
    if not labels_by_layer:
        return {"run_dir": run_dir.name, "skipped": "no HDBSCAN partition"}
    p = run_dir / "activations.npz"
    if not p.exists():
        return {"run_dir": run_dir.name, "skipped": "no activations.npz"}
    acts = np.load(p)["activations"]

    rows = []
    for layer in sorted(labels_by_layer):
        if layer >= acts.shape[0]:
            continue
        X = np.asarray(acts[layer], dtype=np.float64)
        for beta in BETAS:
            rec = measure_layer(X, labels_by_layer[layer], beta, rng)
            if rec is not None:
                rec["layer"] = int(layer)
                rows.append(rec)

    return {
        "run_dir": run_dir.name,
        "timestamp": run_dir.parent.name,
        "checkpoint": checkpoint_of(run_dir.name),
        "labels_from": labels_provenance(run_dir),
        "rows": rows,
    }


def _summarise(rows: list) -> dict:
    def m(key):
        v = np.array([r[key] for r in rows
                      if r.get(key) is not None and np.isfinite(r[key])])
        return round(float(v.mean()), 4) if v.size else None

    ps = [r["clustered_minus_noise_p"] for r in rows
          if r.get("clustered_minus_noise_p") is not None]
    out = {
        "n_units": len(rows),
        "mean_position_r2_raw": m("position_r2_raw"),
        "mean_position_r2_corrected": m("position_r2_corrected"),
        "mean_sink_percentile_raw": m("sink_percentile_raw"),
        "mean_sink_percentile_corrected": m("sink_percentile_corrected"),
        "mean_clustered_minus_noise": m("clustered_minus_noise"),
    }
    if ps:
        E, reject = average_p(ps)
        out["clustered_minus_noise"] = {
            "n": len(ps), "E": round(float(E), 4), "reject": bool(reject),
            "median_p": round(float(np.median(ps)), 4),
            "frac_below_05": round(float(np.mean(np.array(ps) < 0.05)), 4),
        }
    return out


def aggregate(dirs: list) -> dict:
    rows = [(d, r) for d in dirs for r in d.get("rows", [])]
    if not rows:
        return {"n_units": 0}

    out = {"n_directories": sum(1 for d in dirs if d.get("rows"))}
    out.update(_summarise([r for _, r in rows]))

    by_beta = defaultdict(list)
    for _, r in rows:
        by_beta[r["beta"]].append(r)
    out["by_beta"] = {str(b): _summarise(rs) for b, rs in sorted(by_beta.items())}

    by_ckpt = defaultdict(list)
    for d, r in rows:
        if d.get("checkpoint") is not None:
            by_ckpt[d["checkpoint"]].append(r)
    out["by_checkpoint"] = {
        str(s): _summarise(rs) for s, rs in sorted(by_ckpt.items())
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(DATA / "phase12"))
    ap.add_argument("--pattern", default="pythia-410m-*")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DATA / "analysis" / "p10_f12_z.json"))
    args = ap.parse_args()

    root = Path(args.root)
    targets = sorted(
        d for ts in sorted(root.glob("*")) if ts.is_dir()
        for d in sorted(ts.glob(args.pattern)) if d.is_dir()
        and (d / "activations.npz").exists() and read_labels(d)
    )
    if args.limit:
        targets = targets[: args.limit]
    print(f"{len(targets)} directories, betas {BETAS}")

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    dirs = []
    for i, d in enumerate(targets, 1):
        dirs.append(measure_directory(d, rng))
        if i % 25 == 0 or i == len(targets):
            print(f"  [{i}/{len(targets)}] {d.name}  ({time.time() - t0:.0f}s)")

    summary = aggregate(dirs)
    record = {
        "schema": "p10_f12_z/1",
        "row": "F12 — Z_beta,i per token: parked versus pinned",
        "tier": "1 (exploratory, unregistered)",
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(root),
        "betas": BETAS,
        "n_permutations": N_PERMUTATIONS,
        "seed": args.seed,
        "kappa": DEFAULT_KAPPA,
        "alpha": DEFAULT_ALPHA,
        "merger": "arithmetic mean (core.evalues.average)",
        "resolution_floor_e": round(calibrate(1.0 / (N_PERMUTATIONS + 1)), 3),
        # "Could this design have rejected at all?" -- a different question
        # from the resolution floor, and one a permutation null merged by the
        # mean can answer exactly. See `core.evalues.max_attainable_average_E`.
        "max_attainable_E": round(max_attainable_average_E(N_PERMUTATIONS)[0], 3),
        "design_can_reject": bool(max_attainable_average_E(N_PERMUTATIONS)[1]),
        "alternatives": {"clustered_minus_noise_p": "two-sided"},
        "summary": summary,
        "directories": dirs,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    print(f"\nwrote {out}")
    for k, v in summary.items():
        if k not in ("by_beta", "by_checkpoint"):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
