"""How reproducible is the HDBSCAN partition, run to run, on the same model and
the same prompt?

Not a row of `notes-10.md` §8's ladder. A **prerequisite** to reading four of
them, and to reading `CLAIM-C`'s gate, because everything those instruments say
about clusters is downstream of a partition nobody has measured the stability
of.

WHAT MAKES THIS MEASURABLE AT ALL
----------------------------------
Two independent Phase-1 sweeps exist over the same checkpoints and the same
prompts: the 2026-08-12 pilot on `HDD_1TB` (which carries native
`hdbscan_labels.json`) and the 2026-08-31/09-01 sweep in `data/phase12` (whose
partition `tools/run/backfill_hdbscan.py` re-derived). 56 model-prompt
directories overlap.

Their `tokens.txt` are identical and their configs match. Their activations are
NOT bit-identical -- they differ by 2e-07 to 6e-05, mean 3e-08 to 3e-07, which
is ordinary run-to-run float non-determinism and not a difference in what was
computed. So the pair is an almost perfect natural experiment: **the same
question asked twice, differing only by numerical noise.**

WHAT IT MEASURES
-----------------
Per (directory, layer): the adjusted Rand index between the two partitions,
both with HDBSCAN's noise treated as one cluster and with those points dropped
(`notes-10.md` §4.4 names that choice as a hazard, and it moves the answer
here), plus the difference in cluster count and noise fraction.

THIS IS A NOISE FLOOR, NOT A NULL
----------------------------------
There is no hypothesis under test and no p-value here, deliberately. The output
is a **measurement-reproducibility floor**: the amount by which a
partition-derived quantity can differ between two runs that asked the same
question. Any claim that two partitions differ has to clear it, and no null
this project has built accounts for it -- `notes-10.md` §4.4's size-profile
null is about the ARI's variance under random labelling, which is a different
quantity from the ARI's variance under re-running the same measurement.

WHAT IT BEARS ON, CONCRETELY
-----------------------------
`CLAIM-C` reads `cluster_count` and `cluster_membership` from HDBSCAN and from
nothing else (`replication_gate.py`), and its gate compares arms. If re-running
one arm can move those metrics by more than the gate's arms differ, part of a
concordance is run-to-run noise. This runner does not adjudicate that; it
supplies the number that would let someone check.

Run:
    python tools/run/p10_partition_stability.py --out data/analysis/p10_partition_stability.json
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

from core.functional_distance import adjusted_rand_index
from tools.run.backfill_hdbscan import read_labels
from tools.run.p10_anchor import checkpoint_of

#: The sweep that carries NATIVE labels, written while `hdbscan` was installed.
PILOT = Path("/run/media/system/HDD_1TB/Mets_archive/2026-08-12_05-01-35")


def compare_labels(a, b) -> dict:
    """Both ARI conventions for one pair of label vectors, plus the counts.

    Both, because `notes-10.md` §4.4 records that treating HDBSCAN's noise as
    one cluster versus dropping those points gives different answers on the
    same data, and 40-50 % of tokens are noise here. Reporting one would be
    choosing silently.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"label vectors differ in length: {a.shape} vs {b.shape}")
    return {
        "ari": round(float(adjusted_rand_index(a, b)), 6),
        "ari_noise_dropped": round(float(adjusted_rand_index(a, b, ignore_noise=True)), 6),
        "identical": bool(np.array_equal(a, b)),
        "n_clusters_a": int(len(set(a.tolist()) - {-1})),
        "n_clusters_b": int(len(set(b.tolist()) - {-1})),
        "noise_fraction_a": round(float((a == -1).mean()), 4),
        "noise_fraction_b": round(float((b == -1).mean()), 4),
    }


def activation_divergence(dir_a: Path, dir_b: Path):
    """``(max_abs, mean_abs)`` between the two runs' activations, or ``None``.

    Recorded so the reader can see the INPUT difference beside the OUTPUT
    difference. A partition that moves a lot on activations that agree to 1e-07
    is a statement about HDBSCAN; one that moves on activations that disagree
    at 1e-02 would be a statement about the forward pass.
    """
    pa, pb = dir_a / "activations.npz", dir_b / "activations.npz"
    if not (pa.exists() and pb.exists()):
        return None
    A = np.load(pa)["activations"]
    B = np.load(pb)["activations"]
    if A.shape != B.shape:
        return None
    d = np.abs(A.astype(np.float64) - B.astype(np.float64))
    return float(d.max()), float(d.mean())


def compare_directory(dir_a: Path, dir_b: Path, with_activations: bool = True) -> dict:
    la, lb = read_labels(dir_a), read_labels(dir_b)
    shared = sorted(set(la) & set(lb))
    layers = []
    for layer in shared:
        if la[layer].shape != lb[layer].shape:
            continue
        rec = compare_labels(la[layer], lb[layer])
        rec["layer"] = int(layer)
        layers.append(rec)

    out = {
        "run_dir": dir_a.name,
        "checkpoint": checkpoint_of(dir_a.name),
        "source_a": str(dir_a.parent),
        "source_b": str(dir_b.parent),
        "layers": layers,
    }
    if with_activations:
        div = activation_divergence(dir_a, dir_b)
        if div is not None:
            out["activation_max_abs_diff"] = div[0]
            out["activation_mean_abs_diff"] = div[1]
    tok_a, tok_b = dir_a / "tokens.txt", dir_b / "tokens.txt"
    if tok_a.exists() and tok_b.exists():
        out["tokens_identical"] = tok_a.read_text() == tok_b.read_text()
    return out


def _stats(rows: list) -> dict:
    if not rows:
        return {"n": 0}
    ari = np.array([r["ari"] for r in rows])
    arinn = np.array([r["ari_noise_dropped"] for r in rows])
    dk = np.array([r["n_clusters_b"] - r["n_clusters_a"] for r in rows])
    dn = np.array([r["noise_fraction_b"] - r["noise_fraction_a"] for r in rows])
    ident = np.array([r["identical"] for r in rows])

    def q(v, p):
        return round(float(np.percentile(v, p)), 4)

    return {
        "n": len(rows),
        "frac_identical": round(float(ident.mean()), 4),
        "ari_mean": round(float(ari.mean()), 4),
        "ari_p05": q(ari, 5), "ari_p50": q(ari, 50), "ari_min": round(float(ari.min()), 4),
        "ari_noise_dropped_mean": round(float(arinn.mean()), 4),
        "ari_noise_dropped_p05": q(arinn, 5),
        "ari_noise_dropped_min": round(float(arinn.min()), 4),
        "cluster_count_abs_delta_mean": round(float(np.abs(dk).mean()), 4),
        "cluster_count_abs_delta_max": int(np.abs(dk).max()),
        "noise_fraction_abs_delta_mean": round(float(np.abs(dn).mean()), 5),
        "noise_fraction_abs_delta_max": round(float(np.abs(dn).max()), 4),
    }


def aggregate(dirs: list) -> dict:
    rows = [(d, l) for d in dirs for l in d.get("layers", [])]
    if not rows:
        return {"n": 0}
    out = {"n_directories": sum(1 for d in dirs if d.get("layers"))}
    out.update(_stats([l for _, l in rows]))

    acts = [d["activation_max_abs_diff"] for d in dirs
            if d.get("activation_max_abs_diff") is not None]
    if acts:
        out["activation_max_abs_diff_max"] = float(np.max(acts))
    toks = [d["tokens_identical"] for d in dirs if "tokens_identical" in d]
    if toks:
        out["all_tokens_identical"] = bool(all(toks))

    by_ckpt = defaultdict(list)
    for d, l in rows:
        if d.get("checkpoint") is not None:
            by_ckpt[d["checkpoint"]].append(l)
    out["by_checkpoint"] = {str(s): _stats(ls) for s, ls in sorted(by_ckpt.items())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pilot", default=str(PILOT))
    ap.add_argument("--root", default=str(DATA / "phase12"))
    ap.add_argument("--pattern", default="pythia-410m-step*")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-activations", action="store_true",
                    help="skip the input-divergence measurement (it reads every "
                         "pair of activation stacks)")
    ap.add_argument("--out",
                    default=str(DATA / "analysis" / "p10_partition_stability.json"))
    args = ap.parse_args()

    pilot, root = Path(args.pilot), Path(args.root)
    if not pilot.exists():
        raise SystemExit(f"pilot sweep not mounted at {pilot}")

    pairs = []
    for pd in sorted(pilot.glob(args.pattern)):
        if not pd.is_dir():
            continue
        matches = [d for ts in sorted(root.glob("*")) if ts.is_dir()
                   for d in [ts / pd.name] if d.is_dir() and read_labels(d)]
        if matches:
            pairs.append((pd, matches[0]))
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"{len(pairs)} model-prompt directories present in both sweeps")

    t0 = time.time()
    dirs = []
    for i, (a, b) in enumerate(pairs, 1):
        dirs.append(compare_directory(a, b, with_activations=not args.no_activations))
        if i % 10 == 0 or i == len(pairs):
            print(f"  [{i}/{len(pairs)}] {a.name}  ({time.time() - t0:.0f}s)")

    summary = aggregate(dirs)
    record = {
        "schema": "p10_partition_stability/1",
        "question": "how reproducible is the HDBSCAN partition, run to run, on "
                    "the same model and the same prompt?",
        "kind": "measurement-reproducibility floor — NOT a null and NOT a "
                "hypothesis test. There is no p-value here by design.",
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sweep_a": str(pilot), "sweep_b": str(root),
        "labels_a": "native (hdbscan_labels.json)",
        "labels_b": "backfilled (tools/run/backfill_hdbscan.py)",
        "summary": summary,
        "directories": dirs,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    print(f"\nwrote {out}")
    for k, v in summary.items():
        if k != "by_checkpoint":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
