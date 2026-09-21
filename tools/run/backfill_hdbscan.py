"""Recompute the HDBSCAN partition for run directories that were written while
`hdbscan` was missing from this machine, from artifacts ALREADY ON DISK — no
forward pass, no model load.

WHY THIS EXISTS
---------------
`hdbscan` went missing from this machine's venv between 2026-08-12 and
2026-08-31 and was not declared anywhere that would catch it
(`p1_mstate_tracking/clustering.py`'s note, `PROJECT.md` §3.41). Every run in
`data/phase12` was written in that window or after it, so:

    all 152 model-prompt directories of the 19-checkpoint pythia-410m sweep
    carry an EMPTY `hdbscan_labels.json` and a `clustering.json` whose hdbscan
    block reads {"n_clusters": null, "noise_count": 0, "noise_fraction": 0.0}

MEASURED 2026-09-20: 152 of 152, no exceptions. The `kmeans_labels_L*` and
`agglom_mid_labels_L*` arrays are present and fine; it is only the density
partition that is absent, and it is the one Phase 10 is about.

The inputs to recompute it are all there. `activations.npz` holds the
sphere-projected hidden states for every layer (`p1_io.py::_save_activations`),
and `cluster_count_sweep` derives the partition from those and nothing else.
So this is a pure re-derivation, not a re-measurement.

THE TOOLCHAIN GUARD, WHICH IS THE POINT OF THE SCRIPT AND NOT A FORMALITY
--------------------------------------------------------------------------
`clustering.py` records, measured, that HDBSCAN's output here is a property of
the install as much as of the data: the conda `mets` env (py3.10.20, hdbscan
0.8.41, scikit-learn 1.7.2, numpy 2.2.6) reproduces the 2026-08-12 pilot
sweep's `n_clusters` and `noise_fraction` EXACTLY, while `.venv` (py3.14.7,
hdbscan 0.8.44, scikit-learn 1.9.0) does not — 45 -> 41 clusters at one layer.

So this script REFUSES to run outside the reference toolchain, and it does so
by fingerprinting the toolchain rather than by asserting an interpreter path
(which is what every other runner here guards on, correctly, for a different
reason). Backfilling under a second install would put two incomparable
partitions in one sweep and there would be nothing on disk to say which was
which. `--allow-any-toolchain` exists for deliberate cross-version work and
writes the divergence into the artifact.

Re-verified on every invocation, not just documented: `--verify-pilot` replays
N directories of the pilot sweep that DOES carry labels and refuses to write
anything unless they come back bit-identical.

WHAT IT WRITES, AND WHAT IT DOES NOT TOUCH
-------------------------------------------
It writes ONE new file per directory, `hdbscan_backfill.json`, holding the
per-layer labels, the per-layer counts, and the provenance. It does not touch
`hdbscan_labels.json`, `clustering.json` or anything else.

That is deliberate. Filling the canonical file would leave `clustering.json`
still saying `"n_clusters": null` beside it — the same shape of inconsistency
as the forged zeros `b55375e` had to un-write — and it would erase the one
fact a reader most needs, which is that these labels were derived later, by a
different route, from a directory that was written without them. A separate
file cannot be mistaken for a run artifact.

`read_labels` is the reader that prefers the backfill and falls back to the
canonical file, so callers do not each reinvent the precedence.

Run:
    python tools/run/backfill_hdbscan.py --verify-pilot 3 --dry-run
    python tools/run/backfill_hdbscan.py --verify-pilot 3
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", "/run/media/system/WDS_500/Mets"))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))

import numpy as np

#: The install `clustering.py` measured as reproducing the pilot sweep exactly.
REFERENCE_TOOLCHAIN = {
    "python": "3.10",
    "hdbscan": "0.8.41",
    "scikit-learn": "1.7.2",
}

#: The pilot sweep, which DOES carry labels and is therefore the only thing
#: that can verify this script. Git-ignored and irreplaceable (`AXES.md` §2.2).
PILOT = Path("/run/media/system/HDD_1TB/Mets_archive/2026-08-12_05-01-35")

#: Parameters, copied from `cluster_count_sweep` rather than imported, so a
#: later edit there cannot silently change what a backfilled directory means.
#: The test asserts the two agree.
HDBSCAN_PARAMS = {"min_cluster_size": 2, "metric": "precomputed"}

OUT_NAME = "hdbscan_backfill.json"


# ---------------------------------------------------------------------------
# Toolchain
# ---------------------------------------------------------------------------

def toolchain_fingerprint() -> dict:
    """Everything that has been observed to move HDBSCAN's output here."""
    from importlib.metadata import version

    def v(pkg):
        try:
            return version(pkg)
        except Exception:
            return "unknown"

    return {
        "python": platform.python_version(),
        "hdbscan": v("hdbscan"),
        "scikit-learn": v("scikit-learn"),
        "numpy": np.__version__,
        "executable": sys.executable,
    }


def toolchain_divergence(fp: dict) -> list:
    """Which reference fields this interpreter does not match. Python is
    compared on major.minor; the rest exactly."""
    out = []
    for key, want in REFERENCE_TOOLCHAIN.items():
        got = fp.get(key, "unknown")
        ok = got.startswith(want + ".") if key == "python" else got == want
        if not ok:
            out.append(f"{key}: have {got}, reference is {want}")
    return out


# ---------------------------------------------------------------------------
# The computation
# ---------------------------------------------------------------------------

def labels_for_activations(acts: np.ndarray) -> dict:
    """
    ``{layer_index: [labels]}`` for a (n_layers, n_tokens, d) sphere-projected
    stack, by the same route `cluster_count_sweep` takes: cosine distance,
    clipped at 0, HDBSCAN on the precomputed matrix in float64.
    """
    import hdbscan
    from sklearn.metrics import pairwise_distances

    if acts.ndim != 3:
        raise ValueError(f"activations must be (n_layers, n, d); got {acts.shape}")

    out = {}
    for layer in range(acts.shape[0]):
        X = np.asarray(acts[layer], dtype=np.float32)
        cos_dist = np.clip(pairwise_distances(X, metric="cosine"), 0, None)
        lab = hdbscan.HDBSCAN(**HDBSCAN_PARAMS).fit_predict(cos_dist.astype(np.float64))
        out[layer] = np.asarray(lab, dtype=np.int32)
    return out


def summarise(labels: dict) -> dict:
    """Per-layer counts, in the same two fields `clustering.json` records, so a
    backfilled directory can be compared against a natively-written one without
    recomputing anything."""
    return {
        str(layer): {
            "n_clusters": int(len(set(lab.tolist())) - (1 if -1 in lab else 0)),
            "noise_count": int((lab == -1).sum()),
            "noise_fraction": round(float((lab == -1).mean()), 4),
        }
        for layer, lab in labels.items()
    }


# ---------------------------------------------------------------------------
# Verification against the sweep that has labels
# ---------------------------------------------------------------------------

def verify_against_pilot(n_dirs: int, rng_seed: int = 0) -> dict:
    """
    Replay `n_dirs` directories of the pilot sweep and compare label-for-label
    with the labels it already carries.

    Returns a record; the caller refuses to write on anything but a clean pass.
    Comparison is on the LABEL VECTOR, not on the cluster count: two partitions
    can agree on how many clusters there are and disagree about which token is
    in which, and it is membership that Phase 10 reads.
    """
    if not PILOT.exists():
        return {"ran": False, "reason": f"pilot sweep not mounted at {PILOT}"}

    cands = sorted(
        d for d in PILOT.iterdir()
        if d.is_dir() and (d / "hdbscan_labels.json").exists()
        and (d / "activations.npz").exists()
        and json.loads((d / "hdbscan_labels.json").read_text())
    )
    if not cands:
        return {"ran": False, "reason": "pilot sweep carries no non-empty labels"}

    rng = np.random.default_rng(rng_seed)
    picked = [cands[i] for i in rng.choice(len(cands), size=min(n_dirs, len(cands)),
                                           replace=False)]
    per_dir = []
    for d in picked:
        acts = np.load(d / "activations.npz")["activations"]
        historical = json.loads((d / "hdbscan_labels.json").read_text())
        got = labels_for_activations(acts)
        layers_identical = sum(
            1 for layer, lab in got.items()
            if str(layer) in historical
            and np.array_equal(lab, np.asarray(historical[str(layer)], dtype=np.int32))
        )
        n_compared = sum(1 for layer in got if str(layer) in historical)
        per_dir.append({
            "dir": d.name,
            "layers_compared": n_compared,
            "layers_identical": layers_identical,
            "clean": layers_identical == n_compared and n_compared > 0,
        })
        print(f"  verify {d.name}: {layers_identical}/{n_compared} layers identical")

    return {
        "ran": True,
        "source": str(PILOT),
        "n_dirs": len(per_dir),
        "per_dir": per_dir,
        "clean": all(r["clean"] for r in per_dir),
    }


# ---------------------------------------------------------------------------
# Target discovery
# ---------------------------------------------------------------------------

def needs_backfill(run_dir: Path) -> bool:
    """A directory with activations, and with no usable HDBSCAN partition from
    either route."""
    if not (run_dir / "activations.npz").exists():
        return False
    if (run_dir / OUT_NAME).exists():
        return False
    p = run_dir / "hdbscan_labels.json"
    if not p.exists():
        return True
    try:
        return not json.loads(p.read_text())
    except json.JSONDecodeError:
        return True


def discover(root: Path, pattern: str) -> list:
    return sorted(
        d for ts in sorted(root.glob("*")) if ts.is_dir()
        for d in sorted(ts.glob(pattern)) if d.is_dir() and needs_backfill(d)
    )


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def read_labels(run_dir) -> dict:
    """
    ``{layer_index: np.ndarray}`` for a run directory, preferring a backfill
    over the canonical file and returning ``{}`` when neither carries a
    partition.

    The precedence is here so callers do not each reinvent it, and it is
    backfill-first because the canonical file in a backfilled directory is the
    EMPTY one that made the backfill necessary.
    """
    run_dir = Path(run_dir)
    bf = run_dir / OUT_NAME
    if bf.exists():
        rec = json.loads(bf.read_text())
        return {int(k): np.asarray(v, dtype=np.int32)
                for k, v in rec.get("labels", {}).items()}
    p = run_dir / "hdbscan_labels.json"
    if p.exists():
        try:
            raw = json.loads(p.read_text())
        except json.JSONDecodeError:
            return {}
        return {int(k): np.asarray(v, dtype=np.int32) for k, v in raw.items()}
    return {}


def labels_provenance(run_dir) -> str:
    """``"backfill"``, ``"native"`` or ``"absent"`` — so a report can say where
    a partition came from rather than implying every one came from a run."""
    run_dir = Path(run_dir)
    if (run_dir / OUT_NAME).exists():
        return "backfill"
    p = run_dir / "hdbscan_labels.json"
    if p.exists():
        try:
            if json.loads(p.read_text()):
                return "native"
        except json.JSONDecodeError:
            pass
    return "absent"


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(DATA / "phase12"),
                    help="directory of timestamped run directories")
    ap.add_argument("--pattern", default="pythia-410m-*",
                    help="glob for model-prompt directories inside each")
    ap.add_argument("--verify-pilot", type=int, default=3, metavar="N",
                    help="replay N pilot directories first; 0 skips (not advised)")
    ap.add_argument("--allow-any-toolchain", action="store_true",
                    help="write even from a non-reference install, recording the "
                         "divergence in every artifact")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="stop after N directories")
    args = ap.parse_args()

    fp = toolchain_fingerprint()
    divergence = toolchain_divergence(fp)
    print(f"toolchain: python {fp['python']}, hdbscan {fp['hdbscan']}, "
          f"scikit-learn {fp['scikit-learn']}, numpy {fp['numpy']}")
    if divergence:
        print("TOOLCHAIN DIVERGES from the reference install:")
        for line in divergence:
            print(f"  - {line}")
        if not args.allow_any_toolchain:
            raise SystemExit(
                "refusing to backfill: HDBSCAN's output here is a property of the "
                "install (clustering.py's measured note), and a sweep holding two "
                "partitions from two installs has nothing on disk to tell them "
                "apart. Use the conda `mets` env, or pass --allow-any-toolchain."
            )
        print("  ...writing anyway (--allow-any-toolchain); divergence is recorded")

    verification = {"ran": False, "reason": "skipped by --verify-pilot 0"}
    if args.verify_pilot > 0:
        print(f"verifying against {args.verify_pilot} pilot directories...")
        verification = verify_against_pilot(args.verify_pilot)
        if not verification.get("ran"):
            print(f"  could not verify: {verification['reason']}")
            if not args.allow_any_toolchain:
                raise SystemExit("refusing to backfill unverified")
        elif not verification["clean"]:
            raise SystemExit(
                "refusing to backfill: this install does NOT reproduce the pilot "
                "sweep's labels, so anything it writes is a second partition "
                "wearing the first one's name."
            )
        else:
            print("  clean — every layer compared came back identical")

    targets = discover(Path(args.root), args.pattern)
    if args.limit:
        targets = targets[: args.limit]
    print(f"{len(targets)} directories need a backfill under {args.root}")
    if args.dry_run:
        for d in targets[:10]:
            print(f"  would write {d / OUT_NAME}")
        if len(targets) > 10:
            print(f"  ... and {len(targets) - 10} more")
        return

    t0 = time.time()
    written = 0
    for i, d in enumerate(targets, 1):
        acts = np.load(d / "activations.npz")["activations"]
        labels = labels_for_activations(acts)
        record = {
            "schema": "hdbscan_backfill/1",
            "run_dir": d.name,
            "source_timestamp": d.parent.name,
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "derived_from": "activations.npz (sphere-projected hidden states)",
            "why": "the directory was written while hdbscan was missing from "
                   "this machine; see the module docstring",
            "params": HDBSCAN_PARAMS,
            "toolchain": fp,
            "toolchain_divergence": divergence,
            "pilot_verification": verification,
            "n_layers": int(acts.shape[0]),
            "n_tokens": int(acts.shape[1]),
            "per_layer": summarise(labels),
            "labels": {str(k): v.tolist() for k, v in labels.items()},
        }
        (d / OUT_NAME).write_text(json.dumps(record))
        written += 1
        if i % 20 == 0 or i == len(targets):
            print(f"  [{i}/{len(targets)}] {d.name}  ({time.time() - t0:.0f}s)", flush=True)

    print(f"wrote {written} backfill records in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
