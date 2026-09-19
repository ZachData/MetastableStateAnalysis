#!/usr/bin/env python3
"""
tools/audit_p1c_inputs.py — what Phase 1c's four registered predictions can
actually be fed from the Phase 1 artifacts on disk, and what they cannot.

Phase 1c is implemented and validated and has never been run against Pythia
artifacts (`p1c_frames/status-1c.md`). The e-value audit's question for the
phase is narrower than "does it run": for each of `P-gamma2` (1c-A),
`P-gamma1` (1c-B), `P-H1` (1c-E) and `P-S1` (1c-F), is the input the gate
needs PRESENT in a run directory, DERIVABLE from what is present, or absent?
Prose cannot answer that — the keys are in the files — so this reads them.

It computes nothing scientific. No statistic, null, tolerance or clusterer is
chosen here; every verdict is a statement about artifact keys, in the same
spirit as tools/score_claim_c.py, whose record-on-refusal convention this
follows: a record is written either way, because "the input is missing" is a
finding about the tree and belongs in git rather than in a session's prose.

The P-S1 arm check is the exception that needs a second run directory: the
gate refuses when its two arms report different (m, d) (registry, P-S1
null_construction), so whether the arms AGREE is a property of a pair, not of
a directory. Pass --trained and --step0 to measure it.

Usage
-----
    python3 -m tools.audit_p1c_inputs --run-dir data/phase12/<run> [...] \\
        [--trained <run_dir> --step0 <run_dir>]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(REPO))

RECORD_PATH = REPO / "claims" / "audits" / "p1c_inputs.json"

#: Sub-experiment -> (prediction, what it reads). The reads are the module's
#: own requirements, not this file's opinion: A and C need the raw residual
#: stream (p1c_frames/integration_time.py, moments.py), B needs a beta_eff per
#: layer or per head (gamma_null.py, beta_reduction.py), E needs layer
#: activations only (hemisphere_feasibility.py), F needs cluster centroids
#: (centroids.py::load_centroids).
SUBEXP = {
    "A": ("P-gamma2", "raw residual stream + beta_eff"),
    "B": ("P-gamma1", "raw residual stream + beta_eff per head"),
    "E": ("P-H1", "layer activations"),
    "F": ("P-S1", "cluster centroids, both arms at the same (m, d)"),
}


def _cluster_kinds(npz_files: Sequence[str]) -> dict:
    """Which `<kind>_L{i}` families a clusters.npz carries."""
    kinds: dict = {}
    for k in npz_files:
        if "_L" not in k:
            continue
        kind, _, idx = k.rpartition("_L")
        if idx.isdigit():
            kinds.setdefault(kind, []).append(int(idx))
    return {k: sorted(v) for k, v in kinds.items()}


def inspect_one(d: Path) -> dict:
    """The keys one <model>_<prompt> directory carries."""
    geo = json.loads((d / "geometry.json").read_text())
    out = {
        "dir": str(d),
        "model": geo.get("model"),
        "prompt": geo.get("prompt"),
        "n_tokens": geo.get("n_tokens"),
        "d_model": geo.get("d_model"),
        "n_hidden_states": geo.get("n_layers"),
        # A and B: the beta the null is evaluated at. Nothing in Phase 1
        # writes it, which is status-1c.md's open item 1.
        "beta_eff": geo.get("beta_eff"),
        "beta_eff_per_head": geo.get("beta_eff_per_head") is not None,
        # h_attn_only, the frame-correct step-size variant, needs the
        # post-sublayer streams run_1.py writes under --sublayer.
        "sublayer_semantics": geo.get("sublayer_semantics"),
    }
    act = d / "activations.npz"
    if act.exists():
        files = list(np.load(act).files)
        out["activations"] = "activations" in files
        # raw_states refuses without `norms` rather than substituting the
        # unit-norm array (status-1c.md open item 8).
        out["norms"] = "norms" in files
    else:
        out["activations"] = out["norms"] = False
    # beta_eff is derivable from the persisted attention tensor plus the
    # checkpoint's LN parameters — no forward pass. Whether the tensor is
    # there is the part that can be read off disk.
    out["attentions"] = (d / "attentions.npz").exists()
    cz = d / "clusters.npz"
    out["cluster_kinds"] = _cluster_kinds(list(np.load(cz).files)) if cz.exists() else {}
    hj = d / "hdbscan_labels.json"
    out["hdbscan_labels_json_empty"] = (
        not json.loads(hj.read_text()) if hj.exists() else None)
    return out


def counts_by_layer(d: Path, kind: str) -> dict:
    """{layer: number of distinct labels}, noise (-1) excluded."""
    cz = np.load(d / "clusters.npz")
    out = {}
    for k in cz.files:
        prefix, _, idx = k.rpartition("_L")
        if prefix != kind or not idx.isdigit():
            continue
        lab = np.asarray(cz[k]).ravel()
        out[int(idx)] = int(len(set(lab.tolist()) - {-1}))
    return out


def arm_agreement(trained: Path, step0: Path, exclude=("repeated_tokens",)) -> dict:
    """
    How often the two P-S1 arms report the same cluster count, per method.

    The gate refuses a layer-row whose arms disagree on (m, d), by
    construction rather than by tolerance — Q_k's i.i.d. floor is 1/m, so
    "closer to a spherical design" is not a comparison that exists across
    different m. This counts the rows that survive that refusal.
    """
    by_model = {}
    for root in (trained, step0):
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            if not (sub / "geometry.json").exists():
                continue
            geo = json.loads((sub / "geometry.json").read_text())
            by_model.setdefault(root, {})[geo["prompt"]] = sub
    a, b = by_model.get(trained, {}), by_model.get(step0, {})
    prompts = sorted(set(a) & set(b))
    out = {"prompts": prompts, "excluded": list(exclude), "methods": {}}
    for kind in ("kmeans_labels", "agglom_mid_labels"):
        rows = equal = rows_x = equal_x = 0
        for pr in prompts:
            ca, cb = counts_by_layer(a[pr], kind), counts_by_layer(b[pr], kind)
            for layer in sorted(set(ca) & set(cb)):
                rows += 1
                same = ca[layer] == cb[layer]
                equal += same
                if pr not in exclude:
                    rows_x += 1
                    equal_x += same
        out["methods"][kind] = {
            "layer_rows": rows, "same_m": equal,
            "layer_rows_excluding_controls": rows_x,
            "same_m_excluding_controls": equal_x,
        }
    return out


def verdicts(dirs: Sequence[dict]) -> dict:
    """One verdict per sub-experiment over every directory inspected."""
    n = len(dirs)
    have_raw = sum(bool(x["activations"] and x["norms"]) for x in dirs)
    have_attn = sum(bool(x["attentions"]) for x in dirs)
    have_beta = sum(x["beta_eff"] is not None for x in dirs)
    have_beta_head = sum(bool(x["beta_eff_per_head"]) for x in dirs)
    have_sub = sum(x["sublayer_semantics"] is not None for x in dirs)
    have_cent = sum("kmeans_centroids" in x["cluster_kinds"] for x in dirs)
    have_kmeans_lab = sum("kmeans_labels" in x["cluster_kinds"] for x in dirs)
    have_agglom = sum("agglom_mid_labels" in x["cluster_kinds"] for x in dirs)
    have_hdb = sum("hdbscan_labels" in x["cluster_kinds"] for x in dirs)

    beta_note = (
        f"no geometry.json carries beta_eff ({have_beta}/{n}); it is DERIVABLE "
        f"from attentions.npz ({have_attn}/{n}) plus the checkpoint's LN "
        f"parameters (core/ln_frame.py + core/beta_eff.py), which needs no "
        f"forward pass but needs a producer that does not exist"
    ) if have_beta < n else None

    return {
        "A": {
            "prediction": "P-gamma2",
            "runnable": have_beta == n and have_raw == n,
            "raw_stream": f"{have_raw}/{n} carry activations + norms",
            "h_attn_only": (f"{have_sub}/{n} carry sublayer streams; the "
                            f"frame-correct variant is nan without them and "
                            f"they need run_1.py --sublayer, i.e. new forward "
                            f"passes"),
            "blocked_by": beta_note,
        },
        "B": {
            "prediction": "P-gamma1",
            "runnable": have_beta_head == n and have_raw == n,
            "per_head_beta": f"{have_beta_head}/{n} carry beta_eff_per_head",
            "blocked_by": beta_note,
        },
        "E": {
            "prediction": "P-H1",
            "runnable": have_raw == n,
            "note": ("inputs are present; p1c_frames/run_1c.py nevertheless "
                     "SKIPS every run whose beta_used is non-finite, including "
                     "runs where only E was requested and beta is unused"),
        },
        "F": {
            "prediction": "P-S1",
            "runnable": have_cent > 0,
            "kmeans_centroids": (
                f"{have_cent}/{n} carry kmeans_centroids_L*, which "
                f"centroids.py::load_centroids reads for the PRIMARY arm; "
                f"{have_kmeans_lab}/{n} carry kmeans_labels_L*"),
            "agglomerative": (
                f"{have_agglom}/{n} carry agglom_mid_labels_L* — recomputable "
                f"with activations, at that method's own cluster count"),
            "hdbscan": (
                f"{have_hdb}/{n} carry hdbscan_labels_L* in clusters.npz, "
                f"which is where load_centroids looks; the runner writes "
                f"hdbscan_labels.json instead"),
        },
    }


def run(run_dirs: Sequence[Path], trained: Optional[Path] = None,
        step0: Optional[Path] = None) -> dict:
    inspected = []
    for rd in run_dirs:
        for sub in sorted(p for p in rd.iterdir() if p.is_dir()):
            if (sub / "geometry.json").exists():
                inspected.append(inspect_one(sub))
    record = {
        "phase": "1c",
        "predictions": {k: v[0] for k, v in SUBEXP.items()},
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_dirs": [str(p) for p in run_dirs],
        "n_model_prompt_dirs": len(inspected),
        "models": sorted({x["model"] for x in inspected}),
        "subexperiments": verdicts(inspected),
    }
    if trained and step0:
        record["p_s1_arm_agreement"] = {
            "trained": str(trained), "step0": str(step0),
            **arm_agreement(trained, step0)}
    return record


def _print(record: dict) -> None:
    print("Phase 1c input audit — what the four gates can be fed\n")
    print(f"  {record['n_model_prompt_dirs']} model-prompt directories, "
          f"{len(record['models'])} model(s)\n")
    for sx, v in record["subexperiments"].items():
        pred = v["prediction"]
        print(f"  {sx} / {pred:<9} {'RUNNABLE' if v['runnable'] else 'BLOCKED'}")
        for k, val in v.items():
            if k in ("prediction", "runnable") or not val:
                continue
            print(f"      {k}: {val}")
    ag = record.get("p_s1_arm_agreement")
    if ag:
        print("\n  P-S1 arm agreement (the gate refuses a row whose arms "
              "disagree on (m, d)):")
        for kind, m in ag["methods"].items():
            r, e = m["layer_rows"], m["same_m"]
            rx, ex = m["layer_rows_excluding_controls"], m["same_m_excluding_controls"]
            print(f"      {kind:<18} {e}/{r} rows agree "
                  f"({ex}/{rx} excluding {ag['excluded']})")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", action="append", required=True, type=Path,
                    help="a Phase 1 run directory; repeatable")
    ap.add_argument("--trained", type=Path, default=None,
                    help="P-S1's trained arm's run directory")
    ap.add_argument("--step0", type=Path, default=None,
                    help="P-S1's step-0 arm's run directory")
    ap.add_argument("--no-write", action="store_true",
                    help=f"do not write {RECORD_PATH.relative_to(REPO)}")
    args = ap.parse_args(argv)

    for rd in args.run_dir:
        if not rd.is_dir():
            print(f"not a directory: {rd}", file=sys.stderr)
            return 2
    if bool(args.trained) != bool(args.step0):
        print("--trained and --step0 go together", file=sys.stderr)
        return 2

    record = run(args.run_dir, args.trained, args.step0)
    _print(record)
    if not args.no_write:
        RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
        RECORD_PATH.write_text(json.dumps(record, indent=2) + "\n")
        print(f"\nrecord: {RECORD_PATH.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
