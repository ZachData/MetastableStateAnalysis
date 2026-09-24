"""F1 of `p10_cluster_function/notes-10.md` §8: the transport observables, and
the kinematic column of the four-signature table.

`docs/AXES.md` §4 lists this runner as a producer that does not exist and calls
what it would compute *"still the cheapest open action in the tree"* — the code
in `core/dissipation.py` is written and tested, the inputs are in 152/152
directories, and nothing has ever called it. That is what this fixes.

From artifacts ALREADY ON DISK. No forward pass, no model load.

WHAT IT COMPUTES, AND WHY EACH PIECE IS HERE
---------------------------------------------
**Per layer boundary, on the measure.** `w2_identity` is the project's existing
convention — token `i` to token `i` — made explicit as a coupling choice, and
it is an UPPER BOUND on true `W_2`. `w2_optimal` is the true distance by exact
linear assignment. The gap between them is the point:

> displacement that a permutation absorbs is tokens changing places, which
> leaves the measure untouched. Only the residual after optimal matching is
> motion of the distribution.

`swap_absorbed_fraction` is that quantity, and **every per-layer displacement
number this project has recorded is the identity coupling**, so this row
re-reads all of them rather than adding one.

**Per trajectory.** `arc_length` and `straightness = W_2(mu_0, mu_L) /
arc_length`. Near 1 is a trajectory moving steadily in one direction; near 0 is
a long path with little net displacement, **which is what dwelling in a
metastable state looks like measured on the measure rather than inferred from a
clustering algorithm** — an independent read on the same object Phase 1 finds
with HDBSCAN.

**Per particle — the kinematic signature.** `notes-10.md` §3.1's first row:
a parked particle's "displacement per layer is small relative to the layer's".
`tangential_velocity`'s norm is exactly Phase 1c's per-layer step size `h_l`
per particle, so this is that existing quantity split by population rather than
a new convention. The statistic is a standardised difference between the
clustered and unclustered tokens with a label-permutation null.

**The direction is two-sided and fixed here.** H-PARK predicts clustered
particles move LESS (they have been set down). H-CAT's "quiet but load-bearing"
reading is compatible with either. Neither sign was predicted in advance and
neither may be chosen now.

WHAT THIS ROW CANNOT SETTLE
----------------------------
Small displacement is the signature of a parked particle AND of a pinned one —
`attention-10.md` §5, and the whole reason F12 exists beside this row. A
clustered population that moves less is consistent with H-PARK and does not
establish it. The four-signature concordance (F5) is what discriminates, and it
needs the functional and causal columns this row does not touch.

COST
----
`w2_optimal` is an exact O(n^3) assignment. At the sweep's n of 242-512 that is
seconds per directory, and `--identity-only` skips it for a quick pass.

**Tier 1, exploratory, unregistered.** `claims/registry.json` is untouched, and
the merger is the arithmetic mean for the reason the other rows state.

Run:
    python tools/run/transport.py --out data/analysis/p10_f1_transport.json
    python tools/run/transport.py --identity-only --limit 8
"""
import argparse
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

from core.dissipation import (
    tangential_velocity,
    w2_identity,
    w2_optimal,
    wasserstein_arc_length,
)
from core.evalues import (
    DEFAULT_ALPHA,
    DEFAULT_KAPPA,
    average_p,
    calibrate,
    max_attainable_average_E,
)
from core.nulls import label_permutation_null, p_from_null_tolerant
from tools.run.backfill_hdbscan import labels_provenance, read_labels
from tools.run.p10_anchor import checkpoint_of
from tools.run.p10_partition_function import standardised_difference

N_PERMUTATIONS = 2000


def per_particle_step(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    ``||v_i||`` for the step from `X` to `Y` — Phase 1c's per-layer step size
    `h_l`, per particle.

    Goes through `core.dissipation.tangential_velocity` rather than taking a
    raw difference, so this row measures the project's existing quantity split
    by population instead of introducing a second convention for "how far did
    it move".
    """
    return np.linalg.norm(tangential_velocity(X, np.asarray(Y) - np.asarray(X)),
                          axis=1)


def measure_boundary(X, Y, labels, rng, optimal: bool = True) -> dict:
    """One layer boundary: the measure-level transport, and the per-particle
    kinematic split."""
    lab = np.asarray(labels)
    out = {"w2_identity": float(w2_identity(X, Y))}

    if optimal:
        o = w2_optimal(X, Y)
        out["w2_optimal"] = float(o["w2"])
        out["swap_absorbed_fraction"] = (
            round(float(o["swap_absorbed_fraction"]), 4)
            if o["swap_absorbed_fraction"] is not None else None)
        out["swap_fraction"] = round(float(o["swap_fraction"]), 4)
        # `assignment` is an (n,) permutation and is deliberately NOT kept:
        # it is the largest thing this function produces and nothing
        # downstream reads it. Recomputing it is cheaper than storing 152 of
        # them in one JSON.

    step = per_particle_step(X, Y)
    out["mean_step"] = round(float(step.mean()), 6)

    noise = lab == -1
    if lab.size == step.size and noise.any() and not noise.all():
        diff = standardised_difference(step, lab)
        if np.isfinite(diff):
            draws = label_permutation_null(step, lab, standardised_difference,
                                           n_permutations=N_PERMUTATIONS, rng=rng)
            res = p_from_null_tolerant(diff, draws, alternative="two-sided")
            out["clustered_minus_noise_step"] = round(float(diff), 4)
            out["clustered_minus_noise_step_p"] = float(res["p_value"])
            out["degenerate"] = bool(res["degenerate_null"])
            return out

    out["clustered_minus_noise_step"] = None
    out["clustered_minus_noise_step_p"] = None
    out["degenerate"] = None
    return out


def measure_directory(run_dir: Path, rng, optimal: bool = True) -> dict:
    p = run_dir / "activations.npz"
    if not p.exists():
        return {"run_dir": run_dir.name, "skipped": "no activations.npz"}
    acts = np.asarray(np.load(p)["activations"], dtype=np.float64)
    labels_by_layer = read_labels(run_dir)

    arc = wasserstein_arc_length(acts, optimal=optimal)
    # `per_step` duplicates what `boundaries` holds below, at 24 entries per
    # directory times 152 directories. Dropped from the record.
    arc.pop("per_step", None)

    boundaries = []
    for l in range(acts.shape[0] - 1):
        lab = labels_by_layer.get(l)
        if lab is None:
            lab = np.full(acts.shape[1], -1)
        rec = measure_boundary(acts[l], acts[l + 1], lab, rng, optimal=optimal)
        rec["layer"] = int(l)
        boundaries.append(rec)

    return {
        "run_dir": run_dir.name,
        "timestamp": run_dir.parent.name,
        "checkpoint": checkpoint_of(run_dir.name),
        "labels_from": labels_provenance(run_dir),
        "n_tokens": int(acts.shape[1]),
        "trajectory": arc,
        "boundaries": boundaries,
    }


def _summarise(rows: list, dirs: list) -> dict:
    def m(key, source=rows):
        v = np.array([r[key] for r in source
                      if r.get(key) is not None and np.isfinite(r[key])])
        return round(float(v.mean()), 6) if v.size else None

    out = {
        "n_boundaries": len(rows),
        "mean_w2_identity": m("w2_identity"),
        "mean_w2_optimal": m("w2_optimal"),
        "mean_swap_absorbed_fraction": m("swap_absorbed_fraction"),
        "mean_swap_fraction": m("swap_fraction"),
        "mean_step": m("mean_step"),
        "mean_clustered_minus_noise_step": m("clustered_minus_noise_step"),
    }
    trajs = [d["trajectory"] for d in dirs if d.get("trajectory")]
    out["mean_straightness"] = m("straightness", trajs)
    out["mean_arc_length_identity"] = m("arc_length_identity", trajs)
    out["mean_arc_length_optimal"] = m("arc_length_optimal", trajs)

    ps = [r["clustered_minus_noise_step_p"] for r in rows
          if r.get("clustered_minus_noise_step_p") is not None]
    if ps:
        E, reject = average_p(ps)
        out["clustered_minus_noise_step"] = {
            "n": len(ps), "E": round(float(E), 4), "reject": bool(reject),
            "median_p": round(float(np.median(ps)), 4),
            "frac_below_05": round(float(np.mean(np.array(ps) < 0.05)), 4),
        }
    return out


def aggregate(dirs: list) -> dict:
    scored = [d for d in dirs if d.get("boundaries")]
    rows = [(d, b) for d in scored for b in d["boundaries"]]
    if not rows:
        return {"n_boundaries": 0}

    out = {"n_directories": len(scored)}
    out.update(_summarise([b for _, b in rows], scored))

    by_ckpt = defaultdict(list)
    dirs_by_ckpt = defaultdict(list)
    for d in scored:
        if d.get("checkpoint") is not None:
            by_ckpt[d["checkpoint"]].extend(d["boundaries"])
            dirs_by_ckpt[d["checkpoint"]].append(d)
    out["by_checkpoint"] = {
        str(s): _summarise(bs, dirs_by_ckpt[s]) for s, bs in sorted(by_ckpt.items())
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(DATA / "phase12"))
    ap.add_argument("--pattern", default="pythia-410m-*")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--identity-only", action="store_true",
                    help="skip the O(n^3) optimal coupling — a quick pass that "
                         "cannot report swap_absorbed_fraction, which is the "
                         "number this row exists for")
    ap.add_argument("--out", default=str(DATA / "analysis" / "p10_f1_transport.json"))
    args = ap.parse_args()

    root = Path(args.root)
    targets = sorted(
        d for ts in sorted(root.glob("*")) if ts.is_dir()
        for d in sorted(ts.glob(args.pattern))
        if d.is_dir() and (d / "activations.npz").exists()
    )
    if args.limit:
        targets = targets[: args.limit]
    print(f"{len(targets)} directories, optimal coupling: {not args.identity_only}")

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    dirs = []
    for i, d in enumerate(targets, 1):
        dirs.append(measure_directory(d, rng, optimal=not args.identity_only))
        if i % 10 == 0 or i == len(targets):
            print(f"  [{i}/{len(targets)}] {d.name}  ({time.time() - t0:.0f}s)", flush=True)

    summary = aggregate(dirs)
    record = {
        "schema": "p10_f1_transport/1",
        "row": "F1 — transport observables, and the kinematic signature",
        "tier": "1 (exploratory, unregistered)",
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(root),
        "optimal_coupling": not args.identity_only,
        "n_permutations": N_PERMUTATIONS,
        "seed": args.seed,
        "kappa": DEFAULT_KAPPA,
        "alpha": DEFAULT_ALPHA,
        "merger": "arithmetic mean (core.evalues.average)",
        "resolution_floor_e": round(calibrate(1.0 / (N_PERMUTATIONS + 1)), 3),
        "max_attainable_E": round(max_attainable_average_E(N_PERMUTATIONS)[0], 3),
        "design_can_reject": bool(max_attainable_average_E(N_PERMUTATIONS)[1]),
        "alternatives": {"clustered_minus_noise_step_p": "two-sided"},
        "caveat": "small displacement is the signature of a parked particle AND "
                  "of a pinned one (attention-10.md §5). This row cannot tell "
                  "them apart; F12 is the other half and F5 is what "
                  "discriminates.",
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
