#!/usr/bin/env python3
"""
tools/preflight_1c.py — what can actually run, before anything is scheduled.

    python -m tools.preflight_1c --results results/ [--p2-dir results_p2/]

WHY THIS EXISTS

Two of the plan's blockers are questions about the artifacts rather than
about the code, and both can eliminate work:

  * `activations.npz` stores unit-norm activations plus a `norms` key that
    reconstructs the raw residual stream. `norms` was added later.
    **Sub-experiments A and C cannot run at all without it** — the raw
    stream is where h_l's denominator and the entire sink question live —
    and there is no substitute: the unit-norm array would produce plausible
    numbers meaning something else.
  * `beta_eff` may or may not be in `geometry.json`. Without it `run_1c`
    skips the run rather than inventing a beta.

Answering these across 27 checkpoints x 8 prompts by hand is exactly the
kind of thing that gets answered by "probably fine" and then costs a rerun.
This reports it as a table, per sub-experiment, with the reason each run is
excluded.

It reads only file headers and JSON keys — no activations are loaded into
memory, so it is fast enough to run before every sweep.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path


# What each sub-experiment needs. Keeping this as data rather than as
# branching logic means a new sub-experiment adds a row, and — more
# importantly — that the requirements are readable in one place instead of
# being inferred from whichever module happens to raise first.
REQUIREMENTS = {
    "1c-A": {"activations", "norms", "beta_eff"},
    "1c-B": {"activations", "norms", "beta_eff", "ip_mean"},
    "1c-C": {"activations", "norms", "energies"},
    "1c-D": {"activations"},              # + LN weights, checked separately
    "1c-E": {"activations"},
    "1c-F": {"activations", "clusters"},
    "2d-D1": set(),                       # weights only
    "2d-D2": {"activations", "revision", "extraction_convention"},
    "2d-D3": {"activations", "revision", "extraction_convention"},
    "2d-D4": {"activations", "revision", "extraction_convention"},
}

# Human-readable consequence of each missing capability, so the output says
# what to do rather than only what is absent.
CONSEQUENCE = {
    "norms": ("activations.npz predates the norms fix; the raw residual "
              "stream is unrecoverable. Re-extract — do NOT substitute the "
              "unit-norm array."),
    "beta_eff": ("no beta_eff in geometry.json. Either add it or pass "
                 "--beta-fallback deliberately; run_1c will skip otherwise."),
    "clusters": "no clusters.npz, so sub-experiment F has no centroids.",
    "energies": "no energies.json.",
    "ip_mean": "geometry.json layers carry no ip_mean.",
    "revision": ("no revision / checkpoint_step in geometry.json; the "
                 "Phase 2d join refuses rather than guessing from a "
                 "directory name."),
    "extraction_convention": (
        "geometry.json lacks hidden_state_0_is_embedding / "
        "final_hidden_state_is_post_ln. Phase 2d's LN frame resolution "
        "then has to be TOLD the convention by flag, which is the error "
        "class those fields exist to prevent."),
    "activations": "no activations.npz.",
}


def _npz_keys(path: Path) -> set:
    """Array names from an .npz WITHOUT loading any array."""
    try:
        with zipfile.ZipFile(path) as z:
            return {n[:-4] for n in z.namelist() if n.endswith(".npy")}
    except Exception:
        return set()


def inspect_run(run_dir: Path) -> dict:
    """Capabilities present in one Phase 1 run directory."""
    have, notes = set(), []

    geo_p = run_dir / "geometry.json"
    geo = None
    if geo_p.exists():
        try:
            with open(geo_p) as f:
                geo = json.load(f)
        except json.JSONDecodeError as exc:
            notes.append(f"geometry.json unreadable: {exc}")
    if geo:
        if geo.get("beta_eff") is not None or geo.get("beta_effective") is not None:
            have.add("beta_eff")
        for key in ("revision", "checkpoint_step", "model_rev",
                    "checkpoint", "step"):
            if geo.get(key) not in (None, ""):
                have.add("revision")
                break
        layers = geo.get("layers") or []
        if layers and any(l.get("ip_mean") is not None for l in layers):
            have.add("ip_mean")
        if layers and any(l.get("gram_cumulants") for l in layers):
            have.add("gram_cumulants")
        if layers and any(l.get("ip_histogram") for l in layers):
            have.add("ip_histogram")
        if layers and any(l.get("gate_rank") is not None for l in layers):
            have.add("gate_provenance")
        # The extraction convention. p1_io._PROVENANCE_FIELDS writes these
        # at geometry.json's top level, which means Phase 2d's frame
        # resolution can READ the convention instead of being told it —
        # see p2d_io.extraction_convention. A run missing them forces the
        # caller to assert a convention by hand, which is the error class
        # the fields exist to prevent.
        if geo.get("hidden_state_0_is_embedding") is not None:
            have.add("extraction_convention")

    if (run_dir / "energies.json").exists():
        have.add("energies")

    act_p = run_dir / "activations.npz"
    if act_p.exists():
        have.add("activations")
        keys = _npz_keys(act_p)
        if "norms" in keys:
            have.add("norms")

    cl_p = run_dir / "clusters.npz"
    if cl_p.exists():
        keys = _npz_keys(cl_p)
        if any(k.startswith("kmeans_centroids_") for k in keys):
            have.add("clusters")
            have.add("clusters_kmeans")
        if any(k.startswith("agglom_mid_labels_") for k in keys):
            have.add("clusters_agglom")
        if any(k.startswith("hdbscan_labels_") for k in keys):
            have.add("clusters_hdbscan")

    sk_p = run_dir / "sinkhorn.json"
    if sk_p.exists():
        try:
            with open(sk_p) as f:
                sk = json.load(f)
            layers = sk.get("layers") or sk if isinstance(sk, list) else []
            if layers and any(l.get("fiedler_per_head") for l in layers):
                have.add("fiedler_per_head")
        except Exception:
            pass

    return {
        "run_dir": str(run_dir), "name": run_dir.name,
        "model": (geo or {}).get("model"), "prompt": (geo or {}).get("prompt"),
        "n_tokens": (geo or {}).get("n_tokens"),
        "have": have, "notes": notes,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--p2-dir", type=Path, default=None)
    ap.add_argument("--verbose", action="store_true",
                    help="list every excluded run, not just the counts")
    args = ap.parse_args(argv)

    try:
        from p1_mstate_tracking.visualization.loaders import discover_runs
        runs = {k: v for k, v in discover_runs(args.results).items()}
        dirs = sorted(runs.values())
    except Exception:
        # Fall back to a filesystem walk. discover_runs may itself need
        # artifacts this tool exists to check for, and a preflight that
        # cannot run on an incomplete tree is useless.
        dirs = sorted({p.parent for p in args.results.rglob("geometry.json")})

    if not dirs:
        print(f"no Phase 1 runs found under {args.results}", file=sys.stderr)
        return 1

    infos = [inspect_run(d) for d in dirs]
    print(f"{len(infos)} runs under {args.results}\n")

    # --- capability coverage ---
    caps = Counter()
    for i in infos:
        caps.update(i["have"])
    print("CAPABILITY COVERAGE")
    order = ["activations", "norms", "beta_eff", "revision", "ip_mean",
             "energies", "ip_histogram", "gram_cumulants", "gate_provenance",
             "extraction_convention", "clusters_kmeans", "clusters_agglom",
             "clusters_hdbscan", "fiedler_per_head"]
    for c in order:
        n = caps.get(c, 0)
        bar = "#" * int(30 * n / len(infos))
        flag = "" if n == len(infos) else "   <-- incomplete"
        print(f"  {c:<18} {n:>4}/{len(infos)}  {bar:<30}{flag}")

    # --- runnability per sub-experiment ---
    print("\nRUNNABLE PER SUB-EXPERIMENT")
    for sub, req in REQUIREMENTS.items():
        ok = [i for i in infos if req <= i["have"]]
        missing = Counter()
        for i in infos:
            for r in req - i["have"]:
                missing[r] += 1
        line = f"  {sub:<7} {len(ok):>4}/{len(infos)}"
        if missing:
            line += "   blocked by: " + ", ".join(
                f"{k} ({v})" for k, v in missing.most_common())
        print(line)

    # --- consequences, stated once each ---
    blocking = set()
    for sub, req in REQUIREMENTS.items():
        for i in infos:
            blocking |= (req - i["have"])
    if blocking:
        print("\nWHAT TO DO")
        for b in sorted(blocking):
            print(f"  {b}: {CONSEQUENCE.get(b, 'missing')}")

    if args.verbose:
        print("\nPER-RUN DETAIL")
        for i in infos:
            miss = sorted(set().union(*REQUIREMENTS.values()) - i["have"])
            print(f"  {i['name']:<40} missing: {', '.join(miss) or 'nothing'}")
            for n in i["notes"]:
                print(f"      ! {n}")

    # --- Phase 2 side ---
    if args.p2_dir:
        print(f"\nPHASE 2 OPERATORS under {args.p2_dir}")
        sums = sorted(args.p2_dir.glob("ov_summary_*.json"))
        if not sums:
            print("  none found")
        for s in sums:
            stem = s.name[len("ov_summary_"):-len(".json")]
            w = args.p2_dir / f"ov_weights_{stem}.npz"
            keys = _npz_keys(w) if w.exists() else set()
            has_qk = any(k.startswith("wq_head") for k in keys)
            print(f"  {stem:<40} weights={'yes' if w.exists() else 'NO':<4} "
                  f"W_Q/W_K={'yes' if has_qk else 'NO'}"
                  f"{'' if has_qk else '   <-- Phase 2d needs M_h; re-run extraction'}")

    # Exit non-zero when the two headline blockers bite, so this can gate a
    # scheduling script rather than only inform a human.
    n_norms = caps.get("norms", 0)
    n_beta = caps.get("beta_eff", 0)
    if n_norms < len(infos) or n_beta < len(infos):
        print(f"\nBLOCKED: norms {n_norms}/{len(infos)}, "
              f"beta_eff {n_beta}/{len(infos)}. See WHAT TO DO above.")
        return 2
    print("\nAll runs support every sub-experiment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
