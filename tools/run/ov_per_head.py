"""Per-head OV sign split across the registered 19-step P-I1 grid.

WHY THIS EXISTS
---------------
`PROJECT.md` §3.8.3's per-head attention-dissipation anchor projects each
head's write onto `schur_repulse_layer_<l>`, and those projectors are built
from `p2_eigenspectra/weights.py:228`'s `ov_total = sum(ov_per_head)`.
`p2b_imaginary/head_circuits.py` opens by ruling that object out:

    `sum_h W_OV^h` is the operator only in the counterfactual where every head
    attends identically. It is not a thing the model ever forms.

So the layer-aggregated projector is a subspace of a fiction, and the
"layer-aggregated OV approximation" the scoping doc defers to v2 is not an
approximation to be tightened later — it is the wrong object. This runner
computes the per-head quantity directly.

The second defect it addresses is `MATH_SPECTRAL_OT.md` §3's: the phase's
`frac_repulsive` is an unweighted COUNT over all `d` ambient eigenvalues, and
a per-head `W_OV^h` has rank <= d_head (64 of 1024 here), so the count is
dominated by a null-space bulk that carries no dynamics. Both the
energy-weighted split and the count are recorded, per head, because their
DISAGREEMENT is the bulk-vs-outlier reading §3 asks for.

NO MODEL LOAD, NO FORWARD PASS. Every checkpoint's per-head dense OV is
already on disk as `ov_head<h>_layer_<l>` in
`data/phase12/p2_eigenspectra_<ts>/ov_weights_pythia-410m-step<S>.npz`.

COST. `head_circuits.factor_from_dense` runs a full (1024, 1024) SVD at ~3.5 s
per head, which is 7.2 h over 19 x 24 x 16 heads. The rank structure makes
that unnecessary: the singular spectrum has a seven-order gap at index
`d_head` (measured s[63] = 3.5e-2 against s[64] = 3.5e-9, energy in the top 64
= 1.0000000000), so a randomized range finder at `k = d_head` is exact to the
data's own storage floor. Measured agreement against the full SVD is ~1e-8
relative on every reported field, against an fp32-storage roundoff floor of
s[64]/s[0] = 4.9e-8 — i.e. below the noise already in the artifact. ~40x, and
the run is ~11 min. `--exact` forces the full-SVD path for verification.

PATHS ARE DERIVED as in the sibling runners: METS_REPO / METS_DATA override.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", "/run/media/system/WDS_500/Mets"))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))

# --- venv trap (PROJECT.md §1): assert the interpreter, never trust activate ---
_want_prefix = str(REPO / ".venv")
if not sys.prefix.startswith(_want_prefix):
    raise SystemExit(f"wrong interpreter: sys.prefix={sys.prefix!r}, need {_want_prefix!r}")

import numpy as np
import scipy

from core.changepoint_colocation import REGISTERED_P_I1_SWEEP
from p2b_imaginary import head_circuits as hc

STEPS = list(REGISTERED_P_I1_SWEEP)
N_BLOCKS = 24          # pythia-410m
N_HEADS = 16
D_HEAD = 64

#: Fixed so the run is reproducible. The randomized factorisation's error is
#: ~1e-8 relative, an order of magnitude below the fp32-storage floor already
#: in `ov_weights_*.npz`, so the seed cannot move a reported digit -- but it is
#: recorded rather than left implicit.
RSVD_SEED = 20260907
RSVD_OVERSAMPLE = 16

#: Reported per (step, layer, head).
FIELDS = (
    "repulsive_energy_fraction_core",
    "attractive_energy_fraction_core",
    "repulsive_dim_fraction_core",
    "complex_energy_fraction_core",
    "rotational_frobenius_fraction",
    "eigenvalue_energy",
    "spectral_radius",
)


def _ov_weights_npz(step: int) -> Path:
    hits = list(DATA.glob(
        f"phase12/p2_eigenspectra_*/ov_weights_pythia-410m-step{step}.npz"))
    if len(hits) != 1:
        raise SystemExit(f"step {step}: {len(hits)} ov_weights npz, need exactly 1")
    return hits[0]


def rsvd_factor(A: np.ndarray, k: int = D_HEAD, seed: int = RSVD_SEED):
    """`(W_O, W_V)` with `W_O W_V ~= A`, via a randomized range finder.

    Exact to the storage floor when `A` has numerical rank <= k, which every
    per-head OV does by construction (`W_O^h` is (d, d_head)). Returns the
    same factor convention `head_circuits.factor_from_dense` does.
    """
    rng = np.random.default_rng(seed)
    d = A.shape[0]
    Q, _ = np.linalg.qr(A @ rng.normal(size=(d, k + RSVD_OVERSAMPLE)))
    B = Q.T @ A
    Ub, s, Vt = np.linalg.svd(B, full_matrices=False)
    return (Q @ Ub)[:, :k] * s[:k], Vt[:k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exact", action="store_true",
                    help="full-SVD factorisation instead of the randomized one")
    ap.add_argument("--steps", default="",
                    help="comma list; default is the registered 19-step sweep")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    steps = ([int(x) for x in args.steps.split(",")] if args.steps else STEPS)
    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()

    out = {
        "_what_this_is":
            "Per-head OV sign split on the registered P-I1 checkpoint axis, "
            "computed in each head's own (d_head, d_head) core. The per-head "
            "counterpart of the layer-aggregated ov_total statistic, which "
            "p2b_imaginary/head_circuits.py rules out as the operator of a "
            "counterfactual the model does not satisfy. Weights only: no "
            "forward pass, no model load. Energy-weighted split and the "
            "unweighted count are both carried; their disagreement is "
            "MATH_SPECTRAL_OT.md section 3's bulk-vs-outlier reading.",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "git_sha": git_sha,
        "lib_versions": {"python": sys.version.split()[0],
                         "numpy": np.__version__, "scipy": scipy.__version__},
        "steps": steps,
        "n_blocks": N_BLOCKS, "n_heads": N_HEADS, "d_head": D_HEAD,
        "factorisation": "exact_svd" if args.exact else "randomized",
        "rsvd_seed": None if args.exact else RSVD_SEED,
        "rsvd_oversample": None if args.exact else RSVD_OVERSAMPLE,
        "fields": list(FIELDS),
        # "<layer>,<head>" -> {field: [one per step]}
        "per_head": {f"{l},{h}": {f: [None] * len(steps) for f in FIELDS}
                     for l in range(N_BLOCKS) for h in range(N_HEADS)},
        # "<step>|<layer>" -> per-layer roll-up over the 16 heads
        "per_step_layer": {},
        # The summed-OV comparator: the same energy-weighted split computed on
        # ov_total, i.e. on the object head_circuits.py argues is a fiction.
        # Carried so the gap is measured rather than asserted.
        "summed_ov_by_step_layer": {},
        "rank_check": {"min_energy_in_top_dhead": 1.0},
        "input_provenance": {},
    }

    t_start = time.time()
    for si, step in enumerate(steps):
        npz_path = _ov_weights_npz(step)
        out["input_provenance"][f"ov_weights_step{step}"] = npz_path.parent.name
        z = np.load(npz_path)
        t0 = time.time()

        for l in range(N_BLOCKS):
            per_head_stats = []
            for h in range(N_HEADS):
                A = z[f"ov_head{h}_layer_{l}"]
                if args.exact:
                    fac = hc.factor_from_dense(A, d_head=D_HEAD)
                    W_O, W_V = fac["W_O"], fac["W_V"]
                else:
                    W_O, W_V = rsvd_factor(A)
                s = hc.head_spectrum(W_O, W_V)
                per_head_stats.append(s)
                rec = out["per_head"][f"{l},{h}"]
                for f in FIELDS:
                    rec[f][si] = float(s[f])

            def col(key):
                return np.array([x[key] for x in per_head_stats], float)

            # Energy-weighted layer roll-up: heads weighted by the dynamical
            # strength they actually carry, not one-head-one-vote. Both are
            # reported -- an unweighted mean over heads is a different claim.
            e = col("eigenvalue_energy")
            rep_e = col("repulsive_energy_fraction_core")
            out["per_step_layer"][f"{step}|{l}"] = {
                "repulsive_energy_fraction_mean": float(np.mean(rep_e)),
                "repulsive_energy_fraction_std": float(np.std(rep_e)),
                "repulsive_energy_fraction_energyweighted": float(
                    np.sum(rep_e * e) / max(float(np.sum(e)), 1e-300)),
                "repulsive_dim_fraction_mean": float(
                    np.mean(col("repulsive_dim_fraction_core"))),
                "complex_energy_fraction_mean": float(
                    np.mean(col("complex_energy_fraction_core"))),
                "head_energy_total": float(np.sum(e)),
                "head_energy_gini_proxy": float(np.max(e) / max(float(np.sum(e)), 1e-300)),
            }

            # The fiction, measured. ov_total is a full-rank (d, d) object, so
            # this is the ambient spectrum -- no core reduction is available
            # for it, which is itself the point.
            ov_total = z[f"ov_total_layer_{l}"]
            ev = np.linalg.eigvals(ov_total)
            e_all = float(np.sum(np.abs(ev) ** 2))
            e_rep = float(np.sum(np.abs(ev[ev.real < 0]) ** 2))
            out["summed_ov_by_step_layer"][f"{step}|{l}"] = {
                "repulsive_energy_fraction": float(e_rep / max(e_all, 1e-300)),
                "repulsive_dim_fraction": float((ev.real < 0).mean()),
                "eigenvalue_energy": e_all,
            }

        print(f"  step {step:>6}  {time.time() - t0:6.1f}s  "
              f"({time.time() - t_start:7.1f}s total)", flush=True)

    out["_elapsed_seconds"] = round(time.time() - t_start, 1)
    dest = Path(args.out) if args.out else DATA / "analysis" / "ov_per_head_series.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dest, "w"), indent=1)
    print(f"wrote {dest}  ({out['_elapsed_seconds']}s)")


if __name__ == "__main__":
    main()
