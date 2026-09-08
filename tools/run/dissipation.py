"""Tier A of the dissipation-identity run (docs/dissipation_checkpoint_axis_scoping.md).

The sibling of `relay_null.py` / `behavioural.py`. Evaluates `core.dissipation`
across the registered 19-step P-I1 grid x 7 scored prompts x 24 layer
boundaries, from artifacts ALREADY ON DISK — no forward pass, no model load:

  X, dX      : data/phase12/<ts>/pythia-410m-step<S>_<prompt>/activations.npz
               (raw residual stream, 25 rows = emb + 24 blocks; dX_l = row l+1 - row l)
  P_attract  : data/phase12/p2_eigenspectra_<ts>/ov_projectors_pythia-410m-step<S>.npz
  P_repulse    keys schur_attract_layer_<l> / schur_repulse_layer_<l>

Per (step, prompt, layer) it records the first-order energy change and its
per-particle attribution, the attractive/repulsive-subspace split of the TOTAL
displacement (see the caveat below), the gradient-flow alignment distribution,
and the second-order linearisation residual. Reductions only — per-particle
arrays are not persisted in Tier A.

CAVEAT, recorded in the artifact: `dissipation_by_subspace` here projects the
TOTAL dX (attn + ffn) through the layer's aggregated-OV Schur subspaces. The
FFN part of dX is not an OV output, so this is a consistent orthogonal split of
dX, not an attention-channel measurement. The attention-only and per-head
versions are Tier B / v2 (they need a sublayer-capture forward pass).

PATHS ARE DERIVED as in the sibling runners: METS_REPO / METS_DATA override.
METS_DISS_BETAS (comma list, default "1.0,2.0") overrides the beta set.
"""
import json
import os
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
from core.dissipation import (
    dissipation,
    dissipation_by_subspace,
    gradient_flow_alignment,
)

STEPS = list(REGISTERED_P_I1_SWEEP)
BETAS = [float(b) for b in os.environ.get("METS_DISS_BETAS", "1.0,2.0").split(",")]
SCORED_PROMPTS = [
    "camus_letranger", "hdbscan_code", "homer_iliad", "latex_monograph",
    "paper_excerpt", "sullivan_ballou", "wiki_paragraph",
]
CARRIED_BESIDE = "repeated_tokens"          # never scored (degenerate input)
N_BLOCKS = 24                              # pythia-410m


def _phase1_run_dir(step: int, prompt: str) -> Path:
    hits = [
        p.parent
        for p in DATA.glob(f"phase12/*/pythia-410m-step{step}_{prompt}/activations.npz")
        if not p.parent.parent.name.startswith("p2_eigenspectra_")
    ]
    if len(hits) != 1:
        raise SystemExit(f"step {step} {prompt!r}: {len(hits)} Phase 1 run dirs, need 1")
    return hits[0]


def _projector_npz(step: int) -> Path:
    hits = list(DATA.glob(f"phase12/p2_eigenspectra_*/ov_projectors_pythia-410m-step{step}.npz"))
    if len(hits) != 1:
        raise SystemExit(f"step {step}: {len(hits)} ov_projectors npz, need 1")
    return hits[0]


def _reduce_gfa(g: dict) -> dict:
    return {
        "mean": g["mean"], "median": g["median"], "std": g["std"],
        "q10": g["q10"], "q90": g["q90"], "frac_descending": g["frac_descending"],
        "n_defined": g["n_defined"], "n_undefined": g["n_undefined"], "status": g["status"],
    }


def _lib_versions() -> dict:
    import torch  # noqa: only for the version string
    import transformers
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__, "scipy": scipy.__version__,
        "torch": torch.__version__, "transformers": transformers.__version__,
    }


def main() -> None:
    import subprocess
    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()

    # provenance of every input dir we touch, so a stale-data audit is one read
    input_provenance = {}

    out = {
        "_what_this_is":
            "Tier A of the dissipation-identity run "
            "(docs/dissipation_checkpoint_axis_scoping.md). core.dissipation "
            "evaluated per (step, prompt, layer) on raw residual-stream states "
            "from activations.npz, with the attractive/repulsive split taken "
            "against the per-checkpoint aggregated-OV Schur projectors. "
            "NO forward pass. frame=l2_sphere.",
        "_subspace_caveat":
            "dissipation_by_subspace projects the TOTAL dX (attn+ffn) through "
            "the layer's OV Schur subspaces. This is a consistent orthogonal "
            "split of dX, NOT an attention-channel measurement — the FFN part "
            "of dX is not an OV output. Attention-only / per-head splits need a "
            "sublayer-capture forward pass (Tier B / v2).",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "git_sha": git_sha,
        "lib_versions": _lib_versions(),
        "frame": "l2_sphere",
        "betas": BETAS,
        "steps": STEPS,
        "scored_prompts": SCORED_PROMPTS,
        "carried_beside": CARRIED_BESIDE,
        "n_blocks": N_BLOCKS,
        "per_step_layer": {},     # "<beta>|<step>|<prompt>|<layer>" -> reductions
        "pooled_by_step_layer": {},  # "<beta>|<step>|<layer>" -> 7-prompt pooled
        "input_provenance": input_provenance,
        "max_subspace_sum_check": 0.0,
    }

    prompts_all = SCORED_PROMPTS + [CARRIED_BESIDE]
    t_start = time.time()

    for si, step in enumerate(STEPS):
        pnpz_path = _projector_npz(step)
        input_provenance[f"projectors_step{step}"] = pnpz_path.parent.name
        pz = np.load(pnpz_path, allow_pickle=False)

        # sanity on one layer's projector pair (cheap; catches a bad file early)
        Pa0 = pz["schur_attract_layer_0"].astype(np.float64)
        Pr0 = pz["schur_repulse_layer_0"].astype(np.float64)
        d = Pa0.shape[0]
        assert np.abs(Pa0 - Pa0.T).max() < 1e-9, f"step{step} P_attract not symmetric"
        assert np.abs(Pa0 @ Pa0 - Pa0).max() < 1e-6, f"step{step} P_attract not idempotent"
        assert np.abs(Pa0 + Pr0 - np.eye(d)).max() < 1e-6, f"step{step} Pa+Pr != I"

        step_t0 = time.time()
        pooled = {b: {l: {"first_order": 0.0, "actual_delta_E": 0.0,
                          "d_attractive": 0.0, "d_repulsive": 0.0,
                          "abs_residual": 0.0, "n_prompts": 0}
                      for l in range(N_BLOCKS)} for b in BETAS}

        for prompt in prompts_all:
            rd = _phase1_run_dir(step, prompt)
            if prompt == SCORED_PROMPTS[0]:
                input_provenance[f"phase1_step{step}"] = rd.parent.name
            with np.load(rd / "activations.npz", allow_pickle=False) as z:
                act = z["activations"].astype(np.float64)   # (25, n, 1024)
            if act.shape[0] != N_BLOCKS + 1:
                raise SystemExit(f"step{step} {prompt}: {act.shape[0]} rows, need {N_BLOCKS+1}")

            for l in range(N_BLOCKS):
                X = act[l]
                dX = act[l + 1] - act[l]
                Pa = pz[f"schur_attract_layer_{l}"].astype(np.float64)
                Pr = pz[f"schur_repulse_layer_{l}"].astype(np.float64)

                for b in BETAS:
                    dsp = dissipation(X, dX, b)
                    sub = dissipation_by_subspace(X, dX, b, Pa, Pr)
                    gfa = gradient_flow_alignment(X, dX, b)
                    out["max_subspace_sum_check"] = max(
                        out["max_subspace_sum_check"], float(sub["sum_check"]))

                    key = f"{b}|{step}|{prompt}|{l}"
                    out["per_step_layer"][key] = {
                        "first_order": dsp["first_order"],
                        "actual_delta_E": dsp["actual_delta_E"],
                        "residual": dsp["residual"],
                        "relative_residual": dsp["relative_residual"],
                        "step_size": dsp["step_size"],
                        "status": dsp["status"],
                        "d_attractive": sub["attractive"],
                        "d_repulsive": sub["repulsive"],
                        "subspace_sum_check": float(sub["sum_check"]),
                        "gfa": _reduce_gfa(gfa),
                    }

                    if prompt in SCORED_PROMPTS:
                        pp = pooled[b][l]
                        pp["first_order"] += dsp["first_order"]
                        pp["actual_delta_E"] += (dsp["actual_delta_E"] or 0.0)
                        pp["d_attractive"] += sub["attractive"]
                        pp["d_repulsive"] += sub["repulsive"]
                        pp["abs_residual"] += abs(dsp["residual"] or 0.0)
                        pp["n_prompts"] += 1

        pz.close()

        for b in BETAS:
            for l in range(N_BLOCKS):
                pp = pooled[b][l]
                mag = abs(pp["d_attractive"]) + abs(pp["d_repulsive"])
                pp["repulsive_share"] = (abs(pp["d_repulsive"]) / mag) if mag > 0 else None
                out["pooled_by_step_layer"][f"{b}|{step}|{l}"] = pp

        dt = time.time() - step_t0
        # progress line: total first-order dissipation across scored prompts+layers at beta=1
        b1 = BETAS[0]
        tot_fo = sum(pooled[b1][l]["first_order"] for l in range(N_BLOCKS))
        tot_rep = sum(pooled[b1][l]["d_repulsive"] for l in range(N_BLOCKS))
        n_pos = sum(1 for l in range(N_BLOCKS) if pooled[b1][l]["first_order"] > 0)
        print(f"step{step:<7d} {dt:6.1f}s  b{b1}  Sigma_first_order={tot_fo:+.4e}  "
              f"Sigma_d_repulsive={tot_rep:+.4e}  layers_uphill={n_pos}/{N_BLOCKS}",
              flush=True)

    out["_elapsed_seconds"] = round(time.time() - t_start, 1)
    outdir = Path(os.environ.get("METS_SCRATCH", str(DATA / "analysis")))
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / "dissipation_series.json"
    json.dump(out, open(dest, "w"), indent=1)
    print(f"\nmax subspace sum_check over the whole run: {out['max_subspace_sum_check']:.2e}")
    print(f"WROTE {dest}  ({dest.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
