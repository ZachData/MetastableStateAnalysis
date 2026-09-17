"""
tools/run/isometric_path_sweep.py — §2.5's isometric path on `L7H8`
(`MATH_SPECTRAL_OT.md` §2.5), run for real. The only *designed* particle
intervention in this project (`PROJECT.md`'s resume block, item 2 as of
2026-09-16) and still unrun before this module.

WHAT §2.5 BUILDS, IN ONE PARAGRAPH. `L7H8`'s OV operator `M = A @ B`
(`A` `(d_model, d_head)`, `B` `(d_head, d_model)`) has thin SVD
`M = U Sigma V^T`. Because `U` and `V` are both Stiefel points
(`X^T X = I_k`), a smooth path `gamma(t)` from `U` to `V` gives
`M(t) = gamma(t) Sigma gamma(1-t)^T` — an EXACT ISOMETRY at every `t`
(same singular values, same Frobenius norm, same rank as `M`), with
`M(0) = M` and `M(1) = M^T`. `M(1/2) = gamma(1/2) Sigma gamma(1/2)^T` is
symmetric PSD. `L7H8`'s OV core is 100% repulsive (every eigenvalue has
negative real part); the path therefore sweeps

    t = 0     the head's own spectrum        100% repulsive
    t = 1/2   symmetric PSD                   100% attractive
    t = 1     the head's own spectrum again  100% repulsive

WHY THIS IS THE EXPERIMENT `PROJECT.md` §3.12-D WANTED AND THE FOUR-CORNER
PILOT (§3.12-M) COULD NOT GIVE. The naive arithmetic mean `(M + M^T)/2` is
NOT WRITABLE BY A RANK-`k` HEAD (`rank((M+M^T)/2) = 2k` generically, `M`
itself has rank `<= k`) — "a head cannot write its own symmetric part"
(§2.5.1). Interpolating the SVD FRAMES instead of the matrix sidesteps
that obstruction entirely: every intermediate `M(t)` is exactly
realisable as a rank-`k` OV operator, so it can be written into the model
and run for real, at matched energy, rather than only evaluated at the
unrealisable arithmetic-mean corner.

THE CONSTRUCTION, CLOSED FORM (§2.5.2). No geodesic is needed — a chord's
polar retraction suffices: `Y(t) = (1-t) U + t V`,
`gamma(t) = Y(t) (Y(t)^T Y(t))^{-1/2}`. `gamma(0) = U`, `gamma(1) = V`
because the polar factor of a Stiefel point is itself.
**REFUSAL CONDITION, checked before any forward pass**:
`sigma_min(Y(t)) > 0` on `[0, 1]` — fails only if some principal angle
between `U` and `V` reaches `pi`. `check_refusal` computes this on a fine
grid and the sweep does not run if it fails anywhere.

TARGET: `L7H8` on pythia-410m at `step4000` — the same target
`tools/run/induction_rank_sweep.py` (Stage 1 of the bottom-up induction
programme, `PROJECT.md` §3.11) already uses for this exact circuit's OV
analysis, reused rather than re-chosen. `ov_factors`/`write_ov` (weight
extraction/write-back, save-restore around every measurement) and
`measure`/`induction_batch` (the copying-side NLL/KL readout — the
behavioural induction score is a QK quantity OV cannot move within the
layer, so this module reads the COPYING side exactly as Stage 1 does) are
imported from that module rather than reimplemented.

WHAT THE CURVE ANSWERS, AND WHAT IT DOES NOT SEPARATE (§2.5.3). If
second-copy NLL stays near baseline at `t = 0` and `t = 1` (both
100%-repulsive, matched energy) but degrades toward `t = 1/2`
(100%-attractive, same energy), that is a causal readout supporting the
project's central spectral frame: the repulsive character is load-bearing
for copying, not an artifact of magnitude. Symmetry and read/write
alignment move together along this one path by construction (§2.5's own
caveat), so a response curve here shows WHETHER the spectral character
matters, not WHICH of the two (symmetry vs. alignment) carries it — that
separation is §2.5.4's second family (`M_R = U R Sigma V^T`, holding both
subspaces fixed and rotating only the correspondence), not built here.

STATUS: exploratory. No p-value, nothing registered — `PREDICTIONS.md`
names no `P-*` id for this curve as of 2026-09-16, and `claims/
registry.json` is untouched by this module regardless of what it finds.

The pure-math primitives (`polar_frame`, `check_refusal`, `build_M_t`,
`RunRefused`) live in `core/isometric_path.py`, not here — this module
imports `tools.run.induction_rank_sweep`, which imports torch at module
level, so keeping the math here would make it untestable in the pure
tier. `tests/test_isometric_path_sweep.py` covers the math directly;
this module's own real-model sweep is validated by the committed run
(`data/analysis/isometric_path_L7H8_step4000.json`).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

_want_prefix = str(REPO / ".venv")
if not sys.prefix.startswith(_want_prefix):
    raise SystemExit(f"wrong interpreter: sys.prefix={sys.prefix!r}, need {_want_prefix!r}")

from tools.run.induction_rank_sweep import (  # noqa: E402
    EVAL_SEED, N_REP, N_SEQS, VOCAB_LO, VOCAB_HI,
    induction_batch, measure, ov_factors, write_ov,
)
from core.isometric_path import (  # noqa: E402
    RunRefused, check_refusal, build_M_t,
)

LAYER, HEAD = 7, 8
STEP = 4000


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def run_sweep(step: int = STEP, layer: int = LAYER, head: int = HEAD,
              n_t: int = 11, seed: int = EVAL_SEED, rank: int = None) -> dict:
    """
    `rank` : truncate `(U, S, V)` to the top `rank` SVD modes before
    building the path, or `None` for the full `d_head` width. FULL RANK
    REFUSES FOR L7H8 (verified 2026-09-16): its tail singular values are
    numerically zero (~1e-18 relative, machine noise — consistent with
    `PROJECT.md` §3.11's own r*=1 finding that a single direction carries
    97% of this head's causal effect), and Y(0.5) = 0.5(U+V) is singular
    to machine precision in those directions, refusing exactly at t=0.5.
    Truncating to any `rank <= 32` clears the refusal comfortably
    (`min_sigma_min` >= 0.38 at rank 32, checked directly, not assumed).
    `main()`'s default runs several ranks rather than picking one — energy
    and causal relevance diverge for this head throughout this project's
    own findings (`r*_SVD << r*_Schur`), so no single truncation is
    obviously "the" fair one.

    `kl_from_baseline` at every `t`, every rank, is measured against the
    SAME true (untruncated) model baseline — comparable across ranks. Each
    curve's own `t = 0` row is therefore not "no effect": it is the cost
    of truncating to `rank` alone, before any path movement, and is worth
    reading against Stage 1's own already-measured truncation curves
    (`data/analysis/induction_rank_sweep_s{step}_L7H8.json`) as a
    consistency cross-check — verified on disk for steps 1, 2, 4, 8, 32,
    64, 2000, 8000, 16000, 143000, NOT step 4000 specifically, so this
    module's own `t = 0` row is the only same-checkpoint reference until
    one is run there too.
    """
    from core.lm_loading import load_causal_lm

    model, tokenizer = load_causal_lm(
        f"pythia-410m-step{step}",
        device="cpu",
    )
    model.eval()

    A0, B0 = ov_factors(model, layer, head)
    M0 = A0 @ B0
    U_full, S_full, Vt_full = np.linalg.svd(M0, full_matrices=False)
    V_full = Vt_full.T

    if rank is None:
        U, S, V = U_full, S_full, V_full
    else:
        U, S, V = U_full[:, :rank], S_full[:rank], V_full[:, :rank]

    refusal = check_refusal(U, V)
    if not refusal["ok"]:
        raise RunRefused(
            f"§2.5's refusal condition fails at rank={rank}: min "
            f"sigma_min(Y(t)) = {refusal['min_sigma_min']:.3e} <= 0 "
            f"somewhere on [0,1] — a principal angle between U and V "
            f"reaches pi."
        )

    ids = induction_batch(np.random.default_rng(seed))
    true_baseline = measure(model, ids)
    base_lp = true_baseline.pop("logprobs")
    true_baseline["kl_from_baseline"] = 0.0

    rows = []
    for t in np.linspace(0.0, 1.0, n_t):
        built = build_M_t(U, S, V, float(t))
        if built is None:
            rows.append({"t": float(t), "refused": True})
            continue
        A_t, B_t = built
        write_ov(model, layer, head, A_t, B_t)
        try:
            m = measure(model, ids)
        finally:
            write_ov(model, layer, head, A0, B0)  # restore, every iteration
        lp = m.pop("logprobs")
        kl = float((base_lp.exp() * (base_lp - lp)).sum(-1).mean())
        rows.append({"t": float(t), "refused": False, "kl_from_baseline": kl, **m})

    # Baseline re-measured after the whole sweep, asserted equal to the
    # first one -- the save/restore convention induction_rank_sweep.py's
    # own docstring states and this module inherits.
    A_final, B_final = ov_factors(model, layer, head)
    restore_error = float(np.linalg.norm(A_final @ B_final - M0) / np.linalg.norm(M0))
    final_check = measure(model, ids)
    final_check.pop("logprobs")
    weights_restored_exactly = restore_error < 1e-10
    baseline_reproduced = abs(
        final_check["second_copy_nll"] - true_baseline["second_copy_nll"]
    ) < 1e-6

    return {
        "target": {"layer": layer, "head": head, "step": step},
        "rank": rank,
        "eval": {"n_rep": N_REP, "n_seqs": N_SEQS, "seed": seed,
                 "vocab_range": [VOCAB_LO, VOCAB_HI], "n_t": n_t},
        "refusal_check": refusal,
        "true_baseline": true_baseline,
        "rows": rows,
        "weights_restored_exactly": weights_restored_exactly,
        "restore_error": restore_error,
        "baseline_reproduced_after_sweep": baseline_reproduced,
    }


#: No single truncation is obviously "the" fair one for this head (energy
#: and causal relevance diverge throughout this project's own findings) —
#: run a small, informative set rather than pick one. 1 is §3.11's own
#: r* (97% of causal effect); 8 and 16 add back real structure beyond it
#: while staying comfortably clear of the refusal (min_sigma_min >= 0.5,
#: >= 0.43 respectively, verified 2026-09-16).
DEFAULT_RANKS = (1, 8, 16)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step", type=int, default=STEP)
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--head", type=int, default=HEAD)
    ap.add_argument("--n-t", type=int, default=11)
    ap.add_argument("--ranks", default=",".join(str(r) for r in DEFAULT_RANKS))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ranks = [int(r) for r in args.ranks.split(",")]
    all_results = {}
    for rank in ranks:
        print(f"=== rank={rank} ===")
        result = run_sweep(step=args.step, layer=args.layer, head=args.head,
                            n_t=args.n_t, rank=rank)
        all_results[str(rank)] = result
        print(f"  refusal check: ok={result['refusal_check']['ok']} "
              f"(min sigma_min={result['refusal_check']['min_sigma_min']:.3e})")
        print(f"  true baseline: second-copy NLL "
              f"{result['true_baseline']['second_copy_nll']:.4f}  "
              f"first-copy {result['true_baseline']['first_copy_nll']:.4f}")
        for row in result["rows"]:
            if row.get("refused"):
                print(f"    t={row['t']:.2f}  REFUSED")
                continue
            print(f"    t={row['t']:.2f}  second-copy NLL "
                  f"{row['second_copy_nll']:.4f}  KL {row['kl_from_baseline']:.4f}")
        print(f"  weights restored exactly: {result['weights_restored_exactly']} "
              f"(rel error {result['restore_error']:.2e})")
        print(f"  baseline reproduced after sweep: "
              f"{result['baseline_reproduced_after_sweep']}")

    if args.out:
        out_path = REPO / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        git_sha = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip()
        doc = {
            "_what_this_is": (
                "§2.5's isometric path on L7H8 (MATH_SPECTRAL_OT.md §2.5): "
                "an exact-isometry sweep of the OV operator's read/write "
                "frames from the head's own spectrum, through symmetric "
                "PSD, to the transpose. Readout is second-copy NLL / KL on "
                "repeated random sequences, matching Stage 1's OV-focused "
                "convention (induction_rank_sweep.py). Full rank (k=64) "
                "refuses for L7H8 -- its tail singular values are "
                "numerically zero -- so this runs at several truncated "
                "ranks instead of one; see run_sweep's docstring."
            ),
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "git_sha": git_sha,
            "ranks": all_results,
        }
        out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        print(f"wrote {out_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
