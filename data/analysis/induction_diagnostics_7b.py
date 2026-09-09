"""Sub-phase 7b diagnostics — the checks that decide what a valid Stage 3
control is, run BEFORE the differential prediction is written.

WHY THIS EXISTS
---------------
`PROJECT.md` §3.12 blocks C-E. The drafted Stage 3 entry was not registrable:
its copy-matched control is probably unbuildable, its geometry contrast has a
mechanical component with no individuation content, and the direction it
perturbs is identified weights-only. Block D proposes replacing the whole
control design with a norm-preserving spectral surgery. Each check below can
change what that design is, and NONE of them produces a p-value -- so running
them first spends no registration (§6l's timing argument pointed the right way).

THE CHECKS
----------
0. (§3.12-A, 7a) COUNT vs ENERGY on the repulsive collapse. The U-shape is
   energy-weighted; if one dominant eigenvalue carries it the finding is much
   weaker. `head_spectrum` records both, and `ov_per_head.py`'s own docstring
   says their DISAGREEMENT is the bulk-vs-outlier reading. Nobody has compared
   them on the population axis.

1. (§3.12-D) THE S-FLIP IS `-M^T`, verified rather than asserted. For
   `M = S + A` with `S = (M+M^T)/2`: `M^T = S - A`, so `-M^T = -S + A`. That
   flips every eigenvalue's real part while preserving EVERY SINGULAR VALUE and
   the Frobenius norm exactly -- a matched control with no rescaling, no
   matching search and no tolerance. And it is WRITABLE BACK into the model in
   factored form: `W_O' = -W_V^T`, `W_V' = W_O^T` have exactly the parameter
   shapes the model already has. This check confirms all of that numerically at
   the data's own floor, because the entire block-D design rests on it.

2. (§3.12-E2) ROTATED OR SPREAD? `svd@1` falls 0.68 -> 0.20 across
   consolidation while `svd@2` stays high; §3.11 reads that as the copying
   action spreading to a second direction. Equally consistent: the direction
   ROTATED and rank 2 tracks a moving target. Tracked by the overlap of the top
   singular vector between adjacent checkpoints, and by the principal angles of
   the top-2 subspace. A rotating target would also mean "the rank-1 direction"
   is not one object across steps, which the Stage 3 entry assumes.

3. (§3.12-E3) THE ANTI-COPIERS AS A FOUND CONTROL. `L2H10` and `L9H8` have
   NEGATIVE `dOV_nll` -- ablating their OV IMPROVES second-copy prediction --
   while carrying the same representational description as `L7H8` (100%
   repulsive core, no token-identity diagonal). They are a naturally occurring
   negative control, not a constructed one, and therefore not vulnerable to the
   matching problem that killed the two constructed designs. This check asks
   whether anything SPECTRAL separates them from `L7H8` at all.

NOT HERE, and why. The input-whitening check (§3.12-C3 -- is the top singular
direction of `W_OV` still the top direction in the residual-stream metric?)
needs a covariance, `activation_cache/` is empty, and there is no cached
residual stream in this repository. It needs one model load and one forward
pass and is deliberately left to its own runner rather than bolted on here.
The `L5H2 -> L7H8` composition score (§3.12-E1) needs `W_K`, which needs a
model load per checkpoint; same reason.

NO MODEL LOAD, NO FORWARD PASS. Every checkpoint's per-head dense OV is already
on disk as `ov_head<h>_layer_<l>` in
`data/phase12/p2_eigenspectra_<ts>/ov_weights_pythia-410m-step<S>.npz`, and
`np.load` on an npz is lazy, so only the handful of heads named below are read.

PATHS ARE DERIVED as in the sibling runners: METS_REPO / METS_DATA override.
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", "/run/media/system/WDS_500/Mets"))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))

# --- venv trap (PROJECT.md §1): assert the interpreter, never trust activate ---
_want_prefix = str(REPO / ".venv")
if not sys.prefix.startswith(_want_prefix):
    raise SystemExit(f"wrong interpreter: sys.prefix={sys.prefix!r}, need {_want_prefix!r}")

import numpy as np

from p2b_imaginary import head_circuits as hc

OUT = DATA / "analysis" / "induction_diagnostics_7b.json"
SERIES = DATA / "analysis" / "ov_per_head_series.json"

D_HEAD = 64
RSVD_SEED = 0

#: (layer, head, label). L7H8 is the copier; L5H2 its prev-token partner;
#: L2H10 and L9H8 are the two heads with NEGATIVE dOV_nll (PROJECT.md §3.11
#: block B) and are check 3's found control.
HEADS = [(7, 8, "L7H8_copier"), (5, 2, "L5H2_prevtoken"),
         (2, 10, "L2H10_anticopier"), (9, 8, "L9H8_anticopier")]


def _ov_npz(step: int) -> Path:
    hits = sorted(DATA.glob(f"phase12/**/ov_weights_pythia-410m-step{step}.npz"))
    if not hits:
        raise FileNotFoundError(f"no ov_weights npz for step {step}")
    return hits[-1]


def _load_ov(step: int, layer: int, head: int) -> np.ndarray:
    """The dense (1024, 1024) `W_OV` for one head. Lazy: reads one key."""
    with np.load(_ov_npz(step)) as z:
        return np.asarray(z[f"ov_head{head}_layer_{layer}"], dtype=np.float64)


# ---------------------------------------------------------------------------
# check 0 -- count vs energy on the repulsive collapse
# ---------------------------------------------------------------------------

def check0_count_vs_energy() -> dict:
    with open(SERIES) as fh:
        d = json.load(fh)
    steps = d["steps"]
    recs = []

    def walk(o):
        if isinstance(o, dict):
            if "attractive_energy_fraction_core" in o:
                recs.append(o)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(d)

    keys = sorted(recs[0].keys()) if recs else []
    have_dim = "repulsive_dim_fraction_core" in keys

    rows = []
    for i, s in enumerate(steps):
        att = [r["attractive_energy_fraction_core"][i] for r in recs]
        rep_e = [1.0 - a for a in att]
        row = {
            "step": s,
            "n_heads": len(att),
            "median_repulsive_energy": float(np.median(rep_e)),
            "frac_heads_energy_exactly_1": float(np.mean([a == 0.0 for a in att])),
        }
        if have_dim:
            rd = [r["repulsive_dim_fraction_core"][i] for r in recs]
            row["median_repulsive_dim"] = float(np.median(rd))
            row["frac_heads_dim_exactly_1"] = float(np.mean([x == 1.0 for x in rd]))
            # The reading ov_per_head.py's docstring asks for: do the two agree?
            row["median_energy_minus_dim"] = float(
                np.median([e - x for e, x in zip(rep_e, rd)]))
        rows.append(row)

    return {
        "_what": "count-vs-energy on the repulsive collapse; PROJECT.md 3.12-A",
        "has_dim_field": have_dim,
        "available_fields": keys,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# check 1 -- the S-flip is -M^T, and it is writable back
# ---------------------------------------------------------------------------

def check1_sflip(step: int = 4000) -> dict:
    out = {"_what": "S-flip identity and write-back; PROJECT.md 3.12-D",
           "step": step, "heads": {}}
    for layer, head, label in HEADS:
        M = _load_ov(step, layer, head)
        S = (M + M.T) / 2.0
        A = (M - M.T) / 2.0

        flip = -S + A                      # the intended intervention
        negT = -M.T                        # the claimed closed form

        # factored write-back: W_O' = -W_V^T, W_V' = W_O^T
        _f = hc.factor_from_dense(M, d_head=D_HEAD)
        W_O, W_V = _f["W_O"], _f["W_V"]
        M_ref = W_O @ W_V
        W_O_f, W_V_f = -W_V.T, W_O.T
        M_fact = W_O_f @ W_V_f

        sv_M = _f["singular_values"]
        sv_f = np.linalg.svd(negT, compute_uv=False)
        ev_M = np.linalg.eigvals(hc.head_core(W_O, W_V))
        ev_f = np.linalg.eigvals(hc.head_core(W_O_f, W_V_f))

        nrm = np.linalg.norm(M)
        out["heads"][label] = {
            "layer": layer, "head": head,
            # the identity itself
            "max_abs_diff_flip_vs_negMT": float(np.max(np.abs(flip - negT))),
            "rel_diff_flip_vs_negMT": float(np.linalg.norm(flip - negT) / nrm),
            # factorisation fidelity, and the write-back
            "rel_diff_factor_roundtrip": float(np.linalg.norm(M_ref - M) / nrm),
            "rel_diff_factored_writeback_vs_negMT":
                float(np.linalg.norm(M_fact - negT) / nrm),
            # what the intervention preserves
            "frobenius_M": float(nrm),
            "frobenius_flip": float(np.linalg.norm(negT)),
            "max_abs_singular_value_diff": float(np.max(np.abs(sv_M - sv_f))),
            # what it changes: Re(lambda) must flip sign, |lambda| must not move
            "re_lambda_max_M": float(np.max(ev_M.real)),
            "re_lambda_max_flip": float(np.max(ev_f.real)),
            "attractive_energy_fraction_M": float(
                (np.abs(ev_M[ev_M.real > 0]) ** 2).sum()
                / max((np.abs(ev_M) ** 2).sum(), 1e-300)),
            "attractive_energy_fraction_flip": float(
                (np.abs(ev_f[ev_f.real > 0]) ** 2).sum()
                / max((np.abs(ev_f) ** 2).sum(), 1e-300)),
            "max_abs_modulus_diff": float(
                np.max(np.abs(np.sort(np.abs(ev_M)) - np.sort(np.abs(ev_f))))),
        }
    return out


# ---------------------------------------------------------------------------
# check 2 -- rotated or spread?
# ---------------------------------------------------------------------------

def check2_rotate_or_spread(layer: int = 7, head: int = 8) -> dict:
    with open(SERIES) as fh:
        steps = json.load(fh)["steps"]

    tops, top2 = [], []
    for s in steps:
        M = _load_ov(s, layer, head)
        U, sv, Vt = np.linalg.svd(M, full_matrices=False)
        tops.append(Vt[0].copy())            # right singular vector = read direction
        top2.append(Vt[:2].copy())

    rows = []
    for i in range(1, len(steps)):
        ov = float(abs(np.dot(tops[i - 1], tops[i])))
        # principal angles between the two top-2 right-singular subspaces
        s_ang = np.linalg.svd(top2[i - 1] @ top2[i].T, compute_uv=False)
        s_ang = np.clip(s_ang, -1.0, 1.0)
        rows.append({
            "step_from": steps[i - 1], "step_to": steps[i],
            "top1_overlap_abs": ov,
            "top2_subspace_cos": [float(x) for x in s_ang],
            "top2_mean_principal_cos": float(np.mean(s_ang)),
        })
    return {
        "_what": "top singular direction stability across steps; PROJECT.md 3.12-E2",
        "layer": layer, "head": head,
        "reading": ("overlap near 1 across the consolidation steps => the action "
                    "SPREAD to a second direction; overlap decaying => the "
                    "direction ROTATED and rank 2 tracks a moving target"),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# check 3 -- the anti-copiers as a found control
# ---------------------------------------------------------------------------

def check3_anticopiers(step: int = 4000) -> dict:
    out = {"_what": "anti-copiers vs the copier, spectral separation; 3.12-E3",
           "step": step, "heads": {}}
    for layer, head, label in HEADS:
        M = _load_ov(step, layer, head)
        _f = hc.factor_from_dense(M, d_head=D_HEAD)
        W_O, W_V = _f["W_O"], _f["W_V"]
        spec = hc.head_spectrum(W_O, W_V)
        sv = _f["singular_values"]
        e = sv ** 2
        out["heads"][label] = {
            "layer": layer, "head": head,
            "attractive_energy_fraction_core":
                float(spec["attractive_energy_fraction_core"]),
            "repulsive_dim_fraction_core":
                float(spec.get("repulsive_dim_fraction_core", float("nan"))),
            "complex_energy_fraction_core":
                float(spec["complex_energy_fraction_core"]),
            "frobenius": float(np.linalg.norm(M)),
            # the gain concentration that makes L7H8's rank-1 story work
            "top_sv_energy_share": float(e[0] / e.sum()),
            "top2_sv_energy_share": float(e[:2].sum() / e.sum()),
            "participation_ratio": float((e.sum() ** 2) / (e ** 2).sum()),
        }
    return out


def main() -> None:
    res = {
        "_what_this_is": "PROJECT.md 3.12 sub-phase 7b diagnostics. "
                         "EXPLORATORY. No p-value, nothing registered.",
        "check0_count_vs_energy": check0_count_vs_energy(),
        "check1_sflip": check1_sflip(),
        "check2_rotate_or_spread": check2_rotate_or_spread(),
        "check3_anticopiers": check3_anticopiers(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"wrote {OUT}")

    # ---- read-out -------------------------------------------------------
    c0 = res["check0_count_vs_energy"]
    print("\n=== check 0: count vs energy on the repulsive collapse ===")
    if c0["has_dim_field"]:
        print(f"{'step':>7} {'med_rep_energy':>15} {'med_rep_dim':>12} "
              f"{'frac_E==1':>10} {'frac_dim==1':>12} {'E-dim':>8}")
        for r in c0["rows"]:
            print(f"{r['step']:>7} {r['median_repulsive_energy']:>15.3f} "
                  f"{r['median_repulsive_dim']:>12.3f} "
                  f"{r['frac_heads_energy_exactly_1']:>10.3f} "
                  f"{r['frac_heads_dim_exactly_1']:>12.3f} "
                  f"{r['median_energy_minus_dim']:>8.3f}")
    else:
        print("  repulsive_dim_fraction_core NOT in the series -- "
              "count-vs-energy cannot be read from this artifact.")
        print(f"  available: {c0['available_fields']}")

    print("\n=== check 1: S-flip identity (-M^T) and write-back ===")
    for label, h in res["check1_sflip"]["heads"].items():
        print(f"  {label}:")
        print(f"    |(-S+A) - (-M^T)|_max      = {h['max_abs_diff_flip_vs_negMT']:.3e}")
        print(f"    factored write-back rel err= {h['rel_diff_factored_writeback_vs_negMT']:.3e}")
        print(f"    max singular-value change  = {h['max_abs_singular_value_diff']:.3e}")
        print(f"    ||M||_F {h['frobenius_M']:.4f} -> {h['frobenius_flip']:.4f}")
        print(f"    attractive energy frac  {h['attractive_energy_fraction_M']:.3f}"
              f" -> {h['attractive_energy_fraction_flip']:.3f}")
        print(f"    max |lambda| change        = {h['max_abs_modulus_diff']:.3e}")

    print("\n=== check 2: rotated or spread? (L7H8 top singular direction) ===")
    print(f"{'from':>7} {'to':>8} {'top1_overlap':>13} {'top2_mean_cos':>14}")
    for r in res["check2_rotate_or_spread"]["rows"]:
        print(f"{r['step_from']:>7} {r['step_to']:>8} "
              f"{r['top1_overlap_abs']:>13.3f} {r['top2_mean_principal_cos']:>14.3f}")

    print("\n=== check 3: anti-copiers vs the copier (step 4000) ===")
    print(f"{'head':>18} {'attr_frac':>10} {'cplx_frac':>10} {'||M||_F':>9} "
          f"{'sv1_share':>10} {'sv12_share':>11} {'PR':>7}")
    for label, h in res["check3_anticopiers"]["heads"].items():
        print(f"{label:>18} {h['attractive_energy_fraction_core']:>10.3f} "
              f"{h['complex_energy_fraction_core']:>10.3f} {h['frobenius']:>9.4f} "
              f"{h['top_sv_energy_share']:>10.3f} {h['top2_sv_energy_share']:>11.3f} "
              f"{h['participation_ratio']:>7.2f}")


if __name__ == "__main__":
    main()
