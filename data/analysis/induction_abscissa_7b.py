"""The per-head NUMERICAL ABSCISSA — the quantity §2.2 says governs and nobody
has computed per head.

WHY THIS EXISTS
---------------
`MATH_SPECTRAL_OT.md` §2.4.5. Three results combine into one cheap test:

  §2.2  `Re lambda` governs `t -> infinity`; `lambda_max(S)` (the numerical
        abscissa) governs the initial slope. A head can have every eigenvalue
        on the stable side while `S` still has growing directions -- Bendixson
        gives only the INCLUSION `Re lambda(M) in [l_min(S), l_max(S)]`.
  §2.4.1 The first-order change in PAIRWISE distance is `2 delta^T S delta` --
        a quadratic form in `S` alone, with `A` contributing exactly nothing.
        So the readout Stage 3 is built on is governed by `S`, not by `Re lambda`.
  §3.12-F3 `L7H8` (dOV_nll +0.244) and `L2H10` (NEGATIVE dOV_nll) agree to three
        digits on every eigenvalue-derived field -- attractive fraction 0.000
        both, sigma-1 share 0.166 vs 0.176, participation 21.5 vs 19.9.

§2.2 PREDICTS F3's puzzle: agreement on the spectral abscissa constrains the
numerical abscissa only through an inclusion. So the question this runner asks
is the one the derivation hands it:

    does the SYMMETRIC part separate the copier from the anti-copier where the
    EIGENVALUES do not?

If it does, the induction result belongs in the symmetric frame and the whole
attractive/repulsive eigenvalue apparatus is the wrong description of it -- which
is `MATH_SPECTRAL_OT` §5.3(d) settled by measurement rather than by residual.

WHY IT IS NOT ALREADY ON DISK. `p2_eigenspectra/weights.py::eigendecompose`
computes `sym_eigenvalues` and `ov_decomp_*.npz` stores them -- but PER LAYER,
for `sum_h W_OV^h`, which `head_circuits.py` and `ov_per_head.py` both rule out
as "the operator only in the counterfactual where every head attends
identically. It is not a thing the model ever forms." The per-head symmetric
spectrum has never been computed.

ALSO COMPUTED, because §2.4.4 makes it a readout rather than a nuisance: the
self-commutator `[M^T, M] = M^T M - M M^T`, whose norm is the departure from
normality. §2.4.4 shows the S-flip inverts the first-order geometry effect
EXACTLY and that the whole discrepancy is a quadratic form in this commutator --
so its size is the expected size of the deviation.

CONVENTION (§2.4): the discrete residual write `x <- x + Mx`, so GROWTH is
`x^T S x > 0` and the maximum first-order growth rate is `lambda_max(S)`. This
is the opposite sign to §2.1's continuous `x' = -Vx`. Stated, not assumed.

NO MODEL LOAD, NO FORWARD PASS. Reads the dense per-head OV already on disk.
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", "/run/media/system/WDS_500/Mets"))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))

_want_prefix = str(REPO / ".venv")
if not sys.prefix.startswith(_want_prefix):
    raise SystemExit(f"wrong interpreter: sys.prefix={sys.prefix!r}, need {_want_prefix!r}")

import numpy as np

from p2b_imaginary import head_circuits as hc

OUT = DATA / "analysis" / "induction_abscissa_7b.json"
SERIES = DATA / "analysis" / "ov_per_head_series.json"
D_HEAD = 64

#: dOV_nll from PROJECT.md §3.11 block B, for the read-out only. Not recomputed.
DNLL = {"L7H8": +0.244, "L9H9": +0.054, "L1H15": +0.040, "L6H0": +0.021,
        "L2H10": -1.0, "L9H8": -1.0}          # the two negatives: sign only

TARGETS = [(7, 8, "L7H8"), (5, 2, "L5H2"), (2, 10, "L2H10"), (9, 8, "L9H8"),
           (9, 9, "L9H9"), (1, 15, "L1H15"), (6, 0, "L6H0")]


def _ov_npz(step: int) -> Path:
    hits = sorted(DATA.glob(f"phase12/**/ov_weights_pythia-410m-step{step}.npz"))
    if not hits:
        raise FileNotFoundError(f"no ov_weights npz for step {step}")
    return hits[-1]


def _load_ov(step: int, layer: int, head: int) -> np.ndarray:
    with np.load(_ov_npz(step)) as z:
        return np.asarray(z[f"ov_head{head}_layer_{layer}"], dtype=np.float64)


def profile(M: np.ndarray) -> dict:
    """Everything §2.4 says is decision-relevant, for one head's dense W_OV."""
    fro2 = float(np.sum(M * M))
    S = (M + M.T) / 2.0

    # --- the symmetric frame: what governs the first-order pairwise readout ---
    ev_S = np.linalg.eigvalsh(S)                    # ascending, real
    tol = 1e-9 * max(abs(ev_S[0]), abs(ev_S[-1]), 1e-300)
    nz = ev_S[np.abs(ev_S) > tol]
    e_nz = nz ** 2
    pos = nz[nz > 0]

    # --- the eigenvalue frame: what the project currently reports -------------
    f = hc.factor_from_dense(M, d_head=D_HEAD)
    ev_M = np.linalg.eigvals(hc.head_core(f["W_O"], f["W_V"]))

    # --- §2.4.4: the self-commutator, the size of the S-flip's residual -------
    comm = M.T @ M - M @ M.T
    return {
        "lambda_max_S": float(ev_S[-1]),
        "lambda_min_S": float(ev_S[0]),
        "sym_pos_energy_fraction": float((pos ** 2).sum() / max(e_nz.sum(), 1e-300)),
        "sym_pos_dim_fraction": float(len(pos) / max(len(nz), 1)),
        "n_nonzero_sym": int(len(nz)),
        # the eigenvalue-frame fields, for the side-by-side
        "attractive_energy_fraction_core": float(
            (np.abs(ev_M[ev_M.real > 0]) ** 2).sum()
            / max((np.abs(ev_M) ** 2).sum(), 1e-300)),
        "max_re_lambda": float(np.max(ev_M.real)),
        # Bendixson slack: how much room the inclusion leaves
        "bendixson_slack_upper": float(ev_S[-1] - np.max(ev_M.real)),
        # §2.4.4
        "commutator_fro_over_fro2": float(np.linalg.norm(comm) / max(fro2, 1e-300)),
        "frobenius": float(np.sqrt(fro2)),
    }


def main() -> None:
    with open(SERIES) as fh:
        steps = json.load(fh)["steps"]

    res = {"_what_this_is": "per-head numerical abscissa; MATH_SPECTRAL_OT 2.4.5. "
                            "EXPLORATORY. No p-value, nothing registered.",
           "convention": "discrete residual write x <- x + Mx; growth is x^T S x > 0",
           "layer7_at_4000": {}, "targets_at_4000": {}, "targets_series": {}}

    # --- all 16 heads of layer 7 at step 4000: does S rank L7H8 where eig does not?
    for h in range(16):
        res["layer7_at_4000"][f"L7H{h}"] = profile(_load_ov(4000, 7, h))

    # --- the named heads at 4000 (includes the anti-copiers and L5H2) ---------
    for layer, head, label in TARGETS:
        res["targets_at_4000"][label] = profile(_load_ov(4000, layer, head))

    # --- L7H8 and L2H10 across the whole axis --------------------------------
    for layer, head, label in [(7, 8, "L7H8"), (2, 10, "L2H10")]:
        rows = []
        for s in steps:
            p = profile(_load_ov(s, layer, head))
            p["step"] = s
            rows.append(p)
        res["targets_series"][label] = rows

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"wrote {OUT}\n")

    print("=== the question: does S separate what the eigenvalues cannot? "
          "(step 4000) ===")
    print(f"{'head':>7} {'dOV_nll':>9} {'attr_frac':>10} {'l_max(S)':>10} "
          f"{'l_min(S)':>10} {'S_pos_E':>9} {'S_pos_dim':>10} {'||[M^T,M]||':>12}")
    for label, p in res["targets_at_4000"].items():
        d = DNLL.get(label)
        ds = "neg" if (d is not None and d < 0) else (f"{d:+.3f}" if d else "  ?")
        print(f"{label:>7} {ds:>9} {p['attractive_energy_fraction_core']:>10.3f} "
              f"{p['lambda_max_S']:>10.4f} {p['lambda_min_S']:>10.4f} "
              f"{p['sym_pos_energy_fraction']:>9.3f} {p['sym_pos_dim_fraction']:>10.3f} "
              f"{p['commutator_fro_over_fro2']:>12.4f}")

    print("\n=== all 16 heads of layer 7 at step 4000, ranked by lambda_max(S) ===")
    order = sorted(res["layer7_at_4000"].items(),
                   key=lambda kv: -kv[1]["lambda_max_S"])
    print(f"{'head':>7} {'attr_frac':>10} {'l_max(S)':>10} {'S_pos_E':>9} "
          f"{'||M||_F':>9} {'bendixson':>10}")
    for label, p in order:
        star = "  <-- the copier" if label == "L7H8" else ""
        print(f"{label:>7} {p['attractive_energy_fraction_core']:>10.3f} "
              f"{p['lambda_max_S']:>10.4f} {p['sym_pos_energy_fraction']:>9.3f} "
              f"{p['frobenius']:>9.4f} {p['bendixson_slack_upper']:>10.4f}{star}")

    print("\n=== L7H8 vs L2H10 across the axis: lambda_max(S) ===")
    print(f"{'step':>7} {'L7H8 lmaxS':>12} {'L2H10 lmaxS':>13} "
          f"{'L7H8 S_pos_E':>14} {'L2H10 S_pos_E':>15}")
    a = {r["step"]: r for r in res["targets_series"]["L7H8"]}
    b = {r["step"]: r for r in res["targets_series"]["L2H10"]}
    for s in steps:
        print(f"{s:>7} {a[s]['lambda_max_S']:>12.4f} {b[s]['lambda_max_S']:>13.4f} "
              f"{a[s]['sym_pos_energy_fraction']:>14.3f} "
              f"{b[s]['sym_pos_energy_fraction']:>15.3f}")


if __name__ == "__main__":
    main()
