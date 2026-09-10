"""Does ANY weights-only spectral quantity predict the causal copying effect?

WHY THIS EXISTS
---------------
`MATH_SPECTRAL_OT.md` §2.4.5 asked whether the SYMMETRIC frame separates the
copier from the anti-copier where the EIGENVALUE frame does not.
`induction_abscissa_7b.py` answered: **no**. At step 4000 `L7H8` (dOV_nll
+0.244) and `L2H10` (negative) sit at lambda_max(S) 0.2644 and 0.2238, and
`L7H8` ranks 8th of 16 in its own layer on that quantity while carrying ~10x
the layer's causal effect.

That is a two-head anecdote. This runner turns it into a population statement,
because `induction_subspace_characterize*.json` already contains the causal
readout for EVERY head of a layer at seven (step, layer) points -- 112 heads
with a measured `full_ablation_delta_nll` and a dense `W_OV` on disk. So the
question can be asked properly:

    across 112 heads, does any spectral field computed from the weights alone
    RANK-CORRELATE with the causal copying effect?

FIELDS TESTED, one per frame the project has used:
  eigenvalue frame  attractive_energy_fraction_core, max Re lambda
  symmetric frame   lambda_max(S), lambda_min(S), sym positive energy fraction
  gain frame        ||M||_F, sigma-1 share, sigma-12 share, participation ratio
  non-normality     ||[M^T, M]||_F / ||M||_F^2

STATISTIC: Spearman rho against `full_ablation_delta_nll`, computed WITHIN each
(step, layer) population and then reported per-population and pooled, because
the populations sit at different checkpoints with different baseline NLLs and
pooling raw values across them would compare a step-2000 effect to a step-32000
one. EXPLORATORY -- this is a correlation over heads, not a registered test,
and heads within a layer are not independent. It licenses no p-value and none
is emitted.

NO MODEL LOAD, NO FORWARD PASS.
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", "/run/media/system/WDS_500/Mets"))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))
_want = str(REPO / ".venv")
if not sys.prefix.startswith(_want):
    raise SystemExit(f"wrong interpreter: {sys.prefix!r}, need {_want!r}")

import numpy as np
from scipy import stats

from p2b_imaginary import head_circuits as hc

OUT = DATA / "analysis" / "induction_spectral_predicts_7b.json"
D_HEAD = 64

#: (file, step, layer). The characterize runs whose layer_heads block carries a
#: causal readout for all 16 heads of one layer at one checkpoint.
POPS = [
    ("induction_subspace_characterize.json", 4000, 7),
    ("induction_subspace_characterize_s8000_L7.json", 8000, 7),
    ("induction_subspace_characterize_s8000_L9.json", 8000, 9),
    ("induction_subspace_characterize_s2000_L9.json", 2000, 9),
    ("induction_subspace_characterize_s16000_L1.json", 16000, 1),
    ("induction_subspace_characterize_s16000_L6.json", 16000, 6),
    ("induction_subspace_characterize_s32000_L2.json", 32000, 2),
]

FIELDS = ["attractive_energy_fraction_core", "max_re_lambda",
          "lambda_max_S", "lambda_min_S", "sym_pos_energy_fraction",
          "frobenius", "sv1_share", "sv12_share", "participation_ratio",
          "nonnormality"]


def _ov_npz(step):
    hits = sorted(DATA.glob(f"phase12/**/ov_weights_pythia-410m-step{step}.npz"))
    if not hits:
        raise FileNotFoundError(f"no ov_weights npz for step {step}")
    return hits[-1]


def profile(step, layer, head):
    with np.load(_ov_npz(step)) as z:
        M = np.asarray(z[f"ov_head{head}_layer_{layer}"], dtype=np.float64)
    fro2 = float(np.sum(M * M))
    ev_S = np.linalg.eigvalsh((M + M.T) / 2.0)
    tol = 1e-9 * max(abs(ev_S[0]), abs(ev_S[-1]), 1e-300)
    nz = ev_S[np.abs(ev_S) > tol]
    pos = nz[nz > 0]
    f = hc.factor_from_dense(M, d_head=D_HEAD)
    ev_M = np.linalg.eigvals(hc.head_core(f["W_O"], f["W_V"]))
    e = f["singular_values"] ** 2
    comm = M.T @ M - M @ M.T
    return {
        "attractive_energy_fraction_core": float(
            (np.abs(ev_M[ev_M.real > 0]) ** 2).sum()
            / max((np.abs(ev_M) ** 2).sum(), 1e-300)),
        "max_re_lambda": float(np.max(ev_M.real)),
        "lambda_max_S": float(ev_S[-1]),
        "lambda_min_S": float(ev_S[0]),
        "sym_pos_energy_fraction": float(
            (pos ** 2).sum() / max((nz ** 2).sum(), 1e-300)),
        "frobenius": float(np.sqrt(fro2)),
        "sv1_share": float(e[0] / e.sum()),
        "sv12_share": float(e[:2].sum() / e.sum()),
        "participation_ratio": float((e.sum() ** 2) / (e ** 2).sum()),
        "nonnormality": float(np.linalg.norm(comm) / max(fro2, 1e-300)),
    }


def main():
    res = {"_what_this_is": __doc__, "populations": [], "per_field": {}}
    per_field_rhos = {f: [] for f in FIELDS}

    for fname, step, layer in POPS:
        path = DATA / "analysis" / fname
        if not path.exists():
            print(f"  SKIP missing {fname}")
            continue
        with open(path) as fh:
            lh = json.load(fh)["layer_heads"]
        dnll, profs = [], []
        for h in range(16):
            dnll.append(float(lh[str(h)]["full_ablation_delta_nll"]))
            profs.append(profile(step, layer, h))
        dnll = np.asarray(dnll)
        top = int(np.argmax(dnll))

        row = {"file": fname, "step": step, "layer": layer,
               "top_head": top, "top_dnll": float(dnll[top]),
               "layer_mean_dnll": float(dnll.mean()), "rho": {}, "top_rank": {}}
        for f in FIELDS:
            v = np.asarray([p[f] for p in profs])
            rho = float(stats.spearmanr(v, dnll).statistic)
            row["rho"][f] = rho
            # where the causally-top head ranks on this field (0 == largest)
            row["top_rank"][f] = int((v > v[top]).sum())
            per_field_rhos[f].append(rho)
        res["populations"].append(row)
        print(f"  {fname[:46]:<46} step {step:>6} L{layer:<2} "
              f"top=H{top} dnll {dnll[top]:+.3f}", flush=True)

    for f in FIELDS:
        r = np.asarray(per_field_rhos[f])
        res["per_field"][f] = {
            "mean_rho": float(r.mean()), "median_rho": float(np.median(r)),
            "min_rho": float(r.min()), "max_rho": float(r.max()),
            "n_pops": int(r.size),
            "n_pops_rho_above_0.5": int((r > 0.5).sum()),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {OUT}")

    print("\n=== Spearman rho vs causal dOV_nll, within-layer, over "
          f"{len(res['populations'])} populations of 16 heads ===")
    print(f"{'field':>32} {'mean rho':>9} {'median':>8} {'min':>7} {'max':>7} "
          f"{'|rho|>.5':>9}")
    for f in sorted(FIELDS, key=lambda k: -abs(res["per_field"][k]["mean_rho"])):
        s = res["per_field"][f]
        print(f"{f:>32} {s['mean_rho']:>+9.3f} {s['median_rho']:>+8.3f} "
              f"{s['min_rho']:>+7.3f} {s['max_rho']:>+7.3f} "
              f"{s['n_pops_rho_above_0.5']:>9}")

    print("\n=== rank of the causally-TOP head on each field "
          "(0 = the field's own maximum; 8 = middle of 16) ===")
    print(f"{'field':>32} " + " ".join(f"{r['step']}/L{r['layer']}"[:8].rjust(8)
                                       for r in res["populations"]))
    for f in FIELDS:
        cells = " ".join(f"{r['top_rank'][f]:>8}" for r in res["populations"])
        print(f"{f:>32} {cells}")


if __name__ == "__main__":
    main()
