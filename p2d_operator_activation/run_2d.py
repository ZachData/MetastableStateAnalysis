"""
p2d_operator_activation/run_2d.py — Phase 2d driver.

    python -m p2d_operator_activation.run_2d \
        --p2-dir results_p2/ --model EleutherAI/pythia-410m \
        --p1-run results/pythia-410m_step143000/short_heterogeneous \
        --revision step143000 --out results_p2d/ --subexp D1 D2 D3 D4

BOTH REVISIONS ARE REQUIRED AND NEITHER IS INFERRED. The join refuses on a
mismatch or an unknown; see p2d_io. A driver that guessed the revision from
a directory name would be wrong the first time a directory is renamed, and
the failure mode is silent.

SEQUENCING. This should not be run before Phase 1c-B. If T_eff << t*, the
network never integrates far enough for the asymptotic energy argument to
bind, and D1's attribution of the monotonicity break is attributing
something that was not going to happen. The driver prints that reminder
rather than enforcing it, since a sensitivity run before 1c-B is
legitimate — it just should not be reported as an attribution.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .p2d_io import (
    load_operators, join, revision_from_run, JoinRefused, resolve_ln_params,
    extraction_convention,
)
from .gradient_flow_condition import head_regime, adjudicate_p_m1
from .operator_pairing import (
    token_covariance, operator_conditioned_rank, generalized_energy,
    monotonicity_compare,
)
from .table1_predictions import (
    classify_ov_row, projection_modality, modality_stability, adjudicate_p_t1,
)


def violation_counts(run: dict, beta: float) -> tuple:
    """
    Per-layer energy-violation indicator for P-M1.

    `energies.json` carries per-layer E_beta, NOT violation counts, so they
    are derived here with metrics.energy_violation_severity's relative rule
    — the same rule the summary table and checkpoint_scalars now share, so
    the three cannot drift.

    Returns (per_layer_list, info). The list is an INDICATOR (0 or 1) at
    each layer boundary, not a count: a violation is an event between two
    adjacent layers, and there is exactly one such event per boundary. A
    "count per layer" would be a category error, and correlating a
    per-layer regime score against it would silently be correlating against
    a boolean anyway.

    Layer 0 is 0 by construction (no preceding layer), which biases the
    correlation slightly toward zero. Reported rather than dropped, since
    dropping it would misalign the regime series.
    """
    en = (run or {}).get("energies") or {}
    layers = en.get("layers", [])
    if not layers:
        return [], {"note": "energies.json has no `layers`"}

    series = []
    for lay in layers:
        e = {float(b): v for b, v in (lay.get("energies") or {}).items()}
        series.append(e.get(float(beta), float("nan")))

    a = np.asarray(series, dtype=np.float64)
    if not np.isfinite(a).any():
        return [], {"note": f"no finite E_beta at beta={beta}"}

    from core.metrics import ENERGY_VIOLATION_REL_TOL as TOL
    ind = [0.0]
    for i in range(1, len(a)):
        ok = np.isfinite(a[i]) and np.isfinite(a[i - 1])
        ind.append(1.0 if (ok and a[i - 1] - a[i] > TOL * abs(a[i - 1])) else 0.0)
    return ind, {"beta": float(beta), "rel_tol": float(TOL),
                 "n_violations": int(sum(ind)), "n_layers": len(ind),
                 "layer0_is_zero_by_construction": True}


def analyse(joined: dict, subexp: set, betas, center_cov: bool,
            bw_scan: bool) -> dict:
    """Run the selected sub-experiments over every (layer, head) pair."""
    out = {"frame": joined["frame"], "revision": joined["revision"],
           "warnings": list(joined["warnings"]),
           "d_head": joined["d_head"], "per_head": [],
           "energy_series": {"identity": [], "head_mean": []}}

    for pair in joined["pairs"]:
        Y = pair["Y"]
        C = token_covariance(Y, center=center_cov) if {"D2"} & subexp else None
        layer_head_E = []

        for h in pair["heads"]:
            rec = {"layer": pair["layer"], "layer_name": pair["layer_name"],
                   "head": h["head"]}

            if "D1" in subexp:
                rec.update(head_regime(h["wq"], h["wk"], h["ov"],
                                       d_head=pair["d_head"]))

            M = None
            if {"D2", "D3", "D4"} & subexp:
                from .gradient_flow_condition import qk_matrix
                M = qk_matrix(h["wq"], h["wk"], d_head=pair["d_head"])

            if "D2" in subexp:
                rec["pr"] = operator_conditioned_rank(M, C)

            if "D3" in subexp:
                cls = classify_ov_row(h["ov"], M)
                phi1 = cls.pop("phi1")
                mod = projection_modality(Y, phi1)
                if bw_scan and not mod.get("degenerate"):
                    mod["stability"] = modality_stability(Y, phi1)
                rec["table1"] = cls
                rec["row2_candidate"] = cls["row2_candidate"]
                rec["modality"] = mod

            if "D4" in subexp:
                g = generalized_energy(Y, M, betas=betas)
                rec["generalized_energy"] = g
                layer_head_E.append(g["energies"])

            out["per_head"].append(rec)

        if layer_head_E:
            # Mean over heads of the head-specific energy. Stated as a
            # choice: there is no single "the model's energy" for a
            # multi-head layer, and the mean is the least-committal
            # aggregate. D1's adjudicator makes the same choice explicit
            # and reports three alternatives; here the per-head values are
            # all persisted so a different aggregate needs no rerun.
            out["energy_series"]["head_mean"].append(
                {b: float(np.mean([e[b] for e in layer_head_E])) for b in betas})

    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p2-dir", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--p1-run", type=Path, required=True)
    ap.add_argument("--revision", required=True,
                    help="checkpoint revision, e.g. step143000. Applied to "
                         "BOTH sides unless --activation-revision is given.")
    ap.add_argument("--activation-revision", default=None,
                    help="only when the activations genuinely come from a "
                         "different revision, which they should not")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--subexp", nargs="+", default=["D1", "D2"],
                    choices=["D1", "D2", "D3", "D4"])
    ap.add_argument("--betas", nargs="+", type=float,
                    default=[0.1, 1.0, 2.0, 5.0])
    ap.add_argument("--pm1-beta", type=float, default=1.0,
                    help="which beta's energy series P-M1 counts violations "
                         "on. Not swept: P-M1 is a claim about WHERE "
                         "violations sit, and pooling betas would mix "
                         "different violation sets. Report per beta by "
                         "re-running.")
    ap.add_argument("--uncentered-cov", action="store_true",
                    help="D2 on the uncentred token covariance. Meaningful "
                         "but different: the common mode dominates it on an "
                         "anisotropic cloud, so PR_M can read ~1 purely "
                         "because every token shares a direction. Run both "
                         "where kappa_1 is large.")
    ap.add_argument("--bw-scan", action="store_true",
                    help="D3 bandwidth-stability scan. Slower, and P-T1 "
                         "should be adjudicated on the stable count only.")
    ap.add_argument("--ln-which", default="attn", choices=["attn", "ffn"],
                    help="which sublayer's LN frame. 'attn' is the states "
                         "the QK circuit reads and is what every "
                         "sub-experiment here means.")
    ap.add_argument("--assert-convention", action="store_true",
                    help="permit the convention flags below to stand in for "
                         "a geometry.json that does not record the "
                         "extraction convention. Required, because getting "
                         "it wrong applies M_h in the wrong frame silently.")
    ap.add_argument("--keep-embedding", action="store_true",
                    help="activations index 0 is the raw embedding rather "
                         "than block-0 output. Inverts this project's Fix 4 "
                         "convention; only pass it if the extraction did.")
    ap.add_argument("--last-is-post-final-ln", action="store_true",
                    help="the extraction already applied final_layer_norm to "
                         "the last state (core/models.py standard path). The "
                         "correct frame is then the identity, and applying "
                         "final LN again would be wrong.")
    ap.add_argument("--dtype", default="float32",
                    help="fp32 is required: the D3 row classification turns "
                         "on the sign and multiplicity of lambda_1(V) near "
                         "zero, which is what core/models.py's precision "
                         "guard exists to protect.")
    ap.add_argument("--raw-frame", action="store_true",
                    help="apply M_h to the raw residual stream instead of "
                         "the LN'd states. A sensitivity check, never the "
                         "primary measurement.")
    args = ap.parse_args(argv)

    print("NOTE: Phase 2d is sequenced after 1c-B. If T_eff << t*, the "
          "monotonicity break may not be the right thing to attribute.\n")

    try:
        ops = load_operators(args.p2_dir, args.model)
    except (FileNotFoundError, KeyError) as exc:
        print(f"cannot load Phase 2 operators: {exc}", file=sys.stderr)
        return 1

    from p1c_frames.p1c_io import load_run
    run = load_run(args.p1_run)
    if run.get("activations") is None:
        print(f"no activations.npz under {args.p1_run}", file=sys.stderr)
        return 1
    A = np.asarray(run["activations"])
    if run.get("norms") is not None:
        A = np.asarray(run["norms"])[..., None] * A

    ln_params = None
    if not args.raw_frame:
        conv = extraction_convention(run)
        if conv["source"] == "artifact":
            emb_stripped = conv["embedding_stripped"]
            post_ln = conv["last_is_post_final_ln"]
            print(f"  extraction convention read from geometry.json: "
                  f"embedding_stripped={emb_stripped}, "
                  f"last_is_post_final_ln={post_ln}")
            if args.keep_embedding or args.last_is_post_final_ln:
                print("  NOTE: command-line convention flags are being "
                      "IGNORED — the artifact records the convention and it "
                      "is authoritative. Remove the flags.")
        elif args.assert_convention:
            emb_stripped = not args.keep_embedding
            post_ln = args.last_is_post_final_ln
            print(f"  extraction convention ASSERTED by flag "
                  f"(geometry.json does not record it): "
                  f"embedding_stripped={emb_stripped}, "
                  f"last_is_post_final_ln={post_ln}")
        else:
            print("geometry.json does not record the extraction convention "
                  "(hidden_state_0_is_embedding /\n"
                  "  final_hidden_state_is_post_ln), so the LN frame cannot "
                  "be resolved from the artifact.\n"
                  "  Getting it wrong applies M_h in the wrong frame, "
                  "silently. Pass --assert-convention\n"
                  "  together with the flags if you are certain, or "
                  "re-extract.", file=sys.stderr)
            return 2
        try:
            ln_params, ln_info = resolve_ln_params(
                args.model, args.revision, n_hidden_states=A.shape[0],
                which=args.ln_which,
                embedding_stripped=emb_stripped,
                last_is_post_final_ln=post_ln,
                dtype=args.dtype,
            )
        except Exception as exc:
            print(f"cannot resolve LN frame: {type(exc).__name__}: {exc}\n"
                  f"  Re-run with --raw-frame ONLY as a sensitivity check — "
                  f"applying M_h to the raw\n"
                  f"  residual stream measures a different operator on a "
                  f"different space, silently.",
                  file=sys.stderr)
            return 2
        print(f"  LN frame resolved for {A.shape[0]} states "
              f"(which={args.ln_which}): " +
              ", ".join(f"{k}={v}" for k, v in ln_info["frame_counts"].items()))
        if ln_info["identity_indices"]:
            print(f"  identity frame at indices {ln_info['identity_indices']} "
                  f"(already post-final-LN; re-applying would be wrong)")

    act_rev = revision_from_run(run, args.activation_revision or args.revision)
    try:
        joined = join(ops, A, args.revision, act_rev,
                      ln_params_by_layer=ln_params,
                      context=str(args.p1_run))
    except JoinRefused as exc:
        print(f"JOIN REFUSED: {exc}", file=sys.stderr)
        return 3

    for w in joined["warnings"]:
        print(f"  warning: {w}")

    res = analyse(joined, set(args.subexp), args.betas,
                  center_cov=not args.uncentered_cov, bw_scan=args.bw_scan)

    if "D1" in args.subexp:
        viols, vinfo = violation_counts(run, args.pm1_beta)
        res["violation_counts"] = {"per_layer": viols, **vinfo}
        res["p_m1"] = (adjudicate_p_m1(res["per_head"], viols) if viols
                       else {"verdict": f"no usable energy series: "
                                        f"{vinfo.get('note')}"})
    if "D3" in args.subexp:
        res["p_t1"] = adjudicate_p_t1(res["per_head"])

    args.out.mkdir(parents=True, exist_ok=True)
    from p1c_frames.p1c_io import save_p1c
    save_p1c(res, args.out / args.p1_run.name, name="p2d")

    n_gf = sum(1 for r in res["per_head"] if r.get("in_gradient_flow_regime"))
    print(f"\n{len(res['per_head'])} heads, frame={res['frame']}, "
          f"rev={res['revision']}")
    if "D1" in args.subexp:
        print(f"  in gradient-flow regime: {n_gf}")
    if "D3" in args.subexp:
        print(f"  P-T1: {res['p_t1']['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
