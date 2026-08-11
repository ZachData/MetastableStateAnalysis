"""
p1c_frames/run_1c.py — Phase 1c driver.

    python -m p1c_frames.run_1c --results results/ --out results_p1c/ \
        --subexp A B C E F

Every sub-experiment is independently selectable because they have
different input requirements and different costs, and because A and B
should be run and read BEFORE C, E and F are interpreted — the update
plan's sequencing note stands: T_eff determines whether the
energy-monotonicity break is even the right thing to attribute.

WHAT THIS DELIBERATELY DOES NOT DO

It does not adjudicate P-γ1/P-γ2/P-H1/P-S1 across checkpoints in the same
pass that computes them. The adjudicators live in their own modules and
take already-computed per-run results, so the compute step can be rerun
without re-deciding anything, and a verdict cannot be quietly produced by
a code path that also chose the inputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .p1c_io import load_run, raw_states, layer_series, save_p1c
from .gamma_ode import collapse_time
from .integration_time import step_sizes, cumulative_time, verdict
from .gamma_null import residual_curve, time_residual_curve, collapse_fraction
from .beta_reduction import (
    reduction_report, beta_envelope, envelope_verdict, residual_bracket,
)
from .moments import rank_panel, adjudicate_sink_hypothesis, verify_moment_identity, ladder_from_layer
from .hemisphere_feasibility import hemisphere_profile
from .centroids import load_centroids, run_design_test


def _beta_for_run(run: dict, fallback: float) -> float:
    """
    One beta per run, with the source recorded by the caller.

    status-1c open item 1: core/beta_eff.py returns beta per HEAD, and the
    null needs one per layer. The head-to-layer reduction is not decided,
    and the choice moves the null — so this function does NOT invent one.
    It reads a beta_eff already written into geometry.json if present, and
    otherwise uses the explicit --beta-fallback, which must be passed
    deliberately. Silently averaging heads here would bury the open
    decision inside a driver.
    """
    geo = run.get("geometry") or {}
    for key in ("beta_eff", "beta_effective"):
        if geo.get(key) is not None:
            return float(geo[key])
    return float(fallback)


def run_one(run_dir: Path, subexp: set, beta_fallback: float,
            causal: bool, t_target: float,
            f_method: str = "kmeans", f_tmax: int = 3) -> dict:
    run = load_run(run_dir)
    geo = run.get("geometry") or {}
    n_tok = int(geo.get("n_tokens", 0)) or None
    out = {"run_dir": str(run_dir), "model": geo.get("model"),
           "prompt": geo.get("prompt"), "n_tokens": n_tok,
           "available": sorted(run["available"]),
           "provenance": run["provenance"], "skipped": {}}

    beta = _beta_for_run(run, beta_fallback)
    out["beta_used"] = beta
    out["beta_source"] = ("geometry.json" if geo.get("beta_eff") is not None
                          else "fallback_flag")

    X = None
    if {"A", "C"} & subexp:
        try:
            X = raw_states(run)
        except ValueError as exc:
            for s in ("A", "C"):
                if s in subexp:
                    out["skipped"][s] = str(exc)

    # --- A: effective integration time -----------------------------------
    if "A" in subexp and X is not None:
        a = step_sizes(X, beta, causal=causal)
        n_for_t = n_tok or X.shape[1]
        t_star = collapse_time(n_for_t, beta, target=t_target)
        out["A"] = {**a, "t_star": t_star, "verdict": verdict(a, t_star)}

    # --- B: the gamma_beta null ------------------------------------------
    if "B" in subexp:
        ipm = layer_series(run, "ip_mean")
        if "A" in out and np.isfinite(ipm).any():
            t_grid = cumulative_time(out["A"]["h_calibrated"])
            k = min(len(ipm), len(t_grid))
            n_for_t = n_tok or 2
            res = residual_curve(ipm[:k], t_grid[:k], n_for_t, beta)
            res["n_tokens"] = n_for_t
            tr = time_residual_curve(ipm[:k], t_grid[:k], n_for_t, beta)
            out["B"] = {**res, "time_domain": tr,
                        "collapse_fraction": collapse_fraction(res, t_target)}

            # The beta ENVELOPE, when per-head betas are available. This is
            # what makes the residual readable without first settling the
            # head-to-layer reduction: gamma_beta is monotone in beta, so
            # the per-head range brackets the null, and a residual whose
            # sign is the same at both edges holds for EVERY reduction.
            # Without it the residual is a point estimate carrying an
            # unstated error bar the size of the band — which at n=467,
            # T_eff=3 spans 0.20 to 0.46 in gamma.
            per_head = (geo.get("beta_eff_per_head")
                        or geo.get("beta_per_head"))
            if per_head:
                rr = reduction_report(per_head)
                try:
                    env = beta_envelope(t_grid[:k], n_for_t,
                                        rr["beta_min"], rr["beta_max"])
                    out["B"]["beta_reduction"] = rr
                    out["B"]["envelope_verdict"] = envelope_verdict(ipm[:k], env)
                    out["B"]["residual_bracket"] = residual_bracket(ipm[:k], env)
                    out["B"]["envelope_lower"] = env["lower"]
                    out["B"]["envelope_upper"] = env["upper"]
                except AssertionError as exc:
                    out["B"]["envelope_error"] = str(exc)
            else:
                out["B"]["envelope_note"] = (
                    "no per-head beta_eff in geometry.json, so the residual "
                    "is a point estimate at a single reduction. Its error "
                    "bar is the beta envelope and is currently unreported.")
        else:
            out["skipped"]["B"] = ("needs sub-experiment A's step sizes and a "
                                   "finite ip_mean series")

    # --- C: cumulant ladder and the rank panel ---------------------------
    if "C" in subexp and X is not None:
        panels = [rank_panel(X[l]) for l in range(X.shape[0])]
        out["C"] = {"panels": panels,
                    "sink_verdict": adjudicate_sink_hypothesis(panels)}
        layers = geo.get("layers", [])
        en = (run.get("energies") or {}).get("layers", [])
        checks = []
        for i, lay in enumerate(layers):
            lad = ladder_from_layer(lay, n_tok or 2)
            if not np.isfinite(lad.get("kappa1", np.nan)):
                continue
            e = {float(b): v for b, v in (en[i].get("energies", {}) if i < len(en) else {}).items()}
            if e:
                checks.append({"layer": i, "source": lad["source"],
                               **verify_moment_identity(lad, e)})
        out["C"]["moment_identity"] = checks

    # --- E: hemisphere feasibility ---------------------------------------
    if "E" in subexp:
        acts = run.get("activations")
        if acts is not None:
            out["E"] = hemisphere_profile(np.asarray(acts),
                                          d_model=int(np.asarray(acts).shape[-1]))
        else:
            out["skipped"]["E"] = "needs activations.npz"

    # --- F: spherical designs --------------------------------------------
    # Now wired. The concern that blocked it — that the clusterer choice
    # moves the reference through m — was measured and does not hold: the
    # matched-(m, d) baseline makes Q_k/Q_k^random flat at 1 across a 32x
    # range in m. See centroids.py. The clusterer is still FIXED per sweep
    # (default kmeans, the only method whose centroids Phase 1 persists),
    # but it no longer has to be fixed by matching m.
    if "F" in subexp:
        acts = run.get("activations")
        if acts is None:
            out["skipped"]["F"] = "needs activations.npz"
        else:
            acts = np.asarray(acts)
            d_model = int(acts.shape[-1])
            per_layer, errs = [], []
            for l in range(acts.shape[0]):
                try:
                    C, cinfo = load_centroids(run_dir, l, method=f_method,
                                              activations=acts[l])
                except (FileNotFoundError, KeyError, ValueError) as exc:
                    errs.append(f"layer {l}: {type(exc).__name__}: {exc}")
                    continue
                if C.shape[0] < 2:
                    per_layer.append({"layer": l, "n_centroids": int(C.shape[0]),
                                      "note": "fewer than 2 centroids"})
                    continue
                rep = run_design_test(C, d=d_model, t_max=f_tmax)
                rep.pop("Q_random_p95", None)
                per_layer.append({"layer": l, **cinfo, **rep})
            out["F"] = {"method": f_method, "t_max": f_tmax,
                        "per_layer": per_layer, "errors": errs}
            if errs and not per_layer:
                out["skipped"]["F"] = f"no layer produced centroids: {errs[0]}"

    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--subexp", nargs="+", default=["A", "B"],
                    choices=["A", "B", "C", "D", "E", "F"])
    ap.add_argument("--beta-fallback", type=float, default=None,
                    help="beta to use when geometry.json has no beta_eff. "
                         "Required if any run lacks it — there is no safe "
                         "default, since beta is a measured property of the "
                         "model (paper footnote 2), not a convention.")
    ap.add_argument("--non-causal", action="store_true",
                    help="use the paper's unmasked field. Default is causal, "
                         "which is honest for a decoder-only model and a "
                         "departure from the theory; run both once.")
    ap.add_argument("--t-target", type=float, default=0.9,
                    help="gamma threshold defining t*")
    ap.add_argument("--f-method", default="kmeans",
                    choices=["kmeans", "agglomerative", "hdbscan"],
                    help="clusterer for sub-experiment F. kmeans is the "
                         "default because it is the only one whose CENTROIDS "
                         "Phase 1 persists; the others are recomputed from "
                         "labels. Fix one per sweep — if the arms disagree "
                         "about P-S1, the design signal is a property of the "
                         "clustering rather than of the geometry.")
    ap.add_argument("--f-tmax", type=int, default=3,
                    help="highest Gegenbauer degree. A cost choice: each "
                         "degree needs its own baseline simulation. Higher "
                         "degrees are MORE sensitive in relative terms, not "
                         "less — see centroids.py for the measured bands.")
    args = ap.parse_args(argv)

    if args.beta_fallback is None:
        args.beta_fallback = float("nan")

    from p1_mstate_tracking.visualization.loaders import discover_runs
    runs = discover_runs(args.results)
    if not runs:
        print(f"no runs found under {args.results}", file=sys.stderr)
        return 1

    subexp = set(args.subexp)
    print(f"{len(runs)} runs, sub-experiments {sorted(subexp)}, "
          f"{'non-causal' if args.non_causal else 'causal'} field")

    n_ok = 0
    for (model, prompt), d in sorted(runs.items()):
        try:
            res = run_one(d, subexp, args.beta_fallback,
                          causal=not args.non_causal, t_target=args.t_target,
                          f_method=args.f_method, f_tmax=args.f_tmax)
        except Exception as exc:
            print(f"  FAIL {model} / {prompt}: {type(exc).__name__}: {exc}")
            continue
        if not np.isfinite(res.get("beta_used", np.nan)):
            print(f"  SKIP {model} / {prompt}: no beta_eff and no "
                  f"--beta-fallback given")
            continue
        save_p1c(res, args.out / d.name)
        n_ok += 1
        bits = [f"beta={res['beta_used']:.3f}({res['beta_source']})"]
        if "A" in res:
            bits.append(f"T_eff={res['A']['T_eff_calibrated']:.2f}"
                        f"/t*={res['A']['t_star']:.2f}")
        if "B" in res:
            bits.append(f"resid={res['B']['final_residual']:+.4f}")
            br = res["B"].get("residual_bracket")
            if br:
                bits.append(
                    f"bracket=[{br['final_residual_min']:+.3f},"
                    f"{br['final_residual_max']:+.3f}]"
                    f"{'' if br['sign_unambiguous'] else ' SIGN AMBIGUOUS'}")
        if "E" in res:
            bits.append(f"margin_min={res['E']['min_margin']:.4f}")
        if "F" in res and res["F"]["per_layer"]:
            rs = [p["Q_ratio"][0] for p in res["F"]["per_layer"]
                  if "Q_ratio" in p]
            if rs:
                bits.append(f"minQ1ratio={min(rs):.3f}")
        if res["skipped"]:
            bits.append(f"skipped={sorted(res['skipped'])}")
        print(f"  {model} / {prompt}: " + "  ".join(bits))

    print(f"\nwrote {n_ok}/{len(runs)} runs to {args.out}")
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
