"""
p1c_frames/visualization/_fixture.py — a synthetic Phase 1c output directory.

A test aid, and the fastest way to see the whole catalogue without a Pythia
checkpoint. **The numbers are invented and no result should ever be read off
them.** The shapes, the filenames, the dotted-path grammar and the size
threshold that decides which file each array lands in are the real ones,
because everything here is written through `p1c_io.save_p1c` rather than
hand-rolled — which is the only way a fixture can catch the failure it
exists to catch. A fixture that writes its own JSON tests the fixture.

WHAT IS REAL AND WHAT IS INVENTED

Real: the file layout, every key name, the per-layer/per-transition lengths
(`h_*` is n_layers-1, the layer series are n_layers), and the verdict
strings — `verdict`, `adjudicate_sink_hypothesis`, `envelope_verdict`,
`residual_bracket`, `reduction_report` and `wendel_probability` are the
phase's own functions, called on the invented arrays. So a figure quoting a
verdict is exercised against the real vocabulary, and a verdict class the
palette has no colour for shows up here rather than in a blog draft.

Invented: every number. In particular the ODE is NOT integrated —
`integrate_gamma` is a fixed-step Python loop and calling it per layer per
run would make the fixture slower than the analysis it stands in for. The
null curves below are sigmoids of roughly the right shape, and the `theory`
class is where the real ODE gets drawn.

WHAT THE DIRECTORY CONTAINS

Eleven runs: two prompts across two non-checkpoint models, a six-step
`pythia-410m` family on `wiki_paragraph`, and one deliberately degraded run —
`gpt2_short_prompt` — that predates the `norms` fix, carries no `beta_eff`,
and therefore has A, B, C and F all skipped with the driver's own messages.
That run is the fixture's real payload: it is what a directory of old Phase
1 artifacts looks like, and every figure has to degrade to a printed skip
against it rather than a KeyError.

Sub-experiment D is written for ONE run only (`gpt2-large_wiki_paragraph`),
because no driver writes it at all (FIGURES-1c.md G1). Its presence here is
what keeps the `frames` class exercised; its absence everywhere else is what
keeps the skip path exercised.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from p1c_frames.beta_reduction import (
    envelope_verdict, reduction_report, residual_bracket,
)
from p1c_frames.hemisphere_feasibility import wendel_probability
from p1c_frames.integration_time import verdict as a_verdict
from p1c_frames.moments import adjudicate_sink_hypothesis
from p1c_frames.p1c_io import save_p1c

__all__ = ["build_fixture", "build_run", "FIXTURE_STEPS"]

#: The checkpoint family's steps — log-spaced the way the real sweep is, so
#: the step axis in `checkpoints_1c.py` is exercised on a realistic spread
#: rather than on six evenly spaced numbers.
FIXTURE_STEPS = (0, 1000, 8000, 32000, 96000, 143000)

_BETAS = (0.1, 1.0, 2.0, 5.0)


# ---------------------------------------------------------------------------
# Invented but shaped profiles
# ---------------------------------------------------------------------------

def _ip_profile(n_layers: int, resistance: float, rng) -> np.ndarray:
    """
    A layer-wise `ip_mean` with the shape Phase 1 actually reports: a rise
    off the embedding, a mid-network dip below the layer-0 value, and a
    late climb. The dip is deliberate — it is what produces `nan` layers in
    the time-domain residual (observed below the null's own start), and a
    fixture with no unreachable layers would never exercise the code path
    that exists for them.
    """
    x = np.linspace(0.0, 1.0, n_layers)
    base = 0.06 + 0.80 * x ** 2
    dip = 0.16 * np.exp(-((x - 0.42) ** 2) / 0.012)
    return np.clip(base - dip * (0.4 + resistance) + rng.normal(0, 0.004, n_layers),
                   -0.4, 0.995)


def _sigmoid_null(t_grid: np.ndarray, t_star: float, g0: float = 0.0):
    """
    A stand-in for gamma_beta(t): a logistic reaching 0.9 at t_star.

    Shape only. The real curve is `gamma_ode.integrate_gamma`, which the
    `theory` class draws and which no fixture should be re-implementing for
    numbers anyone might read.
    """
    k = 2.2 / max(t_star, 1e-6) * 4.0
    g = 1.0 / (1.0 + np.exp(-k * (t_grid - 0.55 * t_star)))
    g = (g - g[0]) / max(1.0 - g[0], 1e-9)
    return g0 + (1.0 - g0) * g


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def build_run(out_dir: Path, model: str, prompt: str,
              n_layers: int = 12, n_tokens: int = 48,
              seed: int = 0, subexps: Sequence[str] = ("A", "B", "C", "E", "F"),
              per_head_beta: bool = True, attn_streams: bool = True,
              with_d: bool = False, resistance: float = 0.4,
              skipped: Optional[Dict[str, str]] = None) -> Path:
    """
    Write one synthetic `{out_dir}/{stem}/p1c.json` + `p1c_curves.npz`.

    `resistance` scales how far the invented trajectory sits below its null,
    so the checkpoint family can be built with a residual that grows across
    training and `adjudicate_p_gamma1` — imported by K1, never restated —
    has something with a real shape to adjudicate.

    `per_head_beta` and `attn_streams` are the two independent gaps G4 and
    G3, kept as separate flags because they are separate absences: one is a
    missing `geometry.json` key and the other a missing capture, and a run
    can have either without the other.
    """
    rng = np.random.default_rng(seed)
    subexps = set(subexps)
    stem = f"{model}_{prompt}"
    res: dict = {
        "run_dir": f"results/{stem}",
        "model": model,
        "prompt": prompt,
        "n_tokens": int(n_tokens),
        "available": sorted({"geometry", "energies", "activations", "norms"}
                            if "A" in subexps else {"geometry", "activations"}),
        "provenance": {
            name: {"exists": True, "mtime": 1.7e9 + seed, "bytes": 4096 + seed}
            for name in ("geometry.json", "energies.json", "activations.npz")
        },
        "skipped": dict(skipped or {}),
        "beta_used": float(1.0 + 0.4 * rng.random()),
        "beta_source": "geometry.json" if "A" in subexps else "fallback_flag",
    }
    beta = res["beta_used"]
    t_star = 4.2 * (1.0 + 0.15 * rng.standard_normal())
    ip_mean = _ip_profile(n_layers, resistance, rng)

    # --- A -----------------------------------------------------------------
    h_cal = np.clip(0.16 + 0.05 * np.sin(np.linspace(0, 3.5, n_layers - 1))
                    + rng.normal(0, 0.012, n_layers - 1), 0.01, None)
    field_mag = np.clip(0.15 + 0.10 * np.linspace(0, 1, n_layers - 1)
                        + rng.normal(0, 0.008, n_layers - 1), 0.02, 1.0)
    h_disp = h_cal * field_mag
    # No sublayer streams on this run: the frame-correct definition is nan,
    # which is the state status-1c open item 3 warns about.
    h_attn = (h_cal * (0.55 + 0.1 * rng.random(n_layers - 1)) if attn_streams
              else np.full(n_layers - 1, np.nan))

    t_grid = np.concatenate([[0.0], np.nancumsum(h_cal)])

    if "A" in subexps:
        a = {
            "h_displacement": h_disp, "h_calibrated": h_cal,
            "h_attn_only": h_attn, "field_mag": field_mag,
            "T_eff_displacement": float(np.nansum(h_disp)),
            "T_eff_calibrated": float(np.nansum(h_cal)),
            "T_eff_attn_only": float(np.nansum(h_attn)),
            "n_layers": int(n_layers), "beta": beta, "causal": True,
        }
        res["A"] = {**a, "t_star": float(t_star),
                    "verdict": a_verdict(a, float(t_star))}

    # --- B -----------------------------------------------------------------
    if "B" in subexps:
        g_null = _sigmoid_null(t_grid, t_star)
        g_matched = _sigmoid_null(t_grid, t_star, g0=float(ip_mean[0]))
        residual = ip_mean - g_null
        betas = np.full(n_layers, beta)
        betas[1] = np.nan                      # one failed regression
        med = float(np.nanmedian(betas))
        n_fallback = int(np.sum(~np.isfinite(betas)))
        betas = np.where(np.isfinite(betas), betas, med)

        # Time-domain: invert the same sigmoid. Layers whose observed value
        # sits below the null's start have no null time at all and stay nan.
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.clip(ip_mean, 1e-9, 1 - 1e-9)
            t_req = np.where(ip_mean > 0.0,
                             t_star * (0.55 + np.log(frac / (1 - frac)) / 9.0),
                             np.nan)
        t_req = np.where(np.isfinite(t_req) & (t_req > 0), t_req, np.nan)
        time_resid = t_req - t_grid

        res["B"] = {
            "gamma_null": g_null, "residual": residual,
            "beta_per_layer": betas, "beta_median": med,
            "n_beta_fallback": n_fallback, "model": "sa",
            "t_eff_grid": t_grid, "ip_mean": ip_mean,
            "final_residual": float(residual[-1]),
            "max_abs_residual": float(np.nanmax(np.abs(residual))),
            "mean_residual": float(np.nanmean(residual)),
            "sign_convention": "residual = observed - null; negative = less "
                               "clustered than identity-weight dynamics "
                               "predict = resistance",
            "gamma_null_matched": g_matched,
            "residual_matched": ip_mean - g_matched,
            "final_residual_matched": float((ip_mean - g_matched)[-1]),
            "anisotropy_gap": float(np.nanmean(np.abs(g_null - g_matched))),
            "n_tokens": int(n_tokens),
            "time_domain": {
                "t_required": t_req, "t_eff_grid": t_grid,
                "time_residual": time_resid,
                "final_time_residual": float(time_resid[-1]),
                "mean_time_residual": float(np.nanmean(time_resid)),
                "n_unreachable": int(np.sum(~np.isfinite(t_req))),
                "sign_convention": "time_residual = t_null_required - T_eff "
                                   "spent; negative = resistance",
            },
            "collapse_fraction": {
                "t_eff_total": float(t_grid[-1]), "t_star": float(t_star),
                "time_fraction": float(t_grid[-1] / t_star),
                "gamma_reached_by_null": float(g_null[-1]),
                "gamma_fraction": float(g_null[-1] / 0.9),
                "ip_mean_final": float(ip_mean[-1]),
            },
        }

        if per_head_beta:
            per_head = beta * (1.0 + 0.45 * rng.standard_normal(16))
            per_head[3] = np.nan               # one head's regression failed
            rr = reduction_report(per_head)
            spread = 0.10 + 0.05 * rng.random()
            lower = np.clip(g_null - spread, 0, 1)
            upper = np.clip(g_null + spread, 0, 1)
            env = {"lower": lower, "upper": upper, "width": upper - lower,
                   "mean_width": float(np.mean(upper - lower)),
                   "max_width": float(np.max(upper - lower)),
                   "beta_min": rr["beta_min"], "beta_max": rr["beta_max"],
                   "model": "sa", "n": int(n_tokens),
                   "upper_edge_beta": rr["beta_min"],
                   "lower_edge_beta": rr["beta_max"]}
            res["B"]["beta_reduction"] = rr
            res["B"]["envelope_verdict"] = envelope_verdict(ip_mean, env)
            res["B"]["residual_bracket"] = residual_bracket(ip_mean, env)
            res["B"]["envelope_lower"] = lower
            res["B"]["envelope_upper"] = upper
        else:
            res["B"]["envelope_note"] = (
                "no per-head beta_eff in geometry.json, so the residual is a "
                "point estimate at a single reduction. Its error bar is the "
                "beta envelope and is currently unreported.")

    # --- C -----------------------------------------------------------------
    if "C" in subexps:
        norm_pr = np.clip(6.0 - 3.4 * np.linspace(0, 1, n_layers) ** 1.5
                          + rng.normal(0, 0.15, n_layers), 1.05, None)
        panels = []
        for l in range(n_layers):
            raw = float(norm_pr[l] * (1.0 + 0.05 * rng.standard_normal()))
            panels.append({
                "shannon_raw": raw,
                "shannon_normed": float(np.clip(40 - 30 * l / n_layers, 2, None)
                                        + rng.normal(0, 0.8)),
                "pr_rank": float(np.clip(60 - 45 * l / n_layers, 2, None)),
                "norm_pr": float(norm_pr[l]),
                "sink_ratio": float(raw / norm_pr[l]),
                "norm_max_over_median": float(1.5 + 6 * (l / n_layers) ** 2),
                "n_tokens": int(n_tokens),
            })
        checks = []
        for l in range(n_layers):
            row = {"layer": l, "source": "gram_exact" if l % 3 else
                   "ip_histogram_converted"}
            for b in _BETAS:
                err = {0.1: 0.0002, 1.0: 0.0009, 2.0: 0.008, 5.0: 0.266}[b]
                err *= 1.0 + 0.25 * rng.random()
                row[b] = {
                    "measured": float(0.5 + b * 0.3),
                    "two_term": float((0.5 + b * 0.3) * (1 - err)),
                    "three_term": float((0.5 + b * 0.3) * (1 - err * 0.6)),
                    "rel_err_two": float(err),
                    "rel_err_three": float(err * 0.6),
                    "ladder_sufficient": bool(err < 0.01),
                }
            checks.append(row)
        res["C"] = {"panels": panels,
                    "sink_verdict": adjudicate_sink_hypothesis(panels),
                    "moment_identity": checks}

    # --- E -----------------------------------------------------------------
    if "E" in subexps:
        margins = np.clip(0.22 - 0.20 * np.linspace(0, 1, n_layers) ** 1.4
                          + rng.normal(0, 0.006, n_layers), -0.02, None)
        d_model = 1024
        per_layer = []
        for l in range(n_layers):
            m = float(margins[l])
            per_layer.append({
                "n_tokens": int(n_tokens), "margin": m,
                "feasible": bool(m > 1e-8), "boundary": bool(m <= 1e-8),
                "support_size": int(2 + 6 * l / max(n_layers - 1, 1)),
                "min_pairwise_ip": float(-0.6 + 0.5 * l / n_layers),
                "converged": bool(l != n_layers - 2),
                "n_iter": int(120 + 40 * l), "zero_tol": 1e-8,
                "verdict_trustworthy": bool(l != n_layers - 2),
                "d_model": d_model,
                "wendel_p": wendel_probability(n_tokens, d_model),
                "wendel_says_certain": bool(d_model > n_tokens),
            })
        feas = np.array([p["feasible"] for p in per_layer])
        res["E"] = {
            "per_layer": per_layer, "margins": margins,
            "all_feasible": bool(feas.all()),
            "first_infeasible_layer": int(np.argmin(feas)) if not feas.all() else -1,
            "n_infeasible_layers": int((~feas).sum()),
            "min_margin": float(margins.min()),
            "min_margin_layer": int(np.argmin(margins)),
            "layer0_margin": float(margins[0]),
            "final_margin": float(margins[-1]),
        }

    # --- F -----------------------------------------------------------------
    if "F" in subexps:
        t_max = 3
        band = np.array([0.170, 0.015, 0.002])
        per_layer = []
        for l in range(n_layers):
            if l == 1:
                continue                      # a layer whose centroids failed
            m = int(4 + 6 * rng.random())
            sharp = 1.0 - 0.55 * resistance * (l / max(n_layers - 1, 1))
            ratio = np.clip(sharp + rng.normal(0, 0.05, t_max)
                            * np.array([1.0, 0.12, 0.02]), 0.0, None)
            per_layer.append({
                "layer": l, "method": "kmeans", "n_centroids": m,
                "source": "persisted", "d": 1024,
                "Q": (ratio / m).tolist(),
                "Q_random_mean": (np.ones(t_max) / m).tolist(),
                "Q_ratio": ratio.tolist(),
                "random_band": band.tolist(),
                "outside_band": (np.abs(ratio - 1.0) > band).tolist(),
                "t_design_vs_random": int(sum(ratio < 0.5)),
                "t_design_strict": 0,
                "modes": {
                    "n_modes": int(1 + (l % 3)),
                    "mode_locations": [-0.3 + 0.25 * k for k in range(1 + l % 3)],
                    "mass_at_modes": float(0.4 + 0.5 * rng.random()),
                    "ip_mean": float(ip_mean[l]),
                    "ip_std": float(0.1 + 0.05 * rng.random()),
                    "unimodal": bool((1 + l % 3) <= 1),
                },
                "sharp_score": float(np.mean(ratio)),
            })
        res["F"] = {"method": "kmeans", "t_max": t_max,
                    "per_layer": per_layer,
                    "errors": ["layer 1: KeyError: clusters.npz has no "
                               "kmeans_centroids_L1"]}

    # --- D (fixture only — no driver writes this; FIGURES-1c.md G1) --------
    if with_d:
        res["D"] = _fake_frame_block(n_layers, n_tokens, rng)

    return save_p1c(res, Path(out_dir) / stem)


def _fake_frame_block(n_layers: int, n_tokens: int, rng) -> dict:
    """
    A `D` block in the shape `frame_table()` returns, one entry per layer.

    Written here and nowhere else on purpose: this is the only place in the
    repository that asserts what sub-experiment D's artifact would look
    like, so when the driver branch lands it has a target to match — and if
    it lands in a different shape, the `frames` figures break here first.
    """
    per_layer = []
    for l in range(n_layers):
        depth = l / max(n_layers - 1, 1)
        base = 0.05 + 0.6 * depth ** 2
        entry = {}
        for frame, shift, rank in (("l2", 0.0, 145.0), ("ln_plain", -0.04, 96.0),
                                   ("ln_learned", -0.09, 71.0),
                                   ("functional", 0.06, 52.0)):
            entry[frame] = {
                "kappa1": float(base + shift),
                "kappa2": float(0.02 + 0.01 * rng.random()),
                "kappa3": float(0.001 * rng.standard_normal()),
                "pr_rank": float(rank * (1 - 0.6 * depth) + 2),
                "ip_mean": float(base + shift),
                "energies": {b: float(0.4 + b * 0.25 + shift) for b in _BETAS},
                "neg_eigen_mass": float(0.017 if frame == "functional" else 0.0),
                "n_dropped_rows": int(2 if frame == "functional" else 0),
                "n": int(n_tokens),
            }
        entry["l2"]["raw_effective_rank"] = float(5.0 - 2.0 * depth)
        entry["ln_plain"]["norm_cv"] = 3.5e-8
        entry["ln_plain"]["norm_mean_over_sqrt_d"] = 1.0
        entry["ln_learned"]["norm_cv"] = float(0.02 + 0.01 * rng.random())
        entry["gamma_stats"] = {
            "mean": 0.44, "sd": float(0.05 + 0.10 * depth),
            "cv": float((0.05 + 0.10 * depth) / 0.44),
            "min": 0.2, "max": float(0.8 + depth), "abs_min": 0.2,
            "abs_max": float(0.8 + depth),
            "condition_number": float((0.8 + depth) / 0.2),
            "n_channels": 1024, "n_negative": 0,
        }
        entry["bias_floor"] = {
            "energy_floor_frac": {b: float(0.05 + 0.13 * depth) for b in _BETAS},
            "kappa1_with": float(base + 0.20), "kappa1_without": float(base),
            "kappa1_shift": 0.20,
            "bias_norm": 3.1, "signal_norm": 6.2, "bias_norm_ratio": 0.5,
            "ip_mean_with": float(base + 0.20), "ip_mean_without": float(base),
        }
        entry["layer"] = l
        per_layer.append(entry)

    from p1c_frames.frame_table import sphere_license
    lic = sphere_license([e["gamma_stats"] for e in per_layer])
    spreads = [max(e[f]["ip_mean"] for f in ("l2", "ln_plain", "ln_learned",
                                             "functional"))
               - min(e[f]["ip_mean"] for f in ("l2", "ln_plain", "ln_learned",
                                               "functional"))
               for e in per_layer]
    return {"per_layer": per_layer, "sphere_license": lic,
            "frame_disagreement": {"key": "ip_mean", "spread_per_layer": spreads,
                                   "max_spread": float(max(spreads))}}


# ---------------------------------------------------------------------------
# The directory
# ---------------------------------------------------------------------------

def build_fixture(out_dir: Path) -> Path:
    """
    Eleven runs: two static models × two prompts, a six-step pythia-410m
    family on `wiki_paragraph`, and one degraded run.

    Returns `out_dir`, so a caller can pass it straight to `discover_runs`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = 0
    for model in ("gpt2-large", "albert-base-v2"):
        for prompt in ("wiki_paragraph", "code_snippet"):
            build_run(out_dir, model, prompt, n_layers=12, n_tokens=48,
                      seed=seed, with_d=(model == "gpt2-large"
                                         and prompt == "wiki_paragraph"),
                      per_head_beta=(model == "gpt2-large"),
                      attn_streams=(model == "gpt2-large"))
            seed += 1

    # A checkpoint family, with the residual growing across training so K1's
    # imported P-gamma1 adjudicator has a real shape to read.
    for i, step in enumerate(FIXTURE_STEPS):
        build_run(out_dir, f"pythia-410m-step{step}", "wiki_paragraph",
                  n_layers=24, n_tokens=467, seed=100 + i,
                  resistance=0.05 + 0.75 * i / (len(FIXTURE_STEPS) - 1))

    # The old-artifact run: no `norms`, no beta_eff. A, B and C are skipped
    # by the driver with its own messages; F has no clusters.npz. Only E
    # survives, which is exactly what `tools/preflight_1c.py` predicts for a
    # run made before the norms fix.
    build_run(out_dir, "gpt2", "short_prompt", n_layers=8, n_tokens=20,
              seed=200, subexps=("E",), per_head_beta=False,
              attn_streams=False,
              skipped={
                  "A": ("results/gpt2_short_prompt: activations.npz has no "
                        "`norms` key. This run predates p1_io's norm-saving "
                        "fix; the raw residual stream is not recoverable."),
                  "B": "needs sub-experiment A's step sizes and a finite "
                       "ip_mean series",
                  "C": ("results/gpt2_short_prompt: activations.npz has no "
                        "`norms` key."),
                  "F": "no layer produced centroids: no clusters.npz under "
                       "results/gpt2_short_prompt",
              })
    return out_dir
