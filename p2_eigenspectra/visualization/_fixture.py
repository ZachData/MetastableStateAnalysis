"""
p2_eigenspectra/visualization/_fixture.py

Synthetic Phase 2 output directory, for exercising the figure modules
without a 27-checkpoint sweep. Underscore-prefixed because it is a test
aid, not part of the figure pipeline.

The numbers are invented but the SHAPES are the ones the real pipeline
writes: same filenames, same JSON keys, same layer-name grammar, same
null-for-NaN convention, same npz array names. If a loader or figure
breaks against this fixture it will break against a real run.

What the synthetic model does, so a fixture figure is legible rather than
noise: eigenspectra start at the untrained expectation (rep_frac ≡ 0.5,
frac_complex ≈ 1, non-normality ≈ 1) and acquire depth structure over a
logistic ramp centered near step 1,000 — early layers going repulsive,
late layers attractive. That is a GUESS at the phenomenon, deliberately
planted so that a figure which fails to show it is showing a plotting
bug. It is not a prediction and no result should ever be read off it.
"""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np


def _ramp(step: int, midpoint: float = 1000.0, width: float = 1.2) -> float:
    """0 at init, 1 late, logistic in log10(step+1)."""
    x = np.log10(step + 1.0)
    return float(1.0 / (1.0 + np.exp(-(x - np.log10(midpoint)) / width * 4.0)))


def _summary(step: int, n_layers: int = 24, d_model: int = 1024,
             n_heads: int = 16, seed: int = 0, model: str = None) -> dict:
    rng = np.random.default_rng(seed + step)
    depth = np.linspace(0.0, 1.0, n_layers)
    a = _ramp(step)

    # Depth shape: early repulsive, late attractive, crossover ~60% depth.
    shape = -np.tanh((depth - 0.6) * 3.2)
    rep = 0.5 + 0.28 * a * shape + rng.normal(0, 0.012, n_layers)
    rep = np.clip(rep, 0.02, 0.98)

    sym_rep = np.clip(rep + 0.03 * a * np.sin(depth * 6.0)
                      + rng.normal(0, 0.008, n_layers), 0.02, 0.98)
    frac_complex = np.clip(0.995 - 0.35 * a * np.exp(-((depth - 0.5) ** 2) / 0.08)
                           + rng.normal(0, 0.005, n_layers), 0.0, 1.0)
    radius = 0.35 + 1.7 * a * np.exp(-((depth - 0.75) ** 2) / 0.12) \
        + rng.normal(0, 0.02, n_layers)
    radius = np.clip(radius, 1e-3, None)
    nonnorm = 1.02 + 2.4 * a * np.exp(-((depth - 0.45) ** 2) / 0.15) \
        + rng.normal(0, 0.03, n_layers)
    norm = radius * np.clip(nonnorm, 1.0, None)
    qk = 0.8 + 2.5 * a * depth + rng.normal(0, 0.05, n_layers)

    layers = {}
    for i in range(n_layers):
        dim_r = int(round(rep[i] * d_model))
        agree = bool(abs(sym_rep[i] - rep[i]) < 0.10)
        qk_heads = np.clip(qk[i] + rng.normal(0, 0.15, n_heads), 0.01, None)
        # Per-head spectra, as weights.head_core_spectra now writes them.
        # Heads differentiate as the ramp advances: at init they all sit at
        # the d_head-dimensional chance level, and they fan out with
        # training. d_head eigenvalues per head, not d_model — the fixture
        # reproduces the low-rank convention so a figure that assumes the
        # old diluted counts fails here rather than in a real run.
        d_head = d_model // n_heads
        # Init spread is the binomial chance level for d_head eigenvalues,
        # 0.5/sqrt(d_head); the trained fan-out is planted well above it so
        # a differentiation figure has an unambiguous direction to show.
        head_rep = np.clip(
            rep[i]
            + rng.normal(0, 0.5 / np.sqrt(d_head), n_heads)
            + a * rng.normal(0, 0.35, n_heads),
            0.0, 1.0,
        )
        head_rep = np.round(head_rep * d_head) / d_head   # only d_head bins exist
        heads = [{
            "frac_repulsive":  float(hr),
            "frac_attractive": float(1.0 - hr),
            "frac_complex":    float(np.clip(0.9 - 0.3 * a, 0, 1)),
            "spectral_radius": float(max(radius[i] / n_heads, 1e-4)),
            "eig_real_mean":   float(rng.normal(0, 0.05)),
            "n_eigenvalues":   d_head,
            "n_negligible":    0,
        } for hr in head_rep]
        layers[f"layer_{i}"] = {
            "heads":               heads,
            "head_rep_frac_mean":  float(head_rep.mean()),
            "head_rep_frac_std":   float(head_rep.std()),
            "head_rep_frac_range": float(head_rep.max() - head_rep.min()),
            "frac_attractive": float(1.0 - rep[i]),
            "frac_repulsive": float(rep[i]),
            "frac_complex": float(frac_complex[i]),
            "sym_frac_attractive": float(1.0 - sym_rep[i]),
            "sym_frac_repulsive": float(sym_rep[i]),
            "methods_agree": agree,
            "schur_cond": float(np.exp(rng.normal(0, 0.8))),
            "schur_dim_attract": d_model - dim_r,
            "schur_dim_repulse": dim_r,
            "sym_dim_attract": d_model - int(round(sym_rep[i] * d_model)),
            "sym_dim_repulse": int(round(sym_rep[i] * d_model)),
            "ov_spectral_norm": float(norm[i]),
            "ov_spectral_radius": float(radius[i]),
            "qk_spectral_norms_per_head": [float(v) for v in qk_heads],
            "qk_spectral_norm_mean": float(np.mean(qk_heads)),
        }

    return {
        "model": model,         # save_weight_decomposition now fills this
        "d_model": d_model,
        "d_head": d_model // n_heads,
        "n_heads": n_heads,
        "is_per_layer": True,
        "layers": layers,
    }


def _decomp_npz(path: Path, step: int, n_layers: int, d_model: int,
                summary: dict, layers_to_write: Optional[List[int]] = None) -> None:
    """
    Eigenvalue clouds consistent with the summary's rep_frac.

    Only the arrays the cloud figure reads (eig_real / eig_imag /
    sym_evals) are written, and only for the requested layers — the real
    file also carries schur_Z and sym_evecs at (d, d) per layer, which is
    ~200 MB per checkpoint and irrelevant to any figure here.
    """
    rng = np.random.default_rng(1234 + step)
    a = _ramp(step)
    arrays = {}
    idxs = layers_to_write if layers_to_write is not None else range(n_layers)
    for i in idxs:
        s = summary["layers"][f"layer_{i}"]
        rep = s["frac_repulsive"]
        radius = s["ov_spectral_radius"]
        n_neg = int(rep * d_model)
        # Ginibre-like disk, radius-matched, with a trained real-axis
        # condensation that grows with the ramp.
        ang = rng.uniform(0, 2 * np.pi, d_model)
        rad = radius * np.sqrt(rng.uniform(0, 1, d_model))
        re = rad * np.cos(ang)
        im = rad * np.sin(ang) * (1.0 - 0.75 * a)
        # Force the sign split to match rep_frac.
        order = np.argsort(re)
        sign = np.ones(d_model)
        sign[order[:n_neg]] = -1.0
        re = np.abs(re) * sign
        arrays[f"eig_real_layer_{i}"] = re.astype(np.float32)
        arrays[f"eig_imag_layer_{i}"] = im.astype(np.float32)
        sym = np.sort(rng.normal(0, radius / 2, d_model))
        sym[: int(s["sym_frac_repulsive"] * d_model)] *= -np.sign(
            sym[: int(s["sym_frac_repulsive"] * d_model)] + 1e-12)
        arrays[f"sym_evals_layer_{i}"] = sym.astype(np.float32)
    np.savez_compressed(path, **arrays)


def _verdict(model: str, prompt: str, step: int, summary: dict,
             seed: int = 0) -> dict:
    """
    A verdict with `frac_ffn_amplifies_repulsive` absent, so the figure
    code's FFN-unavailable annotation stays covered.

    This was the real Pythia case until decompose.py started dispatching on
    model family; it now describes any architecture without a branch in
    core/sublayer_streams.blocks_of. `ffn_channel_available` reads the data
    rather than the model name, so a fixture rebuilt with the field present
    turns the annotation off by itself.
    """
    rng = np.random.default_rng(hash((model, prompt)) % (2 ** 31))
    a = _ramp(step)
    rep = np.array([summary["layers"][k]["frac_repulsive"]
                    for k in summary["layers"]])
    n_viol = int(rng.poisson(1.0 + 4.0 * a))
    frac_rep = float(np.clip(0.15 + 0.6 * a + rng.normal(0, 0.08), 0, 1)) if n_viol else 0.0
    resc_imp = int(round(n_viol * np.clip(0.2 + 0.7 * a, 0, 1)))
    v = {
        "model": model,
        "prompt": prompt,
        "ov_frac_repulsive_mean": float(rep.mean()),
        "ov_methods_agree_all": bool(all(
            summary["layers"][k]["methods_agree"] for k in summary["layers"])),
        "beta1.0_n_violations": n_viol,
        "beta1.0_frac_overshoot": float(np.clip(rng.normal(0.2, 0.1), 0, 1)),
        "beta1.0_frac_repulsive": frac_rep,
        "beta1.0_frac_self_neg": float(np.clip(rng.normal(0.3, 0.1), 0, 1)),
        "rescaled_improvement_beta1.0": resc_imp,
        "layer_v_crossover": int(np.argmax(rep < 0.5)) if (rep < 0.5).any() else None,
        "layer_v_n_repulsive": int((rep > 0.55).sum()),
        "layer_v_n_attractive": int((rep < 0.45).sum()),
        "violation_rate_repulsive_zone": float(np.clip(0.1 + 0.4 * a, 0, 1)),
        "violation_rate_attractive_zone": float(np.clip(0.1 + 0.05 * a, 0, 1)),
        "violation_rate_transition_zone": float(np.clip(0.1 + 0.1 * a, 0, 1)),
        "rho_repulsive_vs_violations": float(np.clip(0.05 + 0.5 * a
                                                     + rng.normal(0, 0.1), -1, 1)),
        "head_ov_fiedler_rho": float(np.clip(-0.02 - 0.55 * a
                                             + rng.normal(0, 0.08), -1, 1)),
        "head_ov_fiedler_pval": float(max(1e-6, 0.6 * (1 - a))),
        "continuous_repfrac_vs_deltaE_rho": float(np.clip(-0.4 * a
                                                          + rng.normal(0, 0.1), -1, 1)),
        "ov_norm_partial_rho": float(np.clip(-0.1 - 0.35 * a
                                             + rng.normal(0, 0.1), -1, 1)),
        "ov_norm_is_confound": bool(a > 0.5),
        "channel": "mixed",
        "decompose_n_violations": 0,
        "decompose_frac_ffn_drop": 0.0,
        "decompose_mean_ffn_frac": 0.0,
        "falsification": ("no_violations" if n_viol == 0
                          else ("V_repulsive_local" if frac_rep > 0.5
                                else "mixed_or_unattributed")),
    }
    rescaled_frac = resc_imp / max(n_viol, 1)
    v["v_score"] = float(0.40 * rescaled_frac + 0.25 * frac_rep
                         - 0.15 * abs(v["ov_norm_partial_rho"]))
    return v


def build_fixture(
    root: Path,
    steps: Optional[List[int]] = None,
    prompts: Optional[List[str]] = None,
    base: str = "pythia-410m",
    n_layers: int = 24,
    d_model: int = 128,
    n_heads: int = 16,
    decomp_layers: Optional[List[int]] = None,
) -> Path:
    """
    Write a synthetic Phase 2 output directory and return its path.

    d_model defaults to 128 rather than Pythia-410M's 1024 so the fixture
    stays small; nothing in the figure code depends on the value.
    """
    if steps is None:
        steps = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                 1000, 3000, 5000, 9000, 13000, 19000, 60000, 143000]
    if prompts is None:
        prompts = ["wiki_paragraph", "short_heterogeneous", "repeated_tokens",
                   "paper_excerpt", "homer_iliad"]
    if decomp_layers is None:
        decomp_layers = [0, n_layers // 2, n_layers - 1]

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    all_verdicts = []
    for step in steps:
        model = f"{base}-step{step}"
        summary = _summary(step, n_layers=n_layers, d_model=d_model,
                           n_heads=n_heads, model=model)
        (root / f"ov_summary_{model}.json").write_text(json.dumps(summary, indent=2))
        _decomp_npz(root / f"ov_decomp_{model}.npz", step, n_layers, d_model,
                    summary, layers_to_write=decomp_layers)

        for prompt in prompts:
            stem = root / f"{model}_{prompt}"
            (stem / "sub").mkdir(parents=True, exist_ok=True)
            v = _verdict(model, prompt, step, summary)
            (stem / "phase2_verdict.json").write_text(json.dumps(v, indent=2))

            rep = [summary["layers"][k]["frac_repulsive"] for k in summary["layers"]]
            (stem / "sub" / "layer_v_events.json").write_text(json.dumps({
                "name": "layer_v_events", "applicable": True,
                "payload": {
                    "applicable": True,
                    "v_profile": {
                        "repulsive_frac": rep,
                        "qk_mean_norm": [summary["layers"][k]["qk_spectral_norm_mean"]
                                         for k in summary["layers"]],
                        "n_layers": n_layers,
                    },
                    "zones": {"crossover_layer": v["layer_v_crossover"],
                              "n_repulsive": v["layer_v_n_repulsive"],
                              "n_attractive": v["layer_v_n_attractive"]},
                },
                "verdict_contribution": {},
                "error": None,
            }, indent=2))
            n_trans = n_layers - 1
            a = _ramp(step)
            (stem / "sub" / "trajectory.json").write_text(json.dumps({
                "name": "trajectory", "applicable": True,
                "payload": {
                    "profiles": {
                        "n_layers": n_layers,
                        "n_transitions": n_trans,
                        "overshoot_threshold": 0.6,
                        "global_step_mean": 0.3,
                        "global_step_std": 0.15,
                        "per_layer": {
                            "frac_negative": [
                                float(np.clip(0.5 + 0.45 * a * (r - 0.5) * 2, 0, 1))
                                for r in rep],
                            "self_int_mean": [float(-0.1 * a * (r - 0.5)) for r in rep],
                            "subspace_sym_repulse_frac": [float(r) for r in rep],
                        },
                        "per_transition": {
                            "step_mean": [
                                float(0.3 + 0.3 * a * np.sin(t / 3.0))
                                for t in range(n_trans)],
                            "sym_repulse_disp_frac": [
                                float(np.clip(rep[t + 1] + 0.1 * a, 0, 1))
                                for t in range(n_trans)],
                            "total_disp_energy": [1.0] * n_trans,
                        },
                    },
                    "rescaled_comparison": {
                        "beta_1.0": {"approx_error_pct": float(5 + 20 * a)},
                    },
                },
                "verdict_contribution": {}, "error": None,
            }, indent=2))
            all_verdicts.append(v)

    (root / "p2_eigenspectra_cross_run.json").write_text(
        json.dumps(all_verdicts, indent=2))
    return root


if __name__ == "__main__":
    import sys
    out = build_fixture(Path(sys.argv[1] if len(sys.argv) > 1
                             else "results/_p2_fixture"))
    print(f"fixture written to {out}")
