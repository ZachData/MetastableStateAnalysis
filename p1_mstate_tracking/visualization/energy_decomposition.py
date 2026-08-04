"""
visualization/energy_decomposition.py

Single-prompt energy decomposition: attention structure (Phase 1
sinkhorn.json — Fiedler/entropy) vs. attn/FFN energy attribution (Phase 2
`--full` decompose.py's saved raw attn/FFN deltas). Cross-references a
Group C structure run against a matching Phase 2 run for the same
(model, prompt); silently skipped for any model with no Phase 2 run on
disk — additive, never blocks the rest of generate_all.

energy_by_component reimplements decompose.py's energy-attribution
formula rather than importing it, so this module has no hard dependency
on Phase 2's package path. energy_attribution_aggregate.py reuses
_phase2_decomposed, _per_layer_energy_fracs, and energy_by_component from
here for its multi-prompt aggregate — if decompose.py's formula changes,
both copies need updating together.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from core.style import BLOG_STYLE
from core.naming import _safe_model_name
from .loaders import _geo
from .series import _attention_entropy_mean_series, _fiedler_mean_series

def _energy(X_raw: np.ndarray, beta: float) -> float:
    """Mirrors p2_eigenspectra/decompose.py::energy_by_component's inner energy term."""
    X = X_raw / np.maximum(np.linalg.norm(X_raw, axis=-1, keepdims=True), 1e-10)
    G = X @ X.T
    n = G.shape[0]
    return float(np.exp(beta * G).sum() / (2.0 * beta * n * n))


def energy_by_component(
    hidden_before: np.ndarray,
    attn_delta: np.ndarray,
    ffn_delta: np.ndarray,
    beta: float = 1.0,
) -> dict:
    """
    Signed energy attribution for one layer: how much of the realized
    energy drop (hidden_before -> hidden_after) is explained by adding back
    attn_delta alone, ffn_delta alone, or their cross term. Reimplemented
    here (rather than imported) so this module has no hard dependency on
    Phase 2's package path — if decompose.py's formula changes, this copy
    needs updating too.
    """
    h_after     = hidden_before + attn_delta + ffn_delta
    h_attn_only = hidden_before + attn_delta
    h_ffn_only  = hidden_before + ffn_delta

    e_before    = _energy(hidden_before, beta)
    e_after     = _energy(h_after, beta)
    e_attn_only = _energy(h_attn_only, beta)
    e_ffn_only  = _energy(h_ffn_only, beta)

    delta_total = e_after     - e_before
    delta_attn  = e_attn_only - e_before
    delta_ffn   = e_ffn_only  - e_before
    delta_cross = delta_total - delta_attn - delta_ffn

    denom = max(abs(delta_total), 1e-12)
    return {
        "delta_total": delta_total,
        "delta_attn":  delta_attn,
        "delta_ffn":   delta_ffn,
        "delta_cross": delta_cross,
        "attn_frac":   max(0.0, -delta_attn) / denom,
        "ffn_frac":    max(0.0, -delta_ffn)  / denom,
        "cross_frac":  abs(delta_cross) / denom,
    }


def _phase2_decomposed(phase2_run_dir: Path) -> Optional[dict]:
    """
    Load attn_deltas_raw.npz / ffn_deltas_raw.npz / hidden_states.npz from a
    saved Phase 2 `--full` run directory. Same contract as
    p2_eigenspectra/run_2.py::load_decomposed — reimplemented locally.
    """
    run_dir = Path(phase2_run_dir)
    attn_p = run_dir / "attn_deltas_raw.npz"
    ffn_p  = run_dir / "ffn_deltas_raw.npz"
    if not attn_p.exists() or not ffn_p.exists():
        return None

    attn_raw = np.load(attn_p)["attn_deltas"]
    ffn_raw  = np.load(ffn_p)["ffn_deltas"]
    attn_deltas = [attn_raw[i] for i in range(attn_raw.shape[0])]
    ffn_deltas  = [ffn_raw[i]  for i in range(ffn_raw.shape[0])]

    traj_p = run_dir / "hidden_states.npz"
    if traj_p.exists():
        hs  = np.load(traj_p)
        key = list(hs.keys())[0]
        all_hidden = hs[key]
        trajectory = [all_hidden[i] for i in range(all_hidden.shape[0])]
    else:
        n_tokens, d = attn_deltas[0].shape
        h = np.zeros((n_tokens, d), dtype=np.float32)
        trajectory = [h.copy()]
        for a, f in zip(attn_deltas, ffn_deltas):
            h = h + a + f
            trajectory.append(h.copy())

    return {"trajectory": trajectory, "attn_deltas": attn_deltas, "ffn_deltas": ffn_deltas}


def _find_phase2_run_dir(phase2_dir: Path, model: str, prompt: str) -> Optional[Path]:
    """
    Locate the Phase 2 run directory for (model, prompt) under phase2_dir.
    phase2_dir can be either a single `p2_eigenspectra_<ts>/` run, or a
    parent containing several — both are scanned.
    """
    phase2_dir = Path(phase2_dir)
    if not phase2_dir.exists():
        return None
    stem = _safe_model_name(model)
    name_variants = {stem, stem.replace("-", "_")}

    direct = [phase2_dir / f"{s}_{prompt}" for s in name_variants]
    for c in direct:
        if c.is_dir() and (c / "attn_deltas_raw.npz").exists():
            return c

    # phase2_dir is a parent of several timestamped runs — search one level down.
    for sub in sorted(phase2_dir.iterdir()):
        if not sub.is_dir():
            continue
        for s in name_variants:
            c = sub / f"{s}_{prompt}"
            if c.is_dir() and (c / "attn_deltas_raw.npz").exists():
                return c

    # Last resort: any directory under phase2_dir whose name contains both
    # the model stem and the prompt key.
    for c in sorted(phase2_dir.rglob(f"*{prompt}*")):
        if c.is_dir() and any(s in c.name for s in name_variants) \
                and (c / "attn_deltas_raw.npz").exists():
            return c

    return None


def _per_layer_energy_fracs(decomp: dict, beta: float):
    n_layers = len(decomp["attn_deltas"])
    out = {"attn": [], "ffn": [], "cross": []}
    for i in range(n_layers):
        comp = energy_by_component(
            decomp["trajectory"][i], decomp["attn_deltas"][i], decomp["ffn_deltas"][i], beta,
        )
        out["attn"].append(comp["attn_frac"])
        out["ffn"].append(comp["ffn_frac"])
        out["cross"].append(comp["cross_frac"])
    return out, n_layers


def plot_energy_decomposition_trajectory(
    run_dir: Path,
    phase2_run_dir: Path,
    out_dir: Path,
    beta: float = 1.0,
) -> bool:
    """
    Two stacked panels, shared x = layer:
      top    — Fiedler (mean) + attention entropy, twin y-axes
      bottom — attention / FFN / cross-term share of the total energy
               change (% of |ΔE_β|, log y-scale; Phase 2 decompose.py's
               signed energy attribution, recomputed here from the saved
               attn_deltas_raw.npz / ffn_deltas_raw.npz / hidden_states.npz)
    Puts the Group C attention-structure curves directly above the
    energy-attribution curves so a low-Fiedler / low-entropy window can be
    read off against which component (attention, FFN, or their cross term)
    was actually driving the energy drop at that layer.
    Returns False (no figure, no exception) if either side's data is missing.
    """
    geo = _geo(run_dir)
    model, prompt = geo.get("model", run_dir.name), geo.get("prompt", "")
    fiedler_full = _fiedler_mean_series(run_dir)
    entropy_full = _attention_entropy_mean_series(run_dir)
    decomp = _phase2_decomposed(phase2_run_dir)
    if not fiedler_full or decomp is None:
        return False

    fracs, n_layers = _per_layer_energy_fracs(decomp, beta)
    if n_layers == 0:
        return False

    fiedler = fiedler_full[:n_layers]
    entropy = (entropy_full or [np.nan] * n_layers)[:n_layers]
    x = np.arange(n_layers)

    plt.rcParams.update(BLOG_STYLE)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(
        f"{model} | {prompt} — attention structure vs. energy attribution (β={beta:g})",
        fontsize=10, fontweight="bold",
    )

    ax1.plot(x, fiedler, color="#2563EB", marker="o", ms=3, label="Fiedler (mean)")
    ax1.set_ylabel("Fiedler value", color="#2563EB")
    ax1.tick_params(axis="y", labelcolor="#2563EB")
    ax1b = ax1.twinx()
    ax1b.plot(x, entropy, color="#DC2626", linestyle="--", marker="^", ms=3,
              label="Attention entropy")
    ax1b.set_ylabel("Attention entropy", color="#DC2626")
    ax1b.tick_params(axis="y", labelcolor="#DC2626")
    ax1.set_title("Attention structure (Group C)", fontsize=9)

    attn_pct  = 100.0 * np.asarray(fracs["attn"])
    ffn_pct   = 100.0 * np.asarray(fracs["ffn"])
    cross_pct = 100.0 * np.asarray(fracs["cross"])

    ax2.plot(x, attn_pct,  color="#2563EB", marker="o", ms=3, label="attention")
    ax2.plot(x, ffn_pct,   color="#059669", marker="s", ms=3, label="FFN")
    ax2.plot(x, cross_pct, color="#9CA3AF", marker="^", ms=3,
             linestyle=":", label="cross term")
    ax2.axhline(100.0, color="black", lw=0.6, alpha=0.4)
    ax2.set_yscale("log")
    ax2.set_ylabel("Share of |ΔE_β| explained (%, log scale)")
    ax2.set_xlabel("Layer")
    ax2.set_title("Energy attribution (Phase 2 decompose)", fontsize=9)
    ax2.legend(fontsize=8, loc="best")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{_safe_model_name(model)}_{prompt}_energy_decomposition.png"
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fname.name}")
    return True


def plot_fiedler_vs_energy_attribution(
    run_dir: Path,
    phase2_run_dir: Path,
    out_dir: Path,
    beta: float = 1.0,
) -> bool:
    """
    Two scatter panels: Fiedler vs. attn_frac, Fiedler vs. ffn_frac, each
    with the layer-wise Spearman rho/p annotated. Same logic as the
    Fiedler-vs-HDBSCAN-cluster-count check in Group C.2, applied to the
    energy decomposition instead of the geometric one: rho < -0.4 on the
    attn_frac panel is the signal that attention is doing the work
    specifically where it's structurally clustered.
    Returns False (no figure, no exception) if there isn't enough data.
    """
    geo = _geo(run_dir)
    model, prompt = geo.get("model", run_dir.name), geo.get("prompt", "")
    fiedler_full = _fiedler_mean_series(run_dir)
    decomp = _phase2_decomposed(phase2_run_dir)
    if not fiedler_full or decomp is None:
        return False

    fracs, n_layers = _per_layer_energy_fracs(decomp, beta)
    if n_layers < 4:
        return False

    fiedler   = np.array(fiedler_full[:n_layers], dtype=float)
    attn_frac = np.array(fracs["attn"])
    ffn_frac  = np.array(fracs["ffn"])
    valid     = ~np.isnan(fiedler)
    if valid.sum() < 4:
        return False

    rho_attn, p_attn = spearmanr(fiedler[valid], attn_frac[valid])
    rho_ffn,  p_ffn  = spearmanr(fiedler[valid], ffn_frac[valid])

    plt.rcParams.update(BLOG_STYLE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.suptitle(
        f"{model} | {prompt} — Fiedler vs. energy attribution (β={beta:g})",
        fontsize=10, fontweight="bold",
    )

    ax1.scatter(fiedler[valid], attn_frac[valid], color="#2563EB", s=24)
    signal = "signal: attention drives the drop where it clusters" if rho_attn < -0.4 \
             else "no clear signal"
    ax1.set_xlabel("Fiedler value")
    ax1.set_ylabel("attn_frac")
    ax1.set_title(f"ρ={rho_attn:.2f}, p={p_attn:.3f}\n{signal}", fontsize=8)

    ax2.scatter(fiedler[valid], ffn_frac[valid], color="#059669", s=24)
    ax2.set_xlabel("Fiedler value")
    ax2.set_ylabel("ffn_frac")
    ax2.set_title(f"ρ={rho_ffn:.2f}, p={p_ffn:.3f}", fontsize=8)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{_safe_model_name(model)}_{prompt}_fiedler_vs_energy.png"
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fname.name}")
    return True


def generate_energy_decomposition_figures(
    runs: dict,
    phase2_dir: Optional[Path],
    out_dir: Path,
    prompt: str,
    beta: float = 1.0,
) -> None:
    """
    For every (model, prompt) in `runs` that also has a matching Phase 2
    decompose run under phase2_dir, generate both energy-decomposition
    figures. Silent no-op for any model with no Phase 2 data, and a no-op
    entirely if phase2_dir wasn't found — this is additive and never blocks
    the rest of generate_all.
    """
    if phase2_dir is None or not Path(phase2_dir).exists():
        return
    models = sorted({m for (m, p) in runs.keys() if p == prompt})
    found_any = False
    for model in models:
        run_dir = runs.get((model, prompt))
        if run_dir is None:
            continue
        phase2_run_dir = _find_phase2_run_dir(phase2_dir, model, prompt)
        if phase2_run_dir is None:
            continue
        ok1 = plot_energy_decomposition_trajectory(run_dir, phase2_run_dir, out_dir, beta=beta)
        ok2 = plot_fiedler_vs_energy_attribution(run_dir, phase2_run_dir, out_dir, beta=beta)
        found_any = found_any or ok1 or ok2
    if not found_any:
        print(f"  ⚠  no Phase 2 decompose data found under {phase2_dir} for prompt {prompt!r}")

