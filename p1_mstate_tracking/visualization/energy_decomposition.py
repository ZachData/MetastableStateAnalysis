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
from core.metrics import ENERGY_VIOLATION_REL_TOL, interaction_energy
from core.dissipation import dissipation, dissipation_by_channel
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
        ok3 = plot_dissipation_attribution(run_dir, phase2_run_dir, out_dir, beta=beta)
        found_any = found_any or ok1 or ok2 or ok3
    if not found_any:
        print(f"  ⚠  no Phase 2 decompose data found under {phase2_dir} for prompt {prompt!r}")


# ---------------------------------------------------------------------------
# Exact first-order attribution (core.dissipation)
# ---------------------------------------------------------------------------
#
# `energy_by_component` above is a leave-one-in scheme: it adds each delta
# back on its own and calls whatever is left over a "cross term". That cross
# term is not small — `p2_eigenspectra/cross_term_analysis.py` exists because
# it dominates for ALBERT-xlarge on several prompts — and `attn_frac` /
# `ffn_frac` clip with `max(0.0, -delta)`, which is the same sign-destroying
# clip status-2b.md flags at `analysis_p2.py:153`.
#
# The dissipation identity has neither problem. dE = sum_i <G_i, v_i> to
# first order, the tangential projection is linear, and on a parallel-residual
# model dx = dx_attn + dx_ffn exactly — so the split is exactly additive with
# no cross term to attribute and no clipping to hide a sign. What the old
# scheme puts in "cross" is here the *second-order* residual, which is
# reported as its own quantity because it measures whether the continuum
# limit this project assumes holds at that layer.
#
# Both are kept, and they answer different questions rather than competing.
# energy_by_component gives each channel's share of the REALISED DROP, which
# is what its clip is for and is the right reading when the layer does drop.
# This one gives what each channel did to the energy, signed — which is the
# only one of the two that can answer "which channel caused this violation",
# since at a violation layer there is no realised drop to take a share of.


def _per_layer_dissipation(decomp: dict, beta: float) -> dict:
    """
    Per-layer exact first-order energy attribution for one run.

    Returns lists indexed by layer boundary, plus `exact` — False on any
    architecture where dx != dx_attn + dx_ffn, which is the guard that
    replaces a model-name branch.
    """
    n_layers = len(decomp["attn_deltas"])
    out = {
        "attn": [], "ffn": [], "total": [],
        "actual": [], "residual": [], "violation": [],
        "exact": True,
    }
    for i in range(n_layers):
        X = decomp["trajectory"][i]
        a, f = decomp["attn_deltas"][i], decomp["ffn_deltas"][i]

        ch = dissipation_by_channel(X, a, f, beta)
        # Sum in float64. The saved deltas are float32, and `a + f` at that
        # precision differs from the sum dissipation_by_channel forms
        # internally by ~1e-7 relative — enough to break the exact identity
        # between the two curves this figure draws against each other.
        full = dissipation(X, np.asarray(a, dtype=np.float64)
                              + np.asarray(f, dtype=np.float64), beta)

        out["attn"].append(ch["attn"])
        out["ffn"].append(ch["ffn"])
        out["total"].append(ch["total"])
        out["actual"].append(full["actual_delta_E"])
        out["residual"].append(full["residual"])
        out["exact"] = out["exact"] and ch["exact"]

        # The project's relative rule, not a local threshold — status-2b.md
        # known-issue 1 is three hardcoded copies of a different one.
        e_before = interaction_energy(X, beta)
        rise = full["actual_delta_E"]
        out["violation"].append(
            rise is not None and rise > ENERGY_VIOLATION_REL_TOL * abs(e_before)
        )
    return out


def plot_dissipation_attribution(
    run_dir: Path,
    phase2_run_dir: Path,
    out_dir: Path,
    beta: float = 1.0,
) -> bool:
    """
    Exact first-order energy attribution per layer.

      top    — signed dissipation, attention vs FFN, as paired bars. Below
               zero is the layer pushing energy down; ABOVE zero is a layer
               pushing uphill, and the sign is not clipped, so a channel
               that fights the other is visible as two large opposing bars
               rather than as one small total.
      bottom — the linearisation residual as a band around zero, against
               the measured Delta E_beta. Where the band is wide, the
               forward-Euler reading of a residual block is not valid at
               that layer and the top panel should be read with care.

    Energy-violation boundaries are hatched in both panels, so "which
    channel produced this violation" is answerable by looking.

    Returns False (silently) when the run has no saved deltas, matching
    this module's additive-never-blocking contract.
    """
    decomp = _phase2_decomposed(phase2_run_dir)
    if decomp is None:
        return False

    geo = _geo(run_dir)
    model = geo.get("model", Path(run_dir).name) if geo else Path(run_dir).name
    prompt = geo.get("prompt", "") if geo else ""

    d = _per_layer_dissipation(decomp, beta)
    layers = np.arange(len(d["attn"]))
    if len(layers) == 0:
        return False

    with plt.style.context(BLOG_STYLE):
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(11, 7), sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )

        w = 0.4
        ax_top.bar(layers - w / 2, d["attn"], width=w, label="attention", color="#3B7EA1")
        ax_top.bar(layers + w / 2, d["ffn"], width=w, label="FFN", color="#C4622D")
        ax_top.plot(layers, d["total"], "k.-", lw=1, ms=4, label="total (= attn + FFN, exactly)")
        ax_top.axhline(0.0, color="k", lw=0.8)
        ax_top.set_ylabel(r"first-order $\Delta E_\beta$")
        ax_top.legend(fontsize=8, ncol=3)

        exact_note = "" if d["exact"] else "  —  NOT exactly additive on this architecture"
        ax_top.set_title(
            f"Exact first-order energy attribution — {model}"
            f"{(' · ' + prompt) if prompt else ''}  (beta={beta}){exact_note}",
            fontsize=10,
        )

        actual = np.array([np.nan if a is None else a for a in d["actual"]], dtype=float)
        resid = np.array([np.nan if r is None else r for r in d["residual"]], dtype=float)
        ax_bot.plot(layers, actual, color="k", lw=1.2, label=r"measured $\Delta E_\beta$")
        ax_bot.fill_between(layers, d["total"], d["total"] + resid, alpha=0.35,
                            color="#7A7A7A", label="second-order residual")
        ax_bot.axhline(0.0, color="k", lw=0.8)
        ax_bot.set_ylabel(r"$\Delta E_\beta$")
        ax_bot.set_xlabel("layer")
        ax_bot.legend(fontsize=8)

        for ax in (ax_top, ax_bot):
            for i, viol in enumerate(d["violation"]):
                if viol:
                    ax.axvspan(i - 0.5, i + 0.5, color="#B03A2E",
                               alpha=0.12, zorder=0, lw=0)

        fig.tight_layout()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = f"dissipation_attribution_{_safe_model_name(model)}"
        if prompt:
            name += f"_{prompt}"
        fig.savefig(out_dir / f"{name}.png", dpi=150)
        plt.close(fig)
    return True
