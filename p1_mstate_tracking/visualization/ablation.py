"""
visualization/ablation.py

Sublayer-ablation figures: full residual stream vs. post-attention-only
vs. post-FFN-only, same model/prompt, overlaid on four panels (mass-near-1,
cluster membership, effective rank, interaction energy at beta=1.0).
Driven by whichever base models have a saved @attn/@ffn run on disk
(see --sublayer in run_1.py); skipped for any model with neither stream.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from core.style import BLOG_STYLE, ABLATION_STYLE, ABLATION_LABELS
from core.naming import _is_untrained, _color, _random_color, _safe_model_name, _sublayer_groups
from .loaders import _energy_series
from .series import _mass_near_1_series, _cluster_membership_series, _effective_rank_series

def plot_sublayer_ablation(
    runs: dict, out_dir: Path, prompt: str,
    base_model: str, attn_model: Optional[str], ffn_model: Optional[str],
) -> None:
    """
    Full block vs. attention-only vs. MLP-only, same model/prompt,
    overlaid on four panels: mass-near-1, cluster membership, effective
    rank, interaction energy at β=1.0. The @attn/@ffn runs already get
    their own per-model folder like any other variant — nothing puts the
    three side by side. This is that comparison.

    Skips (returns without writing anything) if neither attn_model nor
    ffn_model is given, or if the full-block run for base_model is
    missing. Missing energy data is tolerated — that panel is just empty.
    """
    plt.rcParams.update(BLOG_STYLE)
    if attn_model is None and ffn_model is None:
        return

    variants = [("full", base_model)]
    if attn_model:
        variants.append(("attn", attn_model))
    if ffn_model:
        variants.append(("ffn", ffn_model))

    run_dirs: Dict[str, Path] = {}
    for kind, model in variants:
        rd = runs.get((model, prompt))
        if rd is None:
            print(f"  ⚠  sublayer_ablation: missing run for {model!r} @ {prompt!r}")
            continue
        run_dirs[kind] = rd
    if "full" not in run_dirs:
        return

    panels = [
        ("Mass near 1", _mass_near_1_series, "Fraction ⟨xᵢ,xⱼ⟩ > 0.9", (-0.02, 1.06)),
        ("Cluster membership", _cluster_membership_series,
         "Fraction in a real cluster\n(1 − noise)", (-0.02, 1.06)),
        ("Effective rank", _effective_rank_series, "Effective rank", None),
        ("Interaction energy (β=1.0)", lambda rd: _energy_series(rd, 1.0),
         "Interaction energy  E_β", None),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    color = _random_color(base_model) if _is_untrained(base_model) else _color(base_model)

    any_drawn = False
    for ax, (title, value_fn, ylabel, ylim) in zip(axes, panels):
        for kind, _ in variants:
            rd = run_dirs.get(kind)
            if rd is None:
                continue
            try:
                series = value_fn(rd)
            except Exception:
                series = None
            if not series:
                continue
            series = np.asarray(series, dtype=float)
            xs = np.linspace(0, 1, len(series))
            ax.plot(xs, series, color=color, label=ABLATION_LABELS[kind], **ABLATION_STYLE[kind])
            any_drawn = True
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlabel("Normalized layer depth")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10, fontweight="bold")

    if not any_drawn:
        print(f"  ⚠  sublayer_ablation: no plottable data for {base_model!r} @ {prompt!r}")
        plt.close(fig)
        return

    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(
        f"Full block vs. attention-only vs. MLP-only — {base_model} | {prompt}\n"
        f"solid = full residual stream, dashed/▲ = post-attention only, dotted/■ = post-FFN only",
        fontsize=12, fontweight="bold",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"sublayer_ablation_{_safe_model_name(base_model)}_{prompt}.png"
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {fname}")


def generate_ablation_figures(runs: dict, out_dir: Path, prompt: str) -> None:
    """Sublayer ablation figure for every base model with a saved @attn or @ffn run."""
    models = sorted({m for (m, p) in runs.keys() if p == prompt})
    groups = _sublayer_groups(models)
    if not groups:
        print(f"  ⚠  no sublayer (@attn/@ffn) runs found for prompt {prompt!r}")
        return
    for base_model, attn_model, ffn_model in groups:
        plot_sublayer_ablation(runs, out_dir, prompt, base_model, attn_model, ffn_model)

