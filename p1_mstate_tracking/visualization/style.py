"""
visualization/style.py

Module-wide defaults and plot styling. Nothing in here reads from disk —
pure constants, edited directly to change behavior without touching the
CLI or any plotting code.
"""

from typing import Dict, Union

import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Defaults — edit these to change behavior without touching the CLI
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PROMPT = "wiki_paragraph"     # the wiki article; used unless --all_prompts
DEFAULT_LAYERS = (5, 22, "final")     # early / mid / post-collapse depths
MIN_CLUSTER_SIZE = 4                  # matches the Phase 5 selection-gate size threshold

LayerSpec = Union[int, str]


# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────

MODEL_COLORS: Dict[str, str] = {
    "albert-base-v2":    "#2563EB",
    "albert-xlarge-v2":  "#7C3AED",
    "bert-base-uncased": "#059669",
    "gpt2-xl":           "#DC2626",
    "gpt2-large":        "#EA580C",
    "gpt2-medium":       "#D97706",
    "gpt2":              "#CA8A04",
}
UNTRAINED_COLOR = "#9CA3AF"

# Random-control lines default to UNTRAINED_COLOR (gray, dashed) so the
# whole "-random" family reads as one thing at a glance. Listed here only
# when a specific control needs to be told apart from another random
# control sharing the same overview chart.
RANDOM_COLOR_OVERRIDES: Dict[str, str] = {
    "gpt2-large-random": "#5C3A21",   # deep brown — distinct from
                                       # albert-base-v2-random's default gray
}

PLATEAU_COLOR   = "#FEF08A"
PLATEAU_TINT    = "#FEF9C3"   # light background tint for plateau panels
PLATEAU_BORDER  = "#CA8A04"
DEGENERATE_RANK = 2.0

# Interaction-energy colors, keyed by str(beta) so they line up with the
# keys energies.json actually stores. Falls back to a neutral gray for any
# beta not in BETA_VALUES (e.g. a future sweep adds one).
ENERGY_BETA_COLORS: Dict[str, str] = {
    "0.1": "#93C5FD",
    "1.0": "#2563EB",
    "2.0": "#7C3AED",
    "5.0": "#B91C1C",
}

# Full block / attention-only / FFN-only — all three drawn in the model's
# own color (see _color), distinguished by linestyle + marker rather than
# a separate color scheme, since the comparison is "same model, different
# residual stream" rather than "different models."
ABLATION_STYLE: Dict[str, dict] = {
    "full": dict(linestyle="-",  marker=None, linewidth=2.4, alpha=0.95),
    "attn": dict(linestyle="--", marker="^",  linewidth=2.0, alpha=0.9,
                 markersize=4, markevery=4),
    "ffn":  dict(linestyle=":",  marker="s",  linewidth=2.0, alpha=0.9,
                 markersize=4, markevery=4),
}
ABLATION_LABELS: Dict[str, str] = {
    "full": "full residual stream",
    "attn": "attention-only (post-attn stream)",
    "ffn":  "MLP-only (post-FFN stream)",
}

# Categorical palette for HDBSCAN cluster coloring — 20 distinct hues so
# clusters don't repeat colors as readily as a smaller palette would.
# Module level so every figure uses identical colors for the same cluster id.
CLUSTER_PAL = plt.cm.tab20(np.linspace(0, 1, 20))
NOISE_COLOR = "#D1D5DB"
ADDED_COLOR   = "#16A34A"   # token joined the tracked cluster at this layer
REMOVED_COLOR = "#DC2626"   # token left the tracked cluster at this layer

BLOG_STYLE = {
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#E5E7EB",
    "grid.linewidth":    0.6,
    "grid.alpha":        0.7,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.labelsize":    12,
    "axes.titlesize":    12,
    "legend.fontsize":   9,
    "legend.framealpha": 0.92,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
}
