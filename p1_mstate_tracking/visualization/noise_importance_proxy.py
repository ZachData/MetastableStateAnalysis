"""
visualization/noise_importance_proxy.py

Tests a specific reframing of "HDBSCAN noise": instead of assuming
unclustered tokens are unimportant, check whether they carry MORE of two
cheap importance proxies than clustered tokens do —

  attention received : how much other tokens attend back to this token
                        (sum over heads/queries, excluding self), as a
                        rough proxy for "other computations depend on this
                        token's content."
  content-word status : the inverse of _is_punct_token (already used in
                        cluster_reality.py to filter punctuation-dominated
                        clusters out of the top-k ranking) — punctuation /
                        subword-marker tokens are cheap to discard
                        semantically; content words usually aren't.

Neither proxy is causal. Both are available for free from saved artifacts
(no model reload). The intended use is a fast first pass before the
expensive version: forcibly collapsing noise tokens onto their nearest
cluster centroid mid-stack and reading off the effect on next-token loss
relative to doing the same to already-clustered tokens (see the
collapse_token_onto_centroid sketch added to causal_tests.py — that is
the actual test of "does resisting collapse protect something the
network needs," this module only checks whether the geometry is
*consistent* with that story).

Run this once on a trained run_dir and once on the matched -random
run_dir for the same model/prompt (already produced by the standard
pipeline) and compare panels directly — if the clustered/noise gap in
either proxy is present under random weights at a similar magnitude, that
proxy isn't evidence of a *learned* protective mechanism, since the
architecture-determined / random baseline already does Phase 1's
trained-vs-random comparisons for everything else.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from .style import BLOG_STYLE
from .naming import _safe_model_name
from .loaders import _geo, _hdbscan_labels
from .cluster_reality import _is_punct_token


def _load_attentions(run_dir: Path) -> Optional[np.ndarray]:
    """(n_layers, n_heads, n_tokens, n_tokens) or None. Not in loaders.py
    because nothing else in the package currently needs raw attention."""
    p = run_dir / "attentions.npz"
    if not p.exists():
        return None
    data = np.load(p)
    key = "attentions" if "attentions" in data.files else data.files[0]
    return data[key]


def _received_attention(attn_layer: np.ndarray) -> np.ndarray:
    """
    Sum of attention paid TO each token, excluding self-attention,
    summed over heads and over queries.

    attn_layer : (n_heads, n_tokens, n_tokens), attn[h, query, key]
    Returns    : (n_tokens,) received[key] = sum_h sum_query attn[h, query, key]
    """
    a = attn_layer.copy()
    n = a.shape[-1]
    for h in range(a.shape[0]):
        np.fill_diagonal(a[h], 0.0)
    return a.sum(axis=(0, 1))


def plot_noise_importance_proxies(run_dir: Path, out_dir: Path) -> None:
    plt.rcParams.update(BLOG_STYLE)
    geo = _geo(run_dir)
    model, prompt, n_layers = geo["model"], geo["prompt"], geo["n_layers"]
    tokens = geo.get("tokens", [])
    hdb = _hdbscan_labels(run_dir)
    attn = _load_attentions(run_dir)

    if not hdb or not tokens:
        print(f"  [skip] missing hdbscan labels / tokens for {model}/{prompt}")
        return

    is_punct = np.array([_is_punct_token(t) for t in tokens])
    has_attn = attn is not None

    rel_received_noise = np.full(n_layers, np.nan)
    rel_received_clustered = np.full(n_layers, np.nan)
    punct_frac_noise = np.full(n_layers, np.nan)
    punct_frac_clustered = np.full(n_layers, np.nan)
    noise_fraction = np.full(n_layers, np.nan)

    for layer in range(n_layers):
        labels = np.array(hdb.get(layer, []))
        if labels.size == 0:
            continue
        noise_mask = labels == -1
        clustered_mask = ~noise_mask
        noise_fraction[layer] = float(noise_mask.mean())

        if noise_mask.any():
            punct_frac_noise[layer] = float(is_punct[noise_mask].mean())
        if clustered_mask.any():
            punct_frac_clustered[layer] = float(is_punct[clustered_mask].mean())

        if has_attn and layer < attn.shape[0]:
            received = _received_attention(attn[layer])
            layer_mean = received.mean()
            if layer_mean > 1e-12:
                if noise_mask.any():
                    rel_received_noise[layer] = float(received[noise_mask].mean() / layer_mean)
                if clustered_mask.any():
                    rel_received_clustered[layer] = float(received[clustered_mask].mean() / layer_mean)

    x = np.arange(n_layers)
    n_panels = 3 if has_attn else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 3.4 * n_panels), sharex=True)
    axes = np.atleast_1d(axes)

    panel = 0
    if has_attn:
        ax = axes[panel]
        ax.plot(x, rel_received_clustered, color="#2563EB", linewidth=1.8, label="clustered tokens")
        ax.plot(x, rel_received_noise, color="#DC2626", linewidth=1.8, label="noise (unclustered) tokens")
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.6,
                   label="layer-average (parity)")
        ax.set_ylabel("attention received\n(× layer average)")
        ax.set_title(
            "If noise tokens sit above parity and clustered tokens sit below it, "
            "the unclustered population is what others' computations depend on — "
            "the opposite of 'discardable'",
            fontsize=9,
        )
        ax.legend(fontsize=7, loc="best")
        panel += 1

    ax = axes[panel]
    ax.plot(x, punct_frac_clustered, color="#2563EB", linewidth=1.8, label="clustered tokens")
    ax.plot(x, punct_frac_noise, color="#DC2626", linewidth=1.8, label="noise (unclustered) tokens")
    ax.set_ylabel("fraction punctuation /\nsubword-marker tokens")
    ax.set_title(
        "If clustering is where cheap-to-discard content goes, clustered should "
        "skew punctuation/function-token relative to noise",
        fontsize=9,
    )
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="best")
    panel += 1

    ax = axes[panel]
    ax.plot(x, noise_fraction, color="#4B5563", linewidth=1.8)
    ax.set_ylabel("noise fraction")
    ax.set_xlabel("Layer")
    ax.set_title("Context: how much of the population is unclustered at each layer", fontsize=9)

    fig.suptitle(
        f"Is HDBSCAN noise 'unimportant' or 'protected'? — {model} | {prompt}\n"
        f"correlational only — run on the matched -random model to check whether "
        f"any gap here predates training",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"noise_importance_{_safe_model_name(model)}_{prompt}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}  (attention available: {has_attn})")
