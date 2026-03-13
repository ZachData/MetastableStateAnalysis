"""
figure1_sanity.py
=================
Reproduces Figure 1 from Geshkovski et al. (2024)
"A Mathematical Perspective on Transformers"

Top panel  : pairwise inner product histograms at layers 0, 1, 21, 22, 23, 24
             using the standard 24 albert-xlarge-v2 passes.
Bottom panel: extended run (48 passes of the shared layer) at layers
             25, 26, 27, 46, 47, 48.

Usage
-----
    python figure1_sanity.py                     # standard 24 layers only
    python figure1_sanity.py --extended          # also run 48-layer extended
    python figure1_sanity.py --prompt "custom text here"
    python figure1_sanity.py --model albert-base-v2   # sanity-check smaller model

Output
------
    figure1_reproduction.png     (top panel only, or both panels stacked)
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AlbertModel, AlbertTokenizer

# ---------------------------------------------------------------------------
# Defaults matching the paper
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "albert-xlarge-v2"

WIKIPEDIA_PROMPT = (
    "The transformer architecture was introduced in 2017 and has since become "
    "the dominant paradigm in natural language processing. Self-attention allows "
    "each token to attend to every other token in the sequence, enabling the "
    "model to capture long-range dependencies that recurrent architectures "
    "struggle with. Residual connections and layer normalisation ensure stable "
    "training even at considerable depth, while positional encodings provide "
    "the model with information about the order of tokens in the input sequence."
)

# Layers shown in Figure 1 top row
TOP_LAYERS = [0, 1, 21, 22, 23, 24]

# Layers shown in Figure 1 bottom row (extended 48-pass run)
BOTTOM_LAYERS = [25, 26, 27, 46, 47, 48]

HIST_BINS   = 80
HIST_RANGE  = (-0.5, 1.05)
DENSITY     = True   # paper shows density-normalised histograms


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: str):
    print(f"Loading {model_name} ...")
    tokenizer = AlbertTokenizer.from_pretrained(model_name)
    model = AlbertModel.from_pretrained(
        model_name,
        output_hidden_states=True,
        output_attentions=False,
    ).to(device)
    model.eval()
    print(f"  hidden_size = {model.config.hidden_size}")
    print(f"  num_hidden_layers = {model.config.num_hidden_layers}")
    return model, tokenizer


def l2_normalise(x: torch.Tensor) -> torch.Tensor:
    """Project onto unit sphere — matches RMS layer-norm of the paper."""
    return F.normalize(x, p=2, dim=-1)


def pairwise_inner_products(normed: torch.Tensor) -> np.ndarray:
    """
    normed : (n_tokens, d_model)  L2-normalised
    returns: upper-triangle of gram matrix, shape (n*(n-1)/2,)
    """
    gram = (normed @ normed.T).cpu().numpy()
    n = gram.shape[0]
    idx = np.triu_indices(n, k=1)
    return gram[idx]


# ---------------------------------------------------------------------------
# Standard forward pass: collect all hidden states
# ---------------------------------------------------------------------------

def get_hidden_states(model, tokenizer, text: str, device: str):
    """
    Returns list of (n_tokens, d_model) L2-normed tensors, one per layer
    (including embedding layer 0).
    """
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to(device)

    n_tok = inputs["input_ids"].shape[1]
    print(f"  Prompt tokenised to {n_tok} tokens.")

    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.hidden_states: tuple of (1, n_tokens, d_model), one per layer
    hidden = [l2_normalise(h[0].cpu().float()) for h in outputs.hidden_states]
    print(f"  Collected {len(hidden)} hidden states (layers 0–{len(hidden)-1}).")
    return hidden


# ---------------------------------------------------------------------------
# Extended run: loop the shared ALBERT layer for n_iterations passes
# ---------------------------------------------------------------------------

def get_extended_hidden_states(model, tokenizer, text: str,
                                n_iterations: int, device: str):
    """
    Runs the single shared ALBERT transformer block for n_iterations passes
    and returns ALL intermediate states (0 through n_iterations).
    """
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to(device)

    with torch.no_grad():
        embedding_output = model.embeddings(
            input_ids=inputs["input_ids"],
            token_type_ids=inputs.get("token_type_ids"),
        )
        # ALBERT XLarge uses factored embeddings: embedding_size=128, hidden_size=2048.
        # The encoder applies embedding_hidden_mapping_in to project up before the
        # first transformer layer. We must do the same here or the matmul will fail.
        hidden = model.encoder.embedding_hidden_mapping_in(embedding_output)

        attention_mask = model.get_extended_attention_mask(
            inputs["attention_mask"], inputs["input_ids"].shape
        )
        albert_layer = model.encoder.albert_layer_groups[0].albert_layers[0]

        # Layer 0 = post-projection (equivalent to the first hidden state the
        # transformer block actually sees), matching what hidden_states[1] contains
        # in the standard forward pass.
        trajectory = [l2_normalise(hidden[0].cpu().float())]
        for i in range(n_iterations):
            out = albert_layer(hidden, attention_mask=attention_mask)
            hidden = out[0]
            trajectory.append(l2_normalise(hidden[0].cpu().float()))
            if (i + 1) % 10 == 0:
                print(f"    Extended pass {i+1}/{n_iterations}")

    print(f"  Extended trajectory: {len(trajectory)} states.")
    return trajectory


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_row(axs, hidden_states, layer_indices, row_label=None):
    """
    Plot one row of histograms on the given axes array.
    """
    for ax, layer_idx in zip(axs, layer_indices):
        if layer_idx >= len(hidden_states):
            ax.set_visible(False)
            continue

        ips = pairwise_inner_products(hidden_states[layer_idx])
        mass_near_1 = float((ips > 0.9).mean())

        ax.hist(
            ips,
            bins=HIST_BINS,
            range=HIST_RANGE,
            density=DENSITY,
            color="#2171b5",
            edgecolor="none",
            alpha=0.85,
        )
        ax.set_xlim(HIST_RANGE)
        ax.set_ylim(0, None)
        ax.set_title(f"Layer {layer_idx}", fontsize=9, pad=3)
        ax.tick_params(labelsize=7)

        # Annotate mass near 1 in corner
        ax.text(
            0.97, 0.93,
            f"m>0.9: {mass_near_1:.3f}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=6.5,
            color="#cc0000",
        )

    if row_label:
        axs[0].set_ylabel(row_label, fontsize=8)


def make_figure(top_states, bottom_states=None,
                top_layers=TOP_LAYERS, bottom_layers=BOTTOM_LAYERS,
                model_name=DEFAULT_MODEL, prompt_label="wiki_paragraph",
                save_path="figure1_reproduction.png"):

    n_rows = 2 if bottom_states is not None else 1
    fig, axes = plt.subplots(
        n_rows, 6,
        figsize=(13, 3.2 * n_rows),
        constrained_layout=True,
    )

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    plot_row(axes[0], top_states, top_layers,
             row_label="Standard (24 layers)")

    if bottom_states is not None:
        plot_row(axes[1], bottom_states, bottom_layers,
                 row_label="Extended (48 layers)")

    fig.suptitle(
        f"Pairwise inner products  ⟨xᵢ(t), xⱼ(t)⟩  —  {model_name}  |  {prompt_label}\n"
        "Reproducing Figure 1 of Geshkovski et al. (2024)",
        fontsize=10,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Diagnostics: print mass-near-1 across all layers
# ---------------------------------------------------------------------------

def print_mass_table(hidden_states, label="standard"):
    print(f"\nMass > 0.9 per layer ({label}):")
    print(f"  {'Layer':>6}  {'n_pairs':>8}  {'mass>0.9':>9}  {'ip_mean':>8}  {'ip_max':>7}")
    print("  " + "-" * 50)
    for i, h in enumerate(hidden_states):
        ips = pairwise_inner_products(h)
        print(
            f"  {i:>6}  {len(ips):>8}  {(ips > 0.9).mean():>9.4f}  "
            f"{ips.mean():>8.4f}  {ips.max():>7.4f}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce Figure 1 of Geshkovski et al.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="HuggingFace model name (default: albert-xlarge-v2)")
    parser.add_argument("--prompt", default=WIKIPEDIA_PROMPT,
                        help="Input text (default: Wikipedia-style paragraph)")
    parser.add_argument("--no-extended", action="store_true",
                        help="Skip the 48-iteration extended pass for bottom panel")
    parser.add_argument("--extended-iters", type=int, default=48,
                        help="Number of iterations for extended run (default: 48)")
    parser.add_argument("--out", default="figure1_reproduction.png",
                        help="Output filename")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, tokenizer = load_model(args.model, device)

    print("\n--- Standard forward pass ---")
    top_states = get_hidden_states(model, tokenizer, args.prompt, device)
    print_mass_table(top_states, "standard")

    bottom_states = None
    if not args.no_extended:
        print(f"\n--- Extended {args.extended_iters}-iteration pass ---")
        bottom_states = get_extended_hidden_states(
            model, tokenizer, args.prompt,
            n_iterations=args.extended_iters,
            device=device,
        )
        print_mass_table(bottom_states, f"extended-{args.extended_iters}")

    print("\n--- Generating figure ---")
    make_figure(
        top_states=top_states,
        bottom_states=bottom_states,
        model_name=args.model,
        save_path=args.out,
    )