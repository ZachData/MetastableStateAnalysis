"""
config.py — Global constants, prompt variants, and model registry.

All tuneable parameters live here. Nothing else imports from this module;
everything else imports FROM it.
"""

import numpy as np
from pathlib import Path

import torch
from transformers import (
    AlbertModel, AlbertTokenizer,
    BertModel, BertTokenizer,
    GPT2Model, GPT2Tokenizer,
)

# ---------------------------------------------------------------------------
# Paths & device
# ---------------------------------------------------------------------------

BASE_RESULTS_DIR = Path("metastability_results")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Numerical parameters
# ---------------------------------------------------------------------------

BETA_VALUES           = [1.0, 2.0, 5.0]
DISTANCE_THRESHOLDS   = np.linspace(0.05, 0.6, 12)
K_RANGE               = range(2, 10)
ALBERT_EXTRA_ITERATIONS = [48]

SINKHORN_MAX_ITER = 100
SINKHORN_TOL      = 1e-6
SPECTRAL_MAX_K    = 10

# ---------------------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------------------

PROMPTS = {
    "wiki_paragraph": (
        "The transformer architecture was introduced in 2017 and has since become "
        "the dominant paradigm in natural language processing. Self-attention allows "
        "each token to attend to every other token in the sequence, enabling the "
        "model to capture long-range dependencies that recurrent architectures struggle with."
    ),
    "short_homogeneous": (
        "The cat sat on the mat and looked at the rat."
    ),
    "short_heterogeneous": (
        "Quantum mechanics governs the behavior of subatomic particles. "
        "Meanwhile, the stock market closed higher on Friday."
    ),
    "long_structured": (
        "Although the researchers had initially hypothesized that the model would "
        "fail to generalize beyond its training distribution, the experimental results "
        "demonstrated a surprising degree of robustness, even when the input prompts "
        "were systematically perturbed in ways that humans found trivially easy to handle."
    ),
    "repeated_tokens": (
        "The cat chased the cat because the cat was hungry and the cat wanted food "
        "and the cat finally caught the cat near the fence."
    ),
    "minimal": (
        "Attention is all you need."
    ),
}

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "albert-base-v2": {
        "model_class":     AlbertModel,
        "tokenizer_class": AlbertTokenizer,
        "is_albert":       True,
    },
    "albert-xlarge-v2": {
        "model_class":     AlbertModel,
        "tokenizer_class": AlbertTokenizer,
        "is_albert":       True,
    },
    "bert-base-uncased": {
        "model_class":     BertModel,
        "tokenizer_class": BertTokenizer,
        "is_albert":       False,
    },
    "gpt2": {
        "model_class":     GPT2Model,
        "tokenizer_class": GPT2Tokenizer,
        "is_albert":       False,
    },
}
