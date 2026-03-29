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

BASE_RESULTS_DIR = Path("results")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Numerical parameters
# ---------------------------------------------------------------------------

BETA_VALUES       = [0.1, 1.0, 2.0, 5.0]

# Previously np.linspace(0.05, 0.6, 5, 12) — the 4th positional arg is
# `endpoint` (expects bool), so 12 was coerced to True, producing only 5
# thresholds instead of the likely-intended 12.  Fixed below.
DISTANCE_THRESHOLDS = np.linspace(0.05, 0.6, 12)

K_RANGE           = range(2, 10)
# Run ALBERT once to ALBERT_MAX_ITERATIONS and take snapshots at each depth.
# Because ALBERT shares weights, hidden[i] is identical whether the run
# stops at i or continues to MAX — so a single pass captures every depth.
ALBERT_MAX_ITERATIONS = 60             # single run length (covers full sweep)
ALBERT_SNAPSHOTS      = list(range(6, 62, 2))  # P1-6: dense sweep for phase transition detection
# Legacy subset for quick runs (--fast-albert or manual override)
ALBERT_SNAPSHOTS_LEGACY = [12, 24, 36, 48]

SINKHORN_MAX_ITER = 100
SINKHORN_TOL      = 1e-6
SPECTRAL_MAX_K    = 15

# ---------------------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------------------

PROMPTS = {
    "short_heterogeneous": (
        "Quantum mechanics governs the behavior of subatomic particles. "
        "Meanwhile, the stock market closed higher on Friday."
    ),
    "wiki_paragraph": (
        "Charlotte Nicholls (née Brontë; 21 April 1816 – 31 March 1855), commonly known by her maiden "
        "name Charlotte Brontë, was an English novelist and poet, and was the elder sister of Emily, "
        "Anne and Branwell Brontë. She is best known for her novel Jane Eyre, which was first published "
        "under the pseudonym Currer Bell. Jane Eyre was a great success on publication, and has since "
        "become known as a classic of English literature. Charlotte was the third of six siblings born "
        "to Maria Branwell and Patrick Brontë. Maria died when Charlotte was only five years old, and "
        "three years later, Charlotte was sent to the Clergy Daughters' School at Cowan Bridge in "
        "Lancashire, along with her three sisters, Maria, Elizabeth and Emily. Conditions at the school "
        "were appalling, with frequent outbreaks of disease. Charlotte's two elder sisters fell ill there "
        "and died shortly afterwards; Charlotte attributed her own lifelong ill-health to her time at "
        "Cowan Bridge, and later used it as the model for Lowood School in Jane Eyre. In 1831, Charlotte "
        "became a pupil at Roe Head School in Mirfield, but left the following year to teach her sisters, "
        "Emily and Anne, at home. In 1835, Charlotte returned to Roe Head as a teacher. In 1839, she "
        "accepted a job as governess to a local family, but left after a few months. In 1842, Charlotte "
        "joined the Heger Pensionnat, a girls' boarding school in Brussels, as a student, then later as "
        "a teacher, in the hope of acquiring the skills required to open a school of her own. However, "
        "she was obliged to leave after falling in love with the school's director, Constantin Heger, a "
        "married man, who inspired both the character of Rochester in Jane Eyre, and Charlotte's first "
        "novel, The Professor. Charlotte, Emily and Anne attempted to open a school in Haworth, but "
        "failed to attract pupils. In 1846 the sisters published a collection of poems under the "
        "pseudonyms Currer, Ellis, and Acton Bell. Although Charlotte's first novel, The Professor, was "
        "rejected by publishers, her second novel, Jane Eyre, was published in 1847, attracting both "
        "praise and controversy."
    ),
    "repeated_tokens": (
        "cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat "
        "cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat "
        "cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat "
        "cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat "
        "cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat "
        "cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat "
        "cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat "
        "cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat "
        "cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat "
        "cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat "
        "cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat "
        "cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat"
    ),
    "sullivan_ballou": (
        "My Very Dear Wife: Indications are very strong that we shall move in a few days, perhaps "
        "to-morrow. Lest I should not be able to write you again, I feel impelled to write a few "
        "lines, that may fall under your eye when I shall be no more. Our movement may be one of a "
        "few days duration and full of pleasure and it may be one of severe conflict and death to me. "
        "Not my will, but thine, O God be done. If it is necessary that I should fall on the "
        "battle-field for any country, I am ready. I have no misgivings about, or lack of confidence "
        "in, the cause in which I am engaged, and my courage does not halt or falter. I know how "
        "strongly American civilization now leans upon the triumph of government, and how great a debt "
        "we owe to those who went before us through the blood and suffering of the Revolution, and I "
        "am willing, perfectly willing to lay down all my joys in this life to help maintain this "
        "government, and to pay that debt. But, my dear wife, when I know, that with my own joys, I "
        "lay down nearly all of yours, and replace them in this life with care and sorrows, when, after "
        "having eaten for long years the bitter fruit of orphanage myself, I must offer it, as their "
        "only sustenance, to my dear little children, is it weak or dishonorable, while the banner of "
        "my purpose floats calmly and proudly in the breeze, that my unbounded love for you, my "
        "darling wife and children, should struggle in fierce, though useless, contest with my love of "
        "country. I cannot describe to you my feelings on this calm summer night, when two thousand "
        "men are sleeping around me, many of them enjoying the last, perhaps, before that of death, "
        "and I, suspicious that Death is creeping behind me with his fatal dart, am communing with "
        "God, my country and thee. I have sought most closely and diligently, and often in my breast, "
        "for a wrong motive in this hazarding the happiness of those I loved, and I could not find "
        "one. A pure love of my country, and of the principles I have often advocated before the "
        "people, and the name of honor, that I love more than I fear death, have called upon me, "
        "and I have obeyed."
    ),
    "paper_excerpt": (
        "An important aspect of Transformers is that they are not hard-wired to take into account "
        "the order of the input sequence, contrary to other architectures used for natural language "
        "processing such as recurrent neural networks. In these applications, each token contains "
        "not only a word embedding, but also an additional positional encoding which allows tokens "
        "to also carry their position in the input sequence. Therefore, an input sequence is "
        "perfectly encoded as a set of tokens, or equivalently as the empirical measure of its "
        "constituent tokens. Recall that the output of a Transformer is also a probability measure, "
        "albeit one that captures the likelihood of the next token. As a result, one can view "
        "Transformers as flow maps between probability measures on the sphere. To describe this "
        "flow map, we appeal to the continuity equation, which governs precisely the evolution of "
        "the empirical measure of particles subject to dynamics. This perspective is already present "
        "in prior work, the only modification here being that we add the projection on the sphere "
        "arising from layer normalization. After introducing the continuity equation, we show that "
        "a particular interaction energy functional, which is maximized at any point mass, increases "
        "along solutions thereof. Motivated by this monotonicity property, we propose an illustrative "
        "modified model which has the nice property of being a Wasserstein gradient flow for this "
        "energy. Finally, we demonstrate that the original equation is itself a gradient flow for "
        "the same energy, upon changing the metric underlying the definition of the gradient."
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
    "bert-large-uncased": {
        "model_class":     BertModel,
        "tokenizer_class": BertTokenizer,
        "is_albert":       False,
    },
    "gpt2": {
        "model_class":     GPT2Model,
        "tokenizer_class": GPT2Tokenizer,
        "is_albert":       False,
    },
    "gpt2-medium": {
        "model_class":     GPT2Model,
        "tokenizer_class": GPT2Tokenizer,
        "is_albert":       False,
    },
    "gpt2-large": {
        "model_class":     GPT2Model,
        "tokenizer_class": GPT2Tokenizer,
        "is_albert":       False,
    },
    "gpt2-xl": {
        "model_class":     GPT2Model,
        "tokenizer_class": GPT2Tokenizer,
        "is_albert":       False,
    },
}
