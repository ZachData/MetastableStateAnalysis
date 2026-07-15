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

# Default seed for random-initialisation controls.  Overridden at runtime by
# the --seed CLI argument to run_1 / run_2.  Changing this constant alone is
# NOT sufficient if Phase 1 runs were produced with a different seed — Phase 2
# must be given the matching seed for the OV decomposition to correspond to
# the activations that were actually recorded.
RANDOM_INIT_SEED: int = 0

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
# Single degeneracy gate threshold used by CKA, NN-stability, and energy-drop
# suppression in analysis.py and reporting.py.
# Previously split: CKA used < 3.0, NN used < 2.0.  Unified at 2.
# Below this the token cloud is a near-point-mass on the sphere (rank ≈ 1),
# making NN assignment float-noise and CKA centering noise-dominated.
# At rank 2 the cloud is still 2-D and both metrics remain meaningful.
# Raise to 3 via this single constant if post-rerun rank-2 CKA looks erratic.
DEGENERATE_RANK_THRESHOLD = 2

# Token-count sweep targets for --length-sweep mode.
# wiki_paragraph is truncated at word boundaries to each of these approximate
# token counts and run as separate prompts.  Tests whether plateau width scales
# with n_tokens as the paper's theory predicts.
LENGTH_SWEEP_TOKENS = [50, 100, 150, 200, 300, 400]

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
        ". . . . . . . . . . . . . . . . . . . . . . . . "
        ". . . . . . . . . . . . . . . . . . . . . . . . "
        ". . . . . . . . . . . . . . . . . . . . . . . . "
        ". . . . . . . . . . . . . . . . . . . . . . . . "
        ". . . . . . . . . . . . . . . . . . . . . . . . "
        ". . . . . . . . . . . . . . . . . . . . . . . . "
        ". . . . . . . . . . . . . . . . . . . . . . . . "
        ". . . . . . . . . . . . . . . . . . . . . . . . "
        ". . . . . . . . . . . . . . . . . . . . . . . . "
        ". . . . . . . . . . . . . . . . . . . . . . . . "
        ". . . . . . . . . . . . . . . . . . . . . . . . "
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
    "homer_iliad": (
        "But then, when the tenth night came on me, black as pitch, I burst the doors of the chamber "
        "bolted tight and out I rushed, I leapt the walls at a bound, giving the slip to guards and "
        "women servants. And away I fled through the whole expanse of Hellas and gaining the good dark "
        "soil of Phthia, mother of flocks, I reached the king, and Peleus gave me a royal welcome. "
        "Peleus loved me as a father loves a son, I tell you, his only child, the heir to his boundless "
        "wealth, he made me a rich man, he gave me throngs of subjects, I ruled the Dolopes, settling "
        "down on Phthia's west frontier. And I made you what you are-strong as the gods, Achilles"
        "I loved you from the heart. You'd never go with another to banquet on the town or feast in your "
        "own halls. Never, until I'd sat you down on my knees and cut "
        "you the first bits of meat, remember? You'd eat your fill, I'd hold the cup to your lips and "
        "all too often you soaked the shirt on my chest, spitting up some wine, a baby's way ... a misery. "
        "Oh I had my share of troubles for you, Achilles, did my share of labor. Brooding, never forgetting "
        "the gods would bring no son of mine to birth, not from my own loins. So you, Achilles"
        "great godlike Achilles-I made you my son, I tried, so someday you might fight disaster off my back. "
        "But now, Achilles, beat down your mounting fury! It's wrong to have such an iron, ruthless "
        "heart. Even the gods themselves can bend and change, and theirs is the greater power, honor, "
        "strength. Even the gods, I say, with incense, soothing vows. with full cups poured and the deep "
        "smoky savor men can bring them round, begging for pardon when one oversteps the mark, does "
        "something wrong. We do have Prayers, you know, Prayers for forgiveness, daughters of mighty "
        "Zeus ... and they limp and halt, they're all wrinkled, drawn, they squint to the side, can't "
        "look you in the eyes, and always bent on duty. trudging after Ruin, maddening, blinding Ruin. "
        "But Ruin is strong and swift"
        "She outstrips them all by far, stealing a march, leaping over the whole wide earth to bring mankind "
        "to grief. And the Prayers trail after, trying to heal the wounds. And then, if a man reveres these "
        "daughters of Zeus as they draw near him, they will help him greatly and listen to his appeals. "
    ),
    "hdbscan_code": (
        "def get_plot_data(self, leaf_separation=1, log_size=False, max_rectangle_per_icicle=20):\n"
        "        \"\"\"Generates data for use in plotting the 'icicle plot' or dendrogram\n"
        "        plot of the condensed tree generated by HDBSCAN.\n\n"
        "        Parameters\n"
        "        ----------\n"
        "        leaf_separation : float, optional\n"
        "                          How far apart to space the final leaves of the\n"
        "                          dendrogram. (default 1)\n\n"
        "        log_size : boolean, optional\n"
        "                   Use log scale for the 'size' of clusters (i.e. number of\n"
        "                   points in the cluster at a given lambda value).\n"
        "                   (default False)\n\n"
        "        max_rectangles_per_icicle : int, optional\n"
        "            To simplify the plot this method will only emit\n"
        "            ``max_rectangles_per_icicle`` bars per branch of the dendrogram.\n"
        "            This ensures that we don't suffer from massive overplotting in\n"
        "            cases with a lot of data points.\n\n"
        "        Returns\n"
        "        -------\n"
        "        plot_data : dict\n"
    ),
    "camus_letranger": (
        "À part ces ennuis, je n'étais pas trop malheureux. Toute la question, encore une fois, était "
        "de tuer le temps. J'ai fini par ne plus m'ennuyer du tout à partir de l'instant où j'ai appris "
        "à me souvenir. Je me mettais quelquefois à penser à ma chambre et, en imagination, je partais "
        "d'un coin pour y revenir en dénombrant mentalement tout ce qui se trouvait sur mon chemin. "
        "Au début, c'était vite fait. Mais chaque fois que je recommençais, c'était un peu plus long. "
        "Car je me souvenais de chaque meuble, et, pour chacun d'entre eux, de chaque objet qui s'y "
        "trouvait et, pour chaque objet, de tous les détails et pour les détails eux-mêmes, une "
        "incrustation, une fêlure ou un bord ébréché, de leur couleur ou de leur grain. En même temps, "
        "j'essayais de ne pas perdre le fil de mon inventaire, de [113] faire une énumération complète. "
        "Si bien qu'au bout de quelques semaines, je pouvais passer des heures, rien qu'à dénombrer ce "
        "qui se trouvait dans ma chambre. Ainsi, plus je réfléchissais et plus de choses méconnues et "
        "oubliées je sortais de ma mémoire. J'ai compris alors qu'un homme qui n'aurait vécu qu'un seul "
        "jour pourrait sans peine vivre cent ans dans une prison. Il aurait assez de souvenirs pour ne "
        "pas s'ennuyer. Dans un sens, c'était un avantage. Il y avait aussi le sommeil. Au début, je "
        "dormais mal la nuit et pas du tout le jour. Peu à peu, mes nuits ont été meilleures et j'ai "
        "pu dormir aussi le jour."
    ),
    "latex_monograph": (
        "\\documentclass[11pt,a4paper]{\narticle}\n\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{amsmath,amssymb,amsfonts}\n\\usepackage{geometry}\n\\usepackage{xcolor}\n"
        "\\usepackage{titlesec}\n\\usepackage{microtype}\n\n% Define page geometry\n\\geometry{\n"
        "    a4paper,\n    total={165mm,247mm},\n    left=22mm,\n    top=25mm,\n}\n\n"
        "% Define custom corporate/academic color palette\n\\definecolor{primary}{RGB}{26, 54, 93}     "
        "% Deep slate blue\n\\definecolor{secondary}{RGB}{43, 108, 176} % Accent blue\n"
        "\\definecolor{textdark}{RGB}{45, 55, 72}    % Dark grey for text body\n\n\\makeatletter\n"
        "\\newcommand{\\globalcolor}[1]{%\n  \\color{#1}\\global\\let\\default@color\\current@color\n"
        "}\n\\makeatother\n\\AtBeginDocument{\\globalcolor{textdark}}\n\n% Section styling\n"
        "\\titleformat{\\section}\n  {\\color{primary}\\normalfont\\Large\\bfseries}\n"
        "  {\\thesection}{1em}{}[{\\color{secondary}\\titrule[1pt]}]\n\n\\titleformat{\\subsection}\n"
        "  {\\color{secondary}\\normalfont\\large\\bfseries}\n  {\\thesubsection}{1em}{}\n\n"
        "% Custom styling for title\n\\title{\n    \\vspace{-1.5cm}\n    \\Huge \\textbf{\\color{primary}"
        "{The Principle of Least Action}} \\\\\n    \\large \\textit{\\color{secondary}{A Foundational "
        "Formulation of Classical Mechanics}}\n}\n\\author{\\textbf{Expository Physics Monograph}}\n"
        "\\date{\\small \\today}\n\n\\begin{document}\n\n\\maketitle\n\n\\section{Introduction}\n"
        "The \\textbf{Principle of Least Action}---more accurately termed the \\textit{Principle of "
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
        "random_init":     False,
    },
    # Untrained control: same architecture as albert-base-v2 but with weights
    # randomly re-initialised after loading the architecture.  Used to test
    # whether metastability is a property of trained weights or just of the
    # iterated-map architecture.  Registered as a separate model key so it
    # runs through the full pipeline and produces side-by-side reports.
    "albert-base-v2-random": {
        "model_class":     AlbertModel,
        "tokenizer_class": AlbertTokenizer,
        "is_albert":       True,
        "random_init":     True,
    },
    "albert-xlarge-v2": {
        "model_class":     AlbertModel,
        "tokenizer_class": AlbertTokenizer,
        "is_albert":       True,
        "random_init":     False,
    },
    "bert-base-uncased": {
        "model_class":     BertModel,
        "tokenizer_class": BertTokenizer,
        "is_albert":       False,
        "random_init":     False,
    },
    "bert-large-uncased": {
        "model_class":     BertModel,
        "tokenizer_class": BertTokenizer,
        "is_albert":       False,
        "random_init":     False,
    },
    "gpt2": {
        "model_class":     GPT2Model,
        "tokenizer_class": GPT2Tokenizer,
        "is_albert":       False,
        "random_init":     False,
    },
    "gpt2-medium": {
        "model_class":     GPT2Model,
        "tokenizer_class": GPT2Tokenizer,
        "is_albert":       False,
        "random_init":     False,
    },
    "gpt2-large": {
        "model_class":     GPT2Model,
        "tokenizer_class": GPT2Tokenizer,
        "is_albert":       False,
        "random_init":     False,
    },
    # Untrained control: same architecture as gpt2-large but randomly
    # re-initialised after loading.  Mirrors the albert-base-v2-random entry.
    # Referenced by run_1 --random-baseline and run_2 --random-dir discovery.
    "gpt2-large-random": {
        "model_class":     GPT2Model,
        "tokenizer_class": GPT2Tokenizer,
        "is_albert":       False,
        "random_init":     True,
    },
    "gpt2-xl": {
        "model_class":     GPT2Model,
        "tokenizer_class": GPT2Tokenizer,
        "is_albert":       False,
        "random_init":     False,
    },
}

from core.pythia_registry import build_pythia_model_configs
MODEL_CONFIGS.update(build_pythia_model_configs())