"""
core/config.py — Global constants, prompt variants, and model registry.

All tuneable parameters live here. Nothing else imports from this module;
everything else imports FROM it.
"""

import numpy as np
import os
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

#: Where every phase writes its run directories. Overridable because the
#: artifacts are large — a 12-checkpoint Pythia-410M sweep is ~77 GB of
#: Phase 1 + Phase 2 output in float64 — and the repository's own volume is
#: not always the one with room. Default unchanged, so nothing that does not
#: set it moves.
BASE_RESULTS_DIR = Path(os.environ.get("METS_RESULTS_DIR", "results"))

# Default seed for random-initialisation controls.  Overridden at runtime by
# the --seed CLI argument to run_1 / run_2.  Changing this constant alone is
# NOT sufficient if Phase 1 runs were produced with a different seed — Phase 2
# must be given the matching seed for the OV decomposition to correspond to
# the activations that were actually recorded.
RANDOM_INIT_SEED: int = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------

# dtype every model is loaded in. See core/models.py's module docstring for
# the full argument; the short version:
#
#   Pythia checkpoints are stored fp16 on the Hub. float32 is an exact
#   upcast; bfloat16 is a lossy re-quantisation (fp16 has 10 mantissa bits,
#   bf16 has 7). The V eigenspectrum is eigvals() of a non-normal matrix,
#   so eig_frac_pos_real / eig_frac_neg_real / eig_spectral_radius degrade
#   with input precision in a way singular values do not.
#
#   These quantities were previously described as carrying status-1's
#   "Thm 6.1" falsification. That was a mis-citation twice over: Thm 6.1 is
#   the qualitative statement that d>=3 suffices at any beta, and nothing
#   about V's spectrum bears on it. What the sign structure of V's
#   eigenvalues actually decides is the ATTRACTIVE/REPULSIVE regime
#   (§3.2 and §9.1: V = +I_d gives increasing E_beta, V = -I_d decreasing)
#   and which row of Table 1 (§9.2) a head falls under. Both are sign
#   determinations near zero, which is exactly where reduced precision
#   fails, so the guard stands — it just guards a different claim.
#
# "auto" restores the pre-fix behaviour (bfloat16 on CUDA, float32 on CPU).
# Whatever is set here is written to experiment.txt and to every
# v_eigenspectrum JSON, so a dtype change can never again be an invisible
# term in a cross-run comparison.
MODEL_DTYPE = "float32"

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

# Raised from 100. The n=20 causal baseline needs 232 iterations to reach
# SINKHORN_TOL; at iteration 100 the residual is 4.7e-4 and the cap was hit
# silently on every short-prompt run (status-1 D9). The lambda_2 error was
# negligible for the *uniform* baseline (0.108894 vs 0.108889 converged),
# but real attention is more peaked and converges more slowly, and nothing
# recorded a per-layer residual. sinkhorn_normalize* now return the residual
# and the iteration count alongside P; see p1_mstate_tracking/sinkhorn.py.
SINKHORN_MAX_ITER = 500
SINKHORN_TOL      = 1e-6
SPECTRAL_MAX_K    = 15
# Single degeneracy gate threshold used by CKA, NN-stability, and energy-drop
# suppression in analysis.py and reporting.py.
# Previously split: CKA used < 3.0, NN used < 2.0.  Unified at 2.
# Below this the token cloud is a near-point-mass on the sphere (rank ≈ 1),
# making NN assignment float-noise and CKA centering noise-dominated.
# At rank 2 the cloud is still 2-D and both metrics remain meaningful.
#
# WHICH RANK THIS READS — status-1 defect D10.
# The gate's own justification ("a near-point-mass ON THE SPHERE") is a
# statement about DIRECTIONAL collapse, so it must be evaluated on
# effective_rank_normed. It was previously evaluated on raw effective rank,
# which core/metrics.py's moment section shows is 1/<s^2>_w with
# norm-squared weights — in the near-orthogonal limit it degenerates to the
# participation ratio of the norm distribution alone, carrying zero
# directional content. Three attention sinks in two hundred tokens take raw
# rank from ~112 to ~3.4 with the geometry untouched.
#
# Because this is a GATE and not a reported column, reading raw rank made
# the set of layers entering every gated statistic — energy violations,
# CKA, NN-stability, the Fiedler mean — move with each checkpoint's sink
# structure. Late checkpoints on short_heterogeneous have raw MinRank
# 1.06–1.28 and were being gated off entirely on that basis.
#
# The threshold value below is stated on the NORMED scale and is not
# comparable to the old raw-scale 2. Normed effective rank on this sweep
# spans a different range; re-derive from the actual distribution before
# treating any specific value as calibrated. Set to 2 as a starting point
# because the geometric argument (a 2-D cloud is still meaningful for CKA
# and NN assignment) is scale-free, but this is the weakest-justified
# constant in the phase and is flagged as such.
DEGENERATE_RANK_THRESHOLD = 2

# Which key the degeneracy gates read. "normed" is frame-correct per the
# argument above. Set to "raw" only to reproduce pre-D10 numbers; every
# artifact records this value so a gate change can never be an invisible
# term in a cross-run comparison.
DEGENERATE_RANK_MODE = "normed"

# Layer-inclusion gate for the per-head Fiedler profile
# (reporting_p1._per_head_fiedler_profile, formerly a hardcoded default of
# 10.0 read against RAW rank). Same D10 argument: the docstring justifies
# the gate by directional collapse, so it reads the normed key. Stated on
# the normed scale; re-derive before trusting the value.
FIEDLER_ACTIVE_RANK_THRESHOLD = 10.0

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

    # ------------------------------------------------------------------
    # v2 extension, 2026-09-19. Twelve prompts added under the rule fixed
    # in core/prompts.py BEFORE any of this text was chosen: six genres
    # mirroring v1's, two each, one in v1's short band (1000-1600 chars)
    # and one in its long band (1800-2400), the first passage meeting the
    # band from a predetermined anchor, no candidate run through a model
    # before inclusion. Reason: CLAIM-C's gate stalled at an attainable p
    # of 0.0661 because four of eight rows tied 3-3 and a tied row cannot
    # move the statistic (PROJECT.md 3.41). v1's nine entries above are
    # untouched, so every v1 row is still the same row.
    #
    # Provenance: Wikipedia (CC BY-SA) for the encyclopedic pair; Project
    # Gutenberg public domain for the letters (Speeches & Letters of
    # Abraham Lincoln, ebook 14721) and the narrative pair (Don Quijote,
    # ebook 2000; Moby-Dick, ebook 2701); arXiv open access for the papers
    # (1706.03762, 2604.23740); BSD-licensed source already installed in
    # this venv for the code pair (scikit-learn, scipy); composed for the
    # LaTeX pair, as v1's latex_monograph is.
    # ------------------------------------------------------------------
    "wiki_photosynthesis": (
        "Photosynthesis is a system of biological processes by which photopigment-bearing autotrophic "
        "organisms, such as most plants, algae and cyanobacteria, convert light energy—typically from "
        "sunlight—into the chemical energy necessary to fuel their metabolism. The term photosynthesis "
        "usually refers to oxygenic photosynthesis, a process that releases oxygen as a byproduct of "
        "water splitting. Photosynthetic organisms store the converted chemical energy within the bonds "
        "of intracellular organic compounds (complex compounds containing carbon), typically "
        "carbohydrates like sugars (mainly glucose, fructose and sucrose), starches, phytoglycogen and "
        "cellulose. When needing to use this stored energy, an organism's cells then metabolize the "
        "organic compounds through cellular respiration. Photosynthesis plays a critical role in "
        "producing and maintaining the oxygen content of the Earth's atmosphere, and it supplies most of "
        "the biological energy necessary for complex life on Earth. Some organisms also perform "
        "anoxygenic photosynthesis, which does not produce oxygen."
    ),
    "wiki_byzantium": (
        "The Byzantine Empire, also known as the Eastern Roman Empire, was the continuation of the Roman "
        "Empire centred on Constantinople during late antiquity and the Middle Ages. Having survived the "
        "fall of the Western Roman Empire in the 5th century AD, it endured until the fall of "
        "Constantinople to the Ottoman Empire in 1453. The term 'Byzantine Empire' was coined only after "
        "its demise; its citizens used the term 'Roman Empire' and called themselves 'Romans'. During the "
        "early centuries of the Roman Empire, the western provinces were Latinised, but the eastern parts "
        "kept their Hellenistic culture. Constantine I (r. 324–337) legalised Christianity and moved the "
        "capital to Constantinople. Theodosius I (r. 379–395) made Christianity the state religion and "
        "Greek gradually replaced Latin for official use. The empire adopted a defensive strategy and, "
        "throughout its remaining history, experienced recurring cycles of decline and recovery. The "
        "Byzantine Empire reached its greatest extent under the reign of Justinian I (r. 527–565), who "
        "briefly reconquered much of Italy and the western Mediterranean coast. A plague began around "
        "541, and a prolonged warfare with Persia placed fiscal and military strain on the empire, "
        "contributing to political and strategic challenges in the decades that followed. In the 630s and "
        "640s the Arab conquests defeated Byzantine field armies in Syria and Egypt, resulting in the "
        "permanent loss of those provinces to the Rashidun Caliphate. In 698, Africa was lost to the "
        "Umayyad Caliphate, but the empire stabilised under the Isaurian dynasty. It expanded once more "
        "under the Macedonian dynasty, experiencing a two-century-long renaissance. Thereafter, periods "
        "of civil war and Seljuk incursion resulted in the loss of most of Asia Minor. The empire "
        "recovered during the Komnenian restoration, and Constantinople remained the largest and "
        "wealthiest city in Europe until the 13th century, when it was overtaken by Paris."
    ),
    "lincoln_letter_short": (
        "Dear Colonel, I am told that during my absence last week you passed through this place, and "
        "stated publicly that you were in possession of a fact or facts which, if known to the public, "
        "would entirely destroy the prospects of N.W. Edwards and myself at the ensuing election; but "
        "that, through favour to us, you should forbear to divulge them. No one has needed favours more "
        "than I, and, generally, few have been less unwilling to accept them; but in this case favour to "
        "me would be injustice to the public, and therefore I must beg your pardon for declining it. That "
        "I once had the confidence of the people of Sangamon, is sufficiently evident; and if I have "
        "since done anything, either by design or misadventure, which if known would subject me to a "
        "forfeiture of that confidence, he that knows of that thing, and conceals it, is a traitor to his "
        "country's interest. I find myself wholly unable to form any conjecture of what fact or facts, "
        "real or supposed, you spoke; but my opinion of your veracity will"
    ),
    "lincoln_letter_long": (
        "Dear Madam, Without apologising for being egotistical, I shall make the history of so much of my "
        "life as has elapsed since I saw you the subject of this letter. And, by the way, I now discover "
        "that in order to give a full and intelligible account of the things I have done and suffered "
        "since I saw you, I shall necessarily have to relate some that happened before. It was, then, in "
        "the autumn of 1836 that a married lady of my acquaintance, and who was a great friend of mine, "
        "being about to pay a visit to her father and other relatives residing in Kentucky, proposed to "
        "me that on her return she would bring a sister of hers with her on condition that I would engage "
        "to become her brother-in-law with all convenient dispatch. I, of course, accepted the proposal, "
        "for you know I could not have done otherwise had I really been averse to it; but privately, "
        "between you and me, I was most confoundedly well pleased with the project. I had seen the said "
        "sister some three years before, thought her intelligent and agreeable, and saw no good objection "
        "to plodding life through hand-in-hand with her. Time passed on, the lady took her journey, and "
        "in due time returned, sister in company, sure enough. This astonished me a little, for it "
        "appeared to me that her coming so readily showed that she was a trifle too willing, but on "
        "reflection it occurred to me that she might have been prevailed on by her married sister to "
        "come, without anything concerning me having been mentioned to her, and so I concluded that if no "
        "other objection presented itself, I would consent to waive this. All this occurred to me on "
        "hearing of her arrival in the neighbourhood--for, be it remembered, I had not yet seen her, "
        "except about three years previous, as above mentioned. In a few days we had an interview, and, "
        "although I had seen her before, she did not look"
    ),
    "paper_attention": (
        "Recurrent neural networks, long short-term memory [13] and gated recurrent [7] neural networks "
        "in particular, have been firmly established as state of the art approaches in sequence modeling "
        "and transduction problems such as language modeling and machine translation [35,2,5] . Numerous "
        "efforts have since continued to push the boundaries of recurrent language models and "
        "encoder-decoder architectures [38,24,15] . Recurrent models typically factor computation along "
        "the symbol positions of the input and output sequences. Aligning the positions to steps in "
        "computation time, they generate a sequence of hidden states h t h_{t} , as a function of the "
        "previous hidden state h t − 1 h_{t-1} and the input for position t t . This inherently "
        "sequential nature precludes parallelization within training examples, which becomes critical at "
        "longer sequence lengths, as memory constraints limit batching across examples. Recent work has "
        "achieved significant improvements in computational efficiency through factorization tricks [21] "
        "and conditional computation [32] , while also improving model performance in case of the latter."
    ),
    "paper_svflow": (
        "Despite the Transformer’s dominance across machine learning, its architecture remains largely "
        "heuristic and lacks a unified theoretical foundation. We introduce Score-based Variational Flow "
        "(SVFlow), a continuous-time dynamical system for representation learning in which the state "
        "evolves according to a variational posterior–weighted average of conditional log-likelihood "
        "scores, and provide a principled basis for regularization through variational consistency. We "
        "show that forward Euler discretization of spherical SVFlow exactly recovers the Transformer "
        "architecture. Multi-head attention approximates SVFlow vector field via a vMF kernel-smoothed "
        "posterior, while MoE/FFN approximates it in a relaxed network-based way, and the "
        "residual-normalization block implements a relaxed retraction that maintains spherical geometry. "
        "This unification explains why attention trains stably without explicit regularization while MoE "
        "requires auxiliary balancing losses. Experiments on pre-trained language models with prefix "
        "shuffling show that SVFlow-induced metrics correlate with task performance, reveal "
        "depth-dependent sensitivity, and reflect the intrinsic dynamics of attention. 1 Introduction The "
        "Transformer 33 has become the cornerstone of modern machine learning, yet its design remains "
        "fundamentally heuristic. Recent efforts have retroactively explained its components: attention "
        "as kernel methods 31 or gradient descent 1 , residual connections as neural ODEs 6 , and "
        "normalization as geometric projection 14 . While insightful, these perspectives are fragmented, "
        "leaving many basic phenomena unexplained. For instance, multi-head attention learns stable "
        "representations without explicit regularization, whereas mixture-of-experts (MoE) layers require "
        "auxiliary balancing losses to prevent collapse 24 ; 10 ."
    ),
    "quijote_capitulo": (
        "En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un "
        "hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. Una olla de "
        "algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los sábados, lantejas "
        "los viernes, algún palomino de añadidura los domingos, consumían las tres partes de su hacienda. "
        "El resto della concluían sayo de velarte, calzas de velludo para las fiestas, con sus pantuflos "
        "de lo mesmo, y los días de entresemana se honraba con su vellorí de lo más fino. Tenía en su "
        "casa una ama que pasaba de los cuarenta, y una sobrina que no llegaba a los veinte, y un mozo de "
        "campo y plaza, que así ensillaba el rocín como tomaba la podadera. Frisaba la edad de nuestro "
        "hidalgo con los cincuenta años; era de complexión recia, seco de carnes, enjuto de rostro, gran "
        "madrugador y amigo de la caza. Quieren decir que tenía el sobrenombre de Quijada, o Quesada, que "
        "en esto hay alguna diferencia en los autores que deste caso escriben;"
    ),
    "moby_loomings": (
        "Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my "
        "purse, and nothing particular to interest me on shore, I thought I would sail about a little and "
        "see the watery part of the world. It is a way I have of driving off the spleen and regulating "
        "the circulation. Whenever I find myself growing grim about the mouth; whenever it is a damp, "
        "drizzly November in my soul; whenever I find myself involuntarily pausing before coffin "
        "warehouses, and bringing up the rear of every funeral I meet; and especially whenever my hypos "
        "get such an upper hand of me, that it requires a strong moral principle to prevent me from "
        "deliberately stepping into the street, and methodically knocking people’s hats off—then, I "
        "account it high time to get to sea as soon as I can. This is my substitute for pistol and ball. "
        "With a philosophical flourish Cato throws himself upon his sword; I quietly take to the ship. "
        "There is nothing surprising in this. If they but knew it, almost all men in their degree, some "
        "time or other, cherish very nearly the same feelings towards the ocean with me. There now is "
        "your insular city of the Manhattoes, belted round by wharves as Indian isles by coral "
        "reefs—commerce surrounds it with her surf. Right and left, the streets take you waterward. Its "
        "extreme downtown is the battery, where that noble mole is washed by waves, and cooled by "
        "breezes, which a few hours previous were out of sight of land. Look at the crowds of "
        "water-gazers there. Circumambulate the city of a dreamy Sabbath afternoon. Go from Corlears Hook "
        "to Coenties Slip, and from thence, by Whitehall, northward. What do you see?—Posted like silent "
        "sentinels all around the town, stand thousands upon thousands of mortal men fixed in ocean "
        "reveries. Some leaning against the spiles; some seated upon the pier-heads; some"
    ),
    "sklearn_kmeans_code": (
        "def _kmeans_plusplus(\n"
        "    X, n_clusters, x_squared_norms, sample_weight, random_state, n_local_trials=None\n"
        "):\n"
        "    \"\"\"Computational component for initialization of n_clusters by\n"
        "    k-means++. Prior validation of data is assumed.\n"
        "\n"
        "    Parameters\n"
        "    ----------\n"
        "    X : {ndarray, sparse matrix} of shape (n_samples, n_features)\n"
        "        The data to pick seeds for.\n"
        "\n"
        "    n_clusters : int\n"
        "        The number of seeds to choose.\n"
        "\n"
        "    sample_weight : ndarray of shape (n_samples,)\n"
        "        The weights for each observation in `X`.\n"
        "\n"
        "    x_squared_norms : ndarray of shape (n_samples,)\n"
        "        Squared Euclidean norm of each data point.\n"
        "\n"
        "    random_state : RandomState instance\n"
        "        The generator used to initialize the centers.\n"
        "        See :term:`Glossary <random_state>`.\n"
        "\n"
        "    n_local_trials : int, default=None\n"
        "        The number of seeding trials for each center (except the first),\n"
        "        of which the one reducing inertia the most is greedily chosen.\n"
        "        Set to None to make the number of trials depend logarithmically\n"
        "        on the number of seeds (2+log(k)); this is the default."
    ),
    "scipy_linkage_code": (
        "def linkage(y, method='single', metric='euclidean', optimal_ordering=False):\n"
        "    \"\"\"\n"
        "    Perform hierarchical/agglomerative clustering.\n"
        "\n"
        "    The input y may be either a 1-D condensed distance matrix\n"
        "    or a 2-D array of observation vectors.\n"
        "\n"
        "    If y is a 1-D condensed distance matrix,\n"
        "    then y must be a :math:`\\\\binom{n}{2}` sized\n"
        "    vector, where n is the number of original observations paired\n"
        "    in the distance matrix. The behavior of this function is very\n"
        "    similar to the MATLAB linkage function.\n"
        "\n"
        "    A :math:`(n-1)` by 4 matrix ``Z`` is returned. At the\n"
        "    :math:`i`-th iteration, clusters with indices ``Z[i, 0]`` and\n"
        "    ``Z[i, 1]`` are combined to form cluster :math:`n + i`. A\n"
        "    cluster with an index less than :math:`n` corresponds to one of\n"
        "    the :math:`n` original observations. The distance between\n"
        "    clusters ``Z[i, 0]`` and ``Z[i, 1]`` is given by ``Z[i, 2]``. The\n"
        "    fourth value ``Z[i, 3]`` represents the number of original\n"
        "    observations in the newly formed cluster.\n"
        "\n"
        "    The following linkage methods are used to compute the distance\n"
        "    :math:`d(s, t)` between two clusters :math:`s` and\n"
        "    :math:`t`. The algorithm begins with a forest of clusters that\n"
        "    have yet to be used in the hierarchy being formed. When two\n"
        "    clusters :math:`s` and :math:`t` from this forest are combined\n"
        "    into a single cluster :math:`u`, :math:`s` and :math:`t` are\n"
        "    removed from the forest, and :math:`u` is added to the\n"
        "    forest. When only one cluster remains in the forest, the algorithm\n"
        "    stops, and this cluster becomes the root.\n"
        "\n"
        "    A distance matrix is maintained at each iteration. The ``d[i,j]``\n"
        "    entry corresponds to the distance between cluster :math:`i` and\n"
        "    :math:`j` in the original forest.\n"
        "\n"
        "    At each iteration, the algorithm must update the distance matrix\n"
        "    to reflect the distance of the newly formed cluster u with the\n"
        "    remaining clusters in the forest."
    ),
    "latex_beamer": (
        "\\documentclass[aspectratio=169]{beamer}\n"
        "\\usetheme{metropolis}\n"
        "\\usepackage{amsmath,amssymb,booktabs}\n"
        "\\title{Effective Integration Time in Residual Stacks}\n"
        "\\author{A. Researcher}\n"
        "\\date{\\today}\n"
        "\\begin{document}\n"
        "\\maketitle\n"
        "\\begin{frame}{The question}\n"
        "  A residual block is a forward Euler step,\n"
        "  \\[ h_{\\ell+1} = h_\\ell + F_\\theta(h_\\ell), \\]\n"
        "  so depth is time. How much time?\n"
        "  \\begin{itemize}\n"
        "    \\item Calibrate the step against the field the theory names.\n"
        "    \\item Report $T_{\\mathrm{eff}} = \\sum_\\ell h_\\ell$ against $t^\\ast$.\n"
        "  \\end{itemize}\n"
        "\\end{frame}\n"
        "\\begin{frame}{Method}\n"
        "  \\begin{enumerate}\n"
        "    \\item Extract the residual stream at every layer, unit-normalised.\n"
        "    \\item Fit the step size by projecting the observed displacement onto the\n"
        "      field, $h_\\ell = \\langle \\Delta_\\ell, F(h_\\ell)\\rangle / \\|F(h_\\ell)\\|^2$.\n"
        "    \\item Integrate the null forward to the same time and read the residual.\n"
        "  \\end{enumerate}\n"
        "  The third step is what the literature does not do: the null is integrated to\n"
        "  a \\emph{calibrated} time rather than to a layer index.\n"
        "\\end{frame}\n"
        "\\begin{frame}{Result}\n"
        "  \\begin{table}\n"
        "    \\centering\n"
        "    \\begin{tabular}{lrr}\n"
        "      \\toprule\n"
        "      model & $T_{\\mathrm{eff}}$ & $t^\\ast$ \\\\\n"
        "      \\midrule\n"
        "      small & 0.84 & 4.21 \\\\\n"
        "      large & 1.902 & 4.21 \\\\\n"
        "      \\bottomrule\n"
        "    \\end{tabular}\n"
        "  \\end{table}\n"
        "\\end{frame}\n"
        "\\end{document}"
    ),
    "latex_article": (
        "\\documentclass[11pt]{article}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{amsmath,amsthm,amssymb}\n"
        "\\usepackage[margin=1in]{geometry}\n"
        "\\usepackage{hyperref}\n"
        "\\newtheorem{theorem}{Theorem}[section]\n"
        "\\newtheorem{lemma}[theorem]{Lemma}\n"
        "\\newtheorem{definition}[theorem]{Definition}\n"
        "\\title{Concentration of Pairwise Inner Products under Identity-Weight Attention}\n"
        "\\author{A. Researcher\\thanks{Department of Mathematics.}}\n"
        "\\date{\\today}\n"
        "\\begin{document}\n"
        "\\maketitle\n"
        "\\begin{abstract}\n"
        "We study the evolution of a finite token cloud on the unit sphere under the\n"
        "mean-field dynamics induced by self-attention with identity value and query-key\n"
        "matrices. We show that the pairwise inner products concentrate on a single\n"
        "curve as the ambient dimension grows, and we quantify the rate.\n"
        "\\end{abstract}\n"
        "\\section{Setting}\n"
        "Let $x_1(t),\\dots,x_n(t) \\in \\mathbb{S}^{d-1}$ evolve according to\n"
        "\\begin{equation}\\label{eq:flow}\n"
        "  \\dot{x}_i = \\mathbb{P}_{x_i^\\perp} \\sum_{j=1}^n a_{ij}(t)\\, x_j,\n"
        "  \\qquad\n"
        "  a_{ij}(t) = \\frac{e^{\\beta \\langle x_i, x_j\\rangle}}{\\sum_k e^{\\beta \\langle x_i, x_k\\rangle}},\n"
        "\\end{equation}\n"
        "where $\\mathbb{P}_{x^\\perp} = I - xx^\\top$ is the tangent projection.\n"
        "\\begin{definition}\n"
        "The empirical inner-product measure is\n"
        "$\\mu_t = \\binom{n}{2}^{-1}\\sum_{i<j}\\delta_{\\langle x_i(t), x_j(t)\\rangle}$.\n"
        "\\end{definition}\n"
        "\\begin{theorem}\\label{thm:main}\n"
        "Fix $\\beta \\ge 0$ and let the initial points be i.i.d.\\ uniform. Then for every\n"
        "$t \\ge 0$, $\\mu_t \\to \\delta_{\\gamma_\\beta(t)}$ weakly in probability as\n"
        "$d \\to \\infty$, where $\\gamma_\\beta$ solves the scalar initial value problem\n"
        "$\\dot{\\gamma} = (1-\\gamma)(\\text{terms depending on }\\beta)$, $\\gamma(0)=0$.\n"
        "\\end{theorem}\n"
        "\\begin{proof}[Proof sketch]\n"
        "Only positivity of the coefficients $a_{ij}$ is used. The argument couples the\n"
        "pairwise process to its mean and applies a Gr\\\"onwall estimate.\n"
        "\\end{proof}\n"
        "\\bibliographystyle{plain}\n"
        "\\bibliography{refs}\n"
        "\\end{document}"
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
        "hf_repo":         "albert-base-v2",   # same defect as gpt2-large-random
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
        # The architecture to load before re-initialising. Without it
        # load_model resolves the repo id to the key itself and asks the Hub
        # for "gpt2-large-random", which does not exist — this entry had never
        # loaded (found 2026-09-17, producing CLAIM-C's reference-random arm).
        "hf_repo":         "gpt2-large",
    },
    "gpt2-xl": {
        "model_class":     GPT2Model,
        "tokenizer_class": GPT2Tokenizer,
        "is_albert":       False,
        "random_init":     False,
    },
}

from core.pythia_registry import (
    build_pythia_model_configs,
    PYTHIA_410M_PILOT_STEPS,
    PYTHIA_1_4B_ANCHOR_STEPS,
    PYTHIA_1_4B_EXPENSIVE_STEPS,
)

MODEL_CONFIGS.update(build_pythia_model_configs())

# ---------------------------------------------------------------------------
# Model groups
# ---------------------------------------------------------------------------
#
# MODEL_CONFIGS grew from 10 entries to 47 when the Pythia checkpoint
# registry was merged in. run_1.py used to default --models to
# list(MODEL_CONFIGS.keys()), so a bare `python -m p1_mstate_tracking.run_1`
# now means 27 × 410M + 10 × 1.4B downloads across every prompt — tens of
# gigabytes and hundreds of runs from a command that used to mean "the
# seven Blog 1 architectures". DEFAULT_MODELS pins that original meaning;
# the Pythia schedules are opt-in by group name.

BLOG1_MODELS = [
    "albert-base-v2",
    "albert-xlarge-v2",
    "bert-base-uncased",
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
]

MODEL_GROUPS = {
    "blog1":                 list(BLOG1_MODELS),
    "blog1-random":          ["albert-base-v2-random", "gpt2-large-random"],
    "pythia-410m-pilot":     [f"pythia-410m-step{s}" for s in PYTHIA_410M_PILOT_STEPS],
    "pythia-1.4b-anchors":   [f"pythia-1.4b-step{s}" for s in PYTHIA_1_4B_ANCHOR_STEPS],
    "pythia-1.4b-expensive": [f"pythia-1.4b-step{s}" for s in PYTHIA_1_4B_EXPENSIVE_STEPS],
    # transition plan item 6: the replication gate runs Phase 1 at step 0 and
    # at the final checkpoint and compares against Blog 1's pass criteria.
    # A failed gate stops the sweep, so this is its own group.
    "replication-gate":      ["pythia-1.4b-step0", "pythia-1.4b-step143000"],
}

# Which untrained control belongs to which trained model. --random-baseline
# used to hardcode ("albert-base-v2-random", "gpt2-large-random"), so on a
# Pythia-only selection it silently appended two ALBERT/GPT-2 runs that had
# nothing to do with the sweep. There is no Pythia entry: the published
# step-0 checkpoint is the untrained-weights object, and it is a model in
# its own right rather than a flag.
RANDOM_CONTROLS = {
    "albert-base-v2": "albert-base-v2-random",
    "gpt2-large":     "gpt2-large-random",
}

DEFAULT_MODELS = list(BLOG1_MODELS)

