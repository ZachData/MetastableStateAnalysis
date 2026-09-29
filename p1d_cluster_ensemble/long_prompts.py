"""
p1d_cluster_ensemble/long_prompts.py — six v1 prompts continued to Pythia's
context length, so Phase 1d can ask what length changes (`lit-1d.md` §4:
length first, count later; the user's direction of 2026-09-25 and
2026-09-29).

THE RULE — fixed 2026-09-29, BEFORE any continuation text was chosen
--------------------------------------------------------------------
This docstring is committed on its own, ahead of the commit that adds the
text, as `core/prompts.py`'s v2 rule was, so git shows the rule predates the
text. Nothing below is chosen after running a model on it.

  1. **Which prompts.** The six v1 metastability prompts that can be
     continued from their own source: `wiki_paragraph`, `repeated_tokens`,
     `sullivan_ballou`, `paper_excerpt`, `hdbscan_code`, `latex_monograph`.
     `homer_iliad` (it appears to be Fagles's 1990 translation) and
     `camus_letranger` (1942) are in copyright and are dropped, not
     replaced (user, 2026-09-29).
  2. **Prefix.** Each long prompt is the v1 text byte for byte, then the
     continuation. Attention is causal, so the long run's first ``n_v1``
     tokens are the v1 run's up to float noise, and length is the only
     change. The builder refuses unless the long prompt's first ``n_v1``
     token ids equal v1's.
  3. **Continuation.** The source's text in order, starting at the first
     character after the passage v1 ends on, with no skipping or
     selection, formatted by v1's own convention for that prompt: prose
     paragraphs joined by one space with headings, reference markers,
     footnotes, figures and displayed equations dropped (v1's prose has no
     newlines); code and LaTeX keep their lines.
  4. **Length.** Whole units (a paragraph for prose, a line for code and
     LaTeX) are appended while the total stays at or under ``MAX_TOKENS``
     (2048, Pythia's context, no special tokens added). If the source ends
     first, the prompt ends there and the shortfall is recorded, not
     padded.
  5. **Sources**, with a revision or version recorded at fetch time:
     - `wiki_paragraph`: English Wikipedia, "Charlotte Brontë" (CC BY-SA),
       the revision id the builder fetched.
     - `sullivan_ballou`: the rest of the same letter (14 July 1861, public
       domain) from a public-domain transcription, named in the provenance
       record. The letter is short, so this prompt is expected to stop
       below ``MAX_TOKENS``.
     - `paper_excerpt`: Geshkovski, Letrouit, Polyanskiy & Rigollet, "A
       mathematical perspective on Transformers" (arXiv 2312.10794), the
       arXiv version fetched. If the excerpt is not found verbatim in that
       text, the builder refuses rather than guessing a join.
     - `hdbscan_code`: `hdbscan/plots.py` from the hdbscan package
       installed in the `mets` env (BSD), version recorded.
     - `latex_monograph`: v1's text is composed, not sourced, so the
       continuation is composed too: written once, continuing the same
       document from its last word, before any model sees it, and never
       revised after a run.
     - `repeated_tokens`: v1's `". "` pattern repeated.
  6. **Separate from the battery.** Keys are ``<v1 key>_long``. They are
     *not* added to `core.config.PROMPTS`, so `PROMPT_BATTERY_HASH` and every
     battery check stay as they are; runs record ``LONG_PROMPTS_HASH``
     instead. The v2 held-out prompts are not involved.

Runs go to ``/run/media/system/HDD_1TB/mets_data`` (user, 2026-09-29): about
3 GB of attention per run at 2048 tokens.

Tier 1: exploratory, unregistered.
"""

from __future__ import annotations

#: The v1 prompts continued (rule 1), in battery order.
SOURCE_KEYS = ("wiki_paragraph", "repeated_tokens", "sullivan_ballou",
               "paper_excerpt", "hdbscan_code", "latex_monograph")
#: Pythia's context length; the rule's cap (rule 4).
MAX_TOKENS = 2048
#: Where the long runs are written (user, 2026-09-29).
DATA_ROOT = "/run/media/system/HDD_1TB/mets_data"


def long_key(v1_key: str) -> str:
    return f"{v1_key}_long"
