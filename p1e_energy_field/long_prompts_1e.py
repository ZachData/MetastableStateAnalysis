"""
p1e_energy_field/long_prompts_1e.py — four new long passages, so Phase 1e
reads 8 long passages instead of 1d's 4 (`design-1e.md` "Inputs, fixed
here"; the user's direction of 2026-10-06: move towards only the larger
prompts, and add passages now).

THE RULE — fixed 2026-10-06, BEFORE any source text was fetched
---------------------------------------------------------------
This docstring is committed on its own, ahead of the commit that adds the
text, as `p1d_cluster_ensemble/long_prompts.py`'s rule was, so git shows the
rule predates the text. Nothing below is chosen after running a model on it.

  1. **Which passages** (the user chose these four, 2026-10-06). Each
     replaces a genre 1d's long set lost, or adds a structure v1 lacks:

     | key | source | stands in for |
     |---|---|---|
     | `odyssey_butler_long` | Homer, *The Odyssey*, Samuel Butler's 1900 prose translation, Project Gutenberg eBook #1727; **Book VII** from its first body paragraph | `homer_iliad` (v1's passage, despite its key, appears to be Odysseus's account in *Odyssey* 7, in Fagles's translation, still in copyright) |
     | `horla_long` | Guy de Maupassant, *Le Horla*, the 1887 version (the journal, opening "8 mai"), French, from fr.wikisource | `camus_letranger` (first-person French narrative) |
     | `darwin_origin_long` | Charles Darwin, *On the Origin of Species*, 1st edition (1859), Project Gutenberg eBook #1228; **Chapter IV, "Natural Selection"**, from its first body paragraph | `paper_excerpt` (expository argument) |
     | `hamlet_long` | William Shakespeare, *Hamlet*, Project Gutenberg eBook #1524; **Act I, Scene I** from its first line | none: dialogue with speaker tags, a structure v1 does not have |

     None is a v2 held-out source (`core/holdout.py`: not Moby-Dick, the
     Quijote, the Lincoln letters, the two Wikipedia articles, the two
     papers, the sklearn / scipy code or the two LaTeX files), and no key
     carries a held-out key.
  2. **Start.** The first body text after the named heading. Headings, the
     chapter's or book's summary line, and Gutenberg's header and licence
     are not text. The builder refuses if the heading is not found exactly
     once, or the fetched file's title / author (and, for #1727, the
     translator) is not the one named above.
  3. **Continuation.** The source's text in order from the start, with no
     skipping or selection, through the next book / chapter / scene if the
     named one ends first (its heading dropped as in rule 2). Formatting by
     v1's own convention: prose (`odyssey_butler_long`, `horla_long`,
     `darwin_origin_long`) is paragraphs joined by one space, a paragraph's
     own line breaks turned into spaces, footnote markers and footnotes
     dropped; the journal's dates stay, as the text's own words. Drama
     (`hamlet_long`) keeps its lines, as code and LaTeX do in 1d's set:
     speaker tags and stage directions as the edition prints them, with
     leading indentation stripped.
  4. **Length.** Whole units (a paragraph for prose, a line for drama) are
     appended while the total stays at or under ``MAX_TOKENS`` (2048,
     Pythia's context, no special tokens added). A source that ends first
     is recorded as a shortfall, not padded.
  5. **Provenance.** Each fetched file is cached under
     ``DATA_ROOT/long_prompts_sources_1e/``, with its URL, sha256, and the
     Gutenberg "Release date" / "Most recently updated" lines or the
     Wikisource revision id, in ``long_prompts_1e/provenance.json``.
  6. **Separate from the battery.** Keys are not added to
     `core.config.PROMPTS`, so `PROMPT_BATTERY_HASH` stays as it is. The 8
     long passages (1d's 4 and these 4) carry their own hash,
     ``LONG8_HASH``, which every 1e run records.
  7. **Order.** Built with the tokenizer only; the texts are committed
     before any model sees them, and never revised after a run.

Tier 1: exploratory, unregistered.
"""
