<!-- prompt_library/design-L.md -->
# The prompt library — design (proposed 2026-11-07; D1–D5 are Blocked 140)

Opened 2026-11-07 by `docs/ROADMAP.md` §10.14, the user's: build a larger shared library of real
prompts now, with T5′ paused (`trackT_timaeus/design-T5p.md` "Blocked 139 decided, and T5′
paused"). Row: ROADMAP §5 "Prompt library". Scan: `docs/lit-L.md` (2026-11-07). Infrastructure,
tier 1; nothing is registered, and nothing here touches `claims/registry.json`.

## What it is for

Every ρ_mid on Tracks A and T reads the 7 v1 prompts (`trackA_force/a1a.py` `V1`; `t5.state_stats`),
hand-picked genres, ≤ 512 tokens each. Their spread across prompts is small at most steps and
large exactly where the readings move (`prompt_library/sizing_u0.py`, on T5 U2's records):

| trajectory | SD over the 7, median over steps | max | at step | SE of the mean at the max: k = 7 | 256 | 1024 |
|---|---|---|---|---|---|---|
| `pythia-70m` | 0.023 | 0.069 | 256 | 0.026 | 0.004 | 0.002 |
| `pythia-160m` | 0.031 | 0.117 | 256 | 0.044 | 0.007 | 0.004 |
| `pythia-410m` | 0.029 | 0.183 | 256 | 0.069 | 0.011 | 0.006 |
| `dodecahedron-32m` | 0.033 | 0.228 | 88 | 0.086 | 0.014 | 0.007 |
| `dodecahedron-67m` | 0.041 | 0.207 | 88 | 0.078 | 0.013 | 0.007 |
| `gpt-2-small` | 0.028 | 0.092 | 1100 | 0.035 | 0.006 | 0.003 |

So at 7 prompts a transition's level is known to about ±0.07–0.09 (one SE), the size of the
effects T5 labelled (its bar 0.15). The library is **more prompts, not longer ones**, drawn from
real held-out text by a rule fixed before any model reads them (the user prefers real text to
synthetic sets we choose, `design-T5p.md` "Blocked 139 decided").

**What it cannot do.** Make the 7-prompt results wrong or right on its own: a re-read is a new
reading on new inputs, labelled by the old rule (D5). Make a library item "in distribution" for
every model: one text set is off one family's training data (D1). Remove a document-length
selection (D2). It is not a fix for T5′, which resumes on the inputs `design-T5p.md` fixes
(ROADMAP §5, `/challenge-pr` on #319, findings 1–2).

## Fixed now (no decision waits on these)

| item | fixed as |
|---|---|
| separate from the battery | the library is not added to `core.config.PROMPTS`. The v2 battery, its hash and its rule 6 stay as they are (`core/prompts.py`): adding 1024 keys would change every manifest's `PROMPT_BATTERY_HASH` and widen every reader that loops over the live battery (`docs/battery_consumers.md`, the growth risk) |
| the 7 v1 and the twelve | unchanged. The 7 stay the bridge: a reader moved to the library reports its 7-prompt value beside, once, in the unit that moves it. The twelve keep their status as Phase 10's confirmation set (`core/holdout.py`); the library absorbs neither |
| pinning (`LESSONS.md` 3) | a manifest in git, `prompt_library/library_v1.json`: the dataset revision (`3be90335b66f`), each file's LFS sha256 (`docs/lit-L.md` §1), the permutation seed, the length rule, the tokenizer (`EleutherAI/pythia-410m`'s, by its file hashes), and per item: row index, Pile subset (`meta.pile_set_name`), character span, sha256 of the item's text, its GPT-NeoX token count. The text itself lives outside git (`data/library/v1/`), rebuilt from the pinned file |
| the loader | `core/library.py`: `load(split, k)` returns the first k items in library order after recomputing every item's sha256 against the manifest (refuses on a mismatch, a missing file, an empty or short item); `library_hash(split, k)` hashes the manifest's first k entries, so a reader's record names its exact input (CLAUDE.md "Name the input"). A reader names k in its design; no reader loops over "the whole library" |
| the sealed split | `core/holdout.py` refuses the confirmation split (D4) to every loader call without `allow_sealed=True`, and `tests/test_holdout.py` checks that no file under `tools/`, `trackA_force/`, `trackT_timaeus/` or `prompt_library/` other than the builder passes it |
| `docs/battery_consumers.md` | gains a library section in U1: who reads it, which k, names its input, refuses on drift |
| text handling | no normalisation, no stripping; the item is a prefix of the document string, cut at the end character of its 512th GPT-NeoX token (the fast tokenizer's offset mapping), so the cut never splits a character. Each reader keeps its own tokenizer and special-token convention, and records its own token count per item |
| the draw | no model sees a candidate before inclusion (`core/prompts.py` rule 5's discipline): the builder reads text and a tokenizer only |

## The units, exit and budget

| unit | what | done when | cost |
|---|---|---|---|
| U0 (this) | scan, design, Blocked 140 | the user decides D1–D5 | writing |
| U1, the build | the draw (D1–D4), the manifest, `core/library.py`, the holdout extension, tests, the `battery_consumers.md` section | **first output populated, checked before the rest is drawn**: item 0 is a 512-token prefix with its subset and hash; then all items non-empty, 512 NeoX tokens each, hashes reproduce on a second build from the pinned file; per-subset shares of the drawn items within a binomial 99 % band of the eligible shares the builder counted; the eligible share of each subset reported (D2's selection, stated) | 2 × 338 MB download, CPU minutes |
| U2, the first use (D5) | T5 U2's trajectories re-read on the library | below | ≤ 2 GPU-hours (180 checkpoints × k items; U2 took 474 s for 7) |
| close-out | the one page, card, Parked | CLAUDE.md "Close out" | writing |

**Exit:** the library and its sealed split are built, pinned and loadable with refusing checks,
and (under D5 (a)) T5 U2's labels are read on it. **Budget: 3 units + close-out** (U0, U1, U2).
At the budget it closes at what it knows; under D5 (b) the exit drops U2 and the budget is 2 + close-out.

**U2, fixed now (under D5 (a)).** *Cells:* T5 U2's three scored trajectories (`dodecahedron-32m`,
`-67m`, `gpt-2-small`) at U2's steps; `pythia-70m`, `-160m`, `-410m` read beside, unscored, as U2
had them (`data/trackT/t5u2/2026-10-30/reading.json`, `"scored": false`). *Statistic and rule:* U2's, unchanged
(`trackT_timaeus/t5c.py`): Δ = ρ_mid(first) − min_t ρ_mid(t); spreads if Δ ≥ 0.15 and the
bootstrap 5th percentile > 0.05; does not spread if the 95th < 0.15; else not told apart; the
bootstrap now resamples library items (k = 1024). *Gate, before any library output:* U2's own,
with k: the known answer (the library path fed the 7 v1 texts reproduces U2's records at 410m
0 and 4000 and 32m 1 and 9536 within 1e-6), the synthetic planted 0.15 at k = 1024 seen "spreads"
≥ 90 % and flat ≤ 5 %, with s set from the library's own spread at 9536 on `-32m` (U2 set it from
the 7). *What counts:* a scored label that differs from U2's is a correction to T5 (one line under
`trackT_timaeus/status-T.md` `## Corrections received`); the Pythia labels, the series, per-subset
reads and the 7-prompt overlay are described beside, no rule. U2's labels, for the comparison
(`reading.json`; Δ, bootstrap 5th–95th): scored, `-32m` spreads 0.736 (0.669–0.794), `-67m` spreads
0.734 (0.672–0.778), `gpt-2-small` spreads 0.168 (0.153–0.182); beside, 410m spreads 0.298, 160m
spreads 0.171 (5th 0.146), 70m does not 0.101. `gpt-2-small` and 160m sit nearest the bar.

**Pre-authorised fallbacks.** U1: a download whose sha256 differs from the pinned one refuses (no
fallback: the pin is the point); a missing `pile_set_name` becomes the stratum "unknown", counted;
a subset share outside its band is reported, not redrawn (a redraw is selection). U2: a known-answer
miss is a code defect, fixed and re-gated in the same unit; a synthetic-gate miss at k = 1024 stops
U2 and goes to the user.

## Decisions (Blocked 140, yours)

| # | choice | options | recommended, and why |
|---|---|---|---|
| D1 | source | (a) **one shared set**: `pile-uncopyrighted` `val`, read by every model; each item carries its Pile subset, and the web subset (`Pile-CC`) is flagged, so a FineWeb-trained reader can drop it; (b) **per family**: Pile val for Pythia, the FineWeb `sample-10BT` tail for Dodecahedron and the arms; (c) Pile val, used for Pythia only | **(a).** Held out from Pythia by the Pile's split, and from Dodecahedron and T5's arms by source except where the same web page sits in both (Pile-CC is Common Crawl; FineWeb is Common Crawl), which the flag isolates. T5 compares the families; on one text set the comparison reads the same prompts. (b)'s FineWeb tail is held out from the arms only: FineWeb has no validation split and Dodecahedron's documents are not stated (`docs/lit-L.md` §1). Cost of (a): Pile text is off FineWeb's distribution, so Dodecahedron reads text it fits worse (Paloma; `lit-L.md` §2); per-subset reads beside show how much |
| D2 | an item | (a) **one document**, those with ≥ 512 GPT-NeoX tokens, cut to their first 512; (b) packed 512-token windows across documents (no length selection); (c) whole documents up to 512 tokens, lengths varying | **(a).** It keeps v1's shape (one text, ≤ 512 tokens, one cloud) and one length, so ρ is not read across lengths. (b) mixes documents inside one cloud (T5′'s ICL rows: 0.45 carry a break). (c) puts length into the spread. Cost of (a): short documents are excluded, a selection by length, so U1 reports the eligible share per subset |
| D3 | size, order, mixture | (a) **1024 items in a fixed order**: the val rows permuted once (`default_rng(20261107)`), then the first 1024 that meet D2; the mixture is Pythia's own (proportional, in expectation); a reader names a prefix k (256 for per-item costly readers, 1024 otherwise); (b) 256 items; (c) stratified, an equal count per Pile subset (Paloma's per-domain design) | **(a).** At k = 1024 the transitions' SE falls to 0.006–0.007, at 256 to 0.011–0.014 (table above, from the 7's spread, which real draws may exceed). A prefix of a random permutation is itself a random draw, so every k is a sample of the same population and readers can differ in k without a second library. (c) answers "per domain" but weights rare subsets (EuroParl, Enron) as heavily as Pile-CC; per-subset reads stay possible under (a) where counts allow |
| D4 | the confirmation split | (a) **`pile-uncopyrighted` `test`, the same rule and seed, 1024 items, drawn and hashed in U1 and sealed** (no model reads it until a registry entry names it, CLAUDE.md "Name the family"); (b) a reserved tail of val (rows after the library's); (c) none now | **(a).** A separate official split, held out from the library as well as from training; drawn now, while no reading exists to select it on; costs a download. (b) shares a file and an order with the library. (c) leaves the next confirmatory claim to build its held-out set after seeing exploration output |
| D5 | first use, and re-reads | (a) **one re-read in U2**: T5 U2's three scored trajectories (three Pythia beside) under U2's frozen rule (above); no other reader moves now; new designs take the library by default (T5′ excepted, ROADMAP §5); (b) build only: no reader moves and nothing is re-read until a design asks; (c) re-read every ρ_mid result on Tracks A and T (A1a, A5, T5 U2–U5), one unit each | **(a).** The question that motivated the library is whether the 7-prompt ρ_mid readings hold; T5 U2 is the cheapest re-read with a frozen rule and recorded labels, and it uses the library end to end, which (b) never does before a real design depends on it. Each re-read is a correction to its phase if it changes a label (CLAUDE.md "Stop" step 2), so (c) re-opens closed phases one by one for readings no current question needs |

## Other ways to answer the question (simplest first)

1. Keep the 7 and widen the bootstrap's honesty: report the 7-prompt SE beside every ρ_mid. Free;
   says how thin the readings are, makes none of them thicker.
2. Add twelve to fifty more hand-picked prompts under the v2 rule. Cheap, but chosen text, and the
   user asked for real text.
3. The library as designed (D1–D5 (a)).
4. Per-family libraries (D1 (b)): matched to each family's data, at the price of comparing
   families on different text.
5. A Paloma-style domain benchmark (D3 (c)) with a decontamination pass against each training set.
   The most complete, and several units of work beyond a library.

## Worth challenging (for the PR)

- D1 (a) puts Dodecahedron on text it was not trained on. If ρ_mid depends on how well a model fits
  the text (`lit-L.md` §2, intrinsic dimension tracks loss), a Pythia-vs-Dodecahedron difference
  could be a data-fit difference. The per-subset reads and the 7-prompt overlay are the only check.
- The 512-token cut is in GPT-NeoX tokens. Dodecahedron's tokenizer splits the same text into a
  different count; a reader capped at 512 of its own tokens truncates or under-fills. Cutting by
  characters, or at the shortest count over named tokenizers, was rejected as more machinery
  than the per-item token counts that make it visible.
- The sizing rests on the 7 v1 prompts' spread; random Pile documents (code, legal, mathematics)
  may spread far more, which would favour (a)'s 1024 over 256.
- Near-duplicates between the Pile's train and val/test were not checked. Pythia may have seen a
  val document's twin.
- D5 (a) makes the library's first use a re-read of a closed track's result, which can re-open it
  through a correction. (b) avoids that, and leaves the library unused until N1 or T5′'s successor.
