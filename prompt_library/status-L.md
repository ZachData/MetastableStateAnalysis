<!-- prompt_library/status-L.md -->
# The prompt library — STATUS

<!-- phase-card -->
## Card

- **Question:** How many real, held-out prompts does a token-cloud reading (ρ_mid on Tracks A and T, read today on 7 hand-picked v1 prompts) need, drawn from which source by which rule fixed before any model reads them, and pinned how, so that later readings name their exact input?
- **Inputs:** U1 drew the library from `monology/pile-uncopyrighted` at `3be90335b66f` (`val`, `test` by their LFS sha256) with `EleutherAI/pythia-410m`'s tokenizer at `9879c9b5f8be`; no model read it. U0 read T5 U2's records (`data/trackT/t5u2/2026-10-30/`, the 7 v1 prompts, code `3d425b32db90`) through `prompt_library/sizing_u0.py`, and the Hugging Face API listing of `monology/pile-uncopyrighted` at `3be90335b66f`
- **Results:**
  - Built 2026-11-07 (U1; D1–D5 decided all (a), Blocked 140): 1024 `val` items and a sealed 1024-item `test`, each a single Pile document cut at its 512th GPT-NeoX token, pinned by `prompt_library/library_v1.json` and loadable only through `core/library.py`; a second build reproduced the manifest and texts byte for byte; no subset outside its 99 % band; 42 % of rows meet the length rule (PubMed Abstracts 3 %); 1010 of 1024 items per split re-tokenize to exactly 512, the rest 511–513 — `prompt_library/status-L.md` "U1, the build (2026-11-07)"
  - Opened 2026-11-07 by the user's order in the roadmap's decisions, item 14 (T5′ paused for it): on T5 U2's records the 7 prompts are thin for a level in the transitions but enough in count for T5's scored Δ, and Δ is set mostly by which kind of text (French narrative lowest on Pythia, code on Dodecahedron), so the library's case is coverage of real text, not count; a design proposed with a sealed confirmation split, 3 units + close-out, revised after `/challenge-pr` on #320; D1–D5 are Blocked 140 — `prompt_library/status-L.md` "U0, the design (2026-11-07)"
- **Superseded / wrong:** none
- **Registry:** none, because the library is an input set; D4's sealed split is held for a later registration, which is the user's
- **Depends on:** none
- **Feeds:** none
- **Open threads:**
  - U2, the first use: T5 U2's labels re-read on `val` k = 1024 under U2's rule, beside T5 (`prompt_library/design-L.md` "The units, exit and budget")
  - whether random Pile documents spread more than the 7 v1 prompts (U2 measures it; U1 read no model)
- **After Phase 10:**
  - near-duplicate check of the library against the Pile's train split *(free: CPU, an n-gram pass over one train shard)*
- **Reviewed:** 2026-11-07 · body `62856e4e96`
<!-- /phase-card -->

## Corrections received

None.

## U0, the design (2026-11-07)

Branch `claude/prompt-library-u0`; writing and one reader of existing records. Design
`prompt_library/design-L.md`; scan `docs/lit-L.md`.

| item | value | how to re-run |
|---|---|---|
| across-prompt SD of per-prompt ρ_mid on the 7 v1 prompts, median over steps | 0.023 (70m) – 0.041 (`-67m`) | `METS_DATA=<main>/data python prompt_library/sizing_u0.py` |
| its maximum, and where | 0.069–0.228; Pythia at step 256, Dodecahedron at 88, `gpt-2-small` at 1100 | same |
| SE of a k-prompt mean of the level at the maximum | k = 7: 0.026–0.086; 256: 0.004–0.014; 1024: 0.002–0.007 | same |
| **Δ, U2's scored statistic** (per prompt, first step minus the step where the 7's mean is lowest): SD over the 7 | 0.032 (410m, `gpt-2-small`) – 0.102 (`-32m`) | same |
| SE of Δ's k-prompt mean | k = 7: 0.012–0.039; 256: 0.002–0.006; 1024: 0.001–0.003 | same |
| lowest per-prompt Δ, by trajectory | `camus_letranger` on 70m (0.01 against 0.08–0.15), 160m, 410m, `gpt-2-small`; `hdbscan_code` on `-32m` (0.53 against 0.69–0.82) and `-67m` | same |
| per-prompt median against the recorded ρ_mid | within 0.018 at every record | same |
| held-out text on hand | Pile val and test (`pile-uncopyrighted`, 338 MB each, LFS sha256s in `docs/lit-L.md` §1); FineWeb `sample-10BT` on `HDD_1TB` (no validation split; held out only from T5's arms) | `docs/lit-L.md` §1 |

**Caveats.** The SDs are the 7 hand-picked prompts', not real draws'. The per-prompt median is a
proxy for the recorded statistic; Δ here fixes the minimum's step, where U2's bootstrap retakes it
per resample. T5 U2 scored three trajectories; Pythia's were beside.

**`/challenge-pr` on #320** (accept with changes, at `6117399`), all five findings taken: (1) the
first version sized the library from the level's SE and called it the size of T5's effects; T5
scored Δ, which spreads less, so the Δ rows above were added and D3's 1024 re-argued from per-subset
reads; (2) D5's re-read is recorded beside T5, not as a correction, since Δ tracks the kind of
text; (3) D1's held-out claim narrowed: known only for Pythia; `gpt-2-small`'s data unknown; (4)
U2's gate wording ("before any Δ is read"); (5) the numbers live here only.

## U1, the build (2026-11-07)

Branch `claude/prompt-library-u1`. Blocked 140 decided all (a), the user's (D1–D5, `design-L.md`).
Builder `prompt_library/build_l1.py`; manifest `prompt_library/library_v1.json` (in git); texts
and report `data/library/v1/{val,test}.jsonl`, `build_report.json` (not in git). Loader
`core/library.py`; seal `core/holdout.py` `refuse_sealed_split`; tests `tests/test_library.py`.
No model read anything.

**Input:** `monology/pile-uncopyrighted` at `3be90335b66f`, `val.jsonl.zst` / `test.jsonl.zst`
(sizes and LFS sha256 match `docs/lit-L.md` §1, checked by the builder); tokenizer
`EleutherAI/pythia-410m` at `9879c9b5f8be` (file sha256s in the manifest), no special tokens.

| item | val | test (sealed) |
|---|---|---|
| rows | 179,996 | 180,378 |
| eligible (≥ 512 NeoX tokens) | 75,734 (0.421) | 75,351 (0.418) |
| permuted positions used for 1024 items | 2,398 | 2,461 |
| subsets outside the binomial 99 % band | none of 17 | none of 17 |
| item re-tokenized alone = document's first 512 ids | 1010 / 1024 | 1010 / 1024 |
| item token count (re-tokenized) | 511–512 (14 at 511) | 511–513 (16 not 512) |
| `library_hash(split, 1024)` | `47feaa321fa0…` | `538e14844925…` |
| second build from the pinned files | manifest, texts and report byte-identical | same |

**Eligibility by subset (val; D2's length selection, stated).** Eligible share of rows: DM
Mathematics 1.00, Gutenberg 1.00, ArXiv 0.96, PhilPapers 0.95, PubMed Central 0.89, FreeLaw 0.78,
EuroParl 0.71, Ubuntu IRC 0.71, USPTO 0.54, Github 0.53, Pile-CC 0.47, HackerNews 0.45,
StackExchange 0.42, Wikipedia (en) 0.36, NIH ExPorter 0.24, Enron 0.24, **PubMed Abstracts 0.03**.
Drawn (val, of 1024): Pile-CC 351, StackExchange 177, Github 135, Wikipedia 78, USPTO 76, PubMed
Central 63, FreeLaw 54, DM Mathematics 36, ArXiv 22, PubMed Abstracts 12, HackerNews 10, Enron 6,
NIH 3, PhilPapers 1, EuroParl 0, Gutenberg 0, Ubuntu IRC 0. Subsets with ≥ 32 items (D3's
per-subset reads): Pile-CC, StackExchange, Github, Wikipedia, USPTO, PubMed Central, FreeLaw, DM
Mathematics — 8; ArXiv (22) is not one. The full table, test's too: `build_report.json` "mixture".

**Two things the design did not foresee.**
- *The cut's offsets.* The pinned tokenizer's ByteLevel post-processor trims offsets; trimming
  moves a span's edges inward past whitespace, so a token ending in whitespace after other
  characters (3 in the vocabulary) would get a short end. The builder takes offsets with
  `trim_offsets` off; the ids are unchanged. Recorded in `design-L.md` "Decisions".
- *Not every item is 512 tokens on its own.* The design's done-when said "512 NeoX tokens each";
  its own cut (the document's prefix to the end character of its 512th token) gives that for
  1010 of 1024 per split. In the rest the cut lands in a whitespace run (the document tokenizes
  `\n\n` before a word as `Ċ` + `Ċ…`, the cut text ends `ĊĊ`, one token) or, for test's two 513s,
  presumably inside a multi-byte character (offsets cover the whole character); test's texts were
  not opened. **Kept as drawn**: re-cutting the 28 after seeing them would change the rule, and
  the design already has each reader record its own token count per item (Dodecahedron's
  tokenizer differs anyway). Each item's own count is in the manifest. The alternative, a cut
  rule exact at 512 (e.g. back off to the last pre-token boundary), is in the PR's "Worth
  challenging".

**Caveats.** Eligibility counts tokens on each document's first 32,768 characters (exact for
shorter documents; no longer document fell short there, so none was recounted whole). The
mixture check compares drawn to eligible shares, not to Pythia's training mixture (the Pile
weights subsets by epochs; val is a uniform sample of documents). Near-duplicates against the
Pile's train split are still unchecked (card, "After").

**How to re-run.** `METS_DATA=<main>/data python prompt_library/build_l1.py` (≈ 3 min, 16 CPU
threads, 3.3 GB RAM; `--first` prints item 0 and writes nothing). A second build to compare:
`--data-dir D --manifest D/library_v1.json`, then `cmp` against the committed manifest.

**Parked.** None.
