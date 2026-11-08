<!-- prompt_library/status-L.md -->
# The prompt library — STATUS

<!-- phase-card -->
## Card

- **Question:** How many real, held-out prompts does a token-cloud reading (ρ_mid on Tracks A and T, read today on 7 hand-picked v1 prompts) need, drawn from which source by which rule fixed before any model reads them, and pinned how, so that later readings name their exact input?
- **Inputs:** U2 read T5 U2's six trajectories (`dodecahedron-32m`, `-67m` at 53 steps, `pythia-70m`, `-160m`, `-410m` at 18, `timaeus/gpt-2-small` at 20; 180 checkpoints) on `val` k = 1024 (`library_hash` `47feaa321fa0`), code `4d7962561795`, records `data/library/u2/2026-11-07/`. U1 drew the library from `monology/pile-uncopyrighted` at `3be90335b66f` (`val`, `test` by their LFS sha256) with `EleutherAI/pythia-410m`'s tokenizer at `9879c9b5f8be`; no model read it. U0 read T5 U2's records (`data/trackT/t5u2/2026-10-30/`, the 7 v1 prompts, code `3d425b32db90`) through `prompt_library/sizing_u0.py`, and the Hugging Face API listing of `monology/pile-uncopyrighted` at `3be90335b66f`
- **Results:**
  - Closed 2026-11-07 at its exit, in budget (3 units + close-out; writing only at close): a pinned library of 1024 real held-out Pile documents with a sealed 1024-item test split exists and refuses on drift, and T5 U2's six labels stand on it; why it sits below the 7 v1 prompts in the fall (text or length) and a near-duplicate check against the Pile's train split are carried to the roadmap's §5 row "Library L′" — `prompt_library/status-L.md` "The library on one page (closed 2026-11-07)"
  - Re-read 2026-11-07 (U2; Blocked 141 (a), the user's): on 1024 Pile `val` documents of ≥ 512 tokens T5 U2's six labels all stand under its frozen rule (the three scored "spreads", Δ 0.775, 0.762, 0.174; 70m "does not spread", 160m and 410m "spreads"); Δ is larger than on the 7 v1 prompts on every trajectory (by 0.006–0.043), since the library's troughs are lower; every subset with ≥ 32 items gives each trajectory's label; recorded beside T5 U2, not as a correction to it — `prompt_library/status-L.md` "U2, T5 U2's labels on the library (2026-11-07)"
  - Built 2026-11-07 (U1; D1–D5 decided all (a), Blocked 140): 1024 `val` items and a sealed 1024-item `test`, each a single Pile document cut at its 512th GPT-NeoX token, pinned by `prompt_library/library_v1.json` and loadable only through `core/library.py`; a second build reproduced the manifest and texts byte for byte; no subset outside its 99 % band; 42 % of rows meet the length rule (PubMed Abstracts 3 %); 1010 `val` and 1008 `test` items re-tokenize alone to exactly the document's first 512 ids, the other 30 to 511 or 513 (Blocked 141) — `prompt_library/status-L.md` "U1, the build (2026-11-07)"
  - Opened 2026-11-07 by the user's order in the roadmap's decisions, item 14 (T5′ paused for it): on T5 U2's records the 7 prompts are thin for a level in the transitions but enough in count for T5's scored Δ, and Δ is set mostly by which kind of text (French narrative lowest on Pythia, code on Dodecahedron), so the library's case is coverage of real text, not count; a design proposed with a sealed confirmation split, 3 units + close-out, revised after `/challenge-pr` on #320; D1–D5 are Blocked 140 — `prompt_library/status-L.md` "U0, the design (2026-11-07)"
- **Superseded / wrong:** none
- **Registry:** none, because the library is an input set; D4's sealed split is held for a later registration, which is the user's
- **Depends on:** none
- **Feeds:** none
- **Open threads:**
  - why the library sits below the 7 in the fall (kind of text, or each item's 511–512 tokens against the 7's 242–562), and how much of a Pythia-against-Dodecahedron difference on it is prefix length: not told apart; carried to `docs/ROADMAP.md` §5 "Library L′" — `prompt_library/status-L.md` "The library on one page (closed 2026-11-07)"
- **After Phase 10:**
  - L′ (a): T5 U2's re-read with prefix statistics kept, length against text *(forward pass: ≈ 77 min on the 3080, 180 checkpoints)*
  - L′ (b): near-duplicate check of `val` and `test` against the Pile's train split, before a registry entry names `test` *(free: CPU, an n-gram pass over one train shard, plus its download)*
- **Reviewed:** 2026-11-07 · body `625f645f42`
<!-- /phase-card -->

## Corrections received

None.

## The library on one page (closed 2026-11-07)

**Asked** (`design-L.md`; `docs/ROADMAP.md` §5 "Prompt library" and §10.14, the user's, with T5′
paused for it): every ρ_mid on Tracks A and T read the 7 hand-picked v1 prompts; build a shared set
of real held-out prompts, drawn by a rule fixed before any model reads them and pinned so that a
reading names its exact input, and (D5 (a)) re-read T5 U2's labels on it. **Closed at its exit, in
budget** (U0, U1, U2 + this close-out). Infrastructure, tier 1, nothing registered. Writing only
at close.

**The exit, as read** (numbers in the units' sections; inputs on the card):

| part of the exit | as read | where |
|---|---|---|
| built and pinned | 1024 `val` items and 1024 sealed `test` items, each one Pile document cut at its 512th GPT-NeoX token; the manifest in git; a second build byte-identical; texts rebuilt from the manifest alone | "U1, the build (2026-11-07)" |
| loadable, refusing | `core/library.py` re-hashes every item and refuses on a mismatch, a missing, empty or short item; `test` refused to every caller but the builder (tested) | same |
| T5 U2's labels read on it | **all six stand** under the frozen rule: the three scored spread (`-32m`, `-67m`, `gpt-2-small`); beside, 70m does not, 160m and 410m spread | "U2, T5 U2's labels on the library (2026-11-07)" |

**Established (tier 1):**

1. **A pinned library of real held-out text exists, drawn with no model in the loop.** Its
   subsets' shares match the eligible documents' (none of 17 outside its 99 % band). It is a
   sample of *long* documents, not of the Pile: 42 % of rows meet the length rule, from 3 %
   (PubMed Abstracts) to 100 % (DM Mathematics, Gutenberg).
2. **T5 U2's labels hold on this population of text.** Δ is larger than on the 7 on every
   trajectory (the library's troughs are lower); every subset with ≥ 32 items gives each label,
   and so does the library without Pile-CC. Recorded beside T5 U2, not as a correction (D5 (a)).
3. **The 7's text dependence recurs, in part.** Code and mathematics spread least on Dodecahedron,
   as the 7's code prompt did; DM Mathematics least on Pythia and `gpt-2-small`. No subset stands
   in for the 7's French narrative (EuroParl drew 0 items).
4. **The reading conventions are checked**: neither tokenizer adds a start or end token;
   Dodecahedron's BPE cuts every item at 512 of its own tokens, a shorter prefix than Pythia's.
5. **T5 U2's synthetic gate (and U2's here) cannot fail at any s**: it checks the rule's
   arithmetic, not the labels' robustness (`LESSONS.md` 6; a line in `status-T.md` "Corrections
   received"; no T5 label moves).

**Not established:**

- Why the library sits below the 7 in the fall (up to 0.13): kind of text, or length (511–512
  tokens against 242–562). Nor how much of a Pythia-against-Dodecahedron difference on the library
  is prefix length. Carried (Parked 1).
- That the library is held out from any model but Pythia (FineWeb is a crawl, `gpt-2-small`'s data
  are unknown; D1), or from Pythia at the level of near-duplicates: val and test were not checked
  against the Pile's train split. Carried (Parked 2).
- Checkpoint-to-checkpoint noise: both readings use the same 180 checkpoints and the bootstrap
  resamples items only; `gpt-2-small`'s 0.174 sits 0.024 above the bar, about its step jitter.
- Anything about short documents (58 % of rows excluded by D2), or about lengths other than 512.

**What it gives the plan.** New designs read the library by default (T5′ excepted, ROADMAP §5): a
reader names its k and records `library_hash(split, k)`, and a reader moved to it reports its
7-prompt value beside, once (`design-L.md` "Fixed now"; readers listed in
`docs/battery_consumers.md` "The prompt library (not the battery)"). `test` stays sealed until a
registry entry names it, which is the user's. T5′ resumes on the inputs `design-T5p.md` fixes.

**Units at close:**

| unit | PR | outcome |
|---|---|---|
| U0, design and scan | #320 | proposed; D1–D5 decided all (a), Blocked 140, the user's |
| U1, the build | #321 | built and sealed; the 30 items not 512 alone kept, Blocked 141 (a), the user's |
| U2, T5 U2's labels re-read | #322 | all six stand; the synthetic gate shown unable to fail |
| close-out | this PR | writing only |

Blocked 140 and 141 are decided; none is open for the library.

**Parked at close** (every item carried or done):

1. *Length or text, and Dodecahedron's shorter prefix* (U2, `/challenge-pr` on #322, finding 2) →
   **carried to `docs/ROADMAP.md` §5 "Library L′", (a)**: one ≈ 77 min re-run of `l2.py run`
   keeping prefix statistics (each item at the 7's lengths, and Pythia at the character span
   Dodecahedron's 512 tokens cover); unscheduled. Could change: whether the library's lower fall
   is text, and how much of a family difference read on the library is prefix length. Cheaper
   path: a later library reader that compares families keeps prefix statistics itself.
2. *Near-duplicates against the Pile's train split* (card, U1's caveats) → **carried to the same
   row, (b)**, as a step before a registry entry names `test`. Could change: whether `test` is
   held out from Pythia at the near-duplicate level.
3. *The start-token question* (U1's Parked) → done in U2 (neither tokenizer adds one).

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
Builder `prompt_library/build_l1.py`; manifest `prompt_library/library_v1.json` (in git); `val`'s
texts and the report `data/library/v1/val.jsonl`, `build_report.json` (not in git; `test`'s texts
are not kept on disk, below). Loader
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
| item re-tokenized alone = document's first 512 ids, exactly | 1010 / 1024 | 1008 / 1024 |
| item token count (re-tokenized) | 14 at 511, 1010 at 512 | 14 at 511, 1008 at 512, 2 at 513 |
| `library_hash(split, 1024)` | `47feaa321fa0…` | `538e14844925…` |
| second build from the pinned files | manifest, texts and report byte-identical | same |
| texts rebuilt from the manifest alone (`--from-manifest`) | identical to the drawn texts | every hash reproduced (into scratch, then deleted) |

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
  1010 of 1024 in `val` and 1008 in `test` (30 items in all; the first count, 28, compared only
  the first 512 ids and so missed test's 513s, `/challenge-pr` on #321, finding 2). In the rest
  the cut lands in a whitespace run (the document tokenizes
  `\n\n` before a word as `Ċ` + `Ċ…`, the cut text ends `ĊĊ`, one token) or, for test's two 513s,
  presumably inside a multi-byte character (offsets cover the whole character); test's texts were
  not opened. **Kept as drawn** (Blocked 141 (a) recommended): re-cutting the 30 after seeing them would change the rule, and
  the design already has each reader record its own token count per item (Dodecahedron's
  tokenizer differs anyway). Each item's own count is in the manifest. A cut rule exact at 512
  alone does not exist for the whitespace cases: no prefix within ±60 characters of the cut
  re-tokenizes to 512 on items 5, 156, 196, 296 (`/challenge-pr` on #321, finding 1, checked
  here), so the options left are keep (a) or replace (c).

**Caveats.** Eligibility counts tokens on each document's first 32,768 characters (exact for
shorter documents; no longer document fell short there, so none was recounted whole). The
mixture check compares drawn to eligible shares, not to Pythia's training mixture (the Pile
weights subsets by epochs; val is a uniform sample of documents). Near-duplicates against the
Pile's train split are still unchecked (card, "After").

**After `/challenge-pr` on #321** (accept with changes, at `ddaff99`; all five taken). (1)
Blocked 141's (b) withdrawn, above. (2) The count is 30, not 28; the builder now compares whole
id lists. (3) Rebuilding no longer depends on the random stream or the tokenizer:
`--from-manifest SPLIT` slices the pinned file at each item's row and character end and checks its
sha256; the manifest also records python, numpy and `tokenizers` versions. (4) `test`'s texts are
not kept on disk: the builder deletes them after drawing, and `--from-manifest test
--allow-sealed` rebuilds them once a registry entry names the split; the guard test now counts
any non-`False` value of `allow_sealed`, not only a literal `True`. (5) The manifest test pins the
two library hashes and the exact token counts.

**How to re-run.** `METS_DATA=<main>/data python prompt_library/build_l1.py` (≈ 3 min, 16 CPU
threads, 3.3 GB RAM; `--first` prints item 0 and writes nothing). A second build to compare:
`--data-dir D --manifest D/library_v1.json`, then `cmp` against the committed manifest. Texts only:
`--from-manifest val [--data-dir D]` (≈ 1 min, no tokenizer).

**Parked.**
- *Carried to U2:* whether a U2 reader's tokenizer adds a start token that takes one of 512
  slots (`t5c.py` tokenizes with the HF defaults and `max_length=512`; Pythia adds none,
  Dodecahedron's not checked here). If so, every item loses its last token at read time, which
  matters more than the 30 above (`/challenge-pr` on #321, question). U2's gate states each
  tokenizer's convention and per-item count. *Done in U2:* neither adds one (below).

## U2, T5 U2's labels on the library (2026-11-07)

Branch `claude/prompt-library-u2`. **Blocked 141 decided (a), the user's:** the 30 items are kept as
drawn. Rule and gate: T5 U2's, frozen (`design-L.md` "U2, fixed now"); the reading's details fixed
and pushed before any library output (`2b5294f`, `design-L.md` "U2's details"). Producer
`prompt_library/l2.py` (code `4d7962561795`), tests `tests/test_l2.py`; records
`data/library/u2/2026-11-07/` (main tree; 180 checkpoints, 4605 s on the RTX 3080, runner).
**Input:** `val` k = 1024, `library_hash` `47feaa321fa0…`; T5 U2's six trajectories at its steps
(`t5c.TRAJ`), one item per forward pass, float32, eager, TF32 off, statistics in float64 on the GPU.

**Tokens.** Neither tokenizer adds a start or end token (`check_tokenizer`). Pythia and
`gpt-2-small` read every item at 511–512 NeoX tokens, none cut. **Dodecahedron's BPE gives more
than 512 tokens for every item, so all 1024 are cut at 512 of its own tokens** (U2's ≤ 512
convention): it reads a shorter prefix of each document than Pythia does.

**Gate: passes.** Known answer (the 7 v1 texts through `l2.py` at 410m 0, 4000 and `-32m` 1, 9536
against U2's step records; every per-prompt ρ and ρ⁻⁰, the five means, the token counts):
max 2.3e-15 (`read` checks that `known.json` ran on the records' code). Synthetic at 1024 items, s
0.0194 (`-32m` 9536's item spread on the library, U2's 7 gave 0.0072): planted 0.15 "spreads"
92.4 % (≥ 90 %), flat 0.0 % (≤ 5 %). **This synthetic cannot fail** (`/challenge-pr` on #322,
finding 1, re-run here): at s 0.019, 0.1, 0.3 and 0.6 it gives planted 0.945, flat 0.0 (200
draws each), since a planted draw misses only when the first step's noise falls below the
minimum over the plant's 15-step floor (about 1 in 16). It checks the rule's arithmetic, not
the labels' robustness to noise; the same holds for T5 U2's 94.5 % (`LESSONS.md` 6, a line in
`status-T.md` "Corrections received").

**Reading** (Δ = ρ_mid(first) − min, item bootstrap 5th–95th; T5 U2's on the 7 beside):

| trajectory | first | min (step) | last | Δ library | 5–95 % | label | Δ on the 7 | U2's label |
|---|---|---|---|---|---|---|---|---|
| **`dodecahedron-32m`** (scored) | 0.859 | 0.084 (917) | 0.898 | **0.775** | 0.772–0.778 | **spreads** | 0.736 | spreads |
| **`dodecahedron-67m`** (scored) | 0.836 | 0.074 (671) | 0.910 | **0.762** | 0.758–0.766 | **spreads** | 0.734 | spreads |
| **`gpt-2-small`** (scored) | 0.328 | 0.154 (4300) | 0.169 | **0.174** | 0.173–0.175 | **spreads** | 0.168 | spreads |
| `pythia-70m` (beside) | 0.299 | 0.169 (2000) | 0.339 | 0.130 | 0.128–0.131 | does not spread | 0.101 | does not spread |
| `pythia-160m` (beside) | 0.386 | 0.172 (2000) | 0.370 | 0.214 | 0.212–0.216 | spreads | 0.171 | spreads |
| `pythia-410m` (beside) | 0.458 | 0.151 (4000) | 0.368 | 0.308 | 0.306–0.309 | spreads | 0.298 | spreads |

ρ_mid⁻⁰ gives the same label and minimum step on every trajectory (Δ within 0.002). The minimum's
step moves only on `gpt-2-small` (4300, against 6000 on the 7).

**Beside, no rule.**
- *Per subset* (the 8 with ≥ 32 items): every subset gives each trajectory's label. Lowest Δ:
  on Dodecahedron DM Mathematics and Github (`-32m` 0.726, 0.728; `-67m` 0.725, 0.732; highest
  Wikipedia, Pile-CC and USPTO, 0.79–0.81); on Pythia and `gpt-2-small` DM Mathematics (70m 0.091, 160m
  0.154, 410m 0.272, `gpt-2-small` 0.155) and USPTO. So code and mathematics spread least on
  Dodecahedron, as the 7's code prompt did; no subset stands in for the 7's French narrative
  (EuroParl drew 0 items). The two nearest the bar: `gpt-2-small` on DM Mathematics (0.155, 5th 0.145)
  and 160m on DM Mathematics (0.154, 5th 0.144), both "spreads", n 36.
- *Without Pile-CC* (673 items, D1's flag): `-32m` 0.763, `-67m` 0.750, `gpt-2-small` 0.172, all
  "spreads"; Pythia's labels unchanged.
- *Against the 7:* the library's series has the 7's shape on every trajectory (the early peak
  at the same step, `-67m` one step earlier; the trough at the same step but on `gpt-2-small`; the
  late rise). Firsts agree within 0.012. In the fall the library sits lower, by up to 0.10–0.13 (at
  most 0.05 on 70m); at the minimum lower by 0.013–0.033; at the last step lower by 0.004–0.067
  (160m most). Dodecahedron's early rise is the one place it sits higher (by ≤ 0.04).
- *Across items* (10th, 50th, 90th percentiles of each item's middle-state median): at the first
  step 0.82–0.90 (`-32m`), 0.44–0.48 (410m); at the trough 0.04–0.15 (`-32m`), 0.12–0.18 (410m).

**What it says.** On 1024 random Pile `val` documents of ≥ 512 NeoX tokens (42 % of rows, the
share varying by subset, U1), T5 U2's labels all stand, and Δ is a little larger on each
trajectory. So the labels hold on this population of real text as well as on the 7. Both readings
use the same 180 checkpoints, so neither tests checkpoint-to-checkpoint noise (finding 1).
Recorded **beside** T5 U2 (D5 (a)): a reading of another population of text, not a correction to it.

**Caveats.** The bootstrap resamples items only. Its narrow bands (±0.003) say nothing about
checkpoint-to-checkpoint jitter, and Δ takes a minimum over 18–53 steps. `gpt-2-small`'s 0.174
sits 0.024 above the bar; U2 already noted that its step-to-step jitter is about that size.
Dodecahedron reads shorter prefixes than Pythia (above), so a Pythia-against-Dodecahedron
difference mixes model with text length; the 7 had the same mismatch (32m read 484 tokens of
`hdbscan_code` against Pythia's 242). Why the library sits below the 7 in the fall is not told
apart: kind of text, or length (511–512 tokens against 242–562). Pile text is off Dodecahedron's
training distribution (D1).

**How to re-run.** `HF_HOME=<HDD_1TB>/mets_data/hf METS_DATA=<main>/data python -m
prompt_library.l2 known --out <dir>`, then `run --out <dir>` (≈ 77 min), then `read --out <dir>`
(≈ 1 min, 18 CPU threads; reader `4c65dd7f647a` after #322's review fixes, labels unchanged; CodeRabbit's four taken: `meta.json`
written atomically, a resume refuses another device / torch / transformers, a step resumes only with
both its files, the population named as eligible documents).

**`/challenge-pr` on #322** (accept with changes, at `b552906`; all four taken). (1) The
synthetic cannot fail at any s (above; `LESSONS.md` 6). (2) Length against text: parked, below.
(3) `read` refused nothing when `known.json` came from other code; it now requires the records'
code (`tests/test_l2.py`). (4) `status-T.md`'s line no longer repeats this section's Δs.

**Parked** (homed at the close-out: carried to `docs/ROADMAP.md` §5 "Library L′" (a); "The library
on one page (closed 2026-11-07)").
- *Length or text, and Dodecahedron's shorter prefix* (finding 2): both families are causal, so
  the statistics on a forward pass's first n tokens equal those of the n-token text read alone.
  Reading each item at the 7's lengths, or Pythia at the character span Dodecahedron's 512
  tokens cover, needs no extra forward passes in principle, but these records keep only
  whole-item statistics, so it costs one re-run (≈ 77 min, prefix statistics kept). It could
  change: whether the library's lower fall is length or text, and how much of a
  Pythia-against-Dodecahedron difference is prefix length.
