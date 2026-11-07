<!-- docs/lit-L.md -->
# The prompt library — literature and source scan (2026-11-07)

For `prompt_library/design-L.md` (U0). Web search, abstracts and listing pages, not full texts;
the dataset facts are read from the Hugging Face API on 2026-11-07. Nothing here was run.

## 1. Sources

| source | what it is (as read) | bears on |
|---|---|---|
| `monology/pile-uncopyrighted`, revision `3be90335b66f`, last modified 2023-08-31 | the Pile less Books3, BookCorpus2, OpenSubtitles, YTSubtitles and OWT2 (its card); ships `train/` and **`val.jsonl.zst` (338,045,152 bytes, LFS sha256 `db5e5d1532bf…7c26`) and `test.jsonl.zst` (337,771,483 bytes, `a6cafe820e9c…9c3a`)**. Already used once here: `tools/run/p10_pile_unigram.py` streams its `train/00` | **D1, D4**: a real held-out split for Pythia, and a second one to seal |
| Gao et al. 2020, 2101.00027 (the Pile); Biderman et al. 2023, 2304.01373 (Pythia) | Pythia trains on the Pile's train split (deduplicated or not); val and test are the Pile's own held-out samples | Pile val/test are unseen by Pythia by construction. Near-duplicates across the Pile's splits were **not checked** here |
| Penedo et al. 2024, 2406.17557 (FineWeb) | 15T tokens from 96 Common Crawl snapshots; MinHash dedup **within each snapshot**, not across; **no validation split found** | a FineWeb "held-out" set is held out only from a run whose documents are known. Dodecahedron's card gives its token count (~40B, `docs/lit-T.md` §2), not its documents, so a FineWeb tail is held out from T5's arms (`design-T5.md` (d1)) but not known to be from Dodecahedron |
| `HuggingFaceFW/fineweb` `sample-10BT` at `9bb295dd` (on `HDD_1TB`) | T5's arms trained on its first 2.29B tokens; `t5u4/data/tail.bin` is the unseen continuation, packed 2048-token rows (`design-T5p.md`) | the option (b) source; packed rows cross documents (0.45 of rows 64–1087 carry a break in 40–509, `design-T5p.md` "Blocked 139 decided") |

## 2. How others choose evaluation text

| work | what it says (abstract or summary level) | bears on |
|---|---|---|
| Magnusson et al. 2023, 2312.10523 (Paloma; NeurIPS 2024 D&B) | fit on one held-out distribution does not predict fit on others; 585 domains (546 in the published version) from 18 sources, each sampled per domain; decontaminated against training data, including below the document | **a domain label on every item** (D1, D3): a pooled mean over a mixture hides per-domain spread; Pile subsets are the natural domains here |
| Elazar et al. 2023, 2310.20707 (WIMBD; ICLR 2024) | the Pile intentionally oversamples some sources, so duplication is high; benchmark contamination found in several corpora | the val/test overlap with train is a live caveat (§1), not measured |
| Ethayarajh 2019, 1909.00512; Rudman et al. 2021, IsoScore, 2108.07344; a later study found by search, not identified by name | anisotropy read as mean cosine over sampled pairs; early estimates used ~1,000 pairs, the later study 250,000 and found contextual models far less anisotropic than reported; IsoScore's authors call mean cosine a flawed isotropy measure, and IsoScore is unstable at small samples | ρ is a mean-cosine statistic, so the number of items and how they are drawn move it: **sample size is part of the reading** (D3) |
| Viswanathan et al. 2025, 2501.10573 | intrinsic dimension of a prompt's token cloud peaks in early-to-middle layers, rises when tokens are shuffled, tracks the prompt's loss (Llama, Mistral, Pythia) | per-prompt cloud geometry depends on the text's loss, so a library off a model's training distribution reads a different cloud (D1's cost) |
| Sun et al. 2024, 2402.17762 (massive activations) | a criterion "on more than 90 of 100 sequences"; inputs from short sentences and long RedPajama sequences | 100 real sequences is the order others read activations on; this repo's ρ_mid reads 7 |

**Not found:** any study of how mean-cosine or cluster statistics of a token cloud vary with the
number or choice of prompts. Section 2's sample-size lesson is from pair counts within a corpus,
not prompt counts. So the library's size is set from this repo's own spread
(`prompt_library/sizing_u0.py`), and U1 re-measures it on the library.
