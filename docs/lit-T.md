<!-- docs/lit-T.md -->
# Track T: the search behind ROADMAP §7b (2026-10-29)

**What this is.** The search made at the user's check-in on 2026-10-29, before Track T was
proposed: web search (4 queries, "standard" and "extended") and the Hugging Face API. Abstracts
and search summaries only; **no paper was read as primary text**, so every row is a lead, not a
reading. Track T's first unit to open runs its own scan (`CLAUDE.md`).
**Superseded where they differ by T1** (2026-10-29, primary text and metadata read):
`trackT_timaeus/status-T.md` "T1, the inventory". §2's "9536 × 524k is ~5.0B" read the card's
per-GPU batch as global; 8 × 524k gives the card's 40B.

## 1. The claim T4 rests on

**No published LLC values for any Pythia model were found.** Queries: "local learning coefficient
Pythia checkpoints training LLC estimate"; "\"Pythia\" \"learning coefficient\" SGLD
susceptibilities OR \"Bayesian influence\" checkpoints 2025". What was found instead:

| paper | what it has | on Pythia? |
|---|---|---|
| Hoogland et al., 2402.02364 | LLC along training; stages bigram → n-gram → previous-token → induction | no: a 2-layer attention-only model (C2 §2.2 cites it) |
| Lau et al., 2308.12108; Furman & Lau, 2402.03698 | the estimator; its scaling | no |
| 2507.21449 | a benchmark for local posterior sampling | no |
| 2606.22389 | LLC off equilibrium, chains started at each checkpoint | not known; **read before T4's design** |
| 2601.12703 (spectroscopy) | susceptibilities around trained checkpoints | Pythia-14M, not along training |
| 2509.26544 (Bayesian influence functions) | SGLD covariances, attention weights only | Pythia fine-tuning, not along training |
| 2504.18274 (structural inference) | susceptibilities | a small LM |
| 2510.12071 | influence across checkpoints | a toy model |
| 2601.13548 (patterning) | lists LLC estimators among its measurements | a small LM |
| Timaeus blog 2026-04-21 ("How to scale susceptibilities") | susceptibilities on Pythia-1.4B, ~4,800 H200 hours | trained checkpoint only |

Not ruled out: unindexed work, workshop papers, Timaeus results without a paper.

## 2. Timaeus's released checkpoints (Hugging Face API, 2026-10-29)

| repo | what the listing shows | card |
|---|---|---|
| `timaeus/dodecahedron-32m`, `-67m` | Llama-style, 16 / 20 layers, FineWeb, 53 branches log-spaced steps 1–9536, LR 3.5e-3, warmup 1000, 524k tokens a step | yes; says ~40B tokens (9536 × 524k is ~5.0B) and points at `-v2` repos |
| `timaeus/dodecahedron-32m-archived`, `-67m-archived` | not opened | — |
| `timaeus/dodecahedron-32m-pilot-dense` | 161 branches | none |
| `timaeus/tetrahedron-3m-og` | 501 `.pth` checkpoints, steps 0–50000 every 100, `losses.txt` | none; 2402.02364's model is the unconfirmed guess |
| `timaeus/L2` (and `L0`–`L6`, `H*`, `dm*`) | final model at 75000 plus a `checkpoints/` folder | stub |
| `timaeus/gpt-2-small`, `-log-spacing`, `-init-seed-5` | ~650 checkpoint folders (`gpt-2-small`) | none |

**T2's scan (2026-10-29, one search, before `design-T2.md` freezes):** nothing published on
Dodecahedron's training dynamics or its checkpoints' transitions; the cards are the only source.
The schedule question itself was scanned for C2 (`trackA_force/lit-C2.md`,
`trackC_writeup/lit-C2-unit1.md`).

## 3. Bearing

- T4's novelty rests on §1's negative, which is a search result, not a reading.
- T3's positive control needs a released model with a published curve and its settings; §2 does not confirm one.
- No released Timaeus model was found with a published LLC curve deep enough for V (ROADMAP §7b, "the cheapest overlay").
