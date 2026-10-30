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

## 4. T5's scan (2026-10-30; web search, abstracts and listing pages, not full texts)

What bears on why Dodecahedron's unit cloud sits at mean cosine 0.84 from init (`trackT_timaeus/design-T5.md`).

| work | what it says (abstract level) | bears on |
|---|---|---|
| Noci et al. 2022, 2206.03126 (already cited in C2's draft) | rank collapse at init: stacked self-attention makes tokens near-identical; scaling the residual branches with depth prevents it | **H1's mechanism, published**: T5 tests it on released runs, not in theory |
| Dong et al. 2021, 2103.03404 (cited in `trackC_writeup/lit-C2-unit1.md`) | pure attention loses rank doubly exponentially with depth | U1's positive control (a known collapse) |
| Giorlandino & Goldt 2025, 2505.24333 (ICLR 2026; in `docs/readings/meanfield_reading_list_2026-10-01.md`) | rank collapse and entropy collapse at init; the attention residual strength and the Q/K scale set the regime; the critical strength depends on depth | H1's depth form (r · L); a phase diagram U1's factorial could be placed on (not proposed) |
| Karagodin, Ge, Polyanskiy, Rigollet 2025, 2510.22026 (in `trackA_force/lit-A.md`) | normalization as speed regulation in the particle picture; Pre-LN contracts polynomially | the same Pre-LN family on both models; not a difference between them |
| Sun et al. 2024, 2402.17762 (cited in C2's draft) | massive activations act as fixed biases; **explicit attention biases remove the need to learn them** | **H2, published in its key-bias form**. Dodecahedron has no biases of any kind, and position 0's norm is 13.9–23.8× the median at 9536 |
| Gu et al. 2024, 2410.10781 (ICLR 2025; in `p1d_cluster_ensemble/lit-1d.md`) | attention sinks emerge after effective optimisation; they act like a key bias; sigmoid attention removes them | H2: the sink path U3 measures |
| Timkey & van Schijndel 2021, 2109.04404; Machina & Mercer 2024 (NAACL; Pythia's 143 checkpoints) | 1–5 rogue dimensions dominate cosine similarity; large Pythia models are isotropic through the final LayerNorm, and anisotropy does not grow steadily over training | ρ_mid can be dominated by a few dimensions (m_ℓ and ρ_mid⁻⁰ are read beside for this reason); Pythia's spreading has precedent |
| "Two failure modes", "Curse of Depth" 2502.05795 | Pre-LN variance grows with depth; deep layers near identity | context only |

**Not found:** any measurement of token cosine at init or along training under GPT-NeoX's
depth-scaled init against Llama's flat 0.02 (HF's Llama init has no depth scaling), and anything
published on Dodecahedron's or Timaeus's `gpt-2-small`'s geometry. So T5's comparison is new as
a measurement, not as a mechanism: H1 is Noci et al.'s mechanism and H2 is Sun et al.'s.
