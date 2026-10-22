<!-- trackC_writeup/lit-C2-unit1.md -->
# C2 unit 1 — the reading: full texts, a delta scan, and a suite for 92 (ii) (2026-10-22)

Unit 1 of `design-C2.md` (frozen 2026-10-22). Inputs: `trackS_subspace/lit-S2.md` Parked 1,
`trackS_subspace/lit-S1.md` Parked 1, `trackA_force/lit-C2.md` Parked 1 (Parked 3 was done at
A3: `trackA_force/lit-A3.md` "geometry collapse and recovery", [H]). Not repeated: `lit-C2.md`
(2026-10-18), whose rows this file upgrades where a page was read. About 27 page fetches, eight
web searches, and the Hugging Face hub queried for checkpoint branches (`git ls-remote`, the
models API), all 2026-10-22.

**Marks.** [H] the arXiv HTML full text read through a fetch summary (a fetch reads 100 000
characters; where a page is longer, the part read is named); [A] the abstract page; [R] a
companion repository's file or a model repository read directly (`docs/LITERATURE.md` §0.1);
[S] a search engine's summary only. **`arxiv.org` (abstract and HTML) and `huggingface.co` were
reachable from this session** (`docs/LITERATURE.md` §0.2).

## 1. When is E_β monotone along the dynamics? (`lit-S2.md` Parked 1)

| source | statement | hypotheses | direction |
|---|---|---|---|
| Geshkovski, Letrouit, Polyanskiy, Rigollet, 2312.10794 [H] (§1–§5 read, 100 000 of 220 935 characters) | §3.2 eq. (3.5)–(3.6): E_β[μ] = (1/2β) ∬ e^{β⟨x,x′⟩} dμ dμ, and dE_β/dt = ∫ ‖𝒳[μ]‖² Z_{β,μ} dμ | Q = K = V = I, one head, constant weights, tokens on the sphere, no MLP | **non-decreasing** (V = −I: non-increasing, §3.2) |
| the same, §3.3 Lemma 3.6–3.7, Remark 3.8 | the surrogate (USA), partition function replaced by n, is a Wasserstein gradient flow (ascent) of E_β | V = I; the proof omitted in the paper | non-decreasing |
| the same, §3.4.1 | (SA) is gradient ascent of E_β(X) = (1/2β) Σᵢⱼ e^{β⟨Vxᵢ,xⱼ⟩} in a metric weighting token i by Z_{β,i} | **QᵀK symmetric and V = QᵀK**, one head, constant parameters, sphere | non-decreasing |
| the same, §3.3 after (3.7) | "because of the lack of symmetry" (SA) is not the gradient of a functional in the standard metric | — | — |
| the same, §2.3.1–§2.3.2 | multi-head (2.7) and the MLP block (2.8) are set up; no energy statement for either ("beyond the scope of our current methods" for the MLP) | — | — |
| Pendharkar, 2605.04279 [H] (100 000 of 119 909 characters) | Theorem 11: Σₕ E^h non-decreasing, flat or sphere; Theorem 12: still so for W_V,h = Mₕ + Eₕ with ‖Eₕ‖ ≤ ε*(t); per head, cross-head "radial shadows" block monotonicity (Theorem 17, Condition 8) | every score matrix Mₕ symmetric and **W_V,h = Mₕ** (or within ε), Mₕ constant along the flow; no MLP; §10: the conditions are "never exactly satisfied in trained transformers" | non-decreasing |
| Tomihari, Karakida, 2505.19458 [A] | as in `lit-S2.md` | — | — |

**F5, worded from the texts.** E_β's monotonicity in depth is a theorem only for constant
weights whose score matrices are symmetric and tied to the values (one head, QᵀK = V symmetric:
2312.10794 §3.4.1; many heads, each Mₕ = W_V,h symmetric or nearly: 2605.04279 Theorems
11–12), tokens on the sphere, no MLP. Pythia's weights at every step, including step 0, and
S2's edited weights meet none of these (QK untied from V, 16 heads, MLPs, weights that change
by layer). So the break is a departure from the identity-weight picture's prediction, not a
violated theorem, and L1's monotone steps 0–128 are an empirical fact, not a consequence of one.

**A correction this reading makes.** `lit-S2.md`'s row read "the energy decreases along the
flow". Both texts say non-decreasing. S2's rules counted a drop between layers as a violation
(`core/metrics.py` `energy_violation_severity`), which is the non-decreasing convention, so no
S2 reading moves; the row is corrected in place and routed to S (`status-S.md` "Corrections
received", 2026-10-22).

## 2. 2603.13314 in full (`lit-S1.md` Parked 1)

Shaikh et al. [H]. **Weights, not activations**, for the subspace result: an overlap dimension
OD_H on each layer's concatenated projections (§3). Models Llama-3.1 8B, Falcon3-10B, OLMo-2 7B,
Qwen3-32B; training checkpoints on OLMo-2 only, at about 1B, 10B, 100B, 1T tokens and the final
≈ 4T, the earliest at step 100. Keys grow most ("a steep, almost exponential rise that dominates
late-stage checkpoints"), queries "the same trajectory, albeit with a gentler slope", **values
"remain largely flat, indicating near-orthogonality throughout"**. No onset step for the
overlap (Fig. 3's 50k–100k is where predictors sit, not overlap); no warmup; no Pythia.

**How appendix A states S1 against it.** S1's key row agrees in direction (key sharing grows). S1's
`ov_read` pool, across all of 410m's heads, has its top above a flat Haar spectrum from 512 (λ₁
1.13 → 2.05 at 143000; `trackS_subspace/status-S.md` "S1 unit 2"), where OLMo-2's values stay
near orthogonal within layers. Different model, a within-layer statistic against an all-heads
pool, so stated as different readings, neither confirming nor contradicting.

## 3. `lit-C2.md` Parked 1: the [S] rows that §2 and §8 cite

| row | was | read now | what changes |
|---|---|---|---|
| Li et al., 2509.23024 | [S] | [H] (100 000 of 112 684): RankMe / αReQ on the **last layer's last token**, features "assumed" centred, no normalisation named; FineWeb, 15 000 sequences; Pythia 1B+ in the body (abstract "160M–12B"; App. Fig. 10 reportedly below 1B); no step ranges; the first phase "coinciding with the learning rate ramp-up" | the literature ties its own first geometric phase to the warmup: F1 is the field's confound too (§2, §7). `lit-1.md`'s question (line 263) partly answered: features assumed centred, no normalisation named; `lit-1.md` not edited |
| Hoogland et al., 2402.02364 | [S] | [H] (100 000 of 212 500): a 3M two-layer attention-only LM on the Pile; LM1 (0–900) bigrams; LM2 (900–6.5k) positional use and 3-, 4-grams, first-layer heads start previous-token attention; LM3 (6.5k–8.5k) previous-token heads form; LM4 (8.5k–17k) induction heads; LM5 (17k–50k) no significant change; warmup not mentioned | the stage names are the paper's, now read; their steps are a toy's, so §2 quotes the order, not the steps |
| Wang et al., 2508.00331 ("Embryology") | [S] "lists the five" | [A]: susceptibilities and UMAP give a "body plan" with the induction circuit and a "spacing fin"; the abstract lists no stages | §2 does not cite it for the stage list |
| Yin, Steinhardt, 2502.14010 | [S] | [A]: induction and function-vector heads across 12 models; many FV heads begin as induction heads during training; no steps in the abstract | "induction at ~1000, FV at ~16000" stays [S]; §2 does not quote the steps |
| Wang et al., 2410.02984 | [S] | [A]: refined LLC on a two-layer attention-only transformer; heads differentiate and specialise; a multigram circuit | "staggered by thousands of steps" stays [S]; §8 cites it for head-level differentiation only |
| Herty, Liu, 2608.25055 | [S] | [A]: mean-field theory of coupled token (depth) and parameter (training) dynamics; well-posedness, convergence; no experiments | §8, theory |
| Rigollet, 2512.01868 | [S] "Rigollet et al.", "inner products concentrating along depth in trained models" | [A]: single author (ICM 2026); clustering, metastability, normalisation's effect on contraction, a long-context phase transition; no trained models in the abstract | the trained-model clause is unsupported and dropped; §8 cites it as theory |
| Olsson et al., 2209.11895 | [S] (`lit-10.md` row 3) | [A]: induction heads develop at "precisely the same point" as a sudden sharp rise in in-context learning, visible as a bump in the training loss; causal evidence in small attention-only models, correlational in larger ones | §2.2 cites the abstract's claim; the Pythia step conversion (~1200–2400) stays `lit-10.md`'s |
| Bruno, Pasqualotto, Agazzi, 2410.23228 | [S] | [A]: mean-field token flow on the sphere; near a uniform start, many tokens stay near a meta-stable manifold whose structure is set by β (a rescaled Gegenbauer polynomial); the PDE "can be interpreted as a Wasserstein gradient flow" | §8, theory |
| Bernas et al., ICML 2026 / 2604.08764 | [A] "anisotropy of embeddings along Pythia ... checkpoints" | [H] (all 147 998 characters, in two reads): the main measure is IsoScore* on the covariance of **backpropagated gradients** (tangent against normal directions); activations only in App. D (MLP activations, wikitext), Pythia-410m from **step 10 000** (ranges 10k–22k … 126k–138k); "early" means the first 30 % of training; nothing in steps 0–4000 | **F9 narrows**: published anisotropy along Pythia's training starts at step 10 000; the break's window (128–512) is not read by it (design-C2 F9) |

## 4. Delta scan since 2026-10-18

Searches: the energy along training, token clustering and checkpoints, anisotropy in early
Pythia training, Geshkovski-type energy in trained models, suites with early checkpoints.

| hit | what | bearing |
|---|---|---|
| `andrey-lolobrigida/token-drift` [R] (GitHub, undated, a personal project) | Pythia-70m at 12 log-spaced checkpoints; raw (uncentred) mean cosine per layer at the end 0.54–0.72 (L1–L5; 0.19–0.41 at random init); "training builds almost all of this in the first 1000 steps", a "middle block half-built by step 512" | the nearest thing to F9 inside the window: residual-stream anisotropy rising where the break is, on 70m, unreviewed. Not cited as a finding; it raises 92 (i)'s stakes |
| Ben-Artzy, Schwartz, 2609.08692 [A] (Sep 2026) | trained transformers' representations dominated by one principal direction, against SSMs; no training axis in the abstract | beside F9 (§8) |
| 2608.08922, 2601.23236 | already in `lit-1.md`, `lit-1d.md`, `lit-1e.md` | — |

**Verdict on "nobody reads this energy across training": not refuted, as of 2026-10-22.**

## 5. A suite for 92 (ii): different warmup, checkpoints inside it

Pythia's break is at steps 256–512, about 0.5–1.1B tokens (2 097 152 tokens a step).

| suite | warmup | early checkpoints (checked 2026-10-22) | verdict |
|---|---|---|---|
| Pythia, PolyPythias | 1 % of 143 000 (1430 steps), every size [R] | 0, 1, 2, …, 512, 1000 | the same schedule |
| Stanford CRFM Mistral GPT-2 small / medium, 5 seeds each | 4000 steps, linear; medium peak 1.5e-4 [R, `conf/trainers/gpt2-medium.yaml`]; batch 512 × 1024 | the docs list every 10 steps to 100, every 50 to 2000, as Hub branches; **the Hub repositories now hold only `main`** (`git ls-remote` on beren, eowyn; models API refs on arwen, alias, battlestar) | gone from the Hub |
| OLMo-2 1B (0425) | — | stage 1: step 0, 300 (1B tokens), then 10 000 | one point in the window |
| OLMo-1B-0724, OLMo-1B | — | from step 1000 (2B and 4B tokens) | past the window |
| LLM360 Amber | — | 360 over 1.26T tokens | coarser |
| SmolLM2-135M intermediate, pico-decoder-small | — | from step 240 000; `main` only | none |

**92 (ii) is not feasible: no public suite found with a different warmup and checkpoints inside
its first billion tokens.** Unit 3c is not run; the budget is 6 units + close-out.

## 6. 92 (iii)'s configs [R] (`EleutherAI/pythia` `models/*/pythia-*.yml`)

| size | layers × width, heads | peak LR | min LR | warmup | schedule | batch |
|---|---|---|---|---|---|---|
| 70m | 6 × 512, 8 | 1e-3 | 1e-4 | 0.01 (1430) | cosine, 143 000 | 32 × 32 GPUs = 1024 × 2048 |
| 160m | 12 × 768, 12 | 6e-4 | 6e-5 | 0.01 | cosine | the same |
| 410m | 24 × 1024, 16 | 3e-4 | 3e-5 | 0.01 | cosine | the same |

The design's numbers stand: the same warmup, the same tokens a step, the peak LR the dose.

## Parked

1. **The Mistral GPT-2 checkpoints off the Hub** (why: the docs name only Hub branches, now gone;
   cost: an email to Stanford CRFM, days; could change: 92 (ii), and with it a unit 3c). Carry to
   C2's close-out, where it is carried or dropped.
2. **token-drift's per-step table** (why: a 70m anisotropy series in the window, uncentred;
   cost: one read of its config and outputs; could change: nothing in a rule, only §7's words
   beside unit 3b's 70m readings). Carry to unit 3b.
