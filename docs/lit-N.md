<!-- docs/lit-N.md -->
# Track N — literature scan before N1's design (2026-11-08)

For `trackN_neurons/design-N1.md` (U0). Web search and summarising fetches of abstracts, READMEs,
blog pages and two papers' HTML; one paper's PDF text read (2311.00863). Grades as in
`trackT_timaeus/design-T1.md`: **read** = primary text, **summary** = a fetch's summary of a page
(not checked line by line), **listing** = a search result only. Nothing was run. The essay itself
is the local copy `docs/readings/goodfire_already_clean_2026-10-15.txt` (untracked).

## 1. The essay's method (read: the local copy)

| term | what the essay says (line) | bears on |
|---|---|---|
| pole | "each sign of a neuron read on its own" is ROADMAP's gloss; the essay counts "neuron poles" (34), reads the negative pole as a gate, "not a neuron's opposite" (238, 496), and in ReLU nets reads it through the pre-nonlinearity "votes" (238) | **Q4**: the essay's two-sign reading is SwiGLU's product sign (582); for ReLU it already reads the pre-activation |
| sign regimes | SwiGLU's gate × value: − × − gives a positive output, the "double-negative regime", the whole top of the list for ~4 % of neurons (382, 582–586) | under GeLU there is no such regime (`tools/math_checks/n1_gelu_pole.py` G2) |
| blind maximum | agent 1 explains a neuron; agent 2 sees only the explanation and writes a maximal example; compared with the strongest corpus example (10B tokens); per input bundle, with the other bundles ablated (398–404) | N3's explanation test, if any; it needs an LLM, the corpus maximum and attribution |
| thresholds | no fixed one: "trust what the neuron learns, its natural sparsity and activation ranges" over percentiles (36, 330–334); weak activations mostly ablatable (310–318) | **D1**'s populated-pole bar is ours to set |
| ~90 % one job | Qwen3-32B, "preliminary audits", LLM agents, no planted control stated (34); GPT-2 XL ~70 % (404) | ROADMAP N2's reason |
| over training | InceptionV1 retrained: "the movement is local" (690); recursive generalization predicts a branch "grow[s] out from [a local origin] over training in a particular order" (824); MoE experts not looked at over training (390) | N3's discriminating prediction |
| companion paper | "forthcoming" (70, 346) | **listing**: not out as of 2026-11-08 (evidencepress.org 2026-10-20; a search 2026-11-08); re-check at U1 |

## 2. Neurons across Pythia's training (the question N3 asks)

| source | what it is | grade | bears on |
|---|---|---|---|
| skydancerosel, `github.com/skydancerosel/neuron-lineage` (2026; no paper) | **the essay's "local growth" tested on `pythia-410m` and `-1.4b-deduped`**: MLP layers 6, 12, 18, 512 sampled neurons per layer, top 200 contexts, 11 checkpoints from step 128 to 143000, 2M tokens of `pile-10k`; a "job" is a cluster of ≥ 10 contexts in a frozen embedding (genre, or GPT-2 medium's layer-12 role); a new job's parent is the unit's nearest job one checkpoint back; against 512 random directions, with planted-answer controls per layer and settings fixed on a pilot sample. **410m: role-adjacent 10.4 % vs 6.5 % (p 5e-9), genre 6.5 % vs 4.8 %; 5–28 % of new jobs sprout locally; 72 % of new jobs born at steps 256–2000.** Sign handling not stated | summary | **Q2, Q6**: N3's core test is run on our model, with controls; its birth window is C2's 512–1000 |
| Gao, Zhang, Chen (MBZUAI), "Recursive Generalization and Superposition Are Both Right" (blog, 2026-10; repos `HeartyHaven/pythia-neuron-evolution`, dataset `LangGao/pythia-160m-neuron-evolution`) | `pythia-160m`, 240 neurons, 40 checkpoints, Pile-10K top-10 passages, LLM explanations linked across time by a second agent: recursive generalization alone 14, superposition alone 11, mixed 180; 213 of 240 end-of-training jobs reached by generalization; "superposition" more common early; **a second LLM agreed with the first on ~40 %**; no planted control | summary | **Q2**: the LLM-graded route on Pythia, and its reliability problem stated by its authors |
| Quirke, Heindrich, Gurnee, Nanda 2023, 2311.00863 (NeurIPS workshop) | `pythia-70m`'s German context neuron **L3N669** across checkpoints: detects German (F1 0.9 by "checkpoint 13"), its mean-ablation loss on German grows 0.4 % → over 10 % from checkpoint 20 to 100; many German context neurons (229 with F1 > 0.85 at some checkpoint) form in a "phase transition around checkpoint 10", "coinciding with the LR warm-up", most later unlearned | read | **Q3**: a known-job neuron with a dated history, on 70m. Which step "checkpoint 10" is (an index into Pythia's 154 or its 143 linear checkpoints) is not stated in the text read: **unknown** |
| Kong, Ning, Adler, Shavit, 2509.24198 (v2 2026-03), "Negative Pre-activations Differentiate Syntax" | in GeLU / SiLU models the negative pre-activation region carries computation: zeroing only the negative pre-activations of the 1 % most non-Gaussian ("Wasserstein") neurons hurts BLiMP and TSE far more than perplexity-matched ablations of many more neurons; WD measured on Pythia 70m–12B; **on Pythia 1.4B's checkpoints the cohort's WD rises within ~25k steps and the ablation's harm grows with it** | summary | **Q4**: the negative pole read on the pre-activation is functional on a Pythia, and forms over training |
| Stolfo et al. 2024, 2406.16254 (NeurIPS) | entropy and token-frequency neurons; **`pythia-410m` neuron 23.417 is a token-frequency neuron** (with 3412, 2017, 87 in layer 23 in its appendix H.2); entropy neurons in Pythia 410m and 1B (appendix F, no index given); no checkpoint study found | summary | **Q3**: a known-job neuron on 410m itself, final layer, final checkpoint only |
| 2505.12822 (rare-token neurons), 2506.21468 (TopK LMs), 2605.18789 (SAE features' life histories at 7 checkpoints of `pythia-160m`, `-410m`: most events in the first ~7 % of training), 2410.21272 (arithmetic heuristic neurons across `pythia-6.9b` checkpoints), 2407.10827 (circuits across Pythia's training and scale; Tigges et al.) | further neuron or feature histories on Pythia | listing | beside; none reads poles at C2's dated events |
| Gurnee et al. 2023, 2305.01610 (Neurons in a Haystack), 2401.12181 (universal neurons, GPT-2 seeds); Lecomte et al. 2023, 2312.03096 (incidental polysemanticity) | the neuron-level baselines the essay answers; the last predicts polysemanticity from init, i.e. present at step 0 and not grown | listing | N3's step-0 row: a pole's job present at init is neither branch |

## 3. What the scan changes

- **N3 as written is partly answered on 410m** (neuron-lineage, §2 row 1): local growth above
  random directions, small in absolute terms, most births at 256–2000. What neither neighbour
  reads, as far as the summaries show: births dated against C2's 512 (the break) and 1000 (the
  centred break and induction) on the 18-step grid, the negative pole, and the 9 seeds. That is
  design-N1.md D3.
- **N2 has a natural candidate on 410m** (23.417) and one with a dated history on 70m (L3N669);
  ROADMAP §7c's "no known neuron in the repo" stands, but not "on any Pythia size" in the field.
- **A pole under GeLU** should be read on the pre-activation (§1 row 1, §2 row 4;
  `tools/math_checks/n1_gelu_pole.py`): design-N1.md "What a pole is under GeLU".

**Not found:** any study reading GeLU neurons' negative poles' jobs across training, or any
neuron-level history placed against Pythia's dated events (the warmup, step 512, step 1000).
**Not read** (primary text): neuron-lineage's code and planted controls, 2509.24198's Pythia
sections, Stolfo's appendix F; U1 reads them (design-N1.md Q2–Q4).
