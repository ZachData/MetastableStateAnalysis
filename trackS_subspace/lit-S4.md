<!-- trackS_subspace/lit-S4.md -->
# Track S — literature and inventory scan before S4's design (2026-11-08)

For `trackS_subspace/design-S4.md` (U0). Opened by Blocked 147 (b), the user's. Web search,
summarising fetches of abstracts and two papers' HTML, and the Hugging Face API (configs, yamls,
file lists; no weights downloaded, no pickle opened). Grades as `docs/lit-N.md`: **read** =
primary text, **summary** = a fetch's summary of a page (not checked line by line), **listing**
= a search result only, **confirmed** = a released config or file list read here. Nothing was
run.

## 1. Released checkpoints that already are S4's arms (confirmed: HF API, 2026-11-08)

| repo | what | grade | bears on |
|---|---|---|---|
| `timaeus/L2` … `L6` | attention-only TransformerLens models, `n_layers` 2–6, d_model 256, 8 heads × 32, ctx 1024, vocab 5000 (`georgeyw/TinyStories-tokenizer-5k`, public), **Shortformer** positions (added to Q and K only, never to the residual stream), `model_seed` 0, data seed 42; AdamW 1e-3, wd 0.05, **`lr_scheduler_type: constant`, `warmup_steps: 0`**, batch 32 × 1024, 75,000 steps; **451 checkpoints** each: every step 1–26, then a log grid (`extra_save_steps`) to 75,000, plus every 300; `checkpoints/checkpoint-N.pt` (pickles; final also `model.safetensors`, 15.5 MB); data task `pile13m` | confirmed (`model.yaml`, `training.yaml`, `data.yaml`, file list) | **the untied arm at five depths, for free**; no warmup, so a break here cannot be the warmup's (C2's F1 does not arise); depths ≥ 3 carry B7's read |
| `timaeus/L1` | empty repo (0 files) | confirmed | no 1-layer member of the series |
| `timaeus/L1W256-dense` | 5,003 files, `checkpoints_early/checkpoint_00000N0.pth` every 10 steps from 0; no `model.yaml` | confirmed (files); its config and schedule **unknown** | a 1-layer model cannot form an induction head (Olsson et al.: K-composition needs a second layer): the architectural "induction cannot form" arm, if its recipe is close enough |
| `timaeus/tetrahedron-3m-og` | 2 layers, attention-only, 501 checkpoints every 100 to 50,000; 2402.02364's model (leaning, `trackT_timaeus/status-T.md` "T1" Q2) | confirmed (files) | a second 2-layer run on a different recipe |
| `timaeus/pile13m` | the L-series' data: **401, private** | confirmed (HTTP) | our trained arms cannot replay the series' stream; `georgeyw/pile-13m-space-dedupe` (public; relation **unknown**) and `timaeus/dsir-pile-10m` (public, tetrahedron's) are the candidates |
| `KeremSahin2210/vanilla_1b`, `hapax_1b` (and `hapax_1b_thresh_0.3`, `vanilla_2.8b`, `hapax_2.8b`) | §2 row 1's models: GPT-NeoX, **16 layers, 2048 wide, 8 heads, rotary 0.25, parallel residual, vocab 50,304: Pythia-1b's shape**; fp16; `step=100` … `step=19900` every 100 (199 checkpoints; **no step 0**), 2.0 GB each | confirmed (`config.json`, file list) | **a Pythia-shaped causal arm**: the same model and data with and without the loss on induction-predictable tokens, on a grid every 100 steps (denser than Pythia's inside C2's (128, 512] break window) |

## 2. Papers

| source | what it is | grade | bears on |
|---|---|---|---|
| Sahin, Feucht, Belfki, Brinkmann, Mueller, Bau, Wendler, 2511.05743 (EMNLP 2026), "In-Context Learning Beyond Copying" (HAPAX) | the loss at position j is dropped if some earlier i has (x_{i−1}, x_i) = (x_{j−1}, x_j) ("positions where the correct token can be retrieved by copying the continuation of an earlier matching bigram"; single-token repeats not masked; Eq. 3): 31.7 % of 41.7B tokens. 1B GPT-NeoX "with the same hyperparameters and configuration as Pythia", the Pile, 19,900 steps, checkpoints every 100; seed 1234, two more seeds (bf16, 18,000 steps). **Hapax: "fewer and weaker induction heads", "the absence of the characteristic phase changes"**; extra seeds: Vanilla 8 and 9 of 128 heads above 80 % prefix-matching, Hapax 1 and 2. Code `github.com/KeremSahin22/icl-without-copying`, weights `huggingface.co/KeremSahin2210` | read (§3.1 via the HTML; the hyperparameter appendix not found in the part read: LR, warmup, data order **unknown**) | **the "without repeats" arm as a published intervention**: the inputs are unchanged, only the gradient for copying is removed. Induction is weakened, not removed |
| Gopalani, Hu, 2506.13688 (NeurIPS 2025), "What Happens During the Loss Plateau?" | shallow transformers on algorithmic tasks: during the plateau a repetition bias and "internal representation collapse, where hidden states across different tokens become nearly parallel" (mean pairwise cosine, last layer); attention progress precedes the drop. **On Pythia (ARC-Easy prompts, last layer): cosine 0.4–0.65 at init, > 0.9 within a few steps, "followed by a decrease around step 1000"**; OLMo-2 7B 0.93 at step 150, 0.43 at 600 | summary (HTML, sections named) | **C2's mean term**: a near-parallel cloud has large ‖m‖², and its fall around 1000 is C2's centred date's neighbourhood. Not cited in `paper/` (grep, 2026-11-08). Parked 1 |
| Aoyama, Wilcox, Schneider, 2511.16893 (ICML 2026) | when induction heads form: an expression in batch × context size, independent of model size; bigram repetition frequency and reliability set a boundary | summary (already in `p7_motifs/lit-7.md`, `docs/lit-T.md`) | the toy's induction step is set by batch and context, not depth; the L-series share both, so their induction steps should be close |
| Olsson et al. 2022, 2209.11895; Elhage et al. 2021 | induction needs two layers (K-composition with a previous-token head); the bump; prefix-matching score on repeated random tokens; ICL score (loss at token 500 minus 50) | read before (`docs/lit-T.md` row 83) | the induction instruments; the 1-layer arm's premise |
| Hoogland et al. 2402.02364 | the 2-layer attention-only LM's stages (bigram → n-gram → previous-token → induction) with the LLC | summary (`trackT_timaeus/status-T.md` "T1") | the toy's developmental order, against which the energy's dates are laid |
| 2601.09588 (looped transformers, energy–entropy regularisation) | looped (weight-tied) models solve an induction task with two heads; a single head trains poorly without a regulariser | summary | the tied arm may not form induction at all: a risk U2 measures |
| Blayney et al. 2604.11791 (looped reasoning LMs) | each layer of a looped block converges to its own fixed point (a cyclic trajectory); attention stabilises across recurrences | summary (abstract) | what the tied arm's layer series may look like: a cycle, not a monotone flow |
| 2511.01033 | a 2-attention-layer + linear model's weight structure for induction | listing | U3's circuit reading |
| Zucchet et al. 2505.17863 | repetition speeds sparse attention and the induction head's emergence | listing | beside |

## 3. What the scan changes

- **S4's first unit needs no training.** The untied arm exists at depths 2–6 on one recipe with
  no warmup, 451 checkpoints each, 15.5 MB a checkpoint. The question "is there a break to move
  in a toy?" is a forward-pass read on them, and it should come before any training is paid for.
- **A Pythia-shaped "induction suppressed" arm exists** (HAPAX's pair at 1B). It is closer to C2's
  question than any toy. But Hapax weakens induction rather than removing it, its schedule is not
  yet read, it has no step 0, and C2 dated the break on 410m, not 1B. This is design-S4.md D1.
- **"Without repeats" has a published form.** It removes the loss on bigram-predictable tokens and
  keeps the inputs unchanged. Removing repeats from natural text cannot be done; removing every
  repeated token would change the inputs. This is design-S4.md D2.
- **The tied arm is not released anywhere found**, and looped models may not form induction
  (2601.09588), so it is the arm S4 has to train.

**Not found:** any measurement of E_β, or of a layer-monotonicity count, on a toy induction model
across training, or on HAPAX's pair. **Not read** (primary text): HAPAX's hyperparameter appendix
and code, `L1W256-dense`'s recipe, 2506.13688's Pythia figure (which steps, which layer).

## Parked (why / cost / which decision it could change)

1. 2506.13688's Pythia result (last-layer cosine near 1 early, falling around step 1000) is not in
   C2's `paper/`. It bears on §8's related work and on F-fences about the mean term. Cost: one
   read of its §6 and figure 8. It could change §8's wording and whether "the mean's fall at
   1000" is new. **Carried to C2's rewrite** (status-C "For the user's rewrite").
2. HAPAX's pair as a C2 arm in its own right (does the centred date move when induction is
   suppressed?). It is in S4's U1 under D1 (a). If the user takes D1 (b), it is **carried to
   ROADMAP C5**.
