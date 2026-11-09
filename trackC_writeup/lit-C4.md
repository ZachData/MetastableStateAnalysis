<!-- trackC_writeup/lit-C4.md -->
# C4, the schedule intervention: literature scan before its design freezes (2026-11-09)

Trigger (`CLAUDE.md` "Literature scans"): C4 opens (STATE Blocked 153, revised; after C7 (i)'s
close). Scope: (1) whether anyone has moved a training-dynamics event by changing the LR warmup
alone, and what a flat rate does; (2) how to reproduce Pythia's data order and run for 70m;
(3) what is already known on this project's side. Three web searches (2026-11-09); no full text
read, so every outside row is [S], a search engine's summary only. Not repeated: C2 unit 1's
`lit-C2-unit1.md` (warmups of Pythia and Mistral GPT-2 small), C6's `design-C6.md` (the
released `pythia-70m-warmup01`, read off its weights).

## 1. Moving an event by the schedule

| work | what | bearing |
|---|---|---|
| Qu & Liu, blog, 2026 ("Emergence of induction head depends on learning rate schedule") [S] | a small transformer: **a fixed LR did not produce induction heads**, even with extra steps; a warmup to a suitable plateau did, and a 100-step warmup was fastest; a plateau of 6e-3 never formed them, while 6e-4 did; decay barely mattered | the flat arm's main risk: at Pythia's peak (1e-3) without a ramp, the events C2 dates may not happen at all. That is a reading too ("not formed by 4000"), but the design must say so before the run |
| Olsson et al. 2209.11895 [S] | the induction phase change at 2.5–5B tokens, about 1–2 % of training for large models | the centred event at 1000 (2.1B tokens on Pythia) sits in that window |
| Kalra & Barkeshli 2406.09405 (NeurIPS 2024) [S] (added after `/challenge-pr` on #344, finding 6) | warmup's main benefit is tolerating a larger target LR: preconditioned sharpness starts high and causes instability at high LR, and warmup pushes it down. Depending on the target LR and the initial sharpness, warmup can be cut sharply or removed | the flat arm's divergence risk at 1e-3 is real but not certain. U1 measures the loss over F's first 64 steps before F is launched (a "catapult" there is the paper's sign). Two further titles (Kosson et al., "Analyzing & reducing the need for learning rate warmup in GPT training"; Wortsman et al., "Small-scale proxies for large-scale Transformer training instabilities") were not returned by the search; U1 reads them before F launches |
| 2603.29805, 2505.17863 [S] | emergence timing as power laws in task, architecture and optimiser; spectral early warnings | no warmup-only intervention on a pretrained-scale LM found |

**Verdict:** no controlled warmup-only intervention on a Pythia-scale model that dates the
events C2 dates was found. `pythia-70m-warmup01` (C6) is the only other schedule already
trained, and its centred date read "consistent with cumulative" (`status-C.md` "C6 on one page").

## 2. Reproducing Pythia's run

| item | found [S] | bearing |
|---|---|---|
| data order | `EleutherAI/pile-standard-pythia-preshuffled`: 20 `.bin` shards of about 30 GB each, plus `document.idx`; `utils/batch_viewer.py` in the Pythia repo reads batches; one open issue reports an overflow on the *deduped* merged maps | 70m is the standard (non-deduped) Pile; step 4000 is 4000 × 1024 × 2049 tokens ≈ 8.4B, inside shard 0 (≈ 15B), so one shard (≈ 30 GB) is the download, unverified until U1 |
| the recipe | 70m: peak 1e-3, batch 2M tokens (1024 × 2048), 1 % warmup (1430 steps), cosine to 0.1 × peak; checkpoints at 0, 1, 2, 4, …, 512, then every 1000 | A0's schedule; the flat arm changes only the warmup |
| this project | `trackT_timaeus/t5h.py`, a resumable GPU training loop with a known answer, exact in float64 (`status-T.md` "T5 U4a"), 104.8k tokens/s on a 32m model | the harness starts from it; the earlier 70m harness is gone from disk (ROADMAP §8 C4) |

## What the scan changes in the design

1. **A flat arm can fail to form the events at all** (Qu & Liu). The design fixes now that
   "no date by 4000" is a reading ("not formed"), not a refusal, and that a diverging loss is a
   refusal.
2. **The harness is checked against the released run's first steps** (C6's method: the
   first batch's update signs against `pythia-70m`'s), before any arm trains past step 64.
3. **One shard of the preshuffled Pile** is the data, verified in U1 against the released run.

## Parked

1. **Qu & Liu's peak-dependence** (why: their 6e-3 plateau never formed induction heads; cost:
   a second flat arm at half peak, about 2 GPU-days; could change: whether a flat arm that forms
   nothing is read as "the schedule's" or "the peak's"). It is option D1 (b) in the design.
