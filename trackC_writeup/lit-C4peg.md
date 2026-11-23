<!-- trackC_writeup/lit-C4peg.md -->
# C4′ (e) and (g): literature scan before the design freezes (2026-11-23)

Trigger (`CLAUDE.md` "Literature scans"): C4′ (e) and (g) open (STATE Blocked 169, decided by the
user 2026-11-23: "(e) and (g) first", then (d), (f)). Scope: (1) the tokens' common direction and
word frequency, (2) unigram statistics early in training, (3) whether two runs from one init are
linearly connected, and what a warmup does to that, (4) ablating induction heads. Five web searches
(2026-11-23); no full text read, so every outside row is [S], a search engine's summary only. [K]
marks a work cited from prior knowledge, not checked here. Not repeated: `lit-C4.md`, `lit-C4p.md`,
`lit-C4pc.md`, `lit-C5.md` (massive activations, the mean's coordinates).

## 1. The common direction and frequency

| work | what [S] | bearing |
|---|---|---|
| 2609.02959, "The Direction of Ignorance" | a unigram prior is encoded along one unembedding direction, found by least squares of log-frequencies on the unembedding; scored by a normalised KL (0 exact, 1 uniform); learned early in training, across families | (e)'s planted effect: that least-squares direction is the reference a mean direction is compared with through W_U (not an upper bound on its rank correlation). Its "early" is (e)'s question on two schedules |
| Kobayashi et al. 2305.18294 | GPT-2's prediction head (LN bias, W_U) handles word frequency, promoting frequent tokens | the final LN's bias alone can write a unigram prior; (e) reads it separately from the mean |
| Biś et al., NAACL 2021 | anisotropy is a drift in a common direction, driven by rare tokens' updates through the softmax | a mechanism in the embedding; (e)'s embedding share at state 0 tests whether the mean starts there |
| 2306.07656; Godey et al. 2401.12143 | drift does not explain transformers' anisotropy; it is tied to attention's sharpness | the alternative: attention writes the common direction. (e)'s component split separates the two |
| LessWrong, "uncertainty vector" (GPT-2 scale) | one residual direction makes the output ≈ the unigram distribution; an lm_head bias would remove the pressure for it | Pythia's `embed_out` has no bias (checked: `tie_word_embeddings` false, no bias), so the residual must carry it |

## 2. Unigram first in training

| work | what [S] | bearing |
|---|---|---|
| Belrose et al. 2402.04362 | Pythia learns statistics of increasing order: unigram, then bigram | the unigram phase is early; whether it sits at V's falls on 70m is (e)'s reading |
| 2510.24963 | Pythia's token probabilities correlate first with unigram, then bigram, then higher n-grams; consistent across architecture, data, scale | the same ordering, as behaviour; (e) reads the state, not the output |

## 3. Linear connectivity from one init

| work | what [S] | bearing |
|---|---|---|
| Frankle et al. 1912.05671 | two runs from one state, differing in data order, are linearly connected once training is past an early point (vision) | (g)'s replicate pairs, (A0, A0′) and (F, F′), are this test on each schedule |
| Altıntaş et al. 2023 ("Disentangling LMC") | Adam breaks connectivity more than SGD; adding warmup or lowering the rate restores it; a high rate sends different orders to different regions | **a lean stated before output**: the barrier between F and F′ (no warmup) exceeds that between A0 and A0′ |
| 2210.14102 | pretrained LMs' fine-tuned minima are well connected across data orders | fine-tuning only; no pretraining from scratch |
| Juneja et al. 2205.12411 | linearly connected BERT fine-tunes share a generalisation strategy; disconnected ones differ | the reason (g) puts the barrier beside the ablations: a barrier is read as "another solution", not as worse |
| 2606.23607 | LMC at billion-parameter scale after permutation alignment | (g) does no alignment: F and A0 share init, so no permutation is needed to compare them |

## 4. Induction-head ablation

| work | what | bearing |
|---|---|---|
| Olsson et al. 2022 [K] | ablating induction heads removes in-context learning on repeated sequences | (g)'s measure: the loss on the second copy of repeated random tokens, with each arm's heads ablated |
| Singh et al. 2404.07129 [S] | induction is three interacting subcircuits; ablations through training | ablating the heads ≥ 0.5 removes one subcircuit's members, not the previous-token heads; (g) says so |

## What the scan changes in the design

1. (e)'s alignment is read through W_U against the least-squares unigram direction (§1) as the planted
   effect and random directions as the null, at each checkpoint (W_U trains too).
2. The final LN's bias is a separate component (Kobayashi et al.), never pooled into the mean.
3. (g) reads the barrier for both replicate pairs, so Altıntaş et al.'s warmup claim is a lean on
   pairs C4 already trained, stated here before any interpolation.

## Parked

1. **2609.02959's full text** (why: its normalised KL may be the better statistic than a rank
   correlation, and its "early" may name a step on Pythia; cost: one reading; could change: (e)'s
   primary statistic before U1's code is written). Read before U1's code.
