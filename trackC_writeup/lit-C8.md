<!-- trackC_writeup/lit-C8.md -->
# C8: literature scan before the design freezes (2026-11-27)

Trigger (`CLAUDE.md` "Literature scans"): C8 opens (ROADMAP §8 C8, the user's, 2026-11-24; opened
2026-11-27 at the user's word, "U0, the literature scan and design ... candidate (i) first").
Scope: (1) loss and gradient-norm spikes early in training, and whether a fixed clip marks or
prevents them (candidate (i)); (2) the adaptive edge-of-stability threshold, its form checked
(ii); (3) the stages of statistics a language model learns (v); (4) collapse then expansion on
Pythia (iii), (v); (5) the LLC off equilibrium (vi). Nine web searches (2026-11-27). **No full
text was read, so every outside row is [S]**, a search engine's summary only. [K] marks prior
knowledge, not checked here. Not repeated: `lit-C4.md` (Kalra & Barkeshli 2406.09405, Qu & Liu),
`lit-C4pdf.md` (Lewkowycz 2003.02218, Kalra & Barkeshli 2311.02076, Zhai 2303.06296, Qi
2505.21910, Gupta 2308.04014 read in full, Ibrahim 2403.08763, Molybog 2304.09871, SPAM
2501.06842), `lit-C4pc.md` (Kosson et al. 2410.23922), `lit-C4peg.md` (Belrose et al. 2402.04362),
`docs/lit-T.md` (2606.22389).

## 1. Loss and gradient-norm spikes (for (i))

| work | what [S] | bearing |
|---|---|---|
| Takase et al. 2312.16903, "Spike No More" (COLM 2025) | loss spikes taken to come from sudden growth of the gradient norm, traced to the sub-layers' Jacobian spectral norms; stability needs small sub-layers and a large shortcut; some spikes at the start of training in a 1.7B run | a loss spike and a grad-norm spike are expected together if the transient is an instability. Pythia is pre-LN with parallel attention and MLP [K], the setting they analyse |
| Wortsman et al. 2309.14322 (ICLR 2024) | small models show large-scale instabilities at high rates: attention-logit growth and output-logit divergence; gradient norms fall with scale and rate, so AdamW's default ε becomes too large | the transient's rate dependence (present at a tenth of the peak) is not the instability regime they describe, which needs a high rate. ε at 1e-8 against 70m's gradients is beside, not read |
| ZClip 2504.02507 | with a fixed clip at 1.0, a 1B LLaMA still spikes and diverges early; an adaptive threshold from recent norms fixes it | a fixed clip at 1.0 does not prevent spikes, so the clip's activity in the window is a fact to report, not a trigger by itself |
| Peri-LN 2502.02732 | pre-LN spikes irregularly whatever the rate, most often early | the logs will hold early spikes unrelated to the transient: an arm's own background rate is needed beside its window |
| 2508.12837 (in-context n-grams; sub-n-grams near-stationary) | gradient norms stay low on a plateau and **spike at the jump between plateaus**; on escape a head learns to attend to the (−2)-token | an outside lean for (i): if the transient is a stage change, the grad norm rises at its onset |
| Gopalani & Hu 2506.13688 (NeurIPS 2025) | the right attention pattern is learned gradually during the plateau, before the loss drops; intervening on attention changes the plateau's length | for (iii): the previous-token heads' date can lead the loss, so the loss alone may be silent at a representation event |
| Liu et al. 1908.03265 (RAdam); Ma & Yarats 1910.04209 | the adaptive rate's variance is large in the first updates; Adam's early updates are irregular even at a minimum; untuned linear warmup matches RAdam | for (iv): fresh moments give large, irregular first updates. Both are from init, so neither separates fresh moments from early large steps on a trained state |

## 2. The adaptive edge of stability (for (ii))

| work | what [S] | bearing |
|---|---|---|
| Cohen et al. 2207.14484 (NeurIPS 2023) | full-batch Adam's preconditioned sharpness λ_max(P⁻¹H) rises to and settles at **(2 + 2β₁)/((1 − β₁) η)**, 38/η at β₁ 0.9; the same with the preconditioner frozen | **the form is checked**: the ROADMAP row's ≈ 38/η is exact at β₁ 0.9 (`tools/math_checks/c8_thresholds.py`). "Frozen Adam" settles too, so the preconditioner read at one saved state is enough to place a state against the threshold; that needs v̂ at that state, which the checkpoints do not hold |

The threshold at C8's rates (β₁ 0.9): 3.8e4 at 1e-3 (F, F¹, B_P), 7.6e4 at 5e-4 (F½), 3.8e5 at
1e-4 (F⅒). Cohen et al. is full-batch. C4's batch (1024 × 2048 tokens) is large, but not full
[K], so a crossing is read as a band, not a point (U2's rule).

## 3. Stages of statistics (for (v))

| work | what [S] | bearing |
|---|---|---|
| Edelman et al. 2402.11004 (NeurIPS 2024) | on in-context Markov chains: uniform predictions, then **in-context** unigram statistics, then a sharp move to the in-context bigram solution; the simpler solution delays the right one; a previous-token layer then an induction layer | the stages are in-context statistics of the prompt, not the corpus's. (v)'s lean (ROADMAP: "the collapse is the unigram stage") is about the corpus's unigram (Belrose et al.). So (v) names which it reads, and it measures both: KL to the corpus tables, and the loss on in-context repeats |

## 4. Collapse, then expansion, on Pythia (for (iii), (v))

| work | what [S] | bearing |
|---|---|---|
| Li et al. 2509.23024 (NeurIPS 2025) | three phases on OLMo and Pythia: warmup collapse, entropy-seeking expansion, compression-seeking consolidation | already cited (`draft-C2.md`; `status-C.md` F1: tied to the warmup). F has no warmup and still collapses (er_mid 1.22 at 4), so on F the collapse is not the warmup's |
| Billa 2602.15997 (v4 in `docs/readings/`, the cited title; the search returned v1's, "Anatomy of Capability Emergence") | training starts with a representation collapse to task floors; **collapse propagates top-down** through layers; on Pythia the global pattern replicates, per-task precursors do not | for (iii): each block's relative update in the window gives a top-down or bottom-up order. Beside, not read |

## 5. The LLC off equilibrium (for (vi))

2606.22389 was not found by the search; `docs/lit-T.md` holds it ("LLC off equilibrium, chains
started at each checkpoint; read before T4's design"). Not repeated. (vi) stays conditional on
(ii) and goes to §7b T3/T4 (D3).

## What the scan changes in the design

1. **(i) has two leans that point opposite ways.** If the transient is a stage change, a
   grad-norm spike marks the jump (2508.12837), and an edge-of-stability crossing shows as loss
   rises (Cohen et al. [K]: non-monotone over short spans). But earlier units have already seen
   that the branches' loss spikes come at the restart (257–259) on arms with and without a transient
   (`status-C.md` "unit 3b", "Beside"). That points to a log silent at the onset itself. The
   design states both and reads which.
2. **Each arm's own background spike rate goes beside its window** (Peri-LN: pre-LN spikes early
   whatever the rate). A window event counts only against the chance that every arm's window
   holds one by background.
3. **The clip is reported, not read as a trigger** (ZClip: a fixed clip at 1.0 does not stop
   spikes).
4. **(ii)'s threshold is (2 + 2β₁)/((1 − β₁) η), checked, and it needs v̂ at each state read.**
5. **(v) names its unigram**: the corpus's (Belrose) or the prompt's (Edelman). It reads both.

## Parked

1. **2508.12837's full text** (why: whether its grad-norm spike at the jump survives Adam and a
   clip, and how many steps ahead of the loss it comes; cost: one reading; could change: (i)'s
   lean, not its rule). Read before U1's output is opened, or not at all for U1.
2. **Cohen et al. 2207.14484 §3 (minibatch Adam)** (why: whether the threshold holds at a large
   batch, and how far below it the state sits off the edge; cost: one reading; could change:
   U2's band width). Read before U2's rule freezes.
