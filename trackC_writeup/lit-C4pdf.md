<!-- trackC_writeup/lit-C4pdf.md -->
# C4′ (d) and (f): literature scan before the design freezes (2026-11-23)

Trigger (`CLAUDE.md` "Literature scans"): C4′ (d) and (f) open (STATE Blocked 169, decided by the
user 2026-11-23: "(e) and (g) first", then (d), (f); (e)+(g) closed in #370). Scope: (1) whether
an early-training transient has a threshold in the learning rate, (2) whether the time to form
induction heads scales with the rate, (3) what a sudden rise in the rate does to a trained state
under Adam, (4) what restarting Adam's moments does. Six web searches (2026-11-23). No full text
was read, so every outside row is [S], a search engine's summary only. [K] marks a work cited from
prior knowledge, not checked here. Not repeated: `lit-C4.md` (Qu & Liu, Kalra & Barkeshli
2406.09405), `lit-C4p.md`, `lit-C4pc.md` (Kosson et al. 2410.23922, read at (c)'s close-out),
`lit-C4peg.md`.

## 1. A threshold in the rate early in training

| work | what [S] | bearing |
|---|---|---|
| Lewkowycz et al. 2003.02218 (the catapult mechanism) | three regimes set by the rate against the top NTK eigenvalue **at initialisation**: lazy below 2/λ₀, a catapult (an early loss spike, then a flatter region) between 2/λ₀ and a maximum, divergence above. Proved for wide two-layer linear nets under MSE; carried to deep nets by experiment | a lean for (d): the transient is a threshold in the rate, set by the early state's curvature. It names no dose clock. An Adam, cross-entropy transformer is far from its setting |
| Kalra & Barkeshli 2311.02076 (universal sharpness dynamics) | sharpness can spike in the first steps and fall at once ("sharpness catapult"); a phase diagram of early training over rate, depth and width | the same lean; it also says the spike is an early-steps event, so it bears on (f) (a later state may not spike) |
| Zhai et al. 2303.06296 (attention entropy collapse) | raising the rate readily collapses attention entropy and can diverge training; interventions during warmup give a sharp entropy drop in the first block | F's transient sits in blocks 1–5 (T1–T5), not in block 0, so this names a candidate (attention collapsing early at a high rate), not the transient. Not read for (d) |
| Qi et al. 2505.21910 (training transformers without warmup) | without warmup, rank collapse, entropy collapse and gradient explosion "especially during the initial training stage"; bounding the update against the weight's largest singular value removes the need | the transient's mechanism-level reference after Kosson et al.: early updates too large relative to the weights. That reading predicts a threshold in the update size, which under Adam is the rate |

## 2. Induction's date and the rate

| work | what [S] | bearing |
|---|---|---|
| 2511.01033 (the emergence of induction heads for in-context learning) | emergence time bounded tightly, quadratic in the context length | a clock in steps for a fixed setting; the rate's role is not reported in the summary |
| 2511.16893 (already in `lit-C4pc.md`) | batch size and context length predict the emergence point, independent of model size | the same: nothing on the rate |
| 2410.11474 (two-layer transformer, four phases) | a plateau, then emergence, set by a time-scale separation in the parameters | under gradient flow a plateau's length scales as 1/rate [K]; with Adam and a fixed rate that is the dose clock d2 tests |

No search returned a test of induction's date against the rate at a fixed schedule shape. Qu &
Liu (`lit-C4.md`) is the nearest: a flat rate formed no heads in their small model, too high a
plateau none either. **So d2 has no outside lean; it rests on `P-SCH4`** (induction nearer the
summed rate than the counts, held on the replicate pair).

## 3. A sudden rise in the rate on a trained state

| work | what [S] | bearing |
|---|---|---|
| Gupta et al. 2308.04014 (continual pre-training: how to (re)warm, on Pythia-410M) | re-warming first raises the loss; **starting directly at the maximum rate gives a large initial loss spike** ("chaotic phase", "stability gap") with no lasting cost | a lean for (f): a jump to the peak on a trained state is itself a disturbance. It is a loss spike after the Pile, not a dated early transient; whether their moments were reset is not in the summary |
| Ibrahim et al. 2403.08763 (simple and scalable continual pre-training) | re-warming plus re-decay plus replay matches retraining; **dropping the optimiser's states made no difference**, since β 0.9 and 0.95 forget the old state quickly | (f)'s fresh moments matter for the first tens of updates only. That is exactly the window in which F's transient began (48), so the restart is a confound there, not a negligible one |
| Cohen et al. 2207.14484 (adaptive gradient methods at the edge of stability) | Adam's preconditioned sharpness settles near 38/η (β₁ 0.9); instability inflates the second moment, which lets it move on | a lean for (f): a fivefold rise in η (A0's 0.18 of the peak at 256 to the peak) puts the state far past its threshold, so a spike is expected at any state on the edge. Whether A0 at 256 is on the edge is not known |
| 2604.14108, 2601.16979 | a mid-training rise in the rate acts as a catapult, then training restabilises near the new threshold; critical sharpness tracked on OLMo-2 through mid-training | the same lean, at scale; neither dates a representation-level transient |

## 4. Restarting Adam

| work | what [S] | bearing |
|---|---|---|
| SPAM 2501.06842 (ICLR 2025) | periodic momentum resets with spike-aware clipping stabilise LLM pre-training, 60M–1B | a reset is not by itself destabilising at a modest rate; supports (f)'s own-rate branch as a control near zero |
| Molybog et al. 2304.09871 | loss spikes from Adam's interaction with the state and the batches | background only |

## What the scan changes in the design

1. **Two leans for (f) that point opposite ways, stated before output.** The catapult line puts the
   threshold in the *early* state's curvature: a trained state at 256 has lower sharpness, so a
   jump to the peak there need not repeat F's transient (f: "the state's"). Gupta et al. and the
   adaptive-edge line say a jump to the peak disturbs a trained state anyway (f: "the rate's"). The
   design reads which, and does not take a side in advance.
2. **The restart is a confound in exactly F's window** (Ibrahim et al.: fresh moments matter for
   tens of updates; F's transient began at 48). So (f) keeps an own-rate branch with the same
   restart as its control, and compares the peak branch with F (also a fresh Adam at the peak),
   which changes only the state.
3. **The transient's rule reads a threshold or a dose, not a mechanism.** None of the works dates a
   representation-level transient by step, so "threshold" in (d) means "absent at a rate where the
   dose clock places it", nothing finer.

## Parked

1. **Gupta et al. 2308.04014's full text** (why: whether their direct-to-peak spike was with fresh
   or carried moments, and how many steps it lasted on Pythia-410M; cost: one reading; could change:
   (f)'s wording, and whether the kept-moment branch (D2 (c)) is worth its hours). Read before U3's
   reading, so its answer cannot be fitted to (f)'s result.
