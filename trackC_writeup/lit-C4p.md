<!-- trackC_writeup/lit-C4p.md -->
# C4′ (a) and (b): literature scan before the design freezes (2026-11-15)

Trigger (`CLAUDE.md` "Literature scans"): C4′ opens (STATE Blocked 158 (a), the user's,
2026-11-15). Scope: (1) does a warmup's length move induction's date, (2) the summed rate as a
clock, (3) comparing checkpoints of two runs by KL. Three web searches (2026-11-15); no full text
read, so every outside row is [S], a search engine's summary only. Not repeated: `lit-C4.md`
(Qu & Liu, Olsson et al., Kalra & Barkeshli), C6's `design-C6.md` (the run itself).

## 1. Warmup length and induction's date

| work | what [S] | bearing |
|---|---|---|
| Qu & Liu, blog, 2026 (in `lit-C4.md`) | the plateau rate decides whether induction forms; at a working plateau a 100-step warmup was fastest; decay barely matters | the only direct test found, one small setup; says warmup length moves the date "modestly", which (a) measures on a released 70m |
| Singh et al. 2404.07129 | three subcircuits interact to give the phase change; data properties set its timing | the timing is not the schedule's alone; (a) holds data and init fixed, so only the schedule differs |
| 2505.17863 | emergence time follows power laws in task, architecture and optimiser | no warmup-only arm |

**Verdict:** no controlled warmup-length intervention dating induction on a pretrained-scale LM
was found. `pythia-70m-warmup01` (same init and first batch as `pythia-70m`, a ten times longer
ramp; C6 U0) is such an intervention, already trained.

## 2. The summed rate as a clock

| work | what [S] | bearing |
|---|---|---|
| Tissue, Wang & Wang 2408.11029 | loss ≈ L0 + A·S1^(−α) − C·S2: S1 the summed rate ("forward area"), S2 an annealing area | the loss, not an event's date, is a function of the summed rate; C4's map is that clock for dates. During a warmup S2 is 0, so on F and A0 to 1430 the law says the loss is a function of S1 alone |
| Luo et al. 2503.12811 (ICLR 2025) | the multi-power law: steps replaced by the summed rate, plus a decay term | same clock; (b) asks the stronger question, whether the whole function (not the loss) is matched |

**Bearing on (b):** these laws predict F and A0 match at equal S1 *in loss*. C4's close-out
already shows that is not enough for events (item 10: induction at near-equal loss but on other
heads). (b)'s KL match is the test of the function, with loss beside.

## 3. Comparing checkpoints by KL

| work | what [S] | bearing |
|---|---|---|
| 2505.15353 | KL between models from log-likelihood vectors; Pythia's trajectory a spiral; KL between checkpoints large during the linear warmup | a scale for (b): early checkpoints differ a lot, so a residual must be read against the grid's own step-to-step KL, not an absolute number |
| 2509.26643 | per-token KL across PolyPythias seeds: a near-uniform phase, sharp convergence, divergence, then slow reconvergence around step 2k that coincides with induction; warmup flagged as a possible artifact | the seeds' KL is (b)'s "another path" scale, and the reconvergence near induction says (b) should be read separately before and after induction's date |

## What the scan changes in the design

1. (b)'s residual is read against two scales on the real design points: A0's own step-to-step
   KL (the grid's resolution) and the seeds' KL at the same step (another init and order). The
   released run against A0 is the planted null (same clock), a seed the planted effect.
2. (b) reports the loss at each match beside the KL (S1 laws predict a loss match).
3. (b) is read in two windows, before and after A0's induction date (2509.26643's phases).

## Parked

1. **2509.26643's full text** (why: its phase dates on PolyPythias 70m could serve as (b)'s
   seed scale directly; cost: one reading; could change: whether (b) needs its own seed KL
   pass). Read before U2's code is written.
