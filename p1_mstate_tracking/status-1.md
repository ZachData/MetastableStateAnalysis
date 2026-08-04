# Phase 1 — STATUS

**Last verified:** Pythia-410M checkpoint pilot (execution-order item 8), cross-run report
`llm_cross_run_report.txt`.
**Supersedes:** the 2026-04-23 GPT-2/BERT/ALBERT run (`results/2026-04-23_18-30-06`). That
run's verdict table is retained below only where the Pythia result speaks to the same
prediction; where the two disagree, the Pythia result is stated as the current finding and
the GPT-2 finding is marked as not reproducing rather than silently dropped.

**Scope of this run:** 27 checkpoints × 8 prompts = 216 metastability runs, plus 27
`repeated_tokens` collapse controls. One model size (`pythia-410m`), 25 analyzed layers
(embeddings + 24 blocks). **No random baseline and no 1.4B run.** This pilot therefore
adjudicates PREDICTIONS.md claims (a) and (b) only; claim (c), the hard-stop gate, is
untouched and still pending.

**Overall:** Falsification criterion passes again — plateaus are present in all 216 runs, so
metastability survives on GPT-NeoX as it did on GPT-2. Beyond that, the headline result is
new and is not a replication: the metastability signature is not a fixed property of the
trained network but a **trajectory** with four separable transitions, two of which land in
the window PREDICTIONS.md nominated and two of which do not. Several of the quantities the
verdict table is built on are measured in the wrong frame (see *Measurement defects*), and
three rows below cannot be closed until that is fixed.

---

## The developmental arc

Aggregates over the 8 metastability prompts at each checkpoint.

| step | MaxMass | MinRank (raw) | Σ viol @β=1 | mean severity | Fiedler dev | plateau onset |
|---|---|---|---|---|---|---|
| 0–4 | 0.016 | 13.5 | 1 | 0.000 | +0.0006 | weight-level (SD 0.00) |
| 8 | 0.016 | 6.5 | 0 | 0.000 | +0.0006 | weight-level |
| 16–32 | 0.51–0.58 | 2.1 | 0 | 0.000 | +0.0007 | weight-level |
| 64–256 | 0.09–0.19 | 4.0–7.0 | 0 → 21 | 0.004 | +0.0004 | weight-level |
| 512 | 0.019 | 11.4 | 64 | 0.036 | +0.0001 | **content-driven (SD 3.31)** |
| 1k–5k | 0.016 | **27.9–40.4** | 62–67 | 0.041 | +0.001 → −0.009 | content-driven |
| 7k–19k | 0.016 | 27.8 → 9.6 | 65–79 | 0.065 → 0.131 | −0.012 → −0.018 | content-driven |
| 40k–143k | 0.017–0.021 | 4.7 → 2.3 | 69–83 | 0.170 → 0.101 | −0.022 → −0.026 | content-driven |

Four transitions, at four different times:

1. **Step 8→16 — transient late-layer collapse.** Raw effective rank 6.5→2.1; max IP-mass
   near 1 jumps 0.016→0.58. Confined to the top of the stack (step 32 / `homer_iliad`: mass
   plateau at layers 21–23, mean 0.536, while layers 0–12 sit at 0.016). Fully recovered by
   step 512. Unpredicted, and not in the provisional anchor list.
2. **Step 256→512 — the energy break.** First violations at 256; 21→64 in a single interval.
   Severity then climbs monotonically to step 40k.
3. **Step 512 — plateau onset flips from weight-level to content-driven.** SD 0.00 → 3.31 in
   one interval.
4. **Step 1000→3000 — Fiedler deviation crosses zero** and stays negative, saturating near
   −0.023 by step 40k.

Effective rank peaks at steps 3000–5000 (mean 40.4, individual runs to 60.4) and then falls
monotonically for the remaining 140k steps. That arc — collapse, recovery, overshoot, slow
decline — is the phase's main new object, and it is not visible from any single checkpoint.

---

## Verdict table

| Prediction | Pythia-410M result | Status vs. GPT-2 run |
|---|---|---|
| Tokens cluster over layers | Confirmed, all 216 runs | Reproduces |
| Metastable plateaus exist | Confirmed, all 216 runs | Reproduces |
| Monotone energy $E_\beta$ (Thm 3.4) | **Holds at init** (3 violations across 8 prompts at every β), **broken by training from step 256** | **Reverses.** GPT-2 run recorded "falsified universally, including under random weights" |
| Higher β → stronger metastability | **Not reproduced.** Violation counts are β-independent after step 512. A β gradient exists only at steps 128–256, and its direction is *higher β → fewer violations* (43/33/22/6) | Does not reproduce |
| Higher d → faster convergence (Thm 6.1) | Unsupported; nothing dimensional tracks | Unchanged |
| Metastability is architecture-, not weight-determined | **Contradicted for plateau onset.** Weight-level (SD 0.00) through step 256, content-driven from step 512 onward | Reverses for this quantity |
| Degenerate input collapses | **Reversed by training.** `repeated_tokens` final-layer mass 0.948 at init → 0.379 at step 143000; rank 1.11 → 2.02. Onset ~step 11k–13k (0.718 → 0.335) | New |
| Two-timescale dynamics | Not assessable — the two-timescale ratio is only computed on the collapse controls, and collapse onset is layer 0 in all 27 | Open |
| Effective-rank collapse | **Pending.** Measured in raw mode, which mixes directional collapse with residual-stream norm growth (defect D1) | Blocked on re-report |
| Per-head Fiedler classification | **Vacuous.** All 432 head-rows read STABLE-CLUSTER by construction (defect D2) | Blocked on rerun |
| Cluster-merge counting via spectral $k$ | **Dead metric.** $k = 1.0000$ in all 216 runs at every plateau layer (defect D4) | Does not transfer |

Additional findings not on the original prediction list:

- **Cluster carrying capacity is invariant; turnover is not.** Max-alive holds at 50–55
  across all 27 checkpoints while mean lifespan falls 7.0 → 4.5 and births rise 113 → 164.
- **Mid-network mass drops below the embedding floor.** Step 143000 / `wiki_paragraph`:
  plateaus at layers 9–14 with mean 0.0007, against a layer-0 duplicate-token value of
  0.0149 — a factor of 20. The trained model separates even identical tokens by mid-depth,
  consistent with the `repeated_tokens` result.
- **Nesting retreats to layer 0.** Multi-scale nesting is detected at layers 0–4 early and at
  layer 0 only from step 7000 on.
- **Neighbour structure decouples from embedding geometry.** `ext_sem_frac` 0.81 (init) →
  0.64 (step 5000) → 0.67 (final). This is the old blocker #3, now quantified across a full
  trajectory — but the reference frame is itself checkpoint-dependent (defect D6).

---

## PREDICTIONS.md adjudication

**Claim (a) — collapse-resistance is learned, not initial. → Split.**

| Sub-prediction | Result |
|---|---|
| Monotone energy at steps 0 and 8 | **Confirmed**, cleanly. β-independent |
| Rank collapse at steps 0 and 8 | **Not confirmed.** Raw rank is 13.5 at step 0 and 6.5 at step 8. Collapse to ~2.1 arrives at step **16**, after eight optimizer steps — early, but not initial |
| High stationary Fiedler at steps 0 and 8 | **Unadjudicable.** The report emits only the mask-baseline deviation, never raw λ₂ (defect D2). Deviation ≈ 0 at init is *consistent* with "λ₂ at the mask baseline," but the raw number is not in the artifact |

The claim's failure reading — "resistance is partly architectural/init-borne" — is the wrong
frame for what happened. Energy monotonicity is not init-borne at all; it is destroyed by
training. Rank collapse is not init-borne either, but it appears so early that calling it
"learned" is a stretch. These two components of "random-like" behaviour come apart and should
be tracked separately from here on rather than bundled.

**Claim (b) — resistance emerges at circuit-formation events, ~512–2000. → Mostly confirmed,
one clean falsification.**

- Energy break at 256–512: **inside the window.**
- Fiedler deviation sign change at 1000–3000: **overlaps**, slightly late.
- Plateau-onset content-sensitivity flip at exactly 512: **inside the window**, and a fourth
  marker the claim did not name.
- **Effective-rank transition does not co-locate.** It moves at steps 8–32 and peaks at
  3000–5000. This is its own dynamics, an order of magnitude earlier than the other two.

The claim bundled all three transitions. Two of three hold. The rank signal must be
un-bundled from the energy/Fiedler pair in any restatement.

**Claim (c) — phenomenology transfers across architecture. → Not adjudicated.** No random
baseline and no 1.4B run in this pilot. The hard stop is still armed.

---

## Measurement defects

These are ordered by how much of the verdict table they hold up. D1–D3 must be resolved
before the corresponding rows can be closed.

**D1 — `MinRank` is measured in the wrong frame. (Fix: re-report only.)**
`analysis_p1.py:210` fills `effective_rank` from `effective_rank_from_raw`, and the summary
table's `MinRank` column reads that key. Raw-mode SVD mixes directional collapse with
residual-stream norm growth; a single massive-norm outlier token drives raw effective rank
toward 2 with no directional collapse whatsoever, which is exactly the regime trained
transformers are known to enter. The claim "effective rank collapses to 2.3 by the end of
training" therefore cannot be written as stated.
`effective_rank_normed` — the sphere-spread quantity the theory is actually about — is
computed at `analysis_p1.py:211` and **persisted to `geometry.json`** (`p1_io.py:158`). This
is a report-only fix. No forward passes needed.

**D2 — the per-head Fiedler analysis is both mislabeled and unreproducible. (Fix: rerun.)**
Three separate problems, which together are why the reported values are negative:

- *Mislabel.* The `MeanFiedler` column at `reporting_p1.py:1971` prints `p["mean"]`, which is
  the mean of `cls_vals`, which is the **causal-mask deviation** whenever
  `fiedler_per_head_deviation` is present (`reporting_p1.py:219–222`). λ₂ of a normalized
  Laplacian is non-negative by construction; the deviation is not. Nothing is wrong with the
  eigensolver — the column is carrying a different quantity than its header claims.
  Computed baselines for this run's prompt lengths: λ₂ = 0.0640 (n=242), 0.0654 (n=467),
  0.0658 (n=512), and **0.1089 (n=20)**. So the observed mean deviation of −0.026 at step
  120000 implies a raw λ₂ ≈ 0.039 on the long prompts, and the per-head minimum of −0.0424
  implies λ₂ ≈ 0.023. All non-negative, all far below the baseline. The finding is real —
  trained heads route into more separable graphs than the mask alone forces — it is just not
  a negative Fiedler value.
- *Vacuous thresholds.* CLUSTER/MIXED/MIXING split at 0.3 and 0.7 (`reporting_p1.py:239–244`),
  calibrated for raw λ₂ on [0,1]. On this model the deviations live in ±0.05 **and the raw
  values live in [0.02, 0.07]** — so every head classifies CLUSTER on either quantity. All
  432 head-rows reading STABLE-CLUSTER carries no information about the model. It is not
  evidence for "content-independent heads."
- *Length confound.* `causal_fiedler_baseline(n)` is n-dependent, and the n=20 baseline
  (0.109) is 1.7× the n=512 baseline (0.066) — a spread comparable to the entire signal.
  The cross-prompt mean averages deviations taken against different baselines. At n=20 the
  Sinkhorn fixed point also puts 74.5% of mass on the diagonal, so the short-prompt baseline
  is measuring a mostly-self-loop graph and is not the same object as the long-prompt one.

A fourth, separable problem: **`sinkhorn.json` persists only `fiedler_mean`**
(`p1_io.py:258–273`). `fiedler_per_head`, `fiedler_per_head_deviation`, and
`fiedler_baseline` are never written. `_per_head_fiedler_profile` reads
`sinkhorn["fiedler_per_head"]` and returns `[]` on any reloaded run, so the entire per-head
section silently vanishes when the report is regenerated from artifacts. This section only
exists because the report was written in-session from in-memory results. This is an instance
of the artifact-contract bug class INDEX.md names, and it is why D2 costs a rerun where D1
costs a re-report.

**D3 — the Fiedler layer filter is checkpoint-dependent, with a silent fallback.**
`_per_head_fiedler_profile` excludes layers with raw effective rank < 10
(`reporting_p1.py:170, 208`), then falls back to *all* layers when none qualify
(`reporting_p1.py:212–214`). Raw rank ranges from 40 to 2 across this sweep, so the layer set
entering the Fiedler mean changes with the checkpoint — and the fallback fires silently at
steps 16–32, where every layer is below threshold. The −0.023 saturation curve is confounded
by a moving denominator. Recompute on a fixed layer set, and gate on normed rank if a gate is
kept at all (the docstring's justification — "once tokens collapse to a near-point-mass every
head trivially saturates" — is a statement about directional collapse, so raw mode is the
wrong gate quantity for the same reason as D1).

**D4 — `nMerges` is a dead column.** Spectral $k$ = 1.0000 in all 216 runs at every reported
plateau layer. The eigengap estimator returns a single cluster universally on this
architecture, so the only merges recorded are the trivial `layer 1: k 2→1` and `k 7→1`. Drop
the column. P1-1 cluster tracking already provides real merge counts (25–45 per run) and is
what the merge-location analysis should read from.

**D5 — `MaxMass` is floor-dominated.** Outside the step-16–256 window it equals the layer-0
duplicate-token pair fraction and is essentially constant per prompt across all of training
(`wiki_paragraph` reads 0.0148 at step 0 and 0.0149 at step 143000). The real signal is the
mid-network *minimum* (0.0007), which the max-over-layers reduction discards. Report a
per-layer mass profile, or a min-mass column alongside the max.

**D6 — `ext_sem_frac`'s reference frame trains.** `ext_semantic` is defined by the cosine Gram
of the model's own layer-0/embedding activations against a fixed 0.5 cutoff
(`clustering.py:318–323`). The embedding matrix changes across checkpoints, so the 0.81 →
0.67 decline conflates "deep-layer neighbour structure moves away from embedding space" with
"embedding space moved." Needs a frozen reference — final-checkpoint embeddings, or an
external encoder — before it can be written as a result.

**D7 — two different violation counters, both labelled "violations."** The summary table uses
the relative-tolerance rule (`energy_violation_severity`, `rel_tol = 1e-3`); the ENERGY
MONOTONICITY section prints a raw count. Step 0 reads 1 in one and 3 in the other. Reconcile,
and note that `checkpoint_scalars.py:51` keeps a hand-synced duplicate of the constant.

**D8 — `DEGENERATE_RANK_THRESHOLD = 2` is now biting.** Late checkpoints on
`short_heterogeneous` have raw MinRank 1.06–1.28, so energy and CKA are gated off at those
layers and violation counts are computed over fewer transitions than elsewhere in the table.
`config.py:82` anticipates exactly this moment ("raise to 3 … if post-rerun rank-2 CKA looks
erratic"). Decide the value after D1, since the gate should probably read normed rank.

**D9 — `SINKHORN_MAX_ITER = 100` is hit without a flag.** The n=20 causal baseline needs 232
iterations to reach `SINKHORN_TOL = 1e-6`; at iteration 100 the residual is 4.7e-4. The λ₂
error is negligible for the uniform baseline (0.108894 vs 0.108889 converged), but real
attention is more peaked and converges more slowly, and no per-layer convergence residual is
recorded anywhere. Log the residual.

---

## Blockers / open items

1. **D1, D2, D3** — three verdict rows blocked. D1 is a re-report; D2 needs both a schema fix
   and a rerun; D3 is a re-report once D1 lands.
2. **Claim (c) unadjudicated.** No `pythia-410m-random` / `pythia-1.4b-random` and no 1.4B
   trained checkpoint in this pilot. The hard-stop gate is still pending, and no
   checkpoint-sweep work past item 8 should be treated as cleared.
3. **`step0` and `step1` produce byte-identical output on all 82 lines** where the model name
   appears (`step1` vs `step2` differ on 34). Either the HF `step1` revision resolves to the
   same weights as `step0`, or the loader is caching. Verify with a weight hash before either
   is used as a distinct trajectory point.
4. **Step 8→16 collapse is unexplained** and could be an LR-warmup artifact rather than a
   training event. It is currently resolved by a single interval.
5. **Two-timescale ratio is not measured on the metastability runs**, only on the collapse
   controls, where collapse onset is layer 0 in all 27 and the ratio is therefore degenerate.
   The GPT-2-era "confirmed above a depth threshold" row has no Pythia counterpart yet.
6. **Late-training severity decline is unexplained.** Violation *counts* stay at 69–83 from
   step 19000 on, but mean severity peaks at 0.170 (step 60000) and falls to 0.101 by step
   143000. Count and magnitude come apart; nothing in the current metric set says why.
7. **Final-layer LM-head contamination** — carried over from the GPT-2 run, still not stripped
   from plots. Pythia's untied embedding makes this a different question than it was for
   gpt2-small/medium; not yet checked.

---

## Checkpoint schedule (this pilot's actual job)

Sharpest inter-checkpoint deltas: **8→16** (rank 6.5→2.1, mass 0.016→0.58) and **256→512**
(violations 21→64, plateau-onset SD 0.00→3.31). Both are currently single-interval jumps with
nothing resolving them.

Recommended additions before the 1.4B anchors are fixed:

| Steps | Resolves |
|---|---|
| 10, 12, 24, 48 | Whether the step-8→16 collapse is a training event or an LR-warmup artifact, and where the recovery begins |
| 384, 768 | Separates the energy break from the plateau-onset SD flip — currently confounded in one interval |
| 2000 | The rank peak sits unbracketed between 1000 and 3000 |
| 25000, 30000 | There is a 21k-step gap between 19000 and 40000 across which mean severity moves 0.131 → 0.170 |

The three adaptive slots the v2 plan reserves should go to the first row: the collapse
transient is the only transition in this sweep that was not predicted at all, and it is the
one the current grid resolves worst.

---

## Not yet done

- Items 9–11 (1.4B sweep) remain gated behind the item-6 replication gate, which this pilot
  does **not** satisfy — item 8 tests claim (b), not claim (c).
- The two random baselines (`pythia-1.4b-random`, norm-matched; and true step-0 init as a
  separate developmental object) are still the plan of record. Note that this pilot makes
  step 0 more interesting than "a stand-in for random": energy is monotone there and
  nowhere else, so step 0 is now load-bearing on its own terms.
- `energy_decomposition.py` / `energy_attribution_aggregate.py` still have no Pythia path
  exercised in this run. The parallel-residual decomposition (Δx = attn_out + ffn_out, exact)
  is the natural instrument for open item 6 — attributing the late severity decline to attn
  vs FFN — and is the first thing to build when this phase is next touched.
