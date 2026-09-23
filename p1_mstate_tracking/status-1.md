# Phase 1 — STATUS

<!-- phase-card -->
## Card

- **Question:** Do tokens, read as particles on the sphere moving through depth, pass through metastable multi-cluster states before they collapse, as the identity-weight model predicts, and does that survive learned weights and training?
- **Inputs:** `pythia-410m`: pilot of 27 checkpoints × 8 v1 prompts (battery v1, `1e47918ef77a`) plus the `repeated_tokens` control; the 19-checkpoint sweep (152 dirs). `CLAIM-C` arms: `gpt2-large`, `pythia-1.4b` trained and random, on v1 and on v2 (`06790b90dcfe`), hashed in `claims/audits/claim_c_real_run.json`. The 2026-04-23 GPT-2/BERT/ALBERT run is gone from disk
- **Results:**
  - Plateaus in every pilot run, at the model's width d = 1024, where the paper's own numerics say the metastable band is gone; whether the detector sees the paper's object is open — `status-1.md` "Before the verdict table"
  - A developmental arc with four transitions at four different steps (late-layer collapse 8→16, energy break 256→512, plateau onset turns content-driven at 512, Fiedler sign 1k–3k) — `status-1.md` "The developmental arc"
  - Energy regime attractive at init, repulsive from step 256; reverses the GPT-2 run — `status-1.md` "Verdict table"
  - Claim (a) splits (energy and rank come apart in time); claim (b) holds for two of three markers, rank does not co-locate — `status-1.md` "PREDICTIONS.md adjudication"
  - Cluster carrying capacity invariant across training while turnover rises; `repeated_tokens` collapse is undone by training — `status-1.md` "Verdict table"
  - `CLAIM-C` gate: INSUFFICIENT at chance concordance on the v1 prompts (§3.41); refused on the v2 prompts for want of a tabulated homogeneity correction (§3.46)
  - HDBSCAN output depends on the install: the conda env reproduces the 2026-08-12 sweep exactly and `.venv` does not — `status-1.md` "The arms are all here and the gate still refuses"
- **Superseded / wrong:**
  - Several verdict columns read the wrong quantity (raw-frame rank, per-head Fiedler, dead spectral k, rank gates on raw rank; D1–D10) — `status-1.md` "Measurement defects"
  - "Metastability is an open problem" may be stale (`2410.06833`), and the causal mask is now part of the theory (`2411.04990`) — `lit-1.md` §1.2, `lit-1.md` §1.3
  - The developmental arc is not new: Pythia's warmup-collapse / expansion / consolidation phases are published (`2509.23024`) — `lit-1.md` §2
  - Open item 3 is answered: step0 and step1 are the same weights, so the axis has one point fewer (§3.51)
  - The step-size definition open item 0 relies on (from MATH.md, a file that does not exist) understates the effective integration time several-fold, in the direction that favours "never integrates far enough" — `p1c_frames/status-1c.md` "Findings from implementation, before any data"
  - Cluster counts carry a re-measurement floor: the HDBSCAN partition is not reproducible run to run — `p10_cluster_function/status-10.md` §3
  - The carrying-capacity invariant has a formula now (Lemma C.1) — `p10_cluster_function/math-10.md` §5.4
- **Registry:** `CLAIM-C` e-value, active, real run recorded, INSUFFICIENT, not adjudicated; `CLAIM-A` needs-null, construction specified and not built (`claims/EXPERIMENTS.md`)
- **Depends on:** none
- **Feeds:** 1b, 1c, 2, 2d, 10
- **Open threads:**
  - Effective integration time against the collapse time t*, never measured (`P-gamma2`, Phase 1c A/B) — `status-1.md` "Blockers / open items" item 0
  - Which of the two readings of the d = 1024 plateaus holds (Phase 1c B)
  - Is the step 8→16 collapse a training event or an LR-warmup artifact?
  - Why late-training energy severity falls while violation counts hold
  - Per-head Fiedler is not persisted; D2 needs a rerun
- **After Phase 10:**
  - Re-report D1, D3 and D10 on normed rank, with the rank thresholds re-derived on that scale (free)
  - Attribute the late severity decline to attention vs FFN through the parallel-residual decomposition (free where `sublayer_streams` exist)
  - Fill the checkpoint gaps listed in `status-1.md` "Checkpoint schedule" (forward pass, 410m, all prompts)
  - Extend `CLAIM-C`'s homogeneity calibration to the v2 prompt count and rescore the v2 arms (free, CPU calibration, cost in §3.46; order against the holdout is `docs/PHASE_REVIEW.md` "Open" 3)
  - Build `CLAIM-A`'s null and run it on `pythia-1.4b` steps 0 and 8 (forward pass, two 1.4b checkpoints)
- **Reviewed:** 2026-09-23 · body `c5b6802fd7`
<!-- /phase-card -->

**Registered predictions:** `CLAIM-A` (needs-null — construction specified
below, deliberately not built; decision 2026-09-17) and `CLAIM-C` (e-value —
null built in `replication_gate.py`, calibrated on a known-answer dry run,
**all four required arms produced by 2026-09-19, and the gate still refuses —
on a dead metric rather than a missing arm**; see "E-value audit" below). Their nulls and evidence paths
are `claims/registry.json`; the per-phase view is `claims/EXPERIMENTS.md`.
Nothing in this file is an e-value: the verdict tables below are threshold
comparisons, and the gate that would turn this phase's Pythia sweep into a
p-value has not yet produced one.

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

## Before the verdict table: which regime we are in

The paper's Figure 3 sweeps clustering probability over $(d, \beta)$. At $d = 2$–$8$ there
is a broad band where that probability is strictly between 0 and 1 — the metastable zone.
The band narrows as $d$ grows and is **gone by $d \approx 512$**; the transition is sharp.
The paper attributes this to Theorem 6.9: as $d \to \infty$ every pairwise inner product
concentrates on the single curve $\gamma_\beta(t)$ of Theorem 6.8, leaving no room for a
multi-cluster intermediate state.

Pythia-410M has $d = 1024$ and $n \le 512$, so **every run in this pilot sits in the regime
where the paper's own numerics say metastability disappears.** We find plateaus in all 216.

Two readings, and nothing currently on disk separates them:

1. The plateaus we detect are not the paper's metastability. Our plateau detector fires on
   flatness in cluster-count and IP metrics; the paper's metastability is a specific
   multi-cluster intermediate configuration of the identity-weight dynamics. These may not
   be the same object.
2. The identity-weight concentration argument fails under learned weights. Theorem 6.9's
   hypothesis is $Q^\top K = V = I$; nothing guarantees it survives multi-head attention
   with an FFN.

Both are results. Neither is currently written down anywhere, and the choice between them
is what Phase 1c's $\gamma_\beta$ null model (sub-experiment B) is built to force: if
observed `ip_mean` tracks $\gamma_{\beta_{\rm eff}}(T_{\rm eff}(\ell))$ the concentration
argument holds and reading (1) wins; if it departs, reading (2) does.

**This paragraph governs everything below.** The verdict table's "metastable plateaus exist —
confirmed, all 216 runs" row is not a theorem check. Metastability is **Problem 1** in the
paper, explicitly open, supported by $d=2$ numerics at $\beta = 4$ and $9$ (Fig. 4). The
falsification criterion this phase passes is a test of a conjecture, run outside the
parameter regime where the conjecture's own evidence lives.

---

## Verdict table

Citations below follow `MATH.md` §9. Where a row previously carried a theorem number, the
number was wrong; the corrected referent is given inline.

| Prediction | Pythia-410M result | Status vs. GPT-2 run |
|---|---|---|
| Tokens cluster over layers | Confirmed, all 216 runs | Reproduces |
| Metastable plateaus exist | Confirmed, all 216 runs | Reproduces |
| Which energy regime is each layer in? — eq. (3.6) / Lemma 3.7 | **Attractive at init** (3 violations across 8 prompts at every β), **repulsive from step 256 onward** | **Reverses.** GPT-2 run recorded "falsified universally, including under random weights" |
| Higher β → stronger metastability | **Not reproduced.** Violation counts are β-independent after step 512. A β gradient exists only at steps 128–256, and its direction is *higher β → fewer violations* (43/33/22/6) | Does not reproduce |
| Exponential convergence rate when $d \ge n$ (**Thm 6.3**) | **Untested.** All 8 prompts satisfy $d > n$ ($d{=}1024$, $n \le 512$), so the hypothesis holds everywhere and the rate $\lambda = O(e^{-\beta})$ is a live prediction. Nothing in the current metric set measures a rate | New row |
| Trajectory pinned to $\gamma_\beta(t)$ when $d \gg n$ (**Thm 6.9**) | **Untested.** This is the sharpest available prediction and it is [R]-cost: integrate (6.9) at each run's $(n, \beta_{\rm eff})$ and overlay on `ip_mean`. Phase 1c sub-experiment B | New row |
| ~~Higher d → faster convergence (Thm 6.1)~~ | **Retracted as a prediction.** Theorem 6.1 is qualitative — $d \ge 3$ at any $\beta \ge 0$ implies single-cluster convergence. It makes no dimensional rate claim. The old "unsupported" verdict was testing something the paper does not assert | Withdrawn |
| Metastability is architecture-, not weight-determined | **Contradicted for plateau onset.** Weight-level (SD 0.00) through step 256, content-driven from step 512 onward | Reverses for this quantity |
| Degenerate input collapses | **Reversed by training.** `repeated_tokens` final-layer mass 0.948 at init → 0.379 at step 143000; rank 1.11 → 2.02. Onset ~step 11k–13k (0.718 → 0.335) | New |
| Two-timescale dynamics | Not assessable — the two-timescale ratio is only computed on the collapse controls, and collapse onset is layer 0 in all 27 | Open |
| Effective-rank collapse | **Pending.** Measured in raw mode, which mixes directional collapse with residual-stream norm growth (defect D1) | Blocked on re-report |
| Per-head Fiedler classification | **Vacuous.** All 432 head-rows read STABLE-CLUSTER by construction (defect D2) | Blocked on rerun |
| Cluster-merge counting via spectral $k$ | **Dead metric.** $k = 1.0000$ in all 216 runs at every plateau layer (defect D4) | Does not transfer |

**On the energy row.** It is now phrased as a regime question rather than pass/fail. The
paper proves $dE_\beta/dt \ge 0$ under $Q^\top K = I$, $V = +I_d$; it also states plainly
that $V = -I_d$ makes $E_\beta$ *decrease* along trajectories (§3.2, §9.1). An observed
decrease therefore identifies the repulsive regime — a case the paper treats — rather than
falsifying anything. The sharper condition under learned weights is §3.4's: (SA) is a
gradient flow in the reweighted metric $\langle a,b\rangle_X = \sum_i Z_{\beta,i}\langle
a_i,b_i\rangle$ only when $Q^\top K$ is symmetric **and** $V = Q^\top K$. Heads far from
that condition carry no monotonicity guarantee at all, which converts "the theorem is
violated" into "which heads are outside its hypotheses, and do violations localize there?"
That is prediction **P-M1**, and Phase 2d D1 tests it.

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

## E-value audit, Phase 1 (2026-09-17)

The pass over this phase's two registered predictions (`PROJECT.md` §3.36).
Neither carries an e-value and neither can on the data in the tree.

**`CLAIM-C` — the instrument was complete except for its plumbing, and the
data does not exist.** The gate (`replication_gate.py`) was built and
calibrated on 2026-08-24/25 but nothing assembled its arms from a Phase 1 run
directory, so it had never been pointed at real artifacts.
`tools/score_claim_c.py` now does that — five module-constant arms, the eight
metastability prompts, `repeated_tokens` excluded as the control it is,
artifact hashes and a committed record either way. Run against every Phase 1
run directory on disk it **refuses**: `gpt2-large`, `gpt2-large-random`,
`pythia-1.4b-step143000` and `pythia-1.4b-random` are absent from all of them
(`claims/audits/claim_c_real_run.json`). The tree holds pythia-410m at 19
steps and nothing else; the 2026-04-23 GPT-2 directory cited above no longer
exists. Producing the arms is five `run_1.py` model runs (every one is already
in `MODEL_CONFIGS`, including the norm-matched `pythia-1.4b-random`) and two
HF downloads that `HF_HUB_OFFLINE=1` currently forbids. Order of magnitude
from the 410m sweep's timestamps (~35 min per 9-prompt checkpoint): 6–8 h.
**The hard stop this claim carries was bypassed de facto** — Phase 2's rerun
and Phase 7's 19-step sweep are exactly the "checkpoint-sweep work (items
9–11)" the gate was to precede, and they ran with the gate unrun. That is a
process fact to state, not to relitigate: the sweeps exist, and if the gate
later FAILS-TO-TRANSFER the reading is that they were run on an
architecture whose phenomenology was never shown to match Blog 1's.

**`CLAIM-A` — nothing built, and the 410m reading above cannot be its data.**
The "Split" verdict in the table above is three threshold readings on
pythia-410m steps 0 and 8, already seen. A null designed now and run on the
same artifact would be shaped by that outcome, so the construction must be
calibrated on known-answer inputs and run on a checkpoint pair not yet read —
`pythia-1.4b` steps 0 and 8, which is what the instrument line names and what
`CLAIM-C`'s arms would download anyway. Recommended construction, **not
built** because it is a `design` decision (`CLAUDE.md` scan trigger 2
applies before it lands in the registry): the prediction is a CONJUNCTION of
three criteria, so the valid correction-free combination is the
intersection-union max over three per-criterion one-sided p's, the same
device `CLAIM-C` uses across its metric subsets; exchangeable unit the
prompt, as in `CLAIM-C`; per-criterion null the matched random-weight
baseline the registry already names (`core/nulls.shuffled_dimension_null`),
with the attainable floor checked first — eight prompts under exhaustive
sign-flip gives 2/(2⁸+1) = 0.0078, so the design can reject; the informative-
row floor that bit `CLAIM-C` needs checking per criterion. The table above
also says the three criteria come apart in time (energy ≠ rank ≠ Fiedler),
which an IU max reports as INSUFFICIENT rather than as a partial pass — the
correct reading for a conjunction the data splits.

### Decisions taken, and the first two arms produced (2026-09-17)

**Decisions (user).** `CLAIM-C`: spend the compute, in arm-sized jobs — one
`run_1.py --models <arm>` per invocation, each its own run directory, because
the scorer takes an arm whole from one directory and `run_1.py` cannot resume a
partial one, so a kill costs one arm and never more. `CLAIM-A`: leave
`needs-null`; the construction above is the spec, building it is its own unit
(design decision, lit-scan trigger 2 before registration), and it cannot run
before pythia-1.4b steps 0/8 are on disk. Queued behind `CLAIM-C`'s verdict.

**Produced, in a ~3 h window.** Both reference arms, on all nine prompts
(`repeated_tokens` included in the run, excluded by the scorer):

| arm | run directory | wall time |
|---|---|---|
| `gpt2-large` | `data/phase12/2026-09-17_14-51-40` | 65 min (14:51→15:57) |
| `gpt2-large-random` | `data/phase12/2026-09-17_16-04-54` | 33 min (16:05→16:38) |
| `pythia-1.4b-step143000` | `data/phase12/2026-09-19_09-44-25` | **34 min (09:44→10:18)**, 2026-09-19 |
| `pythia-1.4b-random` | `data/phase12/2026-09-19_10-18-45` | **17 min (10:18→10:35)**, 2026-09-19 |

Per-prompt times are in `data/phase12/claim_c_logs/`. The trained arm ran
~5–7 min per long prompt; the random arm is half that. Both are on the 16-core
CPU in float32. The 6–8 h estimate for all five arms was extrapolated from
410m; measured, the two gpt2-large arms cost 1 h 40 min together. **The 2 h
estimate for a 1.4b arm was also high: the trained one took 34 min** — more
parameters but fewer, cheaper prompt-level analyses than gpt2-large's 36-layer
sweep — so a 3 h window fits all three.

**A trap that cost the first launch of the 1.4b arm (2026-09-19).**
`_pythia_entry` sets `tokenizer_revision: None` deliberately, which means the
tokenizer loads at revision `main` — and `main` was not in the 1.4b cache,
only `step0` and `step143000` were. Under `HF_HUB_OFFLINE=1` the arm died in
seconds with `OSError: We couldn't connect to 'https://huggingface.co'`, which
names no revision and reads like a network fault. pythia-410m has `main`
cached, which is why nothing had hit it before. Fetching the tokenizer at
`main` once, online, fixes it; **any new Pythia size needs the same** before
it can run offline.

**Defect found on the way: `gpt2-large-random` had never loaded.** Its
`MODEL_CONFIGS` entry carried no `hf_repo` / `pretrained_name`, so
`load_model` resolved the Hub repo to the key itself and asked for
`gpt2-large-random`, which does not exist; `run_1.py`'s per-model handler
swallowed it as a protobuf error and the sweep "finished" with zero prompts.
The resolution rule is unchanged since the earliest recorded `core/models.py`,
so this arm has never run through this code path — consistent with its absence
from every run directory on disk. Fixed (`hf_repo: "gpt2-large"`, and
`albert-base-v2-random` likewise) with a smoke-tier test,
`tests/test_random_controls_name_base.py`, that every `random_init` entry
names a base repo of the same model class and that `RANDOM_CONTROLS` agrees.
The fix is not a registry amendment: the arm is still gpt2-large's
architecture, orthogonally re-initialised at seed 0 (144 matrices, 2
embeddings; checksum 105156.8 → 92759.1 in the log).

**The record.** `claims/audits/claim_c_real_run.json` carries **all four
required arms** as of 2026-09-19 — `gpt2-large`, `gpt2-large-random`,
`pythia-1.4b-step143000`, `pythia-1.4b-random`, 8 metastability prompts each,
96 artifact files hashed (`step0` is the sensitivity arm and is not required).
`real_run_record` in the registry stays null: the record is still a refusal,
not a run that produced the statistic. No arm's contrast has been read — the
gate is the reader, and it reads all arms at once.

### The arms are all here and the gate still refuses — on a metric (2026-09-19)

**`REFUSED: no prompt has all six metrics in all four arms`.** Verdict
`INSUFFICIENT`, `hard_stop: True`, `falsified: False`, all eight prompts
dropped for the same reason: `metric 'cluster_count' unavailable in at least
one arm`. Producing the arms was necessary and is not sufficient.

**The blocking metric is dead in every arm, including the two produced on
2026-09-17.** Measured per arm, prompts with a usable series:

| metric | source | gpt2-large | -random | 1.4b-step143000 | 1.4b-random |
|---|---|---|---|---|---|
| `mass_near_1` | `geometry.json: ip_mass_near_1` | 9/9 | 9/9 | 9/9 | 9/9 |
| `effective_rank` | `geometry.json: effective_rank_normed` | 9/9 | 9/9 | 9/9 | 9/9 |
| `cluster_membership` | `clustering.json: 1 - hdbscan.noise_fraction` | 9/9 | 9/9 | 9/9 | 9/9 |
| **`cluster_count`** | `clustering.json: hdbscan.n_clusters` | **0/9** | **0/9** | **0/9** | **0/9** |
| `cka_prev` | `geometry.json: cka_prev` | 8/9 | 8/9 | 8/9 | 8/9 |
| `fiedler_mean` | `sinkhorn.json: fiedler_mean` | 9/9 | 9/9 | 9/9 | 9/9 |

*(`cka_prev`'s missing prompt is `repeated_tokens` in every arm — the collapse
control, which the scorer excludes anyway. It costs nothing.)*

**Why.** `p1_mstate_tracking/clustering.py:33-38` imports the **standalone
`hdbscan` package** and prints "hdbscan not available — skipping HDBSCAN" when
the import fails. It is not installed in `.venv` and **is not named in
`requirements/` at all**, so it has never been installed here; every run in
`data/phase12` was made without it. `n_clusters` is therefore `null` on every
layer of every run, and two of `CLAIM-C`'s six registered metrics come from
that block.

**And only one of the two says so.** The writer emits the HDBSCAN block even
when HDBSCAN did not run — `{"n_clusters": null, "noise_count": 0,
"noise_fraction": 0.0}` — so `cluster_membership = 1 - noise_fraction` is
**exactly 1.0 at every layer of every arm** and passes the gate's availability
test, which asks only "not missing, not all-NaN". A metric that is structurally
constant because it was never computed is indistinguishable, to that test, from
one that was measured and came out flat. Its contrast between any two arms
would be exactly zero. **So if `cluster_count` were dropped, the gate would run
on five metrics of which one is a constant** — the dropped-prompt refusal is
the only thing currently preventing that.

**Resolved 2026-09-19 by installing the package, not by substituting one.**
`scikit-learn` 1.9 ships `sklearn.cluster.HDBSCAN` and would have served, but
swapping implementations changes what the registered metric means, and it was
not necessary: **`hdbscan` 0.8.44 has a cp314 wheel**, installs into `.venv`
under numpy 2.5.2, and runs. It is now named in `requirements/heavy.txt` —
where it never was, which is why its disappearance was silent — and all four
arms are being re-run with it present (~2.5 h at the measured per-arm costs of
65 + 33 + 34 + 17 min, not the 6–8 h originally feared).

**These two metrics are a property of the toolchain, and the toolchain was
never recorded.** Replaying the 2026-08-12 sweep's own activations through
both environments on this machine:

| env | python | hdbscan | scikit-learn | numpy | reproduces 2026-08-12? |
|---|---|---|---|---|---|
| conda `mets` | 3.10.20 | 0.8.41 | 1.7.2 | 2.2.6 | **exactly, at every layer tried** |
| `.venv` | 3.14.7 | 0.8.44 | 1.9.0 | 2.5.2 | no — 45 → 41 clusters at layer 12 of `step11000/sullivan_ballou` |

So `results/2026-08-12_05-01-35` was produced under the conda env, and that is
now checkable rather than remembered. Parameters have not changed since April
and the algorithm is deterministic within one install, so the difference is the
toolchain and nothing else. **Consequences, in the order they bite:** the four
`CLAIM-C` arms must come from ONE install and now do (`.venv`, 0.8.44,
recorded per layer in `clustering.json`'s `impl` / `version` / `params`);
`cluster_count` and `cluster_membership` are **not value-comparable between
`results/2026-08-12` and anything produced in `.venv`**, though nothing
currently compares them; and what makes this checkable at all is the provenance
field, which no artifact written before 2026-09-19 carries.

**The re-run changes the HDBSCAN metrics and nothing else, checked rather than
assumed.** Comparing the 2026-09-17 `gpt2-large` arm against today's re-run of
the same arm, per metric, over the nine prompts: `mass_near_1`,
`effective_rank` and `fiedler_mean` are **bit-identical on 9/9 prompts** and
`cka_prev` on 8/8 (its ninth is `repeated_tokens`, which has none), max
absolute difference **0** in every case. `cluster_membership` differs on 9/9 —
it was the forged constant 1.0 and is now measured — and `cluster_count` has no
old values to compare. So the pipeline is deterministic across the two runs and
the re-run is a clean swap: the old arms and the new ones differ in exactly the
two metrics that were dead, which is what makes it safe to discard the old run
directories rather than keep both.

*(Not in the artifact, and worth adding when nothing is mid-flight: the
`scikit-learn` and `numpy` versions. `pairwise_distances` builds HDBSCAN's
input, so it is part of the fingerprint. The schema was left alone here because
changing it between arms of one gate is the exact inconsistency this section is
about.)*

The alternative that was **not** taken — amend `CLAIM-C` to a five-metric
statistic — is a registry amendment, and it inherits the constant-metric
problem above unless `cluster_membership` goes with it.

### The v2 battery ran too, and the gate refused on its own calibration (2026-09-19)

All four arms re-run on prompt battery v2 (21 prompts), one invocation each:

| arm | run directory | wall time |
|---|---|---|
| `gpt2-large` | `data/phase12/2026-09-19_13-40-48` | 127 min (13:40→15:47) |
| `gpt2-large-random` | `data/phase12/2026-09-19_15-47-26` | 63 min (15:47→16:50) |
| `pythia-1.4b-step143000` | `data/phase12/2026-09-19_16-50-41` | 63 min (16:50→17:53) |
| `pythia-1.4b-random` | `data/phase12/2026-09-19_17-53-27` | 38 min (17:53→18:31) |

**4 h 51 total, against the ~2½ h projected** — 21 prompts rather than 9, and
the v2 additions are longer on average than v1's mix (v1 carried a
115-character prompt and the repeated-token control; v2's twelve are all
1000–1950). HDBSCAN on a precomputed distance matrix is the part that scales
worst with token count.

**Scored, and refused for a third distinct reason:**

    REFUSED: no homogeneity correction is available, and the correction is
    what enters the e-value: no calibration curve is tabulated for 20 prompts
    (tabulated: [6, 7, 8, 9, 10, 11, 12]).

`INSUFFICIENT`, `hard_stop: true`, `falsified: false`, no p. **20 prompts, 120
cells, 54 concordant, sign homogeneity 0.875, 240 artifact files hashed** — the
table is complete and the floor is no longer the binding constraint. What binds
now is that `tools/calibrate_claim_c_homogeneity.py` tabulates the correction
only to twelve prompts, because its own comment says the upper end was "generous
against the eight metastability prompts". **Extending the battery invalidated
that assumption**, and the gate refuses rather than reporting an uncorrected p —
correctly, since the uncorrected null is already measured to be anticonservative
when the prompt sign-rows agree, which at 0.875 they largely do.

**The three refusals in order, because they are a sequence and not a repetition:**
arms absent → a metric dead in every arm (HDBSCAN) → the floor unreachable at
eight prompts → **the correction untabulated at twenty**. Each one was fixed and
revealed the next. None was a fact about the phenomenology.

**The fix is prescribed by the gate itself** — "extend `N_PROMPTS_TABULATED`
and regenerate rather than running uncorrected" — and it is a calibration job,
not a measurement one. Measured cost on this machine (400-draw probe,
extrapolated to the tool's own per-count budget):

| row | full budget | projected |
|---|---|---|
| n = 12 | 40 000 draws | ~6 min |
| n = 20 | 40 000 draws | **~45 min** |
| regenerating rows 6–12 | as stored | ~40 min |
| contiguous 13–20 | — | **~3 h** on top of the regen |

Per-count seeds are `seed + 1000 * i` on the **index** in
`N_PROMPTS_TABULATED`, so appending counts after 12 leaves rows 6–12 with
their original seeds; a regeneration must reproduce them byte-identically, and
if it does not, that is a finding about drift in the gate rather than about
this run. **Decision (2026-09-19, user): not tonight.** The record stands as a
refusal and the extension waits.

### The gate ran, for the first time: INSUFFICIENT at chance concordance (2026-09-19)

All four required arms re-run with HDBSCAN present — `gpt2-large` (61 min),
`gpt2-large-random` (30), `pythia-1.4b-step143000` (33), `pythia-1.4b-random`
(17), 2 h 20 in total — and scored together, one install, 96 artifact files
hashed. **Every prompt is now usable and none is dropped**; the six metrics
exist in all four arms. `claims/audits/claim_c_real_run.json` is the record and
`real_run_record` in the registry now points at it: the gate has been run on
real checkpoints, which is what that field is for, even though it emitted no
p for the conjunction.

**Verdict `INSUFFICIENT`, `hard_stop: true`, `falsified: false`, `p_value:
null`.** Two separate reasons, and they say different things:

| subset | concordant / cells | informative rows | smallest p this table can express | p (transfers) | p (inversion) |
|---|---|---|---|---|---|
| all six metrics | 23/48 | **4 of 8** | **0.0661** | 0.7510 | 0.5019 |
| drop `mass_near_1` | 16/40 | 8 of 8 | 0.0078 | 0.9377 | 0.1167 |
| drop `effective_rank` | 22/40 | 8 of 8 | 0.0078 | 0.2568 | 0.8872 |
| drop `cluster_membership` | 18/40 | 8 of 8 | 0.0078 | 0.8872 | 0.2568 |
| drop `cluster_count` | 21/40 | 8 of 8 | 0.0078 | 0.4202 | 0.7237 |
| drop `cka_prev` | 23/40 | 8 of 8 | 0.0078 | 0.2529 | 0.8872 |
| drop `fiedler_mean` | 15/40 | 8 of 8 | 0.0078 | 0.9572 | 0.0778 |

1. **The full-set row hits the informative-row floor** the registry warned
   about. Six metrics is an EVEN number of cells per prompt, and four of the
   eight prompts split exactly 3–3; such a row contributes the same number to
   the observed sum and to all 256 null patterns, so it is enumerated without
   ever being counted. With four movable rows the smallest expressible p is
   0.0661, above α = 0.05 — the design's own floor over eight prompts is
   0.0078 and this table cannot reach it. **Every leave-one-out subset has
   five metrics, an odd count, so no row can tie and all eight are
   informative** — which is why they all reach 0.0078.
2. **Where the design can express a small p, the data is nowhere near one.**
   The six leave-one-out p-values run 0.2529–0.9572 in the transfer direction
   and 0.0778–0.8872 in the inversion direction; nothing clears α in either
   tail. Overall concordance is **23/48 = 47.9%**, which is the coin.

**Per-metric, and this is the substance:** concordance across the eight
prompts is `fiedler_mean` **8/8**, `mass_near_1` **7/8**, `cluster_membership`
5/8, `cluster_count` 2/8, `effective_rank` **1/8**, `cka_prev` **0/8**. So the
registered question — does the trained-minus-random contrast transfer from
`gpt2-large` to `pythia-1.4b` — has no single answer: two metrics transfer
almost perfectly, two invert almost perfectly, two sit in between. A
conjunction that demands unanimity across leave-one-out subsets reports
INSUFFICIENT on exactly this shape, which is the gate working as designed
rather than failing.

**Read as diagnostics, not as an adjudication.** `claims/adjudications/` is
untouched, `--adjudicate` was not passed, and the per-metric table above must
not be used to re-pick the metric set: choosing metrics after seeing which
ones transfer is the selection the pre-registration exists to prevent. The
metric set is a registry amendment or it is nothing.

**What could change the answer, and what could not.** More prompts: the floor
is set by how many rows can move, so extending the battery (a
`PROMPT_BATTERY_VERSION` bump, `core/prompts.py`) would lift the full-set row
off 0.0661 — at the current rate of four informative rows per eight prompts,
about twelve more prompts would be needed for the full set to reach α, and
they must be chosen without reference to their contrasts. A looser α would
not: the gate's message says it exactly, "needs prompts that come down on one
side, not a different threshold". And the leave-one-out rows say the extra
prompts would have to behave very differently from these eight to move a
concordance sitting at 47.9%.

**Remaining.** All three pythia-1.4b revisions are cached under `data/hf`
(step143000, step0; `-random` norm-matches step143000), so the remaining arms
run offline:

```bash
cd /run/media/system/WDS_500/Mets && source .venv/bin/activate
export HF_HOME=$PWD/data/hf METS_RESULTS_DIR=$PWD/data/phase12 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
python -u -m p1_mstate_tracking.run_1 --models pythia-1.4b-step0    # sensitivity arm, optional
python -m tools.score_claim_c --run-dir data/phase12/2026-09-17_14-51-40 \
    --run-dir data/phase12/2026-09-17_16-04-54 \
    --run-dir data/phase12/2026-09-19_09-44-25 \
    --run-dir data/phase12/2026-09-19_10-18-45 --run-dir <each new dir>
```

All four REQUIRED arms now exist; `step0` is the optional sensitivity arm.
*(This block predates the HDBSCAN re-run. The re-run it waited for is done,
and the scored result is the table at the top of this section. The command's `--run-dir` list names the
pre-re-run directories; the scored ones are in
`claims/audits/claim_c_real_run.json`.)*

`-u` matters: without it the log is block-buffered and shows nothing until
exit. Do not stop an invocation mid-arm. The `-random` arm re-initialises on
load — the log's "96 matrices, 1 embeddings re-init; checksum 173425.590 →
100442.844" line is what says it actually happened, and is the check that
`gpt2-large-random`'s defect above would have failed.

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

**D10 — every rank gate reads the wrong quantity.** Separate from D1, which is about the
reported `MinRank` column. `DEGENERATE_RANK_THRESHOLD` (`config.py:82`) and the Fiedler
`active_rank_threshold = 10.0` (`reporting_p1.py:170, 208`) both gate *layer inclusion* on
raw effective rank. `MATH.md` §6.4 shows what raw rank measures: with $n_i = \|y_i\|$ and
$s_{ij}$ the cosine, participation-ratio rank is $1/\langle s^2\rangle_w$ with weights
$w_i = n_i^2/\sum_j n_j^2$, and in the near-orthogonal limit it tends to
$(\sum_i n_i^2)^2/\sum_i n_i^4$ — the participation ratio of the **norm distribution alone**,
with zero directional content. Numerically: three tokens at 30× norm in a cloud of 200 take
raw rank from 112 to 3.4 with the directional geometry untouched. Our `MinRank → 2.3` at step
143k is most likely a sink count.

Because these are *gates*, this is worse than a mislabeled column: the set of layers entering
every gated statistic — energy violations, CKA, NN-stability, the Fiedler mean — moves with
the checkpoint's sink structure. Both gates should read `effective_rank_normed`, and both
thresholds must be re-derived against the normed scale rather than carried over. Promoted out
of D1/D3 because it changes which layers enter *every* gated statistic, not just one column.
Report the norm-participation ratio next to raw rank to test the sink hypothesis directly.

**D9 — `SINKHORN_MAX_ITER = 100` is hit without a flag.** The n=20 causal baseline needs 232
iterations to reach `SINKHORN_TOL = 1e-6`; at iteration 100 the residual is 4.7e-4. The λ₂
error is negligible for the uniform baseline (0.108894 vs 0.108889 converged), but real
attention is more peaked and converges more slowly, and no per-layer convergence residual is
recorded anywhere. Log the residual.

---

## Blockers / open items

0. **$T_{\rm eff}$ vs $t^\ast$ has never been measured, and it is the single highest-value
   unrun quantity in the project — at [R] cost, no forward passes.** A residual block is a
   forward-Euler step of the paper's ODE with step size $h_\ell = \|P^\perp_{x_\ell}(\Delta
   x_\ell)\|/\|x_\ell\|$ (exact for Pythia's parallel residual), so the network's effective
   integration time is $T_{\rm eff} = \sum_\ell h_\ell$. Numerically integrating (6.9) at
   $n = 467$ gives $\gamma_\beta = 0.9$ at $t^\ast \approx 4.2$, near-invariant in $\beta$
   across two decades. If $T_{\rm eff} \ll t^\ast$, **the network never runs the dynamics
   long enough to collapse and "trained weights resist collapse" is partly an artifact of
   depth** — the correct comparison becomes $\gamma_\beta(T_{\rm eff})$, a specific finite
   number, not $t = \infty$. If $T_{\rm eff} \gtrsim t^\ast$ with no collapse, Blog 1's
   claim stands and is now quantitative. Every input is on disk (`sublayer_streams`,
   `activations.npz`, `core/beta_eff.py`). Phase 1c sub-experiments A and B. This is
   prediction **P-γ2**, and it should be run before the rest of the open list, because the
   answer changes how the verdict table reads.

1. **D1, D2, D3, D10** — three verdict rows blocked, and D10 moves the layer set under all
   of them. D1 is a re-report; D2 needs both a schema fix
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
