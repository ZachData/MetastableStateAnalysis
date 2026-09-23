<!-- p1_mstate_tracking/lit-1.md -->
# Phase 1 — LITERATURE

**Status: LEADS, NOT READINGS.** Produced 2026-09-16 by web search only. **No paper
below has been read.** `arxiv.org`, `semanticscholar.org`, `openreview.net` and every
other scholarly host are blocked by this session's egress proxy, so not even an
abstract page could be fetched — every arXiv id, author list and finding here comes
from a search engine's summary of the page, which is weaker evidence than an abstract
and much weaker than a paper. Verify before citing, arguing against, or dropping a
line of work. §5 is the queue.

**Why this file exists.** `archive/docs/literature_scan_2026-09-10.md` covered §3.12-V only —
the 7d/7e/8 territory. Every other phase was building on citations chosen when the
phase was designed, some of them two years old, with no check on what the field did
since. This is Phase 1's check.

**The headline, bluntly: two of Phase 1's framing statements are stale, and its
single most-quoted new object — the four-transition developmental arc — appears to
have been published in September 2025 on the same model family.**

---

## 0. Verification ledger

| mark | meaning |
|---|---|
| **[S]** | a search engine's summary of the paper's page was read; the paper was not |
| **[N]** | id and title appeared in a result list only; no summary text was seen |

Nothing in this file is marked "read", because nothing was.

---

## 1. The base the phase is built on, and what has happened to it since

### 1.1 The three papers Phase 1 cites

`design-1.md` cites Geshkovski, Letrouit, Polyanskiy and Rigollet throughout, via
`MATH.md` §9 (which does not exist on disk — see `INDEX.md`, "Referenced, not
present"). The citable objects behind it:

- **"The emergence of clusters in self-attention dynamics"**, arXiv **2305.05465**,
  NeurIPS 2023. **[S]** The clustering theorem.
- **"A mathematical perspective on Transformers"**, arXiv **2312.10794**, published
  in the *Bulletin of the AMS* **62(3), 2025**. **[S]** The survey Phase 1's
  Figures 3/4, Theorems 6.1/6.3/6.9 and Problem 1 are quoted from. **The Bulletin
  version is the one to cite**, and the project cites the preprint.

### 1.2 **Stale framing #1 — metastability may no longer be an open problem**

`design-1.md` and `status-1.md` both rest on this sentence: *"the second half is
Problem 1, which the paper poses as open, writing that it observes the behaviour in
numerics and is not able to explain it theoretically."*

- **"Dynamic metastability in the self-attention model"**, arXiv **2410.06833**
  (Geshkovski, Koubbi, Polyanskiy, Rigollet), October 2024. **[S]** The summary
  states the paper **proves the appearance of dynamic metastability**: particles
  collapse to a single cluster only in infinite time, and **remain trapped near a
  several-cluster configuration for an exponentially long period**.

If that is what it proves, then Phase 1's "a phase whose falsification criterion
tests an open problem" is a 2023 statement being repeated in 2026. **What it is
proved *for* is the thing to check first** — the search summary elsewhere notes
"theoretical results are proven for d = 1", and Phase 1 runs at d = 1024. If the
proof is d = 1 with identity weights, Phase 1's regime tension survives intact and
only the word "open" has to change. If it is general, the design doc's second
paragraph needs rewriting and the phase's falsification criterion needs restating.
**This is the single highest-value verification in this file.**

### 1.3 **Stale framing #2 — the theory now has a causal-mask version, and Pythia is causal**

- **"Clustering in Causal Attention Masking"**, arXiv **2411.04990** (Karagodin,
  Polyanskiy, Rigollet), Nov 2024, NeurIPS. **[S]** Modifies the dynamics to the
  causally-masked attention actually used in generative transformers. Three claims
  in the summary, each of which lands on this project:
  1. The masked system **cannot be interpreted as a mean-field gradient flow.**
  2. Asymptotic convergence to a single cluster is proved for **arbitrary
     key-query matrices** with **V = I**.
  3. It connects metastable states to the **Rényi parking problem**.

Phase 1c already exposes `sa_field(causal=)` and defaults to `causal=True`, calling
it "a departure from the theory". **It is not a departure any more; it is a
different theorem, and the project has been comparing against the wrong one.**
Claim (1) also pre-empts Phase 2d's D1: on a decoder-only model the gradient-flow
structure is gone before the symmetry of `QᵀK` is even asked about.

Claim (3) is the most useful thing in this file. The Rényi parking problem predicts
**a number of clusters as a function of n** — a quantitative, published, unchecked
prediction about exactly the quantity Phase 1's cluster tracking counts. See §4.

---

## 2. Finding by finding: what has been done before

| Phase 1 finding | Nearest prior work | Verdict |
|---|---|---|
| Plateaus in cluster count / IP histograms survive at d = 1024 under learned weights (216/216 runs) | 2312.10794 Fig. 3 says the metastable band is gone by d ≈ 512; 2410.06833 **[S]** proves trapping; no empirical d = 1024 test found | **Probably still open.** The searches surfaced numerics at d ∈ {4,5,7,10,20,50} and nothing at 1024 |
| **The developmental arc: rank collapses (step 16), recovers (512), overshoots 3× (3k–5k), declines for 140k steps** | **arXiv 2509.23024**, *Tracing the Representation Geometry of Language Models from Pretraining to Post-training* **[S]**, NeurIPS 2025 / ICML 2025, Google. **Pythia 160M–12B and OLMo 1B–7B.** Reports "a consistent non-monotonic sequence of three geometric phases": **warmup collapse → entropy-seeking expansion** (peak n-gram memorisation) **→ compression-seeking consolidation** | **NOT NEW.** This is Phase 1's arc, on Phase 1's model family, with a name and a mechanism attached |
| Raw effective rank is dragged toward 2 by massive-norm outlier tokens (defect D1) | **arXiv 2510.06477**, *Attention Sinks and Compression Valleys in LLMs are Two Sides of the Same Coin* **[S]**, ICLR 2026. **Proves** massive activations necessarily produce representational compression, with entropy bounds; 410M–120B; BOS-norm / sink-rate / matrix-based-entropy as the metric triple | **NOT NEW as a phenomenon — but see §3.1.** Phase 1 filed it as a measurement defect; the field filed it as the mechanism |
| Mid-network mass drops 20× below the embedding floor; plateaus at layers 9–14 | Same paper's **compression valleys**; and 2311.05928 *The Shape of Learning* **[S]**, which reports a **bell-shaped anisotropy profile peaking in middle layers** | **NOT NEW** |
| Effective rank plateaus near ~200–250 regardless of d_model (the 5c anomaly, cited here) | Intrinsic-dimension literature (2311.05928 **[S]**); no source found stating a d_model-independent plateau value | **Unresolved.** Nothing found confirms *or* denies the specific claim |
| Energy-monotonicity violations absent at init, onset step 256, count saturating at 512, severity peaking at 60k then declining | Nothing found. Searches for E_β / Lyapunov monotonicity on a checkpoint axis returned theory papers only | **Looks new.** The strongest surviving card |
| `repeated_tokens`: init leaves a degenerate input degenerate (mass 0.948), step 143000 actively separates it (0.379), onset ~11k–13k | Nothing found | **Looks new** |
| Plateau onset flips weight-level → content-driven at exactly step 512 | Nothing found under that description | **Looks new** |
| Fiedler deviation crosses zero at 1000–3000 | Nothing found; but the instrument is vacuous as built (defect D2) | Moot until D2 is fixed |
| Cluster carrying capacity invariant (50–55) while turnover rises | Nothing found | **Looks new**, and is the quantity the Rényi-parking link in §1.3 would predict |

---

## 3. What survives, ranked

### 3.1 The frame discipline is a correction to a published result, not a defect report

This is the most valuable inversion in this file, so it is stated carefully.

`status-1.md` files raw-vs-normed effective rank as **defect D1** — an internal
reporting mistake. 2510.06477 **[S]** publishes the same coupling as its **main
theorem**: massive activations ⇒ representational compression. Both cannot be
framed as errors.

The question that makes it ours: **is the published compression a directional fact
or a norm fact?** Phase 1 already computes both — `raw` (scale + direction) and
`normed` (direction on the sphere alone) — and already knows they disagree by an
order of magnitude on this architecture. If the field's matrix-based entropy is
computed on unnormalised activations, then "compression valleys" and the
"compression-seeking phase" of 2509.23024 are **partly measurements of one token's
norm**, and the normed-frame version of those curves is a publishable correction on
a paper at ICLR 2026 and a paper at NeurIPS 2025.

It is also cheap: `effective_rank_normed` is already on disk for all 216 runs
(status-1.md says the D1 fix is a re-report, not a rerun).

**Check first, before any of this is written down:** whether either paper already
L2-normalises, centres, or excludes the BOS/sink token. If they do, this direction
is dead and should be recorded as dead.

### 3.2 The energy axis is unoccupied

Nothing in the searches tracks an interaction energy — the paper's `E_β`, or any
Lyapunov-style functional — **across training checkpoints** of a real model. The
field tracks rank, entropy, anisotropy, intrinsic dimension and sink rate. Phase 1's
energy-violation trajectory (absent → onset 256 → saturating 512 → severity peak 60k
→ decline) is the one headline curve with no near neighbour found.

The caveat is internal, not external: the violations are measured under the
identity-weight proxy, and Phase 2d D4 exists because the model's own energy may
remove some of them. **The unoccupied territory is real; whether our occupancy is
valid is a 2d question.**

### 3.3 The degenerate-input control

The `repeated_tokens` sign flip — training installs a *separating* force — is a
one-line result with no neighbour found, and `design-1.md` already explains why it
was nearly lost (it was excluded from the main table by design, correctly).

### 3.4 What does **not** survive

**Do not build anything on the developmental arc as a new object.** 2509.23024
**[S]** has it, on Pythia, with more sizes, a second model family, and standard
metrics (RankMe, αReQ). `status-1.md`'s "that arc — collapse, recovery, overshoot,
slow decline — is the phase's main new object" must be struck or rewritten as a
replication with a frame correction (§3.1). Rewriting it as a replication is not a
demotion: an independent reproduction on 27 checkpoints in a different frame, with
the energy axis attached, is worth more than the sentence it replaces.

---

## 4. Directions to grow

Ranked by (value × cheapness), with the measurement each needs.

1. **The Rényi parking prediction for cluster count.** 2411.04990 **[S]** links
   metastable states under causal masking to the Rényi parking problem.

   > **CORRECTED TWICE on 2026-09-20, and read the second correction.** The
   > first (`p10_cluster_function/lit-10.md` §5) worked from search summaries
   > and over-shot; the second works from the paper, which was read in full
   > that day by two routes — **`docs/readings/2411.04990.md`** (the dedicated
   > reading note) and **`lit-10.md` §11 / `math-10.md` §7** (the same read
   > inside Phase 10's files). Both are `[R]` and they agree except where
   > `lit-10.md` §11.5 is noted below.
   >
   > The sentence that originally stood here — that parking gives "a **density
   > constant** (the Rényi constant ≈ 0.7476) and hence an expected number of
   > occupied cells as a function of n" — is **half right**, not wrong on both
   > halves as the first correction claimed.
   >
   > - **Wrong: "as a function of n."** There is no `n` in the law. Lemma C.1
   >   gives the expected count in an *infinitely long* sequence, and the
   >   finite-`n` form **saturates** at it. More tokens do not buy more centres
   >   once the sphere is full — **which is this project's own carrying-capacity
   >   finding (max simultaneously-alive clusters invariant at 50–55 across all
   >   27 checkpoints) with a formula attached.**
   > - **Right: the Rényi constant is there.** Appendix C.4 — at `d = 2`, as
   >   `δ → 0`, the average number of **ordinary** Rényi centres approaches
   >   `c·2π/δ` with `c ≈ 0.75` (Dvoretzky–Robbins 1964). It lives in the
   >   ordinary-centre count, not in anything this project currently measures.
   > - **The scaling is `Θ(β^((d−1)/2))`**, in **β and dimension**. At
   >   `d = 1024` the exponent is 511.5 — **but that is the small-`δ` power-law
   >   asymptotic only.** The exact law,
   >   `E[#strong Rényi centres] = E_{x∼μ}[1/μ(B_δ(x))] = 1/σ_{d−1}(B_δ)`, is
   >   **distribution-free and dimension-free**: a reciprocal local-density
   >   average, no exponent, no `β`, only the separation `δ`, which is a
   >   distance and can simply be swept. **So the `d ≫ 1` objection does not
   >   kill the prediction for strong centres**, and this growth direction is
   >   live rather than blocked.
   > - **The `d_eff` form is the paper's own open conjecture**, with
   >   `d₁ = dim L`, the top eigenspace of `V` — which Phase 2's `sym_*` /
   >   `schur_*` projectors already identify, on disk for all 19 checkpoints.
   >   The test is therefore **differential**: predict `d₁` from the OV
   >   spectrum, then check it against the count.
   > - **The nuclei are geometric objects**, not "the earliest member of a
   >   cluster": a *strong Rényi centre* is a token separated by more than
   >   `δ = cβ^{−1/2}` from **every preceding token**. Phase 10's F0 measured
   >   the cluster-earliest-member proxy, which is a different statistic, so
   >   **F0's failure is not evidence against this account** (`lit-10.md`
   >   §11.4).
   > - **Theorem 4.1**: for `V = I_d` and *arbitrary* `Q, K`, all tokens
   >   converge to `x₁(0)` — the first token's initial position. Item 4 below
   >   is also unblocked by the reading.
   >
   > `2411.04990` is now **[R]**, not `[S]`. Phase 1 has cluster counts, per   layer, per prompt length, at 27 checkpoints, already on disk. **Nobody has
   checked a parking-derived cluster-count prediction against a trained
   transformer.** This is re-analysis, costs no forward passes, and it is a
   quantitative theory-vs-measurement comparison of exactly the kind
   `claims/adjudications/` has none of. **Do this one first.**
2. **Re-run the published arc in the normed frame** (§3.1). Re-report only.
3. **Swap the null to the causal theory.** Phase 1c's `sa_field(causal=True)` is
   already the default; what is missing is that the *comparison object* is still
   2312.10794's unmasked dynamics. 2411.04990's masked system is the frame-correct
   null for every Pythia number this project has produced.
4. **The trapping timescale as a quantitative prediction. — UNBLOCKED 2026-09-20.**
   `2411.04990` supplies both ends in ODE units: quasi-stationarity for
   `T_j·s_j < e^{c²/2 − c⁴/(24β)}·ε` (Lemma 5.1) and final collapse at
   `t = exp(Ω(√β))`. Phase 1c's `T_eff` is measured in the same units.
   Note the `s_j`: **a centre's stationary lifetime falls with its own token
   index**, which is testable against the measured lifespan fall 7.0 → 4.5.
   The original text follows.
 If 2410.06833 **[S]**
   gives an exponential trapping time, it is a timescale in the ODE's own units —
   and Phase 1c's `T_eff` is measured in those units. A predicted trapping time
   against a measured integration time is a sharper test than a plateau count.
   Blocked on reading the paper.
5. **Publish the d = 1024 regime result as a negative on Figure 3.** The paper's own
   numerics say the metastable band is gone by d ≈ 512; we find plateaus in 216/216
   runs at d = 1024. Either our detector finds a different object or learned weights
   break the concentration argument — `status-1.md` already states both readings and
   says Phase 1c-B separates them. **The literature has not settled this**, and the
   searches found no empirical test above d = 50.

---

## 5. Verification queue

In priority order. Each line is what to check, not what is true.

1. **2410.06833** — in what dimension is metastability proved, under what weight
   assumptions, and what is the trapping timescale? Decides whether `design-1.md`'s
   "Problem 1 is open" framing must be retracted.
2. **2411.04990** — the exact form of the Rényi-parking correspondence, and what it
   predicts for cluster count vs. n. Decides growth direction 1.
3. **2509.23024** — which Pythia sizes and checkpoints, which layers, and **whether
   representations are normalised before RankMe**. Decides §3.1 and §3.4.
4. **2510.06477** — whether matrix-based entropy is computed on raw or normalised
   activations, and whether BOS is excluded. Same decision.
5. **2312.10794 (Bulletin of the AMS 62(3), 2025)** — replace the preprint citation
   project-wide; check whether the published version renumbers the theorems
   `MATH.md` §9 depends on.
6. **2311.05928** — the anisotropy bell curve and the ID-rises-early result, against
   our layer profile and our step-8–32 collapse.
7. **2305.05465** — confirm the theorem numbering the project quotes.

---

## 6. Search log (2026-09-16)

- `Geshkovski Letrouit Polyanskiy Rigollet mathematical perspective transformers clustering metastability follow-up`
- `metastable clusters self-attention dynamics high dimension d=1024 trained weights empirical test`
- `Clustering in Causal Attention Masking arXiv 2411.04990 decoder-only self-attention dynamics`
- `effective rank trajectory during language model training collapse recovery overshoot decline representation dimensionality`
- `"Tracing the Representation Geometry of Language Models" pretraining post-training RankMe alphaReQ entropy-seeking compression-seeking Pythia`
- `massive activations outlier dimensions attention sink transformers effective rank confound residual stream norm`
- `"compression valleys" attention sinks massive activations same cause entropy layer depth matrix-based entropy paper`
- `intrinsic dimension language model representations plateau 250 dimensions independent of d_model anisotropy`

Also surfaced, not pursued, recorded so they are not re-found from scratch:
**2608.08922** *Clustered Attractor Manifolds and Dynamical Condensation in
Self-Attention* **[N]**; **2601.21942** *Clustering in Deep Stochastic Transformers*
**[N]**; **2509.25611** *Transformers through the lens of support-preserving maps
between measures* **[N]**; **2601.21366** *Perceptrons and localization of
attention's mean-field landscape* **[N]**; **2505.19458** *Recurrent Self-Attention
Dynamics: An Energy-Agnostic Perspective from Jacobians* **[N]**; **2604.08764**
*Revisiting Anisotropy in Language Transformers: The Geometry of Learning Dynamics*
**[N]**; **2510.15511** *Language Models are Injective and Hence Invertible* **[N]**.
