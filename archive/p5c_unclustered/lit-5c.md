<!-- archive/p5c_unclustered/lit-5c.md -->
# Phase 5c — LITERATURE

**Status: LEADS, NOT READINGS.** Produced 2026-09-16 by web search only. **No paper
below has been read**; scholarly hosts are blocked by this session's egress proxy.
Marks: **[S]** search-engine summary read, **[N]** title and id only.

**Phase 5c has no code directory** and its framing was promoted to organise the whole
transition project. `math-5c.md` lives in `p5_single_mstate_analysis/`.

**The headline: the unclustered population has a name in the literature, and the name
is frequency. Phase 5c's two competing stories — overflow vs deliberate individuation
— have a third the phase never listed, and it is the one the field would offer first.**

---

## 1. The story Phase 5c did not list

`math-5c.md` §1 gives two stories for the ~40–50 % of tokens HDBSCAN never assigns:
**(a) overflow** — computationally inert slack; **(b) deliberate individuation** — kept
distinct because collapsing them would cost something.

The literature supplies **(c) frequency stratification**:

- *Not a nuisance but a useful heuristic: **Outlier dimensions favor frequent tokens**
  in language models*, arXiv **2503.21718** **[S]** — outlier dimensions on the last
  layer "are part of an **ad-hoc mechanism boosting the prediction of frequent
  tokens**".
- **[S]**, same pass: "**rare token embeddings become anisotropic during pre-training**,
  with outliers driven by token frequency."
- *Exploring Anisotropy and Outliers in Multilingual Language Models*, **2306.00458**
  **[N]**.

So a token's clustering behaviour may be largely predicted by its **corpus
frequency**, with rare tokens anisotropic and frequent tokens boosted through outlier
dimensions. **Neither (a) nor (b) is the frequency story, and frequency is a
confound for both** — a rare token is both hard to cluster *and* plausibly one the
network needs to keep distinct.

**The corrective measurement is one line and was never run:** regress cluster
membership on log token frequency, per layer, per checkpoint. If frequency explains
most of it, (a) and (b) are competing for the residual. **This should be the first
thing any revival of 5c does**, and it is re-analysis of artifacts already on disk.

---

## 2. The other two anchors

### 2.1 The ~200–250 effective-rank plateau

`math-5c.md` §1.1 flags that `design-5c.md` and Blog 1 Fig. 10 disagree by ~25 % on the
plateau value (≈200 vs ≈250), and that the exact number anchors a quantitative budget
hypothesis. `lit-1.md` §2 records the outcome of a targeted search: **nothing found
confirms or denies a d_model-independent plateau.** The intrinsic-dimension literature
(**2311.05928** **[S]**) reports ID trajectories and anisotropy profiles but no such
invariant.

**So the budget hypothesis's anchor is unverified in both directions**, and the
internal disagreement should be resolved before the external one is pursued. That is a
re-report, not a rerun.

### 2.2 The boundary question

*Is the unclustered population the boundary population?* Phase 1b built the instrument
(`border_vs_noise`, a rank AUC crossing distance-from-the-Fiedler-boundary against
HDBSCAN noise labels) and `lit-1b.md` §2 records that nothing prior was found. **The
5c question has an answer waiting in a Phase 1b output.** Run it and read it.

---

## 3. What survives, and it is the frame

Phase 5c's contribution was never mathematics — `math-5c.md` §0 says so. It is the
reframing: **the residue is the object, and the right unit is a dimensionality budget
rather than an in/out label.** The searches found the field talking about outlier
*dimensions* and outlier *tokens*, and about compression valleys and attention sinks
(`lit-1.md`), but not about a **budget** — a conserved quantity the network allocates
across whatever still needs distinguishing.

That framing is unoccupied. It is also currently unmeasured, which is the problem.

---

## 4. Directions to grow

1. **The frequency regression** (§1). Blocking for the phase's two stories.
2. **Run `border_vs_noise` and read it** (§2.2). Free.
3. **Reconcile 200 vs 250** (§2.1). Free.
4. **Then, and only then, state the budget hypothesis quantitatively:** a conserved
   count, measured per layer per checkpoint, with frequency partialled out. The
   sequence matters — a budget claim made before (1) is a frequency effect wearing a
   budget costume.
5. **Cross against the sink literature.** `lit-1.md` §2 records **2510.06477** **[S]**
   tying mid-layer compression to massive activations. If the clustered population
   collapses because of a sink, the "budget" may be a sink artifact.

## 5. Verification queue

1. **2503.21718** — the frequency/outlier-dimension mechanism.
2. **2311.05928** — ID trajectories, for the plateau anchor.
3. **2510.06477** — sinks and compression (shared with `lit-1.md`).
4. **2306.00458** — anisotropy and outliers, multilingual.

## 6. Search log (2026-09-16)

- `tokens that resist clustering outlier unclustered representations individuation dimensionality budget language model capacity allocation`
- `intrinsic dimension language model representations plateau 250 dimensions independent of d_model anisotropy`
- `massive activations outlier dimensions attention sink transformers effective rank confound residual stream norm`
