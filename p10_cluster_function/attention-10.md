<!-- p10_cluster_function/attention-10.md -->
# Phase 10 — Where the attention actually goes

> **UPDATE 2026-09-20 — row A0 has RUN and the gate it imposed has cleared.**
> Over 3 646 layer-units the raw 1.417× / 0.825× flip becomes **1.034× /
> 0.998×** once §2.2's structural tilt is divided out: **94 % of the gap is the
> causal mask, and 100 % of it at initialisation**, with a learned residual
> appearing only from step ~2000 and the position-bias confound *falling* with
> training. Replicated on a second sweep with an independently-derived
> partition. **Rows A1–A8 are unblocked** — `status-10.md` §5 rates A2 and A4
> highest. Numbers and caveats: `status-10.md` §1.1, `PROJECT.md` §3.51.

**The attentional signature, opened.** `notes-10.md` §3.1 lists four signatures
of a parked particle and says the discrimination lives in the two nobody has
measured. This file works the one that *has* been measured, finds that it has
never been audited, and lays out what the audit costs — which is nothing,
because the tensors are on disk.

Pre-design. Nothing frozen, nothing registered.

---

## 1. The finding, and the user's reading of it is right

`p1_mstate_tracking/visualization/noise_importance_proxy.py` — **live, not
archived** (`notes-10.md` §3.1 says archived; that is wrong and is corrected
there). What it computes, per layer:

```python
received[key] = sum over heads, sum over queries of attn[h, query, key]   # diagonal zeroed
rel_received_noise     = received[noise_mask].mean()     / received.mean()
rel_received_clustered = received[clustered_mask].mean() / received.mean()
```

Results (`archive/p5c_unclustered/status-5c.md`, the phase's strongest):

| | clustered | unclustered |
|---|---|---|
| random GPT-2 / ALBERT | near parity, or clustered-favoured | near parity |
| trained GPT-2-large | **~0.5×** | **~1.6×** |
| trained ALBERT-base | **~0.5×** | **>2×** |

**The reading offered — that the random case is a pure numbers game — is
confirmed by the instrument's own normalisation, and it is worth saying why.**
The reported quantity is a **per-token ratio against the layer mean**, so
population size is already divided out. "Near parity" therefore means attention
per token is roughly uniform, and uniform attention puts **mass in proportion to
population**: if 50 % of tokens are clustered they take ~50 % of the mass, for no
reason but their number. That is exactly the numbers game, and it is what the
measurement says.

The mass version follows by multiplying back: at 1.6× on ~45 % of tokens the
unclustered population holds ~72 % of the mass; at >2× it holds ~90 %. Which is
`math-1.md` §13.1's independently stated figure — *"unclustered tokens absorb a
growing share of attention in trained models (≈90 % of attention mass on ≈50 %
of tokens by late layers)"*. **Two instruments, one number, and the recollection
of "~80 % on 40–50 %" sits between them.**

So the phenomenon is: **training converts a uniform allocation into a targeted
one, and the target is the population that failed to cluster.**

---

## 2. Four things that have never been checked, and any one could be the whole result

None of these is a criticism of the instrument — it is explicitly a *"fast first
pass before the expensive version"* and says so in its own docstring. They are
the audit that was never run because the finding moved straight into
`PREDICTIONS.md` claim (a) as motivating evidence.

### 2.1 Position 0 is the sink, and the sink is unclustered by construction

NeoX tokenizers do not prepend BOS, so **position 0 takes on sink duty and can
carry a residual norm one to two orders of magnitude above every other token**
(`math-1.md` §2.5). A norm outlier at that scale is precisely what HDBSCAN calls
noise. **So the sink is almost certainly inside the "unclustered" bucket**, and
if it commands a large share of all attention it could produce a 1.6× mean on
its own.

`_received_attention` zeroes the diagonal and nothing else. There is no
position-0 arm.

**This project already owns the machine for this.** `core/sink_audit.py` exists
to decide exactly this class of question, and its discipline is better than a
bare exclusion:

- every share is reported **against its structural baseline and as an enrichment
  ratio**, because "position 0 holds 3 % of the energy" is a 4× enrichment on a
  264-token battery where the pairs touching index 0 are 0.76 % of entries;
- the decision rule is **stated before the numbers** — `policy_is_load_bearing`
  (the conclusion moves, so both arms are reported everywhere) /
  `sink_dominates` / `policy_is_cosmetic`.

**Pointing that rule at the attention flip is a few lines and it has never been
done.** If the flip survives with position 0 excluded, it is a real result about
the population. If it does not, it is a sink result — which is `2510.06477`'s
result, not ours (that paper *proves* massive activations produce
representational compression).

### 2.2 The causal mask gives early tokens a mechanical advantage, and it is not divided out

This is the sharpest of the four and it is structural, not empirical.

Attention is causally masked. A token at position `j` is visible to `n − j`
queries; position 0 is visible to all `n`, the last token to one. `received` sums
over queries **without normalising by how many queries could have attended at
all**. So received attention has a built-in `1/j` tilt before any content enters.

**If the unclustered population skews early — and the sink at position 0 is the
extreme case of exactly that — the flip is partly an artifact of the mask.**

**Derived and checked, 2026-09-20 (`math-10.md` §1,
`tools/math_checks/causal_mask_attention_baseline.py`).** Content-free — uniform
within the causal triangle — the baseline is exactly

```
    received(j) = H_n − H_j ,   and   Σ_j received(j) = n
```

so the layer mean is **exactly 1** and `received(j)` *is* the "× layer average"
quantity the flip reports. At the battery's `n = 264` it runs from **6.155×** at
position 0 through **0.691×** at the median to **0.0038×** at the last token —
**about 1 600×, before any content enters.** The observed contrast is 1.6× against
0.5×, a factor of 3.2 sitting inside that.

**And the observed numbers are reproducible with zero content**: the baseline
already equals 1.6× at position ≈ 53 and 0.5× at position ≈ 160. A partition whose
unclustered members average early reproduces the flip exactly with nothing
learned. (The two reported values also pin the split to `f = 5/11 ≈ 45 %`, which
`status-5c` independently reports — an internal-consistency check, not evidence.)

The fix is the same shape as `sink_audit`'s: report received attention **against
its structural baseline**, here `received[key] / (number of queries that can see
key)`, or equivalently against `H_n − H_j`. `sinkhorn.py`
already builds precisely that object for a different purpose — *"a content-free
attention (uniform within the causal triangle)"*, used as a per-head Fiedler
baseline, with classification done **on the deviation, not the raw value**. The
same baseline, the same argument, one population level up.

**Nothing in the flip's measurement carries a mask baseline.** This is the first
thing to check.

### 2.3 It is a mean, and this project has a section about that

`PROJECT.md` §3.13, *When the mean is the wrong instrument*: a mean is the right
summary when an effect is **distributed** across a population and an extremum is
right when it is **concentrated**, and *"choosing between them after seeing the
data is exactly the selection `claims/registry.json` exists to forbid"* — so
**exploratory work reports both, always.**

A 1.6× mean over ~120 unclustered tokens is equally consistent with a broad
shift and with one token at 40× while the rest sit below parity. **Those are
different findings and they license different claims**, and the second is the
sink story again.

The threshold-free shape statistics are already written, in
`p8_scale_ladder/compare_rungs.py`, for the same reason in a different context:
**participation ratio** `(Σd²)²/Σd⁴` — the effective number of tokens carrying
the attention — plus **top-k share** and **gini**. Reuse, not new code. A
participation ratio of 3 on a 120-token population says something very different
from 60.

### 2.4 It has never run on Pythia, and never on a checkpoint axis

Measured on trained/random GPT-2-large and ALBERT-base. The project moved to
Pythia checkpoints in 2026-08 and the flip did not come with it.

**And the data is already there.** `claims/audits/p1c_inputs.json`, over all 19
`pythia-410m` checkpoint directories: **152/152 model-prompt directories carry
`attentions.npz`** — the full `(n_layers, n_heads, n_tokens, n_tokens)` tensor,
19 checkpoints × 8 prompts, alongside `hdbscan_labels.json` and the activations.

> **The flip's developmental curve costs no forward pass. It has been available
> since 2026-09-01 and nobody has plotted it.**

That is the version worth having, because this project's object is a trajectory
and the only question a single checkpoint cannot answer is *when*. And there are
**four known transitions to co-locate against** (`math-1.md` §13.2): the 8→16
transient collapse, the **256→512 energy break**, the **step-512 plateau flip
from weight-level to content-driven**, and the **1000→3000 Fiedler crossing**.

**Co-location discipline applies and is not optional.** `core/changepoint_colocation.py`
exists because the registered permutation null for exactly this kind of claim was
*measured and found to reject under H0 at 0.32–0.45*; the matched-control null is
what adjudicates. And §3.8.2's Tier-B co-location is labelled exploratory
precisely because three anchors were tried without a differential falsifier
registered first. **Same trap, same rule.**

### 2.5 Two confounds already settled, and one named but unchecked

- **Punctuation is ruled out.** The same module's second panel: clustered ~20 %
  punctuation, unclustered ~5 %, **the same ratio under random weights** — so it
  reflects embedding-space geometry, not learned behaviour. A settled negative,
  and it is the reason to take the attention panel seriously in the first place.
- **Token frequency is named and unchecked.** `docs/LITERATURE.md` §6 item 10:
  *"regress cluster membership on log token frequency — the third story 5c never
  listed, and a confound for both it did."* Rare tokens are both harder to
  cluster and more informative to attend to. **This one is cheap and it is a
  genuine alternative explanation for the entire finding.**

---

## 3. Three readings, and they are not the same claim

If the flip survives §2, it still admits three explanations, and the phase should
name them before measuring rather than after.

- **(a) Routing away from parked particles.** The network does not look at what
  it has filed away. This is `H-PARK` (`notes-10.md` §3.1), and the attentional
  signature is doing what the hypothesis says.
- **(b) Routing toward individuated carriers.** Induction, n-gram completion and
  position tracking need *a specific token* rather than its cluster's attractor,
  so attention goes where identity was preserved. This is the
  dimensionality-budget reading (`PUBLICATION_IDEAS.md` idea 3) and it is a
  claim about what the unclustered population is *for*.
- **(c) Routing toward sinks.** Attention mass parked on a few no-op tokens —
  the softmax's "none of the above". `2510.06477` proves sinks and compression
  valleys are two sides of one coin.

**(a) and (c) are both "attention goes where the computation isn't", which is why
the sink arm has to come first**: they are only distinguishable after position 0
is accounted for. (b) is the one that predicts *structure* in where attention
lands — specific heads, specific offsets — and §4 is how to see it.

---

## 4. What the tensor can answer that the scalar cannot

Every item here reduces a tensor already on disk. No forward pass, no new model
load.

### 4.1 Which heads divert — and 410m already has a causal catalogue to join against

`_received_attention` sums over **all heads**. So the finding cannot say whether
the flip is a broad property of the layer or the work of a few heads — and §2.3's
argument applies at the head level as much as the token level.

Per-head is free: the tensor is `(n_layers, n_heads, n, n)`.

> **"Which heads do the diverting" has an answer on disk today**, and at 410m it
> can be joined to `p7d_redundancy`'s 384-head causal sweep and to
> `attention_entropy_per_head`, which `p1_io.py` already stores per layer and
> which nothing has ever read against cluster structure.

If the diverting heads *are* the induction/redundancy-set heads, reading (b)
gains a mechanism. If they are a disjoint set, that set is newly interesting and
nobody has named it.

### 4.2 The population×population mass matrix — the direct test of H-PARK

For each layer (and head), the 2×2 of attention mass:

```
                    key: clustered      key: unclustered
    query: clustered        m_cc               m_cu
    query: unclustered      m_uc               m_uu
```

The scalar flip is the column sums. **The off-diagonal structure is the part
that discriminates**, and it has never been computed:

- **`H-PARK` predicts `m_cc` is low and undifferentiated** — parked particles do
  not attend to each other, because there is nothing to compute between them.
- **`H-CAT` predicts `m_cc` is high and structured** — if a cluster is a computed
  category, its members are each other's context.

`archive/p5c_unclustered/status-5c.md` already has the **inner-product** version
of this decomposition (within/between/noise), and its finding is suggestive:
*within-cluster cohesion stays high and flat while between/noise declines
mid-model then rises near the merge event; the energy plateau is carried entirely
by within-cluster pairs.* **High cohesion with low mutual attention would be the
signature of parking**: particles sitting together without interacting. Nobody
has put the two decompositions side by side, and they are the same shape.

### 4.3 Attention paid versus attention received — and it separates all three readings

The flip is about attention **received** (a column statistic). **Nothing in this
project has looked at attention paid** (the row statistic) by population, even
though `attention_entropy_per_head` is the row-side quantity and is already
stored.

The 2×2 is the discriminator:

| | **pays little** | **pays a lot** |
|---|---|---|
| **receives little** | **parked** — inert both ways | reading, not read |
| **receives a lot** | **sink** — attended, attends to nothing | **carrier** — individuated and in use |

> **Sink, parked and carrier are three different things and the received-only
> statistic cannot tell them apart.** One more reduction of the same tensor does.

And it closes a loop with §5: `math-1.md` §1A.6 identifies the high-`Z` token
with the sink, where `Z` is a **row** quantity and the sink is a **column**
phenomenon. **The paid/received 2×2 is also the test of that identification**,
which is asserted in the math notes and has never been measured.

### 4.4 Normalised depth, because the models have different depths

`POPPER_PLAN.md` §755 item 4: **full normalised depth, no band restriction** —
*"a depth band is a choice with as many options as there are bands."* pythia-70m
has 6 layers, 410m and 1.4b have 24, gpt2-large has 36. Whether the flip appears
at a fixed absolute layer or a fixed **fraction** of depth is the question that
decides whether it is a band phenomenon, and it is only askable with more than
one depth.

---

## 5. `Z_beta,i`: a per-token metric that is already trained, already on disk, and never looked at

The best single unasked question this file found, and it fuses Phase 9 with
Phase 10.

`math-1.md` §1A.6, on the reweighted metric in which (SA) *is* a gradient flow:

> **the partition function is not noise to be normalized away — it is a metric.**
> Softmax reweights how far the configuration has to travel, per token, by how
> much attention mass that token commands. A high-`Z` token (a sink) is one the
> metric makes **expensive to move**. That is a strikingly good match to what
> attention sinks empirically do, and **to my knowledge nothing in this project
> has looked at `Z_beta,i` as a per-token quantity at all** (§15, open question 12).

Why it matters here:

1. **Phase 9's lever is `Γ`, a per-channel metric. `Z_beta,i` is a per-token
   metric.** Together they are the two cheap metric levers the architecture
   already contains, and **`Z` is the one that is measurable from artifacts on
   disk with no intervention at all.**
2. **It separates the two kinds of stationary.** A **parked** particle is
   stationary because nothing pushes it; a **pinned** one is stationary because
   the metric makes it expensive to move. Same displacement, different cause,
   and `Z_beta,i` is the quantity that tells them apart. **That resolves §2.1's
   sink confound with a measurement rather than an exclusion rule** — which is
   strictly better, because excluding position 0 throws away the very particle
   whose behaviour is most informative.
3. **It is the natural weight for every per-particle aggregate in the project.**
   `core/particles.py`'s table has a column for it that nobody has filled.

**Resolved 2026-09-20, and it reverses (`math-10.md` §2).** `Z_beta,i` is
particle `i`'s **row** normaliser while a sink is a **column** phenomenon, and
the mask acts on the two in opposite directions. In the concentration regime:
**unmasked**, `Z_i = n·e^{βγ}` is position-independent, so any spread is content
and §1A.6's reading is reasonable — **masked**, `Z_i = (i+1)·e^{βγ}` is linear in
position and **position 0 is the minimum.** Meanwhile `received(j)` **decreases**
in position, by exactly `1/(j+1)` per step.

> **So under a causal mask the sink is simultaneously the largest received
> attention and the smallest `Z` — on the metric reading, the *cheapest* token to
> move, not the most expensive. §1A.6's identification is an unmasked-model
> statement and Pythia is masked.**

This sharpens rather than weakens §4.3: the sink occupies a **specific corner**
of the paid/received square (low `Z`, high received), distinct from parked (low
both) and carrier (high both). **Measure `Z_i/(i+1)`, not `Z_i`.**

---

## 6. The ladder for this question

All rows read artifacts already on disk unless marked. Slots into
`notes-10.md` §8 as the attentional column's detail.

| # | experiment | cost | decides |
|---|---|---|---|
| **A0** | **Sink and mask audit of the flip.** Recompute with (i) position 0 excluded, (ii) received attention against the causal-mask structural baseline, (iii) `sink_audit`'s enrichment framing and its three-outcome rule stated first. | free | §2.1, §2.2. **Do this before anything else quotes the flip** |
| **A1** | **Distribution, not mean.** Participation ratio, top-k share, gini of received attention within each population, per layer. Both summaries reported, per §3.13. | free | §2.3 |
| **A2** | **The flip on the checkpoint axis**, 19 × 8 at 410m. Where does the sign cross, and does the crossing co-locate with the four known transitions — under `changepoint_colocation`'s matched-control null, with the falsifier named first. | free | §2.4. The developmental version, and the most valuable single run |
| **A3** | **Per-head decomposition**, joined to 7d's 384-head causal sweep and to the stored `attention_entropy_per_head`. | free | §4.1 |
| **A4** | **The population×population mass matrix**, per layer per head, beside 5c's inner-product decomposition. | free | §4.2. **The direct `H-PARK` vs `H-CAT` test on the attentional signature** |
| **A5** | **Paid versus received 2×2.** | free | §4.3. Separates sink / parked / carrier |
| **A6** | **`Z_beta,i` per token**, and whether high-`Z` is the sink. | free | §5 |
| **A7** | **Token-frequency regression** on cluster membership and on received attention. | free | §2.5, the named unchecked confound |
| **A8** | **Cross-rung replication**, threshold-free, on normalised depth. | cheap | §4.4, §7 |

**A0 gates everything.** A2 and A4 are the two with the most information per unit
of work.

---

## 7. Using the rungs: what transfers and what does not

The cross-model question has an answer in this repository already, and it is a
constraint rather than an invitation.

**`p8_scale_ladder/compare_rungs.py`: no absolute threshold transfers between
rungs.** Not a membership bar, not `r* = 12`, not a noise floor. The first 70m
write-up broke that rule — comparing "19 of 48 heads clear +0.05" against 410m —
and the bar had been calibrated on 410m's baseline NLL of 0.585 against 70m's
5.725, *"so the same nats mean different things."*

For the attention flip this is a gift rather than an obstacle, because **the
flip's own statistic is already threshold-free**: it is a ratio against the
layer mean. What must be held to the rule is everything built on top of it —
"the flip appears at layer 9" does not transfer, "the flip's sign crosses at
normalised depth 0.4" might.

Three further constraints, all recorded and all easy to forget:

1. **Ablation mode is not a detail.** Zero-ablation's off-distribution bias
   scales as `1/n_heads`, so it distorts at **8 heads/layer** (70m *and* 1b) and
   barely touches 16 (410m, 1.4b). A cross-rung ablation claim names
   **mean**-ablation.
2. **The rung policy.** Explore on **70m and 410m**; **1b and 1.4b are
   RESERVED** — no measurement there until a prediction naming them is
   registered. `lit-8.md` proposes a rule 4 (*a rung may be externally spent*)
   and it is **not taken**; it is a human call.
3. **Prompts on one model are not independent.** They share the model's weights,
   so a model-wide effect present in every prompt is invisible to the
   enumeration. The prompt is the coarsest unit the design provides.

**The asymmetry worth exploiting.** 70m is 6 layers, `d = 512`, 8 heads — the
cheapest full sweep available, with 19 revisions already on disk (`data/hf`,
step0 → step143000). 410m carries the only complete Phase-1 artifact set. So the
natural division is **structure on 410m, replication on 70m**, with 70m also
carrying the two things 410m cannot reach: the `n > d` cone margin (`d = 512`
against a 2048 window) and the dense-onset bracket of the `lora_ind` sister run
(82 clean checkpoints, 4-step spacing through the induction onset — §3.9, and it
is **reachability, not developmental**, a constraint Mets inherits).

> **Check before assuming: whether a Phase-1 clustering sweep exists for 70m at
> all.** Phase 8 ran the head catalogue and the invariants there, which is
> head-level ablation, not a clustering run. If it does not exist, it is the
> cheapest large artifact the project could add, and every row of §6 gains a
> second rung.

---

## 8. What this file does not claim

- Not that the flip is real. §2 lists four reasons it might not survive, two of
  them structural.
- Not that the flip is an artifact. §2.5 records that the obvious alternative —
  punctuation — was checked and **ruled out**, which is why it is worth auditing
  rather than dismissing.
- Not a registration. Nothing here names a `P-*` id, and A2's co-location arm in
  particular must register its falsifier **before** it is run, or it repeats
  §3.8.2's exploratory label for the same reason.
