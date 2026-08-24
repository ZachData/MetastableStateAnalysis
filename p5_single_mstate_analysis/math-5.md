# Phase 5 — MATH (study notes)

## 0. What this document is

Companion to `math-1.md` … `math-2d.md`. Phase 5 **inverts the project's method.** Every prior
phase worked in aggregate — 35 model×prompt verdicts, thousands of features ranked by
F-statistic, bulk V-alignment scores. Phase 5 takes **a single HDBSCAN cluster trajectory** and
reconstructs, end to end, the mechanism that creates, maintains, and dissolves it,
cross-referenced against every framework the project has built.

> The deliverable is an interpretable narrative of one piece of the model's computation, not
> another aggregate statistic.

That is a real methodological shift and it deserves its own document, but it also means Phase 5
inherits *every* upstream defect at once — and it did. Three of six investigation groups are
blocked by producer/consumer mismatches, the effective-$\beta$ estimator it uses is the one with
the indexing bug (`math-1.md` §3.4), and its cluster-selection score had **two of six criteria
silently contributing zero**. §9 is about that class of failure, because Phase 5 is where the
project learned to name it.

The v2 reframe also relocates the phase's centre of gravity: the object of study becomes the
**per-particle table** (§8), with clustering as an annotation rather than the unit of analysis.

---

## 1. Selecting the object of study

### 1.1 Six criteria, not one axis

`select_cluster.py` scores candidate trajectories on six gates: **lifespan** ($\ge6$ layers),
**merge participation**, **semantic content** (Phase 1's mutual-NN/HDBSCAN agreement tag,
excluding induction artifacts — `math-1.md` §9.4), **prompt context** (`sullivan_ballou` /
`paper_excerpt` preferred, `repeated_tokens` avoided), **size** ($\ge4$ tokens), and **sibling
availability**.

This is deliberately *not* "pick the longest-lived cluster." Each gate is individually motivated
by a known project pitfall — the induction-head confound and the repeated-tokens degeneracy both
come straight from Phase 1 — which makes the selection **auditable rather than arbitrary.** The
stability check is the right one too: each model's runner-up shares the same prompt and scores
within 0.3 points, which is evidence the selection is not cherry-picked.

### 1.2 …of which two were dead, and the arithmetic does not close

`anchors.py` documents that **the merge criterion (weight 3.0 of 9.0, the second-largest term)
contributed 0.0 on every model**, for the same reason the semantic criterion did:

> `p1_io.load_phase1_run` returns an `events` key, **and it is the wrong events.**

Phase 1 emits two different event schemas:

| source | schema |
|---|---|
| `trajectory.json` → `cluster_tracking.events` | **the real one**: `{"layer_from": int, "layer_to": int, "merges": [[prev_ids], curr_id], …}` |
| `events.json` (a Phase 3 bridge file) | `{"merge_layers": [2,5], "energy_violations": {"1.0": [3,4]}}` |

`load_phase1_run` reads the second and normalizes it into
`[{"type":"merge","layer_name":"5","layer_from":"5"}]` — **`layer_from` a string, and no
`merges` key at all.** Any consumer asking "which merge did this trajectory participate in" gets
`None` for every trajectory, **silently, forever.**

Note also the earlier repair was aimed at the wrong cause: `status-5.md` records FIX-B7 as having
fixed blocker 1 by routing `merge_verdict` through `select_cluster`'s own computation instead of
re-deriving it — *the right plumbing change against the wrong cause*, since
`select_cluster._merge_event_for_trajectory` was already reading the broken `run["events"]`.

**An arithmetic inconsistency, already on record.** `design-5.md` reports "all six models achieve
near-perfect scores (**9.000** for 4/6)" on a **9.0** scale, while `anchors.py` reports **5.0 of
that 9.0 was dead**. Both cannot be true: with 5.0 contributing zero, the maximum attainable is
4.0. `PHASE5_PYTHIA.md` §4 already flags this in the same terms — *"9.000 is the sum of all six
weights, and is unreachable if the 2.0 semantic term is structurally zero. Either those numbers
came from a different code path or the reported maximum is wrong. Resolve before any Study A claim
about selection stability is repeated."* **Unresolved, and load-bearing, because the 9.000 figure
is the stated evidence that selection was principled.**

**And the six-criterion design is itself superseded.** `PHASE5_PYTHIA.md`'s decisions D-1/D-2 drop
*both* the semantic and the `preferred_prompt` criteria, leaving **four** criteria with
`SCORE_WEIGHTS` summing to **1.0** — "a perfect score is now 1.0 by construction." §1.1 above
describes the older design; read it as history.

---

## 2. Group A — is this a coherent object at all?

Compactness, silhouette, CKA on the trajectory's member tokens. **Prerequisite for everything
else meaning anything**: if the selected trajectory is not measurably a coherent set, Groups B–F
are describing an arbitrary subset of a cloud.

The quantities are `math-1.md`'s (mass-near-1 restricted to the cluster, cosine silhouette,
layer-to-layer CKA), applied to a masked subpopulation rather than the full token set.
`core.metrics.mass_near_1` takes exactly this `mask` argument and the cluster-cohesion reading
uses `threshold=0.95` rather than the population default 0.9 — one function, two documented
readings, no second implementation to drift (`math-1.md` §5).

---

## 3. Group B — the paper's predictions, at the level of one cluster

### 3.1 The subspace decomposition, and a clamp that may be hiding a real problem

The core operation splits a vector (a centroid, a displacement $\Delta\bar x$, a fusion
direction) into attractive, repulsive, and orthogonal parts using Phase 2's OV eigen-subspace
bases $U_{\rm att}, U_{\rm rep}$:

$$
\lVert P_U v\rVert^2 = \lVert U^\top v\rVert^2, \qquad
\texttt{orth\_sq} = \max\big(0,\ \lVert v\rVert^2 - \texttt{attr\_sq} - \texttt{rep\_sq}\big)
$$

**That subtraction is only valid if $U_{\rm att} \perp U_{\rm rep}$**, and whether it holds
depends entirely on which decomposition produced the bases:

- From the **symmetric part** $\mathrm{Sym}(W_{OV})$: eigenvectors are orthogonal, the split is a
  genuine orthogonal decomposition, and $\texttt{attr} + \texttt{rep} + \texttt{orth} =
  \texttt{total}$ exactly.
- From **$W_{OV}$'s own eigenbasis**: the operator is non-normal — Phase 2b measures large Henrici
  departure from normality, and most of the spectrum in complex pairs (`math-2b.md` §2.4; the
  headline 84–97.5% is the *per-block energy* convention and is computed on the summed operator,
  which §2.4a argues the model never forms — so treat it as indicative, not exact) — and
  non-orthogonal eigenvectors mean the projected energies can **sum to more than the total**, and
  `orth_sq` goes negative.

The `max(0, ·)` then silently clamps it. **This is the same clamping-hides-a-defect pattern as
Phase 2's `max(0, n_phase1 − n_rescaled)`** (`math-2b.md` §7), and it has the same fix: report the
unclamped value so overcounting is visible as a negative number. Nothing in the artifact currently
records which basis was used, which is precisely the frame-ledger gap `core/frames.py` exists to
close (`math-1.md` §3.2).

### 3.2 The ~50/50 finding, and the theorem it is not in tension with

Group B reports an approximately even attractive/repulsive centroid split, universally, and both
`design-5.md` and `status-5.md` flag this as "a mild tension with **Theorem 6.3**'s prediction
that cluster tokens sit primarily in the attractive subspace during stable phases."

**Theorem 6.3 makes no such prediction.** It is the $d \ge n$ exponential-rate result
(`math-1.md` §1.5): convergence to a single cluster at rate $O(e^{-\beta})$, given the cone
condition. It says nothing about intra-cluster mass or about which eigen-subspace tokens occupy.
`UPDATE_PLAN.md` §1 flags this mis-citation at four sites (`v_alignment.py:185-196`,
`run_5.py:535-542`, `design-5.md:57`, `status-5.md:14`) and leaves it for a decision.

So the "mild tension" is **a tension with a claim the paper does not make** — structurally the
same error as the retracted "Theorem 6.1: higher $d$ → faster convergence" verdict row
(`math-1.md` §1.5) and as P-T1's omitted second condition (`math-2d.md` §3.2). Third instance,
same shape. The 50/50 observation itself stands; only its adjudication against a theorem is void.

### 3.3 The effective-$\beta$ numbers here are the broken ones

Group B measures "effective $\beta$ from the attention softmax (regress logits on inner
products)." **This is the estimator `core/beta_eff.py` was written to replace**, and it is the one
that regressed over `np.triu_indices(n, k=1)` — *exactly the entries causal attention masks* —
returning $-1.8\times10^{-14}$ on synthetic data with a known $\beta = 6.0$, "for every head, on
every model, independent of the data" (`math-1.md` §3.4).

Any $\beta$ Group B reported is that number. It should be recomputed through
`estimate_beta_from_gram` with causal pair selection, row fixed effects, offset control, and the
$1/\sqrt{d_{\rm head}}$ scale divided out — and with the Gram matrix passed in an explicitly
recorded frame.

### 3.4 Merge geometry — blocked

The central Group B output (fusion direction vs. attractive subspace at a merge event) is
untested: `merge_verdict` is `n/a` on all six models because `merge_events` never reaches
`merge_event_geometry()`. Root cause is §1.2's schema mismatch.

---

## 4. Groups C1/C2 — which components cause the cohesion

The split into attention heads (C1) and FFN (C2) exists because Phase 2's OV-centric framework
and the FFN pathway are mechanistically distinct enough to need separate treatment — and it is
exactly this split that surfaces the architecture-dependent result.

**C1's mechanism grounding is missing.** OV values are universally `n/a`; `cohesion_source` reads
`inward_mass_fallback` on every model. So **head rankings are valid *relative* signals but are not
grounded in the OV mechanism** — the causal story ("this head attracts because its OV circuit
projects inward") is not what was measured. Likely a miskeyed Phase 2 weights load.

**Findings, with their stated confidence:**

- **Single dominant attractor head — universal.** Top head carries 2–4× the cohesion of second
  place, sharper in larger models. Robust to C1's grounding gap, since it is a ranking.
- **FFN role is architecture-dependent.** GPT-2: attention-dominant or co-dominant. ALBERT:
  **FFN-cohesive, attention-disruptive** — explained by weight sharing, since the *same* attention
  weights applied iteratively produce fragmented attention while the FFN, also shared but applied
  to a *changing* residual state, stabilizes. BERT: **attention-cohesive, FFN-disruptive**
  ($-13.19$) despite architectural similarity to ALBERT, attributed speculatively to
  bidirectional masked-LM pretraining producing a different balance of computational roles.
  **Flagged as unresolved rather than asserted** — correctly.

Note this is the *same* attn-vs-FFN question Phase 2 answers globally (`math-2.md`'s two regimes),
now at trajectory resolution — and on Pythia it would be exactly separable thanks to the parallel
residual (`math-1.md` §2.1), which none of these six models offers.

---

## 5. Group D — feature signatures (blocked)

Connects back to the crosscoder/low-rank-autoencoder work: does this cluster's identity appear in
learned feature space? Blocked on all six models — Phase 4 outputs never reach Phase 5, a
path/naming mismatch in `p5io.load_phase4()` (or Phase 4 never wrote the cache). Same class as
§1.2 and §4; see §9.

---

## 6. Group E — the tuned lens, and a subtle failure worse than the obvious one

### 6.1 The obvious blocker

Stored probabilities round to $0.000$. The tuned lens was never trained, so the logit-lens
fallback is in use (`used_tuned_lens=false` everywhere). **Top-1 token stability (76–100%)
stands; probability mass does not.**

### 6.2 Why "just train the lens" is not the fix

A 2026-07-19 addendum records a documented pathology for *exactly* this class of object. A tuned
lens is a per-layer affine translator $A_\ell h_\ell + b_\ell$ trained to match the **final-layer**
representation. Because that objective is **correlational**, the trained lens "skips ahead": from
early layers onward it decodes the model's *eventual output token* rather than the layer's
intermediate content, recovering almost none of the known mid-layer intermediates that the logit
lens and averaged-Jacobian lens surface.

**Why this hits Group E specifically.** Group E's entire purpose is decoding what a mid-layer
cluster centroid represents *at that layer*. A lens biased toward the final prediction would
report converged, output-like distributions at every depth — **plausible-looking numbers that
erase precisely the per-layer differences the sibling-contrast KL and top-$k$ stability
measurements exist to detect.** Fixing blocker 4 the naive way would replace an *obviously* broken
result (probs 0.000) with a *subtly* wrong one, which is strictly worse.

This is the most sophisticated version in the project of a recurring theme: **an instrument whose
failure mode produces exactly the shape of the result you are looking for.** Compare
`elim_signed = 1.0` being the value an early-truncating frame produces for free
(`math-2b.md` §5), and `h_displacement` biasing $T_{\rm eff}$ toward the direction that would
make the headline an artifact (`math-1c.md` §1.2).

### 6.3 The J-lens alternative, and why it drops in cleanly

The averaged-Jacobian lens is

$$
J_\ell = \mathbb E\left[\frac{\partial h_{\rm final}}{\partial h_\ell}\right]
$$

a per-layer linear map **averaged over prompts** rather than fitted to them. Three properties make
it the preferred option:

1. **Format-compatible.** Set $A_\ell = J_\ell$, $b_\ell = 0$ in the same
   `{A_L{i}, b_L{i}}` npz — `load_tuned_lens` / `apply_tuned_lens` need no changes, only the
   fitting script does (backward passes instead of ridge regression).
2. **Not underdetermined at small $n$.** Because it is *averaged*, not *fitted*, it sidesteps the
   existing `n_tokens < d` ridge-domination warning entirely. Ablations show it beating both logit
   and tuned lens with as few as ~10 prompts.
3. **Causal rather than correlational**, which is why it does not skip ahead: it measures how the
   final representation actually *responds* to a perturbation at layer $\ell$.

Cost is $d_{\rm model}$ backward passes per prompt (batched) through the `ForCausalLM` at the
pinned revision — and `core/lm_loading.load_causal_lm` already guarantees provably the same
checkpoint as the extraction.

---

## 7. Group F — causal interventions, and Group G — the control that licenses everything

### 7.1 F: from correlational profile to causal necessity

Five interventions: head ablation; steering along the centroid; steering along an LDA direction;
activation patching (overwrite a cluster member's residual with the sibling's); feature ablation.
Measured against cluster cohesion, membership, and merge timing. **Deliberately last**, being the
most expensive and most directly falsifying group.

**The blocker has since been diagnosed, and the interesting reading is dead.** `mean_frac_together`
is *identical across all four interventions*, per model. `PHASE5_PYTHIA.md` records this as **a
metric artifact, not duplication**: `recluster_after_intervention` iterates the *full* chain while
the hook is installed at `target_layer`, so **everything upstream of the hook is byte-identical and
pinned at `frac_together = 1.0` by construction** — roughly half the chain for a mid-layer target.
That rules out the "all four interventions hit the same causal bottleneck" reading. The headline "causal robustness scales with GPT-2 model size"
(xl 0.80 > large 0.53 > albert-base 0.15 ≈ albert-xlarge 0.14 > gpt2-medium 0.00) is explicitly
conditional on this not being a bug. The proposed reading — larger models distribute cluster
identity across more components, so no single intervention dissolves it — is offered **as a
hypothesis**. Note gpt2-medium at exactly 0.00 next to gpt2-large at 0.53 is itself suspicious.

The migration to `core/intervention.py` closed the architecture gap: standard per-layer models
(GPT-2, GPT-NeoX/Pythia, bare and LM-head wrappers) route through `run_model_with_hook` via
`forward_pre` hooks; ALBERT keeps its legacy path because the standard HF forward has no way to
run a shared layer more times than `config.num_hidden_layers`, which extended-iteration ALBERT
requires. **GPT-NeoX gained real support in the process — the legacy loop never handled it at
all.** One documented behavioural difference: on the standard path the final trajectory entry is
post-`ln_f` (matching `core/models.py`'s convention, i.e. what Phase 1's labels were built on)
where the manual GPT-2 loop recorded pre-`ln_f`. Recording that rather than silently absorbing it
is the right call — it is exactly the off-by-one class of `math-1.md` §2.2.

### 7.2 G: the distinction between a control tier and a null

This is the sharpest methodological idea in the phase, and conflating the two is the easy error:

| | what it is | what its spread means |
|---|---|---|
| **control tier** | **ONE** fixed random draw, frozen at selection time, measured at **every checkpoint** | a *developmental trajectory* — "these particular unrelated particles did this over training." Its variance across checkpoints is **signal** |
| **label-permutation null** | **200** random draws at **ONE** checkpoint, discarded immediately | a *distribution* — "what would any size-matched subset have scored here." Its spread is the **yardstick** |

**Reporting only the control gives an ordering with no error bar. Reporting only the null loses
the developmental comparison. Both are produced.** The original Group G had the ordering
(primary > sibling > random, cleanly across all six models) but *no significance statement*;
`core/nulls.py` now lets it be stated as "N$\sigma$ from null" and put in a falsification table.

**Which null for which question**, also worth internalizing (`math-1.md` §3, `math-1b.md` §6):

- `label_permutation_null` — permute *membership*, activations fixed. Answers *"is this set more
  compact than a random size-matched subset of the same tokens?"* The right chance baseline for
  every set-level statistic here.
- `shuffled_dimension_null` — permute each feature dimension across tokens. Answers a question
  about the *geometry*, not the labelling.

---

## 8. The particle table — the v2 reframe

`core/particles.py`: *"cluster- and population-level results become aggregations (groupby /
filter) over this table rather than separate code paths."* One row per
`(model, checkpoint_step, prompt_key, layer, token_position)`, with the token set's role and the
sweep geometry as columns. Pure numpy, no pandas.

### 8.1 The question this makes answerable

Phase 1's sharpest unexplained result: across all 27 checkpoints, **maximum simultaneously-alive
clusters holds at 50–55**, while **mean lifespan falls 7.0 → 4.5 and births rise 113 → 164**
(`math-1.md` §13.2). Carrying capacity is invariant; turnover is not. Two readings:

**(a)** the *same* particles cycle through clusters faster, or
**(b)** *different* particles cluster at late checkpoints.

**Cluster-level statistics cannot distinguish these — both produce identical births/deaths/lifespan
curves.** They are separable only per particle, and only if particle identity carries across
checkpoints, which is exactly what frozen token positions plus the byte-identical NeoX tokenizer
give. `turnover_decomposition` is that test, and **it is a groupby, not a new experiment.**

`particle_biography` reads the same table down a different axis: per token position, the layer at
which it first joins a stable cluster, and how that date moves across the sweep.

### 8.2 The complement is retained deliberately

**Every token gets a row at every layer** — including tokens in no role and tokens in no cluster.
The unclustered population is Phase 5c's object ("not a failure mode but a distinct phase"), and a
table restricted to cluster members would make it unrecoverable. This is the same instinct as
`math-1b.md` §5.4's `border_vs_noise`: the unclustered population is roughly half the tokens and
carries most of the attention mass late in the stack (`math-1.md` §13.1).

### 8.3 Two anchors, two different questions

`anchors.py` emits one `TokenSet` per anchor plus their overlap:

- **`anchor_final`** (step 143000) — *what became of these particles?*
- **`anchor_init`** (step 0) — *what happened to particles that started together?*

These are genuinely different questions and the overlap between the two sets is itself the
measurement: a large overlap means cluster membership is largely determined at initialization; a
small one means training reassigns it.

---

## 9. The blocker class, named

Blockers 2 (OV values) and 3 (Group D) are explicitly classified as **one** underlying problem:

> **producer/consumer mismatch** — one phase's writer and the next phase's reader disagree on
> names, shapes, or paths, without either side erroring loudly.

Blocker 1 (§1.2) is the same class. The v2 response is structural rather than local:
`core/artifacts.py` declares each phase's output contract **once**, and every consumer imports
those constants — killing the bug *class* rather than three instances. The instruction is
explicit: **don't fix these as isolated one-offs before that lands.**

This is the correct diagnosis, and it generalizes the rule Phase 1 arrived at from the other
direction — *if a quantity appears in a report, it is persisted* (`math-1.md` §14). Phase 5's
addition: **it is not enough to persist it; the reader and the writer must agree on the contract,
and disagreement must raise.** Every one of these three blockers failed silently and returned a
plausible empty value (`n/a`, `None`, `0.0`) that flowed into a score.

---

## 10. Code map

| File | Group | Role |
|---|---|---|
| `select_cluster.py` | — | Six-criterion selection score; `_merge_event_for_trajectory` |
| `anchors.py` | — | Run discovery → frozen token sets; `load_cluster_tracking` (the *correct* Phase 1 event reader, which belongs upstream in `p1_io`) |
| `token_sets.py` | — | What the object of study is, frozen |
| `cluster_profile.py` | A | Compactness, silhouette, CKA on the masked subpopulation |
| `v_alignment.py` | B | Subspace splits, energy restricted to cluster pairs, S/A local test, merge-event geometry, effective $\beta$ (§3.3) |
| `head_contributions.py` | C1 | Per-head cohesion; OV grounding currently missing |
| `ffn_contributions.py` | C2 | FFN pathway |
| `feature_signature.py` | D | Feature-space identity (blocked) |
| `train_tuned_lens.py`, `tuned_lens_cluster.py` | E | Lens fitting and application (§6) |
| `causal_tests.py` | F | Five interventions, dispatched per architecture through `core/intervention.py` |
| `sibling_contrast.py`, `tiers.py` | G | Sibling and random control; the control-vs-null distinction (§7.2) |
| `particle_join.py` | — | The particle table; `turnover_decomposition`, `particle_biography` |
| `sweep_geometry.py`, `constants.py`, `p5_io.py`, `report.py`, `run_5.py` | — | Support, IO, assembly, CLI |

---

## 11. Open questions

Tracked: the six blockers (§1.2, §3.4, §4, §5, §6, §7.1); the Theorem 6.3 mis-citation at four
sites (§3.2); and the deferred semantic-decode enrichment, correctly gated behind blocker 4 since
nothing built on the tuned-lens output is trustworthy until it is fixed.

Surfaced by writing this document:

1. **The selection-score arithmetic does not close, and the discrepancy is load-bearing** (§1.2).
   `design-5.md`'s "9.000 for 4/6" on a 9.0 scale is incompatible with `anchors.py`'s "two of six
   criteria dead, 5.0 of the 9.0 scale." Since the 9.000 figure is the stated evidence that
   selection was principled rather than arbitrary, **and Groups A–F are all conditioned on having
   selected a real object**, this needs reconciling before any Phase 5 result is cited. It is also
   a five-minute check: recompute the score with the fixed event reader and see whether the same
   trajectory wins.

2. **`orth_sq`'s clamp should be unclamped, and the projector basis should be recorded** (§3.1).
   If $U_{\rm att}$ and $U_{\rm rep}$ come from $W_{OV}$'s own eigenbasis, they are **not
   orthogonal** — Phase 2b measures the operator as strongly non-normal — so the projected
   energies can exceed the total and the `max(0,·)` silently absorbs it. A negative unclamped
   `orth_sq` would then be a **basis diagnostic** — structurally the same move `rotational_schur.py`
   makes in returning `henrici_absolute_unclamped` alongside the clamped value, where a materially
   negative number signals that the block parse disagrees with $T$. Report it, and attach the
   FrameSpec saying which decomposition produced the bases.

3. **Group F's metric should be restricted to layers at or after the hook.** Now that
   `PHASE5_PYTHIA.md` has diagnosed the identical values as an averaging artifact (§7.1) — every
   pre-hook layer pinned at 1.0 by construction — the fix is to compute `mean_frac_together` over
   **post-intervention layers only**, and to report the layer profile rather than the mean. The
   causal-robustness ordering (xl 0.80 > large 0.53 > … > medium 0.00) was computed under the
   contaminated average and should be recomputed before it is cited; note that averaging in a
   constant 1.0 over roughly half the chain compresses all differences toward each other, so the
   *true* spread between models is likely **larger** than reported, not smaller.

4. **Group B's S/A local test may be measuring something rotation-invariant, like Phase 2b's did.**
   `math-2b.md` §3.1 shows every Gram-based quantity is invariant under an orthogonal map, and the
   "locally rotational, universal — no cluster shows a locally non-rotational profile" finding has
   the same suspicious universality as the withdrawn `elim_rotation = 0.0`. **The check is the one
   from `math-2b.md` §3.2: what would this statistic read on data where the hypothesis is
   maximally false?** If a synthetic cluster built with a deliberately non-rotational update
   profile still reads "locally rotational," the test is not a test. Given that `design-5.md`
   presents this as *closing a possible gap* left by Phase 2b, and Phase 2b's result has since been
   withdrawn as an identity, this inheritance should be checked rather than assumed.

5. **The particle table makes a Phase 1 open question answerable, and nothing says so.**
   `math-1.md` §15.8 asks whether the step 3000–5000 effective-rank overshoot and the
   `repeated_tokens` separating force are the same mechanism. Both are *per-particle* phenomena
   (which particles gained directional spread; which identical tokens got separated), both are
   already columns in the join, and the answer is a groupby of exactly the same shape as
   `turnover_decomposition`. The particle table's value is being stated only in terms of the
   turnover question; it is more general than that.

6. **Two anchors are defined but their overlap has no registered prediction** (§8.3). "How much of
   final cluster membership was determined at initialization" is a sharp, cheap, developmentally
   meaningful quantity, and it bears directly on `PREDICTIONS.md` claim (a) — is the structure
   learned or initial? — which Phase 1 could only partially adjudicate. It deserves a registered
   prediction with a falsifier before the numbers exist, on the same discipline as every other
   claim in this project.
