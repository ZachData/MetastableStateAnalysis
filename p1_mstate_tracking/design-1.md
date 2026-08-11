# Phase 1 — DESIGN

## Core question

Geshkovski et al. (*A Mathematical Perspective on Transformers*) model token representations
as interacting particles on $\mathbb{S}^{d-1}$ and prove that they converge to a single
cluster in the long-time limit, passing through metastable multi-cluster states on the way.
The proof holds for a simplified model ($Q^\top K = V = I_d$). Phase 1 asks whether
metastability survives in trained architectures with learned weights, multi-head attention,
and FFN layers.

Falsification criterion: if no plateaus appear in cluster count or inner-product histograms
across consecutive layers, metastability doesn't survive at this scale and the project stops.
It didn't. The criterion passed on GPT-2/BERT/ALBERT and passes again on Pythia-410M — all
216 runs of the checkpoint pilot show plateaus.

## What the theory actually claims

The paragraph above said the paper "proves that they converge to a single cluster, passing
through metastable multi-cluster states on the way." The first half is proved. **The second
half is Problem 1, which the paper poses as open**, writing that it observes the behaviour in
numerics and is not able to explain it theoretically. Its supporting evidence is Figure 4, at
$d = 2$, $\beta = 4$ (two clusters) and $\beta = 9$ (three).

The distinction is load-bearing here, for two reasons.

**First, it changes what a pass means.** Plateaus in 216 runs do not confirm a theorem; they
are evidence bearing on a conjecture. A phase whose falsification criterion tests an open
problem is doing something worthwhile, but it is not replication and should not be written as
though a proved statement had been checked.

**Second, we are outside the regime the conjecture's evidence comes from.** Figure 3 shows the
metastable zone — the $(d,\beta)$ band where clustering probability is strictly between 0 and
1 — narrowing as $d$ grows and vanishing by $d \approx 512$, which the paper attributes to
Theorem 6.9's concentration of all pairwise inner products onto $\gamma_\beta(t)$. Pythia-410M
runs at $d = 1024$ with $n \le 512$.

So the criterion is restated: **does the plateau phenomenon the paper conjectures at low $d$
and high $\beta$ survive at $d = 1024$ under learned weights?** We observe that it does, which
means either our plateaus are a different object from the paper's metastability, or the
identity-weight concentration argument fails badly once weights are learned. Phase 1c's
$\gamma_\beta$ null model is the instrument that separates these: it integrates (6.9) at each
run's own $(n, \beta_{\rm eff})$, reparameterizes by effective integration time, and reports
the residual against observed `ip_mean`.

The other citations this doc inherited are corrected in `MATH.md` §9. The two that mattered
for design decisions: monotone $E_\beta$ is eq. (3.6) / Lemma 3.7, not Proposition 3.4, and
it is proved for **(SA)** — the normalized dynamics — so the "monotonicity is USA-only" hedge
that appeared in earlier drafts is dropped. Theorem 6.1 is qualitative ($d \ge 3$ at any
$\beta$); the rate results are Theorems 6.3 and 6.9.

## The sphere assumption is a measurement, not a convention

The theory lives on $\mathbb{S}^{d-1}$ because layer normalization, in continuous time,
becomes the tangent projection $P^\perp_x y = y - \langle x,y\rangle x$. But RMS-norm
multiplies by a *trained diagonal matrix*, so the true state space is a time-varying
axis-aligned ellipsoid. The paper sets that matrix to $I$ — and justifies it empirically
(§2.2): in ALBERT XLarge v2 the diagonal is essentially constant across layers, mean $0.44$,
sd $0.008$.

**That is a measurement, and it is reproducible on any model.** Which reframes what
`core/ln_frame.py` is for. It has been described in this project as a departure from the
paper's frame; it is the opposite — it is the paper's own licensing check, run on a model
where it may fail. If Pythia's LayerNorm $\gamma$ has wide dynamic range per layer, the
correct manifold is the ellipsoid and **every sphere-frame metric in this phase inherits a
distortion**, including `ip_mean`, `ip_mass_near_1`, and the interaction energy.

Two further consequences follow from the same place, and both are Phase 1c sub-experiment D:

- Plain LayerNorm ($\gamma = 1$, $\beta_{\rm LN} = 0$) is *exactly* sphere projection in the
  mean-zero subspace: $\mathrm{LN}(x) = \sqrt{d}\,P_{\mathbf 1}x/\|P_{\mathbf 1}x\|$,
  constant norm $\sqrt d$. So the LN frame structurally restores uniform token weights and
  removes the sink domination that D10 identifies in raw effective rank. This is a reason to
  prefer the LN frame that has nothing to do with fidelity to the paper.
- The learned LN *bias* adds a fixed vector to every token — pure common mode. It inflates
  $\langle G\rangle$ by roughly $\|\beta_{\rm LN}\|^2$ regardless of input, and
  $\langle G\rangle/2$ is the dominant term in the small-$\beta$ expansion of $E_\beta$.
  **The learned LN bias puts a floor under the interaction energy that has nothing to do with
  the tokens**, which is a candidate confound for every absolute energy number this phase
  reports.

## What changed: the object of study is now a trajectory

The GPT-2-era design swept *architectures* — 7 models × 8 prompts, with ALBERT run at four
iteration depths because its shared weights make depth a free variable. Every quantity was
measured on a finished network, and "is this architectural or learned?" was answered by a
10-seed random-weight sweep: architecture-level quantities are those invariant under any
weight draw.

That design has a blind spot the Pythia pilot makes obvious. A random-weight twin tells you
whether a property survives destroying the weights. It does not tell you when the property
arrives, in what order relative to other properties, or whether it arrives monotonically. On
Pythia-410M all three answers are surprising:

- **Energy monotonicity is not destroyed by randomization — it is destroyed by training.**
  At step 0 there are 3 violations across 8 prompts at every $\beta$. By step 512 there are
  64. The GPT-2 run's "Theorem 3.4 falsified universally, including under random weights" was
  a statement about GPT-2's initialization, not about transformers — and it cited the wrong
  result. Monotonicity is eq. (3.6) for (SA) and Lemma 3.7 for (USA); Proposition 3.4
  characterizes the extremizers of $E_\beta$ (uniform measure is the unique global minimizer,
  Diracs are the maximizers) and says nothing about trajectories.
- **The transitions do not co-locate.** Effective rank moves at steps 8–32. Energy breaks at
  256–512. Plateau onset becomes content-driven at exactly 512. Fiedler deviation crosses
  zero between 1000 and 3000. A single-checkpoint study necessarily reports these as one
  bundled "trained" state.
- **Nothing is monotone.** Effective rank collapses (step 16), recovers (512), overshoots its
  initial value by 3× (step 3000–5000), then declines for 140k steps. Violation severity
  rises to step 60000 and then falls. Reading any of these off two endpoints gives the wrong
  sign.

So the phase's unit of analysis is now a checkpoint series, and the random-weight twin is
demoted from *the* trained-vs-untrained control to *one point* on a trajectory that also
contains a true step-0 init. The v2 plan's two-baseline policy already required keeping
`pythia-1.4b-random` (norm-matched to the final checkpoint) separate from step-0 init,
because GPT-NeoX's init variance-scaling is not comparable to GPT-2's and attraction dynamics
scale with weight norms. The pilot strengthens that: step 0 is not merely a stand-in for
"random," it is the only checkpoint in the entire sweep where the network sits in the
attractive regime eq. (3.6) describes. It has to be its own object.

## Why the prompt set and controls are what they are

Eight metastability prompts spanning 20 to 512 tokens, four natural languages/registers, code,
and LaTeX. The length spread exists to separate token-count effects from content effects; the
register spread exists so that "content-driven" means something more than "different tokens."

`repeated_tokens` is run as a separate control and excluded from every metastability analysis,
because a degenerate initial distribution tests collapse *speed*, not metastability. This
exclusion earns its keep in the pilot: the control turns out to carry the cleanest single
result in the sweep. At init the network leaves a degenerate input degenerate (final mass
0.948, rank 1.11); by step 143000 it actively separates it (mass 0.379, rank 2.02), with the
change arriving around steps 11k–13k. Training installs a *separating* force. That is the
direct empirical counterpart to the attractive/repulsive tension the paper's framework
predicts, and it would have been invisible had the control been folded into the main table.

## Measurement frames — the part that needs the most care

Most of the pilot's ambiguity is not in the phenomena but in the definitions. Three of the
phase's headline metrics turn out to be measured in a frame that does not survive the move to
a trained GPT-NeoX model. The rationale for each, and where it breaks:

**Effective rank: raw vs. normed.** `core/metrics.py` exposes one function with an explicit
mode. `raw` runs the SVD on unnormalized activations and captures both scale and directional
collapse; `normed` runs it on L2-normed activations and measures directional spread on the
sphere alone. The theory is about particles on $\mathbb{S}^{d-1}$, so `normed` is the
frame-correct quantity — but `raw` is the right gate for degeneracy checks, because a
near-zero-norm cloud makes NN assignment float-noise regardless of direction. Both are
computed per layer and both are persisted. The defect is that the *report* reads `raw` for
its headline `MinRank` column. Trained transformers develop massive-norm outlier tokens; a
single one drags raw effective rank toward 2 with no directional collapse at all. Any claim
of the form "representations collapse to rank 2 by end of training" must be made on `normed`
or not at all. Keeping both keys rather than replacing one with the other was the right call
(`p1_io.py` comments this explicitly); the mistake was letting the report choose silently.

**Fiedler value: why there is a baseline, and why the thresholds don't survive it.**
GPT-2-style attention is lower-triangular. Sinkhorn-normalizing a triangular matrix and
symmetrizing $(P+P^\top)/2$ forces a low-connectivity graph regardless of content, which was
manufacturing "100% STABLE-CLUSTER" independent of what the model had learned. Fix 3
subtracts a content-free baseline: build uniform causal attention ($A_{ij} = 1/(i{+}1)$ for
$j \le i$), Sinkhorn it, take its $\lambda_2$, and classify each head on *actual − baseline*.
The question the deviation answers is the right one — "does this head route into clusters
beyond what the mask already forces?" — and the deviation is legitimately signed. That is why
the pilot shows negative values: $\lambda_2$ of a normalized Laplacian is non-negative by
construction, and nothing is wrong with the eigensolver, but the reported column carries the
deviation while its header says `MeanFiedler`.

What the baseline subtraction broke, and nobody re-derived, is the classification. The
CLUSTER/MIXED/MIXING cutoffs at 0.3 and 0.7 were calibrated for raw $\lambda_2$ on $[0,1]$.
For this run's prompt lengths the computed baselines are $\lambda_2 = 0.0640$ ($n{=}242$),
$0.0654$ ($n{=}467$), $0.0658$ ($n{=}512$), $0.1089$ ($n{=}20$) — so both the deviations
(±0.05) and the raw values (0.02–0.07) sit two orders of magnitude below the CLUSTER cutoff.
Every head classifies CLUSTER on either quantity, at every checkpoint, on every prompt. The
"100% STABLE-CLUSTER" result is a restatement of the thresholds, not a finding about the
model. Any future version has to derive cutoffs from the baseline scale — the natural
normalization is deviation / baseline, which bounds the quantity in $[-1, \infty)$ and makes
the floor ($\lambda_2 = 0$, total separation) interpretable.

The baseline's $n$-dependence is a second, separable problem. At $n{=}20$ the Sinkhorn fixed
point puts 74.5% of mass on the diagonal, so the short-prompt baseline is measuring a
mostly-self-loop graph — qualitatively a different object from the $n{=}512$ case, and 1.7×
larger. Averaging deviations across prompts of different lengths averages against different
baselines, and the spread between them is comparable to the entire trained signal. Per-length
reporting, or the ratio normalization above, is required before cross-prompt means mean
anything.

**Mass near 1: the floor problem.** `ip_mass_near_1` is the fraction of pairwise inner
products above threshold, and the summary table reduces it with max-over-layers. At layer 0
every duplicate token pair has IP = 1 exactly, so the max is pinned to a prompt-determined
floor. `wiki_paragraph` reads 0.0148 at step 0 and 0.0149 at step 143000 — a constant, and
not a fact about the model. The only checkpoints where genuine layer-wise clustering exceeds
that floor are steps 16–256, where the transient collapse pushes it to 0.5–0.8. Meanwhile the
actual signal is in the other direction: mid-network mass at step 143000 drops to 0.0007, a
factor of 20 *below* the embedding floor, which is the same separating force the
`repeated_tokens` control shows. Max-over-layers is exactly the wrong reduction for this
metric on a trained model.

**Spectral $k$ does not transfer.** The eigengap cluster-count estimator returns $k = 1.0000$
in all 216 runs at every reported plateau layer, so the `nMerges` column — defined as
spectral-$k$ drop events — is dead on this architecture. The Gram matrix is clipped to
positive with the diagonal filled to 1 before the Laplacian is taken; under Pythia's pervasive
anisotropy every pairwise IP is positive and large, giving one connected component with no
gap. P1-1 cluster tracking, which was added to replace spectral-$k$ counting with token-level
accounting, is doing that job well (25–45 merges per run, with layer locations) and should be
the sole merge instrument going forward.

The general lesson, and the reason this section exists: **every threshold in this phase was
calibrated on GPT-2/BERT activations, and thresholds do not port across architectures.**
`DEGENERATE_RANK_THRESHOLD = 2`, the Fiedler `active_rank_threshold = 10.0`, the 0.3/0.7
classification cutoffs, `ext_sem_threshold = 0.5`, `ENERGY_VIOLATION_REL_TOL = 1e-3` — each
is a free parameter that was set once, against one activation distribution, and then
inherited. Where a threshold gates layer inclusion it is worse than a wrong number, because
the gated set then varies with the checkpoint and the developmental curve acquires a moving
denominator. Fix 7's tolerance sweep was scoped for exactly this and has not been run against
Pythia.

## Silent fallbacks are worse than failures

Two of the pilot's confounds are fallbacks that fire without leaving a trace:

- `_per_head_fiedler_profile` filters to layers with raw effective rank ≥ 10, then falls back
  to *all* layers if none qualify. Raw rank ranges 40 → 2 across this sweep, so the filter is
  checkpoint-dependent, and the fallback fires at steps 16–32 where everything is below
  threshold. Nothing in the output distinguishes a filtered profile from a fallback one.
- `sinkhorn_cluster_count` falls back to a hard $\lambda > 0.5$ count when no clear eigengap
  exists. On a model where no clear eigengap ever exists, that fallback is the whole metric.

Both should emit the branch taken. The design principle: any filter, gate, or fallback whose
behaviour depends on the data must record what it did in the artifact, because the alternative
is a curve that looks like a finding and is partly a filter.

## The artifact contract

`io_utils.py` writes one JSON per metric family (each < 100KB) rather than a monolithic
`metrics.json`, so downstream phases load only what they need. Everything Phase 2 needs —
activations, plateau windows, merge indices, violation layers, energy-drop token pairs — is
saved specifically so Phase 2 never re-runs a forward pass.

The pilot exposed a hole in that contract. `sinkhorn.json` persists `fiedler_mean` only;
`fiedler_per_head`, `fiedler_per_head_deviation`, and `fiedler_baseline` are computed and
never written. `_per_head_fiedler_profile` reads `sinkhorn["fiedler_per_head"]` and returns
an empty list on any reloaded run, so the entire per-head Fiedler section — one of the
phase's headline results — silently disappears when the report is regenerated from artifacts.
It exists in the pilot's output only because that report was written in-session from
in-memory results.

This is the artifact-contract bug class the transition plan names, and it has a cost the plan
predicted: the effective-rank fix is a re-report, because `effective_rank_normed` is on disk;
the Fiedler fix is a rerun, because the per-head values are not. **The rule going forward: if
a quantity appears in a report, it is persisted.** A derived statistic that can only be
computed in the session that produced it is not a result.

## Module architecture

- `run_1.py` — CLI orchestrator. `--random-baseline` adds the untrained control (part of the
  standard run); `--sublayer` captures post-attention/post-FFN streams separately
  (supplementary, excluded from cross-run comparison to avoid conflating decomposition
  granularities). ALBERT extended mode ran one forward pass to max depth and sliced at
  snapshots rather than re-running per depth, since the iterated map is deterministic given
  the input — retained but inert for Pythia.
- `analysis_p1.py` — one pass per layer, computing every metric off pre-normed activations and
  a single Gram matrix, so the same pairwise structure isn't rebuilt per metric.
- `metrics.py` / `clustering.py` — scalar metrics and clustering/projection kept separate:
  metrics are cheap and always computed; clustering (HDBSCAN, multi-scale nesting, pair
  agreement) is heavier and independently interpretable.
- `sinkhorn.py` — Sinkhorn-Knopp normalization plus Fiedler analysis, isolated because the
  doubly stochastic form is the gradient-flow object in the paper's Section 3.3 and the gap
  between raw attention and that form is itself a measurement.
- `reporting_p1.py` — the cross-run report is the primary artifact for downstream analysis,
  not the per-run report, because every later phase cross-references across the full
  model×prompt grid. On a checkpoint sweep "model" is a checkpoint, so this file is now also
  the trajectory instrument, and its reductions (max-over-layers, mean-over-prompts) need the
  scrutiny the previous section gives them.
- `checkpoint_*.py` — the sweep layer added for this pilot. `checkpoint_scalars.py` keeps a
  hand-synced copy of `ENERGY_VIOLATION_REL_TOL` because it deliberately avoids importing
  torch; that duplication is a known cost and is flagged in both files.
- `core/config.py`, `core/models.py` — global registry and loading factored out early because
  every subsequent phase re-imports the same model list and extraction logic. This is also why
  the duplicate `config.py` in `p1_mstate_tracking/` vs `core/` is flagged for collapsing.

## Output format

Per run: `geometry.json`, `energies.json`, `clustering.json`, `spectral.json`,
`sinkhorn.json`, `activations.npz`, `attentions.npz`, `clusters.npz`,
`centroid_trajectories.npz`, `fiedler_vecs.npz`, `llm_report.txt`. Session-level:
`llm_cross_run_report.txt`, `experiment.txt`.

## Relationship to Phase 2

Phase 1 measures the *outcome* of the attractive/repulsive tension the paper predicts —
softmax attention pulls together, $V$'s mixed-sign eigenspectrum pushes apart. Phase 2
measures the tension itself. That division (outcome vs. mechanism) is why Phase 2 is a
separate directory rather than an extension of this analysis loop.

The pilot sharpens what Phase 2 is now for. Energy violation was already the observation
Phase 2 existed to explain; it is now a *dated* observation — absent at init, onset at step
256, saturating in count by step 512, with severity peaking at step 60000 and then declining
while the count stays flat. Count and magnitude coming apart late in training is a mechanism
question Phase 1 cannot answer with the metric set it has.

Pythia's parallel residual is the instrument for it. $\Delta x = \text{attn\_out} +
\text{ffn\_out}$ from the same input makes the attn-vs-FFN decomposition exact, with no
ordering confound — something GPT-2's sequential architecture never offered. The
energy-attribution extension (`energy_decomposition.py`,
`energy_attribution_aggregate.py`) previously had no real Pythia story and a skip-fallback;
it should now be built natively against the parallel residual and pointed at the severity
decline first.
