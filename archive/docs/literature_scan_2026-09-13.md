<!-- docs/literature_scan_2026-09-13.md -->
# Literature scan, 2026-09-13 — the self-repair thread before registration

Run under `CLAUDE.md`'s **trigger 2**: an entry was about to be proposed for
`claims/registry.json` off §3.27, and registration freezes the wording, the
statistic and the null, so this is the last moment a literature fact can change
any of them. It changed them.

**Every id below was fetched and read as full text**, not from a search
summary. §3.16 records why that rule exists: the 2026-09-10 scan talked this
project out of a real finding on an unverified reading, and
`docs/literature_scan_2026-09-10.md` is marked superseded for it. Two papers
here were only readable by pulling the PDF and extracting the text locally; the
arXiv abstract pages were not sufficient to answer a single one of the questions
that mattered.

## Verdict in one line

**The core statistic of §3.22 is published** — independently derived here, but
published in July 2026 — **and the MLP result of §3.23–§3.25 sits precisely in
the gap that paper names as future work.**

---

## 1. `2607.01940` — *Conditional Co-Ablation: Recovering Self-Repair Backups in Transformer Circuits* (Gong, Zeng, Yuen, Lim; NTU Singapore; v1 2 Jul 2026)

**This is the paper §3.16 flagged as "looks structurally like
`pairwise_interaction_matrix.py`. Read that one first." It was not read. It
should have been.**

**Its score is §3.22's statistic.** Definition 1: the conditional ablation
effect of a unit `u` given an ablated set `S` is `‖δz_{u|S}‖²`, and *"the CoAx
score is the growth of this energy under conditioning"* — i.e. how much a unit's
ablation effect **grows once a primary set has been removed**. That is the
second-order object §3.22 computes as
`interaction = dNLL(S+u) − dNLL(S) − dNLL(u)`. The metrics differ (theirs is a
Fisher energy over logit coordinates, ours is ΔNLL on the second-copy readout);
the construction does not.

**Its central thesis is §3.22's headline.** From the abstract: first-order
scoring *"becomes misleading when a transformer self-repairs: after a primary
component is removed, a dormant backup can take over, muting the primary's
measured effect while the backup itself appears irrelevant on the intact
model."* §3.22's `L4H9` — solo ΔNLL **−0.001**, marginal **+1.018** — is an
instance of exactly this, and we presented it as a finding rather than as a
reproduction.

**It covers our rung, and induction.** *"The same label-free procedure transfers
to induction across eight models"* in six architecture families, including
**Pythia-160M, Pythia-410M and Pythia-1.4B**, Llama-3.1-8B, Qwen2.5-7B,
Gemma-2-2B, GPT-2 medium/large. The induction pipeline is *"detect
induction-head primaries by their own effect, seed CoAx on them, recover a
compensating set"*, with an attribution factor from **2.1× (Pythia-160M) to 12×
(Pythia-410M)**.

**What it explicitly does NOT do, in its own words:**

- **Head-level only.** *"The signal is instantiated primarily at attention-head
  granularity: it transfers to the attention-mediated induction circuit but not
  to the"* MLP-dominated case. On the greater-than circuit *"head-level CoAx
  does not recover"* the repair, which they read as *"consistent with
  greater-than's self-repair being mediated by MLPs that our head-level units do
  not"* cover.
- **MLP-mediated repair is future work.** A *"preliminary FFN-group probe"*
  (their Appendix C.4.3, 96 neuron groups on GPT-2-small) finds only limited
  signal, and they name *"a full FFN-level treatment of strongly MLP-mediated
  self-repair"* as outstanding, and *"which we leave to future work."*
- **No attention-pattern readout.** The score is *"output-grounded"*. Attention
  patterns appear twice and illustratively (an averaged pattern over 40
  template fillings; one token-level figure), never as the measurement.
- **No residual-stream geometry.** Their "Fisher geometry" is a metric over
  *logit* coordinates for scoring, and their directional language is about
  writing *"a correlated logit direction"*. Nothing characterises what a backup
  writes into the residual stream.

## 2. `2402.15390` — *Explorations of Self-Repair in Language Models* (Rushing & Nanda, ICML 2024)

**MLP participation in self-repair is published.** Self-repair is attributed to
*"changes in the final LayerNorm scaling factor and sparse sets of neurons
implementing Anti-Erasure"* — erasure neurons that normally contribute
negatively and become less negative when an upstream component is ablated.
Models: gpt2-small/medium/large and **pythia-160m, pythia-410m, pythia-1b**
(not 70m).

**What it does not do:** no characterisation of a compensating component's
residual-stream write **direction** or any rotation of it (searched; absent),
and no attention-pattern readout — previous-token and induction heads appear
once, anecdotally.

**It is also a rival hypothesis, and §3.23 already discriminates against it.**
Their mechanism is LayerNorm *scaling*. §3.23 rejected the scale reading for
MLP 6 three independent ways: `zero` and `mean` leave the residual entering
layer 7 at 43.07 vs 43.46 while giving attention 0.045 vs 0.643; a norm-matched
random constant reproduces `zero` exactly; and MLP 5 removes as much residual
norm as MLP 6 for a fraction of the damage. So the direction result is not their
effect measured differently.

## 3. Checked and not a hit

- `2604.01094` (*Temporal Dependencies in In-Context Learning*) — the search
  summary's "many-to-many wiring between previous-token and induction heads"
  does **not** appear in the abstract, and the abstract's actual claim is that
  removing high-induction-score heads reduces a +1 lag bias while random heads
  do not. Full text not obtained; **treated as unresolved, not as support.**
  Recorded because quoting that phrase from the summary would have been the
  2026-09-10 failure repeated.
- `2307.15771` (Hydra effect) and `2502.14010` (induction → function-vector)
  were already in §3.16 and are unchanged by this scan.

---

## What this does to our claims

**Reclassified as replication** (independently derived, published first):

1. §3.22's conditional marginal-vs-solo interaction statistic — **is CoAx**.
2. §3.22's framing that single-unit ablation scores mislead under self-repair,
   and that a dormant backup looks irrelevant on the intact model.
3. That backups for induction recover across model scale — CoAx, eight models,
   Pythia-410M included.
4. That MLPs participate in self-repair at all — Rushing & Nanda.

**Survives, and is sharpened by sitting in a named gap:**

1. **An MLP as the *dominant* backup for a specific head** — MLP 6 at **+6.26**,
   above every head including `L7H8`'s +4.07, on a solo effect of +0.14. CoAx is
   head-level, its FFN probe preliminary, and it names this as future work.
2. **The mechanism is an active rotation of a constant direction**: `mu_clean`
   restores nothing (0.038, on a par with zero and with noise) while `mu_cond`
   restores 0.643. Neither paper characterises a compensator's residual write,
   and the published mechanism (LayerNorm scaling) is ruled out here three ways.
3. **The rotation is aimed at the redundancy set's shared key read-space**
   (7–8× chance, six of the top nine of 272 heads are members) **and moves the
   heads it points at** (layer-centred Spearman +0.325 / +0.337).
4. **Attention/TV readouts and the ceiling-immunity argument.** Both papers are
   output-grounded. This is what made the 70m port possible where §3.17's
   ceiling censors every ΔNLL cell.
5. **The 70m/410m structural contrast** — the matcher sits in layer 0 *upstream*
   of the relay at 70m with a negative causal effect, and the MLP ordering
   inverts. No prev-token→matcher wiring analysis in either paper.

**What it does to §3.27's registration candidate.** "Relay support concentrated
on the causally-defined set, replicating across rungs" is adjacent to CoAx's
induction transfer across eight models — not identical (ours is the relay's
*support* over downstream heads, theirs is *backups of* an ablated primary), but
close enough that the cross-scale-replication-of-backup-structure card is
substantially weaker than it looked. **Registering it as written would have
claimed novelty this scan does not support.** The MLP-direction result is the
stronger card and it is the one the literature leaves open.

**Recommended next step, for a human decision:** do not register §3.27 as
worded. If anything from this thread is to be registered, the candidate with
literature cover is the **MLP-mediated, direction-carried repair** — and it
needs its own scan of the FFN/neuron-level interpretability literature, which
this one did not cover.
