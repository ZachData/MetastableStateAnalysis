<!-- p7_motifs/lit-7.md -->
# Phase 7 — LITERATURE

**Status: LEADS, NOT READINGS.** Produced 2026-09-16 by web search only. **No paper
below has been read**; `arxiv.org` and every other scholarly host are blocked by this
session's egress proxy. Marks: **[S]** search-engine summary read, **[N]** title and
id only.

**The headline: motif analysis over attention graphs exists, with null models, and
somebody has already written the "network science meets mechanistic interpretability"
paper — for graph transformers. The part of Phase 7 that has no neighbour is the one
`design-7.md` identified as the point of the phase: *the edge carries the force, not
the attention weight.***

---

## 1. The territory, and who is in it

### 1.1 Attention graphs as network-science objects

*Towards Mechanistic Interpretability of Graph Transformers via Attention Graphs*,
arXiv **2502.12352** **[S]**: "mechanistically interprets GNNs and Graph Transformers
by analyzing their attention patterns **from the perspective of network science**.
While previous work on Transformer circuits focused on discrete feature interactions,
this approach captures **continuous information flow patterns** between nodes through
Attention Graphs."

That is Phase 7's framing sentence, for a different model class. **Read it for what it
did with nulls and for its stated limitation**, which the summary gives verbatim and
which lands directly on `interaction_graph.py`:

> "Aggregating heterogeneous attention patterns across heads and temporal patterns
> across layers into a single matrix may **oversimplify** complex model dynamics."

Phase 7 aggregates across heads in exactly that way at the graph level. The
`extra__` event columns are the intended escape, but the caveat is real and published.

### 1.2 Motifs with statistical nulls

The GNN-explainability line has the motif machinery and the null discipline:
*MotifExplainer* **[S]** ("recurrent and statistically significant patterns … better
human-understandable explanations than methods based on nodes, edges, and regular
subgraphs"); *MAGE: Model-Level GNN Explanations via Motif-based Graph Generation*,
**2405.12519** **[N]**; *Studying and Improving GNN-based Motif Estimation*,
**2506.15709** **[S]**, which explicitly discusses "motif scores with clear
statistical relevance and connections to traditional estimation **based on a null
model**."

**Consequence for `design-7.md`:** the "motif zoo" hazard the design defends against
by pre-committing to seven motifs is a known hazard with known partial solutions. The
pre-commitment is still the right call — but it should be presented as a choice among
known options, not as a discovery.

### 1.3 The induction-circuit target

- **2606.02378** *When Do Attention Circuits Form?* **[S]** — identifies **induction,
  previous-token and BOS-attractor heads** across 10 log-spaced revisions on three
  1B-class models, via a participation-ratio spectral signal and a selectivity
  screen. It has `prev_token` and `match` as head classes on a checkpoint axis.
  See `lit-8.md` §1, where this paper matters most.
- **2511.16893** *Predicting the Formation of Induction Heads* **[S]**, NeurIPS 2025 /
  ICML 2026 — **a simple equation in batch size and context size predicts the
  formation point**, plus a Pareto frontier in bigram repetition frequency and
  reliability. See §4.1.
- **2404.07129** *What needs to go right for an induction head?* **[N]** — a
  mechanistic account of induction-head formation. Unread and directly on target.
- **2310.04625** *Copy Suppression* **[S]** — the `repulsor` motif has a documented
  instance (`lit-2.md` §1.2).

---

## 2. Finding by finding

| Phase 7 element | Nearest prior work | Verdict |
|---|---|---|
| Attention as a graph over tokens, analysed with network science | **2502.12352** **[S]** | **NOT NEW** |
| Motifs with matched nulls | MotifExplainer / MAGE / **2506.15709** **[S]** | **NOT NEW** |
| Pre-committed motif alphabet rather than exhaustive enumeration | Implicit in the motif literature; not found stated as an anti-fishing discipline | **Ours as a discipline, not as a technique** |
| **The edge weight is the force `A_ij · V x_j`, not the attention weight** | **Nothing found.** Every attention-graph source found uses post-softmax attention as the edge weight | **NEW, and it is the phase's whole claim to independence** |
| **Edges typed by sign channel (`U_pos`/`U_neg`) and rotational channel (`U_S`/`U_A`)** | Nothing found | **NEW** |
| `relay` = induction head restated as a two-stage particle motif | The two-stage prev-token→matcher decomposition is standard mechinterp (Elhage 2021, Olsson 2022); restating it as a force motif is not | **Restatement of a known circuit in a new vocabulary — the value is entirely in whether the vocabulary predicts something the old one does not** |
| Particle events (`capture`, `hold`, `escape`, `relay_target`, `moved_fraction`) as `extra__` columns on the particle table | Nothing found | **NEW** |
| Rotary-offset nulls N1/N2/N3 for `Δ`-confounded pair sets | **2607.06621** *Fingerprint, Not Blueprint: How Positional Schemes Set the Default Spectral Algebra of Attention* **[N]** is adjacent; the specific offset-matched null is not found | **Looks new**, and `core/qk_offset_null.py` predates this scan |
| The tautology warning (behavioural induction score vs an attentive-edge motif are the same number) | Nothing found stating it | **Ours, and it is the single most important paragraph in `design-7.md`** |

---

## 3. What survives

### 3.1 Force-typed edges

Stated plainly, because it is the finding of this scan: **the field's attention graphs
are routing diagrams; Phase 7's is a force diagram.** `design-7.md` already argues the
distinction ("two heads with identical attention patterns and opposite-signed OV
circuits produce opposite motion"), and the searches found nobody doing it. Combined
with `lit-2.md` §1.1 — the sign channel is the copying/anti-copying axis, which is
well established — the construction is **novel in composition, from parts the field
already trusts.** That is a good position.

### 3.2 The tautology warning is transferable

`design-7.md`'s requirement — *any result claiming an association between motif and
behaviour must state which of the three independence sources is carrying it* — is a
general hazard for every "circuit X explains behaviour Y" claim where the circuit is
defined by the same pattern the behaviour is scored on. It is stated more sharply here
than anything the searches returned. It belongs in any methods write-up.

### 3.3 What does not survive contact

**"Mechinterp phenomena as particle motifs" is not, by itself, a result.**
`design-7.md` says so, and the literature confirms it: the translation table's value
is entirely in the entries that bottom out in a measured quantity with a null. Four of
the seven motifs (`sink`, `hub`, `mutual`, `repulsor`) currently have a definition and
no measurement.

---

## 4. Directions to grow

### 4.1 Check the formation-point equation against our window — free, external, quantitative

**2511.16893** **[S]** gives *"a simple equation combining batch size and context size
predicts the point at which IHs form."* Pythia's training configuration is public and
fixed (batch size and sequence length are in the Pythia paper and the model cards).
7d measured five of six members forming inside `(512, 2000]`.

> **Compute the equation's predicted formation step for Pythia-410m and Pythia-70m and
> compare it against the measured windows.**

This costs nothing, uses published constants, and is a genuine external
adjudication — and `claims/adjudications/` holds zero against 39 registrations. It
also transfers straight to Phase 8, where the ladder changes nothing about batch size
or context but changes everything else. **Do this one first.**

### 4.2 Run the force-typed graph against the attention-typed graph, head to head

The cleanest demonstration that §3.1's distinction matters: **find a pair of heads with
high attention-pattern similarity and opposite sign-channel typing**, and show they
appear identical in an attention graph and opposite in a force graph. 7d already
supplies candidates — §3.12-S found write subspaces overlapping at chance while
residual effects were 87 % aligned, which is the same dissociation from the other
side. One figure, weights and one forward pass.

### 4.3 The BOS/sink motif is not optional any more

`sink` is in the alphabet because Phase 6 found the same-content null collapsing onto
the sink column. Since then, **2510.06477** **[S]** (`lit-1.md`) has tied sinks to
mid-layer compression with a theorem, and **2606.02378** **[S]** has made
"BOS-attractor head" a classified head type with its own emergence curve — and found
that **capability-circuit formation and attention-sink formation are two different
transitions, separated by 10–20× in tokens** on DCLM models. Phase 7's `sink` motif
now has a literature, a timing claim, and a reason to be measured before the others:
**if sinks and induction form at different times, a motif analysis that cannot
separate them will attribute one to the other.**

### 4.4 Address 2502.12352's aggregation caveat explicitly

Per-head graphs, or a stated argument for why aggregation is safe here. "Effective `n`
is heads, not edges" (`design-7.md`'s standing constraint) is the same insight; make it
a measurement rather than a caution.

---

## 5. Verification queue

1. **2511.16893** — the formation-point equation in closed form. Decides §4.1.
2. **2502.12352** — what nulls it uses and how it handles head aggregation.
3. **2606.02378** — its `prev_token` and BOS-head definitions, against our
   `prev_token` and `sink` motifs.
4. **2404.07129** *What needs to go right for an induction head?* — the mechanistic
   account; likely overlaps `relay` directly.
5. **2506.15709** — motif-score nulls, for whether our N1/N2/N3 have a standard
   counterpart.
6. **2608.22007** *The Communication Map of a Transformer* **[N]** — surfaced under
   the 7d searches; the title suggests an interaction-graph object.

---

## 6. Search log (2026-09-16)

- `attention graph motif analysis network motifs transformer circuits subgraph over-representation null model interpretability`
- `"Predicting the Formation of Induction Heads" batch size context size bigram repetition frequency Aoyama Wilcox`
- `induction head formation Pythia checkpoints developmental interpretability arXiv`
- (inherited) `copy suppression head negative eigenvalues OV circuit McDougall anti-copying attention head GPT-2 10.7`
