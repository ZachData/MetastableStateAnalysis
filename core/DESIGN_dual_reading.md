# Dual-reading primitive — DESIGN

Transition plan v2, core analysis primitives, item 4 of 4 (last of the four:
population selector, merged intervention+logits runner, tracking-module
merge, this one). Plan text: "given a point of interest (token, cluster,
checkpoint), return a paired geometric reading (V-attractive/repulsive
projection, real/imaginary split, effective-rank contribution) and semantic
reading (frozen-head decode top-k + entropy via `embed_out`, LDA/probe
membership). Its output schema is written into DESIGN.md before any
implementation — this is the primitive most at risk of becoming a
god-function. No SAE/LRAE features in the semantic half."

This document is that schema. Written before `core/dual_reading.py`, per the
plan's own instruction — not a description of code that already exists.

## What a "point of interest" is

The plan lists three things a point of interest can be: a token, a cluster,
a checkpoint. Read literally as three *alternative kinds* of subject, this
is ambiguous ("a checkpoint" isn't a location within a forward pass the way
a token or cluster is). Resolved as follows, tying directly to
`core/particles.py`'s existing schema rather than inventing a new one:

- **Token** = one particle record: `(model, checkpoint_step, prompt_key,
  layer, token_position)` — a single activation vector.
- **Cluster** = a *set* of particle records sharing `(model, checkpoint_step,
  prompt_key, layer, cluster_label)` — read at its centroid, the same
  convention `tuned_lens_cluster.py::decode_cluster_trajectory` already uses
  for cluster-level decoding (mean activation, renormalized).
- **Checkpoint** is not a third kind of subject — it's the `checkpoint_step`
  field already in the particle-record key. Every token or cluster reading
  is already scoped to a specific checkpoint via that field; "a checkpoint"
  as a point of interest is asking for a reading characterized at the
  population level (e.g. "all clusters at step 16000"), which this
  primitive serves by being called once per cluster/token and aggregated by
  the caller — not by adding a fourth, different code path here.

This keeps "point of interest" to exactly two shapes (one vector, or a set
of vectors reduced to a centroid vector) rather than three, and both
resolve to the same downstream computation once you have "the vector to
read." That collapse is what keeps this primitive thin.

## What this primitive is, mechanically

**An orchestrator, not a computer.** Every non-trivial computation it
reports already exists somewhere else in the codebase. This primitive's own
new code is the plumbing that calls those existing functions on one point
and assembles the result — plus exactly one genuinely new, small
definition (effective-rank contribution — see below), because no existing
function computes a *per-point* contribution to a population-level metric.

It does **not**: fit a probe, fit an LDA direction, build V-subspace
projectors, run clustering, or load a model. All of those are supplied by
the caller as already-computed inputs (a projector dict, an LDA direction,
optionally a fitted probe classifier, a loaded model+tokenizer for the
semantic half). This is the specific, deliberate guard against the
god-function risk the plan calls out: a function that builds its own
projectors/probes/models on demand accretes responsibilities indefinitely;
a function that only *reads with what it's handed* cannot.

## Inputs

```python
def dual_reading(
    vector: np.ndarray,          # (d,) — the point's own activation vector,
                                  # or a cluster's centroid (caller's choice
                                  # of which; this function doesn't know or
                                  # care which one it was given)
    population: np.ndarray,      # (n, d) — every token's activation at the
                                  # same (checkpoint, prompt, layer) as
                                  # `vector`, INCLUDING vector's own
                                  # contributor(s) — the population
                                  # effective-rank contribution is measured
                                  # against.
    projectors: dict,             # this layer's {"U_pos", "U_neg", "U_A",
                                  # "U_S"} — same dict shape
                                  # probe_subspace.py / eigenspace_degeneracy.py
                                  # already consume (subspace_build's output).
    point_membership_mask: np.ndarray | None = None,
                                  # (n,) bool — which rows of `population`
                                  # this point of interest corresponds to,
                                  # for the leave-these-out effective-rank
                                  # delta. A single token: one True. A
                                  # cluster: every member True. None ->
                                  # effective_rank_contribution is NaN
                                  # (can't define "leave it out" without
                                  # knowing what "it" is among `population`).
    lda_direction: np.ndarray | None = None,   # (d,) unit vector, or None
    probe: object | None = None,  # a fitted sklearn-like classifier with
                                  # .predict(X) taking (1, d), or None
    model=None,                   # HF model for the semantic half, or None
    tokenizer=None,               # required together with `model`
    top_k: int = 20,
) -> dict:
```

Every one of `projectors`, `point_membership_mask`, `lda_direction`,
`probe`, `model`/`tokenizer` is optional independently. Missing input means
the corresponding output field(s) are `None`/`NaN`, never an exception and
never a silently-wrong default — a reading with half its fields `None`
because the caller didn't supply a probe is a correct, informative result,
not a degraded one.

## Output schema

Two top-level keys, `geometric` and `semantic`, each a flat dict — paired,
per the plan's own framing ("a paired geometric reading ... and semantic
reading"), not merged into one namespace, so a caller who only wants one
half can `result["geometric"]` and ignore the other without the fields
being interleaved.

```python
{
  "geometric": {
    "v_attractive_frac": float,        # ||U_pos^T v||^2 / ||v||^2
    "v_repulsive_frac":  float,        # ||U_neg^T v||^2 / ||v||^2
    "real_frac":         float | None, # ||U_S^T v||^2  / ||v||^2  (None if
                                       # projectors has no "U_S")
    "imag_frac":         float | None, # ||U_A^T v||^2  / ||v||^2
    "effective_rank_contribution": float,  # NaN if point_membership_mask
                                       # is None — see definition below
  },
  "semantic": {
    "decode_entropy":       float | None,  # None if model/tokenizer absent
    "decode_top1_id":       int | None,
    "decode_top1_token":    str | None,
    "decode_top1_prob":     float | None,
    "decode_top_k":         list[dict] | None,  # full {token,id,prob} list
                                       # — NOT written into a ParticleTable
                                       # column (see "Particle-table
                                       # projection" below); present here
                                       # for direct/ad-hoc callers.
    "lda_projection":       float | None,  # v . lda_direction (unit vector
                                       # assumed; None if lda_direction absent)
    "probe_predicted_label": int | None,   # probe.predict(v)[0], None if
                                       # probe absent
  },
}
```

`v_attractive_frac`/`v_repulsive_frac` are named to match
`core/particles.py`'s already-reserved `v_attractive_proj` /
`v_repulsive_proj` columns (that file was written expecting this primitive
to fill them — see its docstring: "once core analysis primitives item 3
lands) a V-projection and dual-reading-primitive output"). The `_frac`
suffix here (fraction of squared norm) vs. `_proj` there (the file's own
placeholder name) is a naming choice this doc fixes now, before
implementation, precisely so the two don't drift — `dual_reading`'s
`geometric.v_attractive_frac` is the value that goes into a
`ParticleTable`'s `v_attractive_proj` column.

## Particle-table projection

`core/particles.py`'s `extra` columns are `{name: (n_rows,) array}` — one
scalar per row, not nested structures. `decode_top_k` (a list of dicts)
doesn't fit that contract and isn't meant to: a caller writing many rows
into a `ParticleTable` uses only the scalar subset —

```
v_attractive_proj   <- geometric["v_attractive_frac"]
v_repulsive_proj    <- geometric["v_repulsive_frac"]
extra__real_frac                    <- geometric["real_frac"]
extra__imag_frac                    <- geometric["imag_frac"]
extra__eff_rank_contribution        <- geometric["effective_rank_contribution"]
extra__decode_entropy               <- semantic["decode_entropy"]
extra__decode_top1_id               <- semantic["decode_top1_id"]
extra__decode_top1_prob             <- semantic["decode_top1_prob"]
extra__lda_projection               <- semantic["lda_projection"]
extra__probe_predicted_label        <- semantic["probe_predicted_label"]
```

`decode_top1_token` and `decode_top_k` stay out of the particle table
(string / nested, not scalar-numeric) — available from the direct
`dual_reading()` return value for single-point inspection, not from bulk
columnar storage. This is a real, accepted loss of information in the bulk
path, not an oversight: the plan's own artifact-contract discipline (core
infrastructure item 2) is specifically about every producer/consumer
knowing exactly what shape they're getting, and a table column that's
sometimes a string, sometimes a list, defeats that.

## effective_rank_contribution — the one new definition

No existing function computes a per-point contribution to a
population-level metric; `core.metrics.effective_rank` takes a whole
population and returns one scalar. Defined here as a **leave-out delta**:

```
effective_rank_contribution = effective_rank(population)
                             - effective_rank(population[~point_membership_mask])
```

i.e. how much the population's effective rank *drops* when this point (or
every member of this cluster) is removed. Positive means the point was
propping the rank up (an individuated point, not redundant with the rest of
the population); near zero means removing it costs nothing (redundant with
what's left — consistent with the collapse-dynamics framing everywhere
else in this project). This reuses `core.metrics.effective_rank` verbatim,
twice, rather than a new eigendecomposition — the only new logic is the
subtraction and the mask-handling around it. Chosen over alternatives
(e.g. a leverage-score / eigenvector-loading approach) because it needs
zero new machinery beyond a function that already exists and is already
trusted (oracle-tested per the plan's testing policy), at the cost of being
O(population size) per query rather than O(1) — acceptable for a primitive
meant to be called per cluster or per sampled token, not per token at
every layer of every checkpoint.

**Caveat found while testing this definition (not while designing it):**
`effective_rank`'s `mode="raw"` is scale-sensitive — a point with much
larger norm than the rest of the population can dominate the *whole*
population's raw singular spectrum, which can make the full-population
effective rank *lower* (more collapsed-looking) than the same population
without it, inverting the naive expectation that "a big outlier is highly
individuated, so removing it should cost a lot." What actually drives a
large, positive `effective_rank_contribution` is the point occupying a
direction underrepresented in the rest of the population — not raw
magnitude by itself. A caller comparing contributions across points with
very different norms should keep this in mind, or normalize `population`
and `vector` consistently before calling if magnitude-independence is
what's wanted (nothing here does that normalization automatically, since
`mode="raw"` vs `mode="normed"` is a real, meaningful choice belonging to
the caller, not a decision this primitive should make for them).

## Explicitly excluded

- **No SAE/LRAE features anywhere in the semantic half** — per the plan's
  explicit instruction, restated here as a hard constraint on this
  primitive specifically, not just the frozen phase-3/phase-4 code. If a
  semantic reading is ever wanted from a sparse dictionary, that is a
  different primitive with a different name, not an optional argument
  bolted onto this one.
- **No probe/LDA fitting.** Supplied fitted, not fit here (see "mechanically,"
  above).
- **No decision logic** (no "if entropy > X then verdict Y"). This
  primitive returns a reading, not a verdict — verdicts belong in each
  phase's own falsification table, per the plan's methodology section.

## Pairwise geometric field (2026-09-16 extension — unblocks `P-I5`)

Every field above is **per-point**: one vector in, one reading out.
`claims/registry.json`'s `P-I5` (*"ablating an induction head changes the
pairwise-distance distribution among the particles it couples — the matched
positions — not only the logit at the copied token"*) needs a field defined
on a **pair** of points, and its `null_construction` names the gap exactly:
*"every current geometric field is per-point and this needs a pairwise
one."* Written here first, per this document's own rule, before
`core/dual_reading.py` carries the implementation.

**Scope, stated so it isn't overrun:** this is the missing *reading*, not
the null. Building the matched-magnitude random-direction ablation arm, the
permutation test over the two-dimensional (geometric delta, logit delta)
statistic, and running it against real checkpoints is `P-I5`'s adjudication
gate — separate, larger, future work that *consumes* this field. Landing
the field is what turns `needs-null` from "no instrument exists" into "the
instrument exists, the gate is unbuilt."

**Design choice: generalize by substitution, not by new machinery.** A pair
of points has exactly one thing a single point doesn't: a displacement,
`vector_a - vector_b`. Every existing per-point field is a squared-norm
fraction of the vector against a subspace (`_squared_norm_frac`); the
pairwise field is the *same* function applied to the displacement instead
of to the point. No new decomposition is defined — this is substitution,
not new machinery, so it inherits every existing property (including the
"fractions need not sum to 1 when `U_pos`/`U_neg`/`U_S`/`U_A` aren't a full
orthogonal decomposition of `R^d`" caveat already true of `geometric_reading`).

```python
def pairwise_geometric_reading(
    vector_a: np.ndarray,        # (d,) — first particle (e.g. the query
                                  # position an induction head reads from)
    vector_b: np.ndarray,        # (d,) — second particle (e.g. the position
                                  # it copies from)
    projectors: dict | None = None,   # same {"U_pos", "U_neg", "U_S", "U_A"}
                                  # shape as geometric_reading; None or a
                                  # missing key -> None for that field
) -> dict:
```

```python
{
  "raw_distance":              float,        # ||vector_a - vector_b||, the
                                              # scale-sensitive L2 reading —
                                              # same raw/normed duality
                                              # effective_rank already carries
                                              # (see its mode= caveat above);
                                              # a caller wanting the
                                              # magnitude-independent number
                                              # normalizes both vectors first.
  "cosine_similarity":         float | None, # None iff either vector has
                                              # ~zero norm (undefined angle) —
                                              # same guard as
                                              # _squared_norm_frac's `denom`
                                              # check, applied to both inputs.
  "cosine_distance":           float | None, # 1 - cosine_similarity
  "distance_attractive_frac":  float | None, # ||U_pos^T(a-b)||^2 / ||a-b||^2
  "distance_repulsive_frac":   float | None, # ||U_neg^T(a-b)||^2 / ||a-b||^2
  "distance_real_frac":        float | None, # ||U_S^T(a-b)||^2  / ||a-b||^2
  "distance_imag_frac":        float | None, # ||U_A^T(a-b)||^2  / ||a-b||^2
}
```

**Not folded into `dual_reading()` or `geometric_reading()`.** Those two
take one vector; overloading them to sometimes take two would make the
single combined-entry-point's signature ambiguous about which mode it's in
(exactly the god-function risk this document's "mechanically" section
guards against). A separate, single-purpose function keeps `dual_reading()`'s
existing schema and test coverage untouched.

**Not vectorized over many pairs.** `P-I5`'s "distribution" is built by a
caller invoking this once per matched `(query_position, copied_position)`
pair and collecting the results — the same division of labour as
`effective_rank_contribution` (this primitive reads one point/pair;
aggregating across many is the caller's job, per "What this primitive is,
mechanically" above) and as `p7d_redundancy/member_subspace_geometry.py`'s
`cell()` helper, called once per pair in a loop rather than batched. A
batched form is a real future optimization (`functional_distance.py`'s KL
matrix shows the pattern: one matmul instead of a pair loop) but is not
needed to unblock `P-I5`'s gate, which reads a handful of matched pairs per
prompt, not all pairs.

**Not added to `to_particle_row_fields`.** `ParticleTable`'s schema is one
row per *point*; a pairwise reading has no single row to attach to without
inventing a pair-keyed table this project doesn't have yet. `P-I5`'s gate
consumes `pairwise_geometric_reading()`'s dict directly.

## Architecture note carried over from the plan's own audit list

The plan's "Architecture compatibility" section already flags this
primitive by name: "audit everything that decodes through the unembedding
(dual-reading primitive's frozen-head decode, tuned lens) — Pythia's
embeddings are untied, use `embed_out`." `tuned_lens_cluster.py::
frozen_head_decode` currently searches `model.predictions` /
`model.lm_head` / `model.cls` (ALBERT/GPT-2-era) and does not check
`embed_out` (Pythia/GPT-NeoX's untied unembedding). This primitive's
semantic half calls that function as-is in this pass; the `embed_out` audit
is Pythia model-support work (plan item 5), tracked there, not duplicated
or silently worked around here.
