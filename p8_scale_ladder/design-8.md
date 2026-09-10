<!-- p8_scale_ladder/design-8.md -->
# Phase 8 — DESIGN: the Pythia scale ladder

## Core question

7d and 7e measured one model. Six members, one redundancy set, an alignment
peak at step 5000, a rank-1 head and a full-rank anti-ordered one — all of it
`n = 1`. **Which of those are properties of induction, and which are properties
of pythia-410m?**

> Do the structural signatures 7d/7e found reappear at other scales — and which
> ones survive?

That question is the reason the phase exists, and it is also the only route by
which this work becomes publishable as more than a case study.

## The rung policy — **explore low, validate high**

| model | status | role |
|---|---|---|
| **pythia-70m** | untouched on the induction axis | **exploration** |
| **pythia-410m** | spent (7d/7e, §3.12) — exploratory forever | **exploration** |
| **pythia-1b** | not in the registry, never measured | **RESERVED** |
| **pythia-1.4b** | measured on Phase-1 phenomenology only (CLAIM-C) | **RESERVED** |

**pythia-1.4b needs no cleanup.** CLAIM-C measured it on `mass_near_1`,
`effective_rank`, `cluster_membership`, `cluster_count`, `cka_prev` and
`fiedler_mean` — not one of which is an induction-head quantity. It is clean on
the axis this phase uses, and deleting that analysis would cost a registered
claim and buy nothing.

**The policy, stated so it can be violated deliberately rather than by
accident:** no induction measurement may be run on 1b or 1.4b until a
prediction naming that model is in `claims/registry.json`. Exploration on the
low rungs is free and unlimited. This is the ordinary spent-artifact rule
(`check_registry` rule 3) applied *forward* instead of discovered afterwards,
and it is what lets the ladder produce adjudications instead of another pile of
exploratory findings — of which the project has plenty and, in
`claims/adjudications/`, zero of the other kind.

**On P-I7.** Its wording — "a transformer NOT YET MEASURED BY THIS PROJECT" —
is broader than intended and the user has said it is not to be treated as
binding to the letter. Under this policy it is satisfiable on either reserved
rung. Whether to adjudicate it on 70m *before* exploring there, or to let 70m
go to exploration and adjudicate on 1b, is a **human call that should be made
explicitly and recorded**, not drifted into by the first sweep that runs.

## What transfers, and what cannot

**Nothing transfers by head name.** `L5H2`, `L7H8`, `L11H14` are 410m
coordinates. The comparison is between *structural invariants*.

**No absolute threshold transfers either** (§3.9 says this for the 70m grid and
it generalises): not the +0.05 membership bar, not `r* = 12`, not the noise
floor. Each rung gets its own measured null and its own scale.

**`d_head` varies across the ladder**, so a rank budget is not comparable in
absolute terms. Expected — *verify from `model.config` on first load, do not
trust this table*:

| model | layers | d_model | heads/layer | total heads | d_head |
|---|---|---|---|---|---|
| pythia-70m | 6 | 512 | 8 | 48 | 64 |
| pythia-410m | 24 | 1024 | 16 | 384 | 64 |
| pythia-1b | 16 | 2048 | 8 | 128 | 256 |
| pythia-1.4b | 24 | 2048 | 16 | 384 | 128 |

**Every rank is reported as `r*/d_head`**, never as `r*`. 70m and 410m happen to
share `d_head = 64`, which makes them directly comparable and makes the reserved
rungs the ones where the normalisation actually bites.

## The six invariants

Each is a 7d/7e finding restated as something that could fail at another scale.
These are the candidate registrable predictions; **none is registered yet**, and
the wording here is a design sketch, not registry text.

1. **Membership is a small, heavy-tailed set.** 410m: ~4 substantial members,
   ~10 with any effect, of 384; **median head moves the readout by 0.001**. The
   invariant is the *shape* — a heavy tail with an identifiable end — not the
   count.
2. **Members form in one narrow window, coinciding with induction onset.** 410m:
   five of six inside `(512, 2000]`, on a 143,000-step axis, exactly where
   second-copy NLL goes 12.63 → 4.91.
3. **Ordering versus window** — §3.14.4-A's cascade-vs-recruitment. If the
   *order* reproduces across differently-seeded models, cascade; if the *window*
   reproduces and the order scrambles, recruitment. **This is the one 7d
   question that always needed a second model.**
4. **Born aligned, then fanning out.** 410m: alignment present at birth (centered
   CKA 0.693 vs null 0.128 at step 1000), peaking at step 5000, losing 56 % by
   143000 **while delta norms grow** — with the max pair pinned at 0.87–0.97 and
   the min pair falling to −0.19.
5. **A low-rank majority with at least one full-rank, anti-ordered member.**
   410m: `r*/d_head` of 1/64, 1/64, 2/64, 12/64, 24/64 — and `L11H14` at 64/64
   with bottom-`r` beating top-`r` at every rank. **The anti-ordered member is
   the sharpest and most falsifiable of the six.**
6. **One redundancy set, magnitude-dominated, direction-decoupled.** 410m: 44/45
   cells positive, no block structure, r²(interaction, `d_a·d_b`) = 0.74–0.81,
   r²(interaction, δ-cosine) = 0.07–0.09.

## Instruments — already built, need de-hardcoding

7d/7e's runners are nearly scale-generic already. The blocker is three constants:

```
tools/run/induction_rank_sweep.py:83
    D_MODEL, D_HEAD, N_HEADS = 1024, 64, 16
```

which `redundancy_catalog.py`, `member_formation_curves.py`,
`pairwise_interaction_matrix.py`, `member_subspace_geometry.py`,
`ambient_budget.py` and `useful_rank.py` all import at module level. **Read them
from `model.config` instead**; that single change makes every instrument in
7d/7e work at any scale. `N_REP`, `VOCAB_LO/HI` and `EVAL_SEED` are probe
parameters, not architecture, and should stay fixed across rungs so the readout
is comparable — except that `VOCAB_HI = 40000` must be checked against each
model's vocabulary.

`core/pythia_registry.py` needs `PYTHIA_70M_REPO` and `PYTHIA_1B_REPO` entries;
it already carries 410m and 1.4b and the pattern is one line each. **Adding 1b
to the registry is not measuring it** — the reserve holds.

## What must not be lost

- **The rung policy.** One careless `--top 10` sweep on 1b costs the phase its
  reason to exist.
- **§3.12-V5's changing-membership artifact.** It caught 7d three times and
  always manufactured a rising trend. It is *worse* here: rungs have different
  head counts and different formation windows, so any mean over "the members"
  changes membership across both training *and* scale. **Fixed sets, or print
  `n`.**
- **§3.14.4-D's ceiling.** `ΔNLL` is bounded by `ln V`, and every rung shares
  Pythia's 50304 vocabulary, so the bound is the same 10.83 while the baselines
  differ. Small models sit closer to it.
- **§3.13's report-both rule**, per rung and per checkpoint.
- **Measured nulls, not isotropic ones.** §3.12-V3: the ambient participation
  ratio was **22 of 1024** at 410m, so `k/d_model` is not a chance value. Each
  rung needs its own control heads.
- **Reachability vs development.** The 70m dense bracket from the sister project
  cold-starts Adam at step 512; it is an independent draw of an ordering, not
  pythia-70m's trajectory. Published-checkpoint 70m and the retrain bracket are
  **two different artifacts** and must be labelled as such everywhere.

## Sequencing

1. **De-hardcode the architecture constants**, add 70m and 1b to the registry.
   Cheap, and it gates everything else.
2. **Reproduce 7d's Q1 on pythia-70m** — the full 48-head causal ablation sweep.
   Cheapest possible test of invariant 1, and it establishes the rung's own
   scale and null.
3. **Invariants 2–6 on 70m**, reusing the 7d/7e runners unchanged.
4. **Invariant 3 against the dense bracket** — the sister project's ordered
   cascade (3.6 → 3.1 → 4.6 → 4.7 → 3.0 → 3.5) versus 70m's published-checkpoint
   ordering. Two draws, one architecture.
5. **Register whichever invariants survive**, then and only then measure 1b.

## Relation to the sister project

`Lora_inductionhead` stays an independent upstream repo. Its training half —
EC2 spot workers, S3, `METRIC_VERSION` CI, the LoRA fitting itself — does **not**
come here; this box has no GPU. What comes here is artifacts and analysis, one
way. What goes back is instrument fixes, of which the **copying score** is
outstanding and immediately actionable (§3.9-A): theirs is a full-vocab argmax
hit rate reading ≤ 2.8e-4 for every head including a known positive, and
`tools/run/copying_score_sweep.py` is the working replacement.

**Do not import G3's failed gate or the M1–M8 block.** Take the bracket, the
cascade and the φ question; leave the gate structure upstream.

### What to keep from the sister project, and what to drop

User instruction 2026-09-10: keep what is good, drop what is not, but **record
the dead ends so they are not rediscovered**. Dead ends are cheap to write down
and expensive to walk into twice.

**KEEP — carries real value:**

- **The dense onset bracket** (82 checkpoints, stride 4, 249 weight snapshots)
  and the **48-head re-probe** at `n_eval=512` — with the fork label attached.
- **The ordered cascade**: 3.6 (640) → 3.1 (652) → 4.6 (696) → 4.7 (724) →
  3.0 (760) → 3.5 (832), and prev-token head 2.1 closing 0.389 → 0.947 in-window.
- **The φ question itself** — does the antisymmetric fraction differ between the
  QK and OV halves? It is posed in Mets' own `S + Λ` decomposition
  (`core/dual_reading.py`, `p2b_imaginary`) and is the genuine conceptual
  overlap.
- **The reachability result**, which is a finding and not just a blocker: `B`'s
  own `W_Q` for the target head does not move `R`, upper-bounding any trained
  update.
- **The localization dissociation**: grafting blocks 0–2 restores PMS to 0.895
  while `R` stays 0.10. Matching and recovery come apart — the same shape as
  §3.12-U and §3.12-V1.

**DROP — do not port, do not re-run:**

- The **EC2 spot / S3 / worker-bootstrap infrastructure**. No GPU here.
- The **G0–G3 / M1–M8 gate lattice**. Mets adjudicates against a registry;
  running two governance systems corrupts both.
- The **full-vocab argmax copying score** — a known-positive at the floor. Mets'
  `tools/run/copying_score_sweep.py` replaces it.
- The **old fork retrain protocol** (fresh batching seeds). If a dense axis is
  rebuilt, rebuild it on **Pythia's published data order** — see `status-8.md`.

**RECORD ONLY — dead ends, kept so they are not rediscovered:**

- `layer_host_plus_ln_final` — **closed for good**, 8/8 spot reclaims.
- Four negative G3 diagnostics, each disfavouring its own hypothesis: broken
  objective, insufficient rank, frozen `W_K`, wrong head. Three alternative
  heads at lr 1e-2 / rank 64 / 100 steps all plateaued *below* the target head's
  own `R = 0.0095`.
- **Cor 17.2's gradient-gating mechanism is ruled out** — σ_OV is nonzero at all
  2048 query positions at `A`, so Prop 17.5's flat-`R(r)` story is not what the
  diagnostics measured.
