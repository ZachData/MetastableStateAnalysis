<!-- p8_scale_ladder/status-8.md -->
# Phase 8 — STATUS

**Last verified:** 2026-09-10 (opened this day).
**Overall:** **nothing has run.** The phase exists as a design and a directory;
no measurement on any rung has been taken under it. Read `design-8.md` first —
it carries the rung policy, which is the phase's whole epistemic value.

## What is decided

- **The ladder is Pythia only**, by user decision 2026-09-10: the suite is
  already wired into `core/pythia_registry.py`, shares one data order and one
  checkpoint schedule across sizes, and is convenient enough that other families
  are not worth the setup cost right now.
- **Rung policy — explore low, validate high.** 70m and 410m are exploration;
  **1b and 1.4b are reserved** and may not be measured on any induction quantity
  until a prediction naming them is registered.
- **1.4b is NOT to be cleaned up.** It was briefly proposed to delete its
  existing analysis to keep it pristine. Unnecessary: CLAIM-C measured it on
  Phase-1 phenomenology metrics only (`mass_near_1`, `effective_rank`,
  `cluster_membership`, `cluster_count`, `cka_prev`, `fiedler_mean`), none of
  which is an induction quantity, so it is already clean on this phase's axis.
  Deleting it would have cost a registered claim for nothing.
- **P-I7's "not yet measured by this project" is not binding to the letter**
  (user, 2026-09-10) — it was written on a whim and still roughly stands. Under
  the rung policy it is satisfiable on either reserved rung.
- **70m is the training rung** (user, 2026-09-10). It is the only size this box
  can train, so it is also where **activations** and **custom checkpoints** come
  from — if a question needs a checkpoint Pythia never published, 70m is where
  that is affordable. 1b and 1.4b are validation only and are never trained here.

### The fork problem, and the fix that makes a dense axis faithful

**The existing retrain is a fork, not pythia-70m.** Beyond §3.9's
"reachability, not development" caveat, the **dataset batching seeds differ**, so
it is not the trajectory published pythia-70m would have had — it is a sibling
that shares only checkpoint A (step 512). The user's own read, 2026-09-10: the
training "probably needs to be redone."

That splits cleanly into two artifacts, and **both are useful for different
questions**:

| artifact | dense through `(512, 1000]`? | developmental? | good for |
|---|---|---|---|
| published pythia-70m | **no** — same sparse gap as 410m | yes | invariants 1, 2, 4, 5, 6 |
| the existing fork | yes, stride 4 | **no** | invariant 3 only (ordering as an independent draw) |

**A differently-seeded fork is exactly what invariant 3 wants** — cascade
predicts the order reproduces, recruitment predicts the window reproduces and
the order scrambles. So the fork is not damaged goods for that question; it is
the right instrument. It is damaged goods for anything developmental.

**If a dense *and* faithful axis is wanted, the fix is to retrain on Pythia's
published data order** rather than a fresh seed — EleutherAI released the exact
batch ordering, so a continuation from step 512 that replays the true sequence
is a faithful dense interpolation of the published trajectory rather than a
sibling. That is the version worth spending GPU time on. **Do not re-run the
old fork protocol.**

## Literature scan — read before measuring

`docs/literature_scan_2026-09-10.md`. **Leads, not readings**; no paper has been
read and every arXiv id needs verifying. Three of §3.12-V's four headlines are in
populated territory: SVD-ordering-is-a-poor-importance-proxy is **FWSVD, 2022**
(2207.00112); super-additive co-ablation is the **Hydra-effect self-repair
signature** (2307.15771), with 2607.01940 looking structurally like our own
pairwise matrix; and the developmental axis is covered recently by 2502.14010 and
**2606.02378**, the latter across three 1B-class models **including Pythia-1B**.

**Two consequences for this phase.** Do not frame invariants 5 or 6 as novel
phenomena — they are replications, which is still worth doing across a ladder but
is a different claim. And **check what 2606.02378 measured on Pythia-1B before
treating that rung as untouched by the field** — our reserve protects it from
*us*, not from everyone.

The strongest remaining card is methodological: **causally-defined membership
plus the demonstration that structural proxies fail**, and the measured-null
discipline (§3.12-V3, §3.12-V5).

## What is open — in order

1. **De-hardcode the architecture constants.** `tools/run/induction_rank_sweep.py:83`
   sets `D_MODEL, D_HEAD, N_HEADS = 1024, 64, 16` at module level, and **every**
   7d/7e runner imports them from there. Read from `model.config` instead. This
   is the gate on everything below and is small.
2. **Registry entries** for `PYTHIA_70M_REPO` and `PYTHIA_1B_REPO` in
   `core/pythia_registry.py` — 410m and 1.4b are already there and the pattern
   is one line each. **Adding 1b is not measuring it.**
3. **7d's Q1 on pythia-70m** — full 48-head causal ablation sweep. The cheapest
   test of invariant 1 and it establishes the rung's own scale and null.
4. **Invariants 2–6 on 70m**, reusing 7d/7e's runners unchanged once (1) lands.
5. **Invariant 3 against the dense bracket** — the sister project's cascade
   versus 70m's published-checkpoint ordering.
6. **Register what survives, then measure 1b.** Not before.

**Undecided and needing a human call, recorded rather than drifted into:**
whether P-I7 is adjudicated on 70m *before* exploration touches it, or 70m goes
to exploration and P-I7 moves to 1b. Either is consistent with the rung policy.

## Inputs this phase depends on

| what | where | note |
|---|---|---|
| 410m results | `p7d_redundancy/status-7d.md`, `p7e_consolidation/status-7e.md`, `PROJECT.md` §3.12-V | the baseline all six invariants are drawn from |
| 70m dense bracket | `/var/home/iron/Desktop/lora_ind/data/retrain/cb25e3f6c2185c1e/` | 67 GB local, 249 checkpoints, stride 4 |
| 70m cascade re-probe | `data/reprobe_merged.json` on `origin/main` of the sister repo | 82/82, all 48 heads, `n_eval=512` |
| the sister project | `git@github.com:ZachData/Lora_inductionhead.git` | **upstream, stays separate** — §3.9-A |

**The 70m bracket is `reachability, not development`** — Adam cold-starts at step
512, so it diverges from published pythia-70m from the first step. It is an
independent draw of an *ordering*; it is not pythia-70m's trajectory. Published
70m and the retrain bracket are **two different artifacts** and must be labelled
as such in every result.

## Reproducing

Nothing to reproduce yet. When (1) and (2) land, the 7d/7e commands in
`p7d_redundancy/status-7d.md` and `p7e_consolidation/status-7e.md` become the
per-rung commands with a `--model` argument.

**Machine note carried from 7d/7e:** use `--chunk 2` and `OMP_NUM_THREADS=4`;
those runners were killed for memory twice at `--chunk 4` on this box. Disk is
the other constraint — `/run/media/system/WDS_500` has ~95 GB free with
`data/hf` already at 51 GB for 410m alone, while `/var/home` has ~634 GB. **A
full 1b or 1.4b checkpoint grid will not fit beside the 410m cache**; plan
`HF_HOME` placement before pulling a second large rung.
