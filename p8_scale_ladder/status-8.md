<!-- p8_scale_ladder/status-8.md -->
# Phase 8 — STATUS

**Last verified:** 2026-09-11.
**Overall:** the de-hardcoding blocker is cleared and the first exploration-rung
measurement is in — the 48-head causal ablation sweep on pythia-70m. Read
`design-8.md` first — it carries the rung policy, which is the phase's whole
epistemic value.

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

1. ~~De-hardcode the architecture constants.~~ **Done 2026-09-11.**
   `tools/run/induction_rank_sweep.py`'s `ov_factors`/`write_ov` now read
   `(d_model, d_head, n_heads)` from `model.config` via a new `arch_dims(model)`
   helper; `truncate` derives them from the factor shapes it is handed instead
   (which also fixes the pre-existing `d_head=64` mismatch
   `induction_qk_sweep.py`'s docstring flagged for its 48-dim static core — see
   that file's inline comment, now stale but left as a paper trail). Propagated
   into the six runners the design doc names: `redundancy_catalog.py`,
   `member_formation_curves.py`, `pairwise_interaction_matrix.py`,
   `member_subspace_geometry.py`, `ambient_budget.py`, `useful_rank.py`.
   Verified against the real pythia-410m cache (matches `PROJECT.md`'s recorded
   numbers to the sequence-count difference) and the full test suite (2839
   passed, 37 skipped, 0 failures).
2. ~~Registry entries~~ for `PYTHIA_70M_REPO` and `PYTHIA_1B_REPO`. **Done
   2026-09-11**, in `core/pythia_registry.py::build_pythia_model_configs`, over
   `PYTHIA_ALL_STEPS` (every published step) for both — unlike 410m/1.4b, which
   carry earlier phases' historical schedules, these two exist for the ladder
   alone and the ladder wants one shared schedule. **Adding 1b is not measuring
   it** — nothing reads the 1b entries until a prediction names that rung.
3. ~~7d's Q1 on pythia-70m~~ — full 48-head causal ablation sweep. **Done
   2026-09-11**, see "Pythia-70m Q1 results" below. `redundancy_catalog.py`
   gained a `--model` flag (default `pythia-410m`, so the existing 410m
   artifact and its filename are untouched) to make this runnable at all.
4. ~~Invariants 2–6 on 70m~~ — **first pass done 2026-09-11**, see below. All
   five remaining runners gained the same `--model` treatment
   `redundancy_catalog.py` got, plus two correctness fixes the multi-rung case
   exposed: `member_formation_curves.py`'s `--pair` and
   `member_subspace_geometry.py`'s `--controls` both defaulted to 410m head
   coordinates that don't exist at 70m's 6 layers (`L7H8`, `L12H15`) — both
   now refuse a mismatched default rather than silently measuring the wrong
   heads or crashing obscurely, and `members()` (shared by all five) now reads
   a model-keyed catalogue instead of always the 410m one. This is a single
   pass, one checkpoint (or the 19-step formation grid for invariants 2/4) —
   not the full per-invariant workup 7d/7e gave 410m.
5. **Invariant 3 against the dense bracket** — the sister project's cascade
   versus 70m's published-checkpoint ordering.
6. **Register what survives, then measure 1b.** Not before.

### Invariants 2, 4, 5, 6 — first pass on pythia-70m (2026-09-11, exploratory)

Same six heads throughout (item 3's top-6 by `|dNLL|`, step 16000):
`L2H1, L0H0, L3H6, L3H1, L0H2, L3H5`. `L2H1` is the previous-token partner, not
an induction head itself — included because the catalogue ranks by causal
effect, not role, the same convention 410m's own set used for `L5H2`.

**Invariant 2 (narrow formation window) — replicates, more sharply.**
`member_formation_curves.py --model pythia-70m --pair L2H1,L3H6`, 19 steps.
Every one of the six heads sits at noise (`|dNLL| < 0.007`) through step 512,
then **all six** turn on between step 512 and step 1000 — `L2H1` +0.002 → +7.28,
`L3H6` +0.002 → +5.21, `L3H1` −0.002 → +2.08 — a single-step jump on the
published grid, narrower than 410m's `(512, 2000]` window. Consistent with the
sister project's own dense bracket (`PROJECT.md` §3.9), whose onset at step 620
falls inside this same published gap.

**Invariant 4 (born aligned, then fans out) — replicates in shape, smaller in
degree.** Same run's pair cosine (`L2H1`×`L3H6`): ~0 through step 256 (noise),
**+0.499 already at step 1000** (birth, not a gradual climb from zero), peaks
**+0.777 at step 2000**, then declines to **+0.574 by 143000** — a 26 % loss
from peak, against 410m's 56 % (design-8.md invariant 4, peak at step 5000).
Same shape (aligned at birth, peaks shortly after, partially decoheres), a
smaller and slower fan-out.

**Invariant 5 (low-rank majority, one full-rank exception) — does NOT
replicate.** `useful_rank.py`, step 16000, `--controls 2`:

| head | effect `dNLL` | `r*` (of 64) | control `r*` |
|---|---|---|---|
| `L0H2` | +1.385 | **12** | never |
| `L2H1` | +6.166 | **24** | 48 |
| `L3H6` | +2.150 | **48** | never |
| `L3H1` | +1.889 | **64** | never |
| `L3H5` | +1.248 | **64** | never |
| `L0H0` | +2.701 | **64** | 3 |

410m's set (`status-7e.md`): `r*` = 1, 1, 2, 12, 24, and one exception at 64,
five of six summing to 40 -- comfortably inside one head's budget. 70m's set is
the mirror image: **three of six are full-rank**, and even the two lowest
(12 + 24 = 36) leave nothing to spare once the 48 and two 64s are added — sum
276 against a 64-dim budget. Design-7e's consolidation premise (fold a
low-rank aligned core into one head) would fail at 70m on the usefulness
criterion alone, before energy is even asked. `L0H0`'s own curve is worth a
second look: real recovery goes **as negative as −1.44** at r=8 (a rank-8
truncation is worse than deleting the head outright) before climbing back to
0.96 at r=64 -- a candidate second anti-ordered head, though `--bottom` was
not run so this is not yet the direct measurement `L11H14` got.

**Invariant 5's energy criterion (design-7e's other capacity test) also fails
harder at 70m.** `ambient_budget.py`, step 16000, same six heads: ambient PR
**7.3 of 512**, joint effect needs
**170** dims in its own basis and **357** in ambient ordering for 90% energy,
and only **49.6%** of the joint effect's energy sits in the top-64 ambient
directions. 410m's canonical run (`status-7e.md`, same budget size since
`d_head = 64` at both rungs): **73.2%** (step 16000), **78.0%** (143000). Same
instrument, same absolute budget, `d_head`-matched rungs -- and 70m recovers
two-thirds as much. **DOES NOT FIT**, where 410m's did.

**Invariant 6 (redundancy set structure) — causal route is UNREADABLE here,
not merely negative.** `pairwise_interaction_matrix.py --model pythia-70m
--steps 16000`: baseline NLL 5.73 leaves only 5.1 nats of headroom (410m's is
~10.2), and **13 of 15 cells** land within the ceiling-warn band or past the
uniform ceiling itself (joint NLL up to 12.94 against `ln 50304 = 10.83`).
Checking the two-head arm (`L2H1`+`L3H6`) across the whole 19-step formation
grid confirms it is not a step-16000 accident: joint NLL exceeds the ceiling at
**every** measured step from 1000 through 143000. The naive raw-`dNLL`
interaction sign at 70m (11/15 cells nominally sub-additive) is **exactly**
the manufactured-sub-additivity artifact `PROJECT.md` §3.14.4-D/§3.12-M5
warns about, not a scale finding, and must not be read as one.

The ceiling-immune geometric route (`member_subspace_geometry.py`, same six
heads, three near-median controls `L2H0`/`L4H5`/`L5H2`) gives a real if
smaller-magnitude answer: **centered CKA 0.374 (members) vs 0.205 (null)**,
same direction as 410m's own emphasis that centered CKA is the quantity to
trust (`status-7d.md`: birth-step centered CKA 0.693 vs null 0.128 -- not the
same checkpoint, so not a magnitude comparison, only a directional one). Every
one of the 15 member pairs' `cka_centered` (0.23–0.77) sits above the
three-control-pair floor. **Uncentered CKA does not discriminate** (0.506 vs
0.546) -- the same trap 410m's own status doc names ("quote `cka`/
`cka_centered`, never the unweighted `subspace_*_cos` alone") reproduces
itself at this rung. `coverage_by_others` saturates to 1.000 for every head
including controls, as expected at `d_model = 512` with ~100–190-dim effect
subspaces per head -- unusable here for the same reason it was unusable at
410m.

**Net for the rung policy:** invariant 2 and invariant 4's *shape* replicate;
invariant 5's low-rank majority does NOT -- both its causal (`useful_rank`) and
energy (`ambient_budget`) legs fail harder at 70m than they did at 410m,
where the same design-7e capacity question resolved the other way; invariant 6
needs its ceiling-immune instrument at this scale even at a single checkpoint,
where 410m only needed it inside the formation window. All of this is `n = 1`
per invariant per rung and unregistered -- exactly the case this phase exists
to accumulate before anything here can be adjudicated.

### Pythia-70m Q1 results (2026-09-11, exploratory — not registered)

`redundancy_catalog.py --model pythia-70m --step 16000 --seqs 8`, all 48 heads
(6 layers × 8 heads/layer — pythia-70m's *entire* head count, not a sample).
Step 16000 matches 410m's own Q1 default exactly, and is well past the sister
project's located onset (`PROJECT.md` §3.9: PMS crosses 0.10 at step 620, on a
72-step-per-4-checkpoint bracket that ends at 852) — a fair "settled, not mid-
formation" comparison point. Full row data:
`data/analysis/redundancy_catalog_pythia-70m_step16000.json`.

**Median +0.0156, not +0.001.** > **SUPERSEDED the same day — see "The
ablation-mode A/B" below. The +0.05 comparison in this paragraph imports a bar
`design-8.md` forbids transferring across rungs, and it reverses when the
count is taken against each rung's own bulk instead.**
410m's Q1 (`PROJECT.md` §3.12-Q/T) found the
median head moves `ΔNLL` by ~0.001 of 384; here the median is **16x larger**
against a head count 8x smaller. 19/48 heads (40%) clear +0.05, against 410m's
handful of 384 (~1-3%). **Invariant 1's heavy tail is present but far less
extreme** — at 70m, mattering-at-all is closer to the rule than the exception.
Whether that is a real scale effect or an artifact of 70m having 8x fewer
heads to spread the same induction computation over is not decidable from one
rung; it is exactly the question the ladder exists to ask, and this is the
first data point.

**Large NEGATIVE effects, more of them than 410m's.** > **SUPERSEDED — these
are mostly zero-ablation artifacts. Under mean-ablation `L0H6` goes −3.34 →
−0.22 and `L0H5` −2.81 → −0.06. The paragraph below argues explicitly against
dismissing them, and was wrong to.**
16/48 heads (33%) clear
−0.05 (410m has isolated anti-copiers, `L2H10`/`L9H8`, `PROJECT.md` §3.14.3
M4/§3.12). Layer 0 is where they concentrate: `L0H4` −1.74, `L0H5` −2.81,
`L0H6` −3.34 (the single largest-magnitude effect of any sign), `L0H7` −2.03.
Ablating these heads makes the second-copy prediction *easier*, which is a
real finding to carry into invariant 6 (redundancy-set direction-coupling) —
it should not be waved off as noise; `−3.34` is far outside anything readable
as a null at 8 sequences.

**Independent cross-validation of the sister project's cascade — unprompted,
different method entirely.** The sister project (`PROJECT.md` §3.9-A) located
`L3H6` as pythia-70m's induction head and `L2H1` as its previous-token partner
via PMS onset tracking on a differently-seeded retrain fork. This causal
ablation sweep, on the **published** checkpoint, at a **later** step, with a
**different instrument** (`ΔNLL`, not PMS), ranks:

| head | this sweep's `ΔNLL` (rank) | sister project's role |
|---|---|---|
| `L2H1` | **+6.17 (1st of 48)** | previous-token partner |
| `L3H6` | **+2.15 (3rd)** | induction head (cascade position 1, PMS onset 640) |
| `L3H1` | +1.89 (4th) | cascade position 2 (onset 652) |
| `L3H5` | +1.25 (6th) | cascade position 6 (onset 832) |
| `L3H0` | +0.64 (12th) | cascade position 5 (onset 760) |
| `L4H7` | +0.12 | cascade position 4 (onset 724) |
| `L4H6` | below +0.05 print threshold, see json | cascade position 3 (onset 696) |

All six cascade heads and the previous-token partner land in the causally
meaningful region at a checkpoint 15,148 steps after the onset bracket ends,
on the actual published trajectory rather than the reachability fork. This is
independent agreement, not a repeated measurement — worth citing as such, but
it is one model's worth of it (`n = 1` for the cross-validation itself) and
carries no p-value. **The causal-effect ordering at step 16000 does NOT match
the PMS onset order** (`L2H1`/`L3H6` swap to top by causal effect while
`L3H6` was PMS-first; `L4H6`/`L4H7` fall from mid-cascade to near-null) — which
is itself invariant-3-relevant: onset order and later causal-effect order are
already dissociating within a single model, before any cross-model ordering
question is asked. `L0H0` (+2.70, 2nd overall) is causally prominent and
appears in neither the cascade nor the previous-token role — unexplained,
flagged rather than investigated further here.

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

### The ablation-mode A/B (2026-09-11) — **how much of the above survives**

Run because zero-ablation puts the residual stream off-distribution, and one
head is **1/8** of a pythia-70m layer against **1/16** at 410m, so the
off-distribution component of `ΔNLL` cannot cancel in a cross-rung comparison.
`redundancy_catalog.py --ablation {ov,zero,mean}`; instrument in
`induction_rank_sweep.ablate_heads`; statistics in `compare_rungs.py`
(participation ratio, Gini, top-k share, own-bulk count — all threshold-free,
because `design-8.md` forbids importing a bar across rungs and the first
write-up imported one anyway).

**Two instrument facts, both checked rather than assumed.** `write_ov`'s
zero-factor path zeroes `W_V`'s *weight* and leaves its *bias*, so the head
goes on writing a constant — 0.1007 at every position at 70m `L3H6` against an
unablated 2.1–3.2. It is therefore a **bias-ablation, not a zero-ablation**,
and every 7d/7e/8 number to date is one. It is also **immaterial**: true
zero-ablation via the activation hook reproduces the `ov` path's NLL
*bitwise*. Separately, the de-hardcoding refactor reproduces the stored 410m
catalogue bitwise on spot-checked heads (`L5H2` +1.966230, `L7H8` +1.018922,
`L11H14` +0.189625), so the historical artifact is a valid matched arm.

| rung | mode | median | PR | PR/n | gini | top-5 share | > own bulk |
|---|---|---|---|---|---|---|---|
| 70m | `ov` | +0.01557 | 5.05 | 0.105 | 0.677 | 0.465 | 8/48 |
| 70m | `mean` | +0.01953 | **2.12** | 0.044 | 0.779 | 0.659 | 7/48 |
| 410m | `ov` | +0.00107 | 1.71 | 0.004 | 0.828 | 0.567 | 60/384 |
| 410m | `mean` | +0.00091 | **1.47** | 0.004 | 0.859 | 0.592 | 58/384 |

**The mode sensitivity is itself scale-dependent, which is the finding.** 410m
barely moves (PR 1.71 → 1.47, same five top heads). 70m moves a lot (PR
5.05 → 2.12, Gini 0.677 → 0.779) **and its top-5 changes identity**: `L0H6`,
`L0H5`, `L0H0`, `L1H4` drop out and the cascade heads `L3H1`, `L3H6`, `L3H5`
come in. Zero-ablation was surfacing layer-0 heads whose apparent importance is
off-distribution response, and hiding the real induction set behind them.

**What that does to the three Q1 claims.** (a) The layer-0 negatives are
artifacts — `L0H6` −3.34 → −0.22, `L0H5` −2.81 → −0.06 — and the paragraph
above arguing they were not noise is withdrawn. (b) `L0H0`, flagged there as
causally prominent and unexplained, is +2.70 → **+0.27**: ~90 % was artifact,
and the anomaly dissolves rather than needing a mechanism. (c) The "far less
extreme heavy tail" claim half-survives: the median ratio falls 49.1x → 27.5x
and the Gini gap 0.151 → 0.080, so roughly half the apparent difference was
the instrument.

**And read absolutely, invariant 1 replicates better than the first pass
said.** Invariant 1 claims a *shape* — "a heavy tail with an identifiable end,
not the count". Under mean-ablation both rungs concentrate their effect into
**2.12 (70m) and 1.47 (410m) effective heads**, out of 48 and 384 respectively.
The *fraction* differs 11x; the *effective count* is the same small number at
both scales, which is what the invariant actually asserts. The count of heads
clearing each rung's own bulk agrees: **14.6–16.7 % at 70m, 15.1–15.6 % at
410m** — indistinguishable, against the 40 %-vs-1–3 % the imported bar produced.

**Forward-looking, and it bears on a reserved rung.** `pythia-1b` is 16 layers
× **8 heads/layer** (`design-8.md`), the same 1/8 exposure as 70m — so it
inherits 70m's vulnerability to this artifact, not 410m's robustness. Any
prediction registered against 1b should name **mean-ablation**, or it will
spend the rung on an instrument this A/B has already shown to distort at that
head count.

### Invariants 2/4/5/6 re-read under mean-ablation (2026-09-11)

All five runners now carry `--ablation`, and `head_means` takes the means on a
**clean** pass so a multi-head arm cannot depend on layer order (checked: the
joint arm is identical forward and reversed; taken in-pass it differed by 0.17
nats). Single-head arms are unchanged by the precomputation, as they should be
(gap 1e-8). Every re-read below is 70m, step 16000, 8 sequences.

**Invariant 2 (narrow window) — unchanged, replicates.** All six heads still sit
at `|dNLL| < 0.02` through step 512 and turn on together by step 1000 (`L2H1`
+7.25, `L3H6` +5.01, `L3H1` +2.00). The onset location is an ablation-mode
invariant, as expected — the mode changes effect *size*, not when effect exists.

**Invariant 4 (born aligned, then fans out) — replicates BETTER than the `ov`
read suggested.** Pair cosine (`L2H1`×`L3H6`):

| step | 512 | 1000 | 2000 | 4000 | 8000 | 16000 | 32000 | 143000 |
|---|---|---|---|---|---|---|---|---|
| `ov` | 0.041 | 0.499 | **0.777** | 0.713 | 0.611 | 0.497 | 0.172 | 0.574 |
| `mean` | 0.041 | **0.934** | 0.928 | 0.897 | 0.793 | 0.709 | 0.682 | 0.685 |

Under `mean` the peak is **at birth** and the decline is orderly — 0.934 → 0.685,
a 27 % fan-out — against 410m's birth-pair rank-1 cosine **0.857** and 56 %
fan-out (`status-7d.md`). The `ov` read's wrinkle, a climb to a later peak at
step 2000, was instrument noise. "There is no private-subspace phase" holds at
this rung too.

**Invariant 5 (low-rank majority) — still fails, and slightly harder.** The
denominator logic runs the opposite way to what I first wrote: `mean` gives a
*smaller* `d0`, so `recovery = 1 - dNLL(r)/d0` falls and `r*` **rises**.

| head | `ov` effect | `ov` `r*` | `mean` effect | `mean` `r*` | control |
|---|---|---|---|---|---|
| `L2H1` | +6.166 | 24 | +5.970 | 24 | 48 |
| `L3H6` | +2.150 | 48 | +1.268 | **64** | never |
| `L3H1` | +1.889 | 64 | +2.070 | 64 | never |
| `L3H5` | +1.248 | 64 | +1.124 | 64 | never |
| `L0H0` | +2.701 | 64 | +0.273 | — | 24 |
| `L0H2` | +1.385 | 12 | −0.264 | — | 1 |

Restricted to the four heads still above noise under `mean`, `r*` is
**24, 64, 64, 64** against 410m's **1, 1, 2, 12, 24** plus one exception at 64.
The invariant does not replicate, under either mode.

**Instrument limit found in doing this, and it is the reason for the two
dashes.** `useful_rank`'s `r = 64` row carries a known float32 refactorisation
residue (`__doc__`: ~1.7e-4 at 410m, but −0.147 for 70m `L0H2`). That is
harmless while `|d0|` is large and fatal once `mean` shrinks `d0` to the same
order — `L0H2`'s residue is 56 % of its own denominator, so `recovery` never
approaches 1 and `r*` is not meaningful. **`useful_rank` needs `|d0| >>` the
full-rank residue**, and mean-ablation and that requirement pull against each
other. `L0H0`/`L0H2` are below the geometry runner's noise floor under `mean`
anyway, so nothing is lost here — but at a rung where a real member has a small
effect this would silently produce a garbage `r*`.

**Invariant 6 — partially RESCUED, and the sign flips to agree with 410m.**
Ceiling-contaminated cells fall **13/15 → 7/15**; all five cells involving
`L2H1` stay censored, because that head alone (+5.97 on a 5.73 baseline) exhausts
the headroom regardless of mode. Among the **8 readable** cells: 5 super-additive,
2 sub, 1 ~0. Over the whole matrix the count goes 4/15 super under `ov` to
**8/15 under `mean`**. The earlier "11/15 sub-additive at 70m" — which I was
careful not to call a scale finding — was indeed the manufactured
sub-additivity of §3.12-M5, and **the direction now matches 410m's 44/45
super-additive** rather than inverting it. Ratios are mild (0.99–1.16) against
410m's headline 2.2x, but 410m's headline pair is prev-token × matcher and
70m's analogue (`L2H1`×`L3H6`) is exactly the pair that stays censored, so the
direct comparison is still not available.

**Geometry — survives, weaker.** Centered CKA members vs null goes 0.374/0.205
(`ov`) to **0.232/0.174** (`mean`): the gap narrows from 0.169 to 0.058 but keeps
its sign. Subspace cosine 0.746 vs null 0.556. Union ratio over formed members
0.361 vs null 0.654. `L0H0` and `L0H2` now both fall below the noise floor, so
the formed set is four heads, not five.

**Energy leg — conclusion unchanged.** Joint effect energy inside the top-64
ambient directions: **0.556** under `mean` (0.496 under `ov`), against 410m's
**0.732**. Still `DOES NOT FIT`; 70m still fits materially less than 410m into
the same absolute budget at the same `d_head`.

**Net.** Of the four invariants re-read, the instrument changed one conclusion
(6, from unreadable-and-apparently-inverted to partially readable and agreeing
with 410m), strengthened one (4), left one intact (2), and left the failure of
invariant 5 standing slightly harder. The mode is not a detail at this rung.

## Reproducing

```
redundancy_catalog.py        --model pythia-70m --step 16000 --seqs 8 --ablation mean
useful_rank.py               --model pythia-70m --heads ... --ablation mean --controls 2
pairwise_interaction_matrix.py --model pythia-70m --heads ... --ablation mean
member_subspace_geometry.py  --model pythia-70m --heads ... --controls L2H0,L4H5,L5H2 --ablation mean
ambient_budget.py            --model pythia-70m --heads ... --ablation mean
member_formation_curves.py   --model pythia-70m --pair L2H1,L3H6 --ablation mean [--append]
compare_rungs.py --files data/analysis/redundancy_catalog*.json
```

**Machine note, 2026-09-11:** background jobs on this box were repeatedly killed
by a memory watchdog that appears to read `free` (~1 GB, because reading
checkpoints fills the page cache) rather than `available` (~23 GB). Long runs
survive better as short foreground calls — `member_formation_curves.py` takes
`--append`, so a 19-step grid splits cleanly into batches of four.

The 7d/7e commands in `p7d_redundancy/status-7d.md` and
`p7e_consolidation/status-7e.md` are now per-rung commands with a `--model`
argument. **Only `redundancy_catalog.py` carries `--ablation` so far** — the
other five runners are still bias-ablation only, so every invariant 2/4/5/6
number above is on the mode this A/B just showed distorts at 70m's head count,
and none of them has been re-read under `mean`.

**Machine note carried from 7d/7e:** use `--chunk 2` and `OMP_NUM_THREADS=4`;
those runners were killed for memory twice at `--chunk 4` on this box. Disk is
the other constraint — `/run/media/system/WDS_500` has ~95 GB free with
`data/hf` already at 51 GB for 410m alone, while `/var/home` has ~634 GB. **A
full 1b or 1.4b checkpoint grid will not fit beside the 410m cache**; plan
`HF_HOME` placement before pulling a second large rung.
