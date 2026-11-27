<!-- trackC_writeup/design-C8.md -->
# Track C — design: C8, what triggers the transient, and what it does

**Status: proposed 2026-11-27 (U0, writing only; nothing computed, nothing trained).** Opened by
the user 2026-11-27 ("U0, the literature scan and design ... candidate (i) first"). The row is
the user's, from 2026-11-24 at C4′ (d)+(f) U3b: "something ... seems to trigger the repulsive
stage ... deeply linked to induction, but still a little mysterious". Decisions D1–D5 are in STATE
Blocked 182. Scan: `lit-C8.md`. Arithmetic: `tools/math_checks/c8_thresholds.py`. (i)'s rule is
fixed here in full. (ii)–(v) are fixed in shape here, and each unit pushes its constants in code
before its output, as C4′ (d)+(f) did.

## What it is for

The transient is a run of states where some transition outside the end set {T1, T2} is majority
violated (rule T, `c4pdf_rules.transient`; `design-C4pdf.md` "T"). What is known (ROADMAP §8 C8;
`status-C.md` "C4′ (d)+(f) on one page"):

| arm | start | rate | moments | transient (onset at τ 9.61e-5 / 1e-4 / 3e-4) |
|---|---|---|---|---|
| F | init | 1e-3 flat | fresh | (32, 48] / (32, 48] / (48, 64] on C4's grid |
| F¹, F¹′ | init | 1e-3 flat | fresh | (44, 48] / (48, 52] / (48, 52]; (44, 48] / (44, 48] / (48, 52] |
| F′ | init, other rows | 1e-3 flat | fresh | yes, T3–T5 (`P-SCH2`'s L7); onset by rule T not yet computed |
| F½ | init | 5e-4 flat | fresh | (56, 60] / (56, 60] / (60, 64] |
| F⅒ | init | 1e-4 flat | fresh | (752, 784] at all three, still running at 1000 |
| B_P | A0 at 256 | 1e-3 flat | fresh | (272, 280] / (272, 280] / (280, 288] |
| A0, A0′ | init | Pythia's ramp | fresh | none (A0′: one state, {T3} at 640) |
| B_O | A0 at 256 | A0's own | fresh | none in 256 updates |
| B_K, BKO | A0 at 256 | 1e-3 / A0's own | kept | none in 128 updates |

No single condition covers every arm. On a trained state the transient came only with fresh
moments at the peak. From init it comes at a tenth of the peak, after ~750 updates. C8 asks which
quantity crosses a line at the onset on every arm that has one, and on none that lacks it, and
what the network does inside the window. Its candidates, cheapest first, are the ROADMAP row's.
The plan is in D1.

## Seen before this design (so not blind)

Some log numbers were already seen: F's loss and grad norm at 1, 4 and 64 (11.08, 9.11, 6.27;
3.67 → 0.39; `status-C.md` "C4 unit 1"). The branches' loss and grad-norm peaks at 257–259, and
their values at 384 (`status-C.md` "unit 3b", "Beside"). F⅒'s and F½'s at 1000, and F⅒'s at 481
(in a runner's report; not recorded). **No file records any log value inside a window below.**
Whether a session's transcript showed one was not checked. The branches' spikes at 257–259 lie
inside the 8 updates that (i) excludes as the restart's.

## U1, candidate (i): loss and grad-norm spikes against the onset (CPU, logs only)

**Input.** Each arm's `train_log.jsonl` (one row per update s: `loss`, the batch's mean loss before
update s; `grad_norm`, the total norm before the clip at 1.0, as `clip_grad_norm_` returns it;
`lr`). Under `HDD_1TB/mets_data/c4/{A0,F}`, `c4pc/{A0p,Fp}`, `c4pdf/{F1,Fh,Ft,BP,BO,BK,BKO}`,
`c4pdf/rep2/F1`. A resumed segment's repeated step keeps its last row. The reader writes the sha256 of
every log and record it reads into its output. This is the carried item from `/challenge-pr` on
#378, finding 3.

**The references.** The onsets above, read from `c4pdf/2026-11-25-u3c/reading.json`. F's and F′'s
come from rule T on their own records (`c4_read`'s per-checkpoint records on C4's grid), computed
and written to `reference.json` **before any log is opened**.

**Conditions read** (five, one per distinct run): **F** (it reads for F, F¹ and F¹′: same init,
data and rate, so they are not independent; F¹'s 128-update log leaves too little background,
`c8_thresholds.py` (2)), **F′**, **F½**, **F⅒**, **B_P**. F¹ and F¹′ stand beside, with their own
references.

**The window**, for an onset (t⁻, t]: updates s with t⁻ − 16 < s ≤ t (16 = rule T's `SPAN_MIN`).
On a branch it also excludes the restart's first 8 updates (s ≤ 264), which every branch shares
(seen). Lengths: F 32, F½ 20, F⅒ 48, B_P 16 (24 at 3e-4).

**Events**, at each update s ≥ 9 (s ≥ 265 on a branch):
- **loss rise**: L_s − min(L_{s−8}, …, L_{s−1}) > δ_L;
- **grad spike**: g_s > κ · median(g_{s−8}, …, g_{s−1}).

An event is either. Each kind is reported beside. Under a falling loss, a rise above the trailing
minimum is conservative: a fast fall cannot trigger it.

**The constants, placed before any window of a with-transient arm is opened**, on A0's updates
2001–4000 (a control arm, smooth there): σ̂_L = 1.4826 · MAD(ΔL)/√2 and ĉ_g = 1.4826 ·
MAD(Δ ln g)/√2. Then δ_L = max(0.02, 6 σ̂_L) and κ = max(1.5, exp(6 ĉ_g)).

**The background**, per condition: the arm's log from 9 (265 on a branch) to min(end, 1000),
cut into non-overlapping blocks of the window's length, starting at the span's first update. A
block is dropped if it comes within 16 updates of the window. b_a is the share of the remaining
blocks that hold an event. At least 4 blocks are needed (each condition has 8–46,
`c8_thresholds.py`); with fewer, the condition is refused.

**The gate (U1, before the read):**

| gate | what | bar |
|---|---|---|
| G-null | synthetic logs on each condition's own layout (a smooth decay, iid noise at σ̂_L and ĉ_g), 200 draws | "marked" in ≤ 5 % of draws |
| G-plant | the control arms' **real** logs at matched positions: A0 at F's, F½'s and F⅒'s windows; A0′ at F′'s; B_O, B_K and BKO at B_P's. A loss rise of 2 δ_L and a grad jump of 2κ are planted at each update of the window in turn | detected at ≥ 90 % of positions in every window |
| G-half | the same at δ_L/2 and √κ | reported (power), no bar |
| H-ref | F's and F′'s rule-T references reproduce C4′ (c)'s L7 sets (T3–T5 on both) | exact |

If a gate fails, U1 reports a refusal. The constants may be re-placed once, and only before any
with-transient window has been opened. That re-placement is recorded.

**Rule R_i, at each gated τ:**

| label | when |
|---|---|
| **marked** | every one of the five conditions has an event in its window, Π_a b_a ≤ 0.05, and none of B_O, B_K, BKO has one in B_P's window |
| **silent** | none of the five has an event in its window |
| **mixed** | otherwise; each condition's label shown |

The joint reading holds only where it holds at all three gated τ. Otherwise it is "mixed across
τ". The Π bar fails if every b_a exceeds ≈ 0.549 (`c8_thresholds.py` (3)).

**Clip label C, per condition, a description with no bar:** the share of window updates with
g_s > 1.0 (clipped). "clipped" if ≥ ½, "unclipped" if none, "partly" otherwise. This answers the
row's "is the clip at 1.0 active in the window?".

**Beside, never read:** each event's lead t − s, the loss and grad-norm curves over (t⁻ − 64, t +
64], F¹'s and F¹′'s windows, A0's and A0′'s matched windows, and the first update with g < 1 per
arm.

**Leans, stated before output:**
- R_i [seen, scan]: **silent or mixed.** The branches' spikes come at the restart, on arms with a
  transient and without. So far nothing in the logs has marked an onset. Against this lean: a
  stage change spikes the grad norm at the jump (2508.12837), and an edge-of-stability crossing
  shows as short loss rises (Cohen et al. [K]).
- C [K]: F⅒ "unclipped" (1e-4, after ~750 updates). F and B_P not predicted.

**What each reading changes:**
- *marked*: the event gives a per-update date for the onset, finer than any saved grid. U2 reads
  sharpness at the saved states around it. U3's arms are read by the log first (free), then by
  rule T.
- *silent*: at C4's batch, the onset leaves no mark on the loss or the grad norm. The
  edge-of-stability lean loses its plainest visible prediction, though Adam can absorb a crossing
  by inflating v (Cohen et al.). U2 still runs (D1) with its lean restated, and whether (iii) and
  (v) come before it goes back to the user.
- *mixed*: each condition's label carries into U2, with nothing joint.

## U2, candidate (ii): sharpness against the stability threshold (GPU)

- **Raw λ_max(H)** by Hessian-vector power iteration on a fixed held-out batch, at the saved
  states in each window and around it (two states either side). Arms: F¹, F½, F⅒, B_P, B_O, B_K
  and A0's grid to 128. About 1 minute a state on 70m (the ROADMAP's figure; U2 measures it).
- **Preconditioned λ_max(P⁻¹H)**, P = diag(√v̂ + ε), against (2 + 2β₁)/((1 − β₁)η): 3.8e4 at
  1e-3, 7.6e4 at 5e-4, 3.8e5 at 1e-4 (`c8_thresholds.py` (1)). This needs v̂, which the
  checkpoints do not save. So **F² (F¹'s recipe) to 128 and B_P², B_K² (256 → 384) are re-run,
  saving weights and v̂ at their grids** (≈ 1 h each; 61 states × 2 × 282 MB ≈ 34 GB). F⅒'s v̂ at 752–784 would need ~800 updates
  (≈ 6 h), so it is not taken. F⅒ reads raw λ only.
- **Rule shape**, constants (the band θ, the batch, the iteration count) pushed in U2 before
  output, with a gate on a planted quadratic of known λ: **"crosses at the onset"** if, on F² and
  B_P², the first saved state with λ_P ≥ θ · threshold lies in the window, and B_K² stays below
  θ · threshold through 128. **"crosses, not at the onset"** if a crossing falls outside the
  window. **"never crosses"** otherwise. θ is a band, since Cohen et al. is full-batch (`lit-C8.md`
  Parked 2, read before the freeze).
- **Lean** (the ROADMAP's): crosses at the onset.

## U3, candidate (iv): causal arms (GPU), with (viii) if D2 (a)

- **F_w**: F with Adam's moments warmed by k = 64 updates at rate 0 (`c4_train.step_once`: an
  update at lr 0 still moves the moments; v̂ then holds 0.96 of its weight on the init's
  gradients, `c8_thresholds.py` (4)), then flat at 1e-3 to 128 on grid D. It reads C4's batches
  from 0 for the real updates, so F's data order is unchanged. The k warm-up batches come from
  rows no arm trains on; U3 names them and checks that.
- **B_K↑**: B_K (kept moments, A0 at 256) with a short rate overshoot over its first 16 updates,
  sized so its drift after 16 matches B_P's (`drift.json`: 0.033 against 0.011 after one update).
  This tells fresh moments from large early steps on a trained state (`/challenge-pr` on #375,
  finding 1). The size is placed in U3 before training.
- **Rule shape:** rule T on each, at the three gated τ. The joint reading:

| F_w | B_K↑ | reads |
|---|---|---|
| none | transient | large early steps, not fresh moments as such |
| transient | transient | large early steps at either state |
| none | none | fresh moments themselves |
| transient | none | the init and the trained state differ: no joint claim |

- **D5:** F_w continued to 1000 if it has no transient by 128. Its induction date is then the
  direct test of the induction link: does induction form without the transient?
- **(viii), D2 (a):** B_O continued from its `resume.pt` at 512 at A0's rate to 1000 (≈ 7 h). By
  rule T, does a low-rate restart make the transient after a dose like F⅒'s?
- Per-group warmup and frozen MLPs (the row's other (iv) arms) are not taken. They are carried
  at close.

## U4, candidates (iii) and (v): what it does (CPU, GPU minutes)

- **(iii)**, with instruments in hand, all beside: (e)'s attention-vs-MLP split of the energy
  fall at T3–T5 in the window; er_mid through it; the previous-token heads' date (L1H7, L1H3 on F)
  against the onset; each block's relative update ‖Δθ‖/‖θ‖ between grid-D states, with a
  top-down or bottom-up order (Billa 2602.15997).
- **(v)**, a rule pushed in U4 before output: per state, the KL from the model's next-token
  distribution to the **corpus's** unigram and bigram tables, built from the rows the arm has
  seen (C4's shard, CPU). Beside it, the loss on **in-context** repeated tokens (Edelman et al.'s
  stages; `lit-C8.md` §3). The ROADMAP's lean to break: the collapse is the corpus-unigram stage
  (KL to unigram at its minimum at the collapse) and the transient is the move away from it (KL to
  unigram rising, KL to bigram falling, inside the window). B_P then has to re-approach unigram
  after its jump, which its first 32 updates test. The row's per-neuron variance split is not a
  job count, so it is not gated on §7c N2, but it is outside U4's budget. It is carried at
  close.

## Exit condition and budget

**C8 closes when** (i), (ii) and (iv) are each read by their frozen rules, or refused with the
cause named. (viii) must be read by rule T if D2 (a), and (iii) and (v) reported, with (v)'s lean
labelled held or broken by its U4 rule. **Or C8 closes at the budget, at what it knows.**

**Budget: 4 units + close-out** (U1–U4, the ROADMAP's guess of 3–4). GPU ≈ 15 h: U2 ≈ 3 h of
re-runs + ≈ 2.5 h of λ_max; U3 ≈ 2 h of arms + 0.5 h of records + 7 h for (viii). F_w to 1000
under D5 adds ≈ 6.5 h, so the most is ≈ 21.5 h. An extension is the user's, on the record (C4′
(d)+(f) went from 4 units to 7 that way). **Disk:** HDD_1TB has 37 GB free (2026-11-27), so new
outputs go to `WDS_500/Mets_hdd_storage/c8/` (→ `SSUNG_1TB`), ≈ 60 GB (U2 34, F_w to 1000 22,
B_K↑ 4). That keeps the folder
under its 250 GB cap with C9's 57 GB. (viii) resumes from HDD_1TB and writes there.

## Carried in

| item | from | here |
|---|---|---|
| (vii) F⅒ continued 1000 → 2000 (≈ 7.7 h) | C4′ (d)+(f) Parked 3 | D2 |
| (viii) B_O continued 512 → 1000 (≈ 7 h) | Parked 4b, `/challenge-pr` on #379 finding 1 | D2; U3 if (a) |
| the reader hashes its inputs | Parked 2, `/challenge-pr` on #378 finding 3 | U1's reader (and every C8 reader) |
| the two `c2_seeds` refusals | C9's close, Blocked 181 (ii) (a) | **not triggered**: C8 reads C4's three gated τ and derives no centred tolerance arm. Recorded at each unit's freeze, carried at close |
| C9's standing note (A3's null biased on a max over heads) | Blocked 180 (e) | not triggered: no C8 gate takes a max over heads (U4's previous-token date is beside) |

## Decisions (each yours; recommendation first; STATE Blocked 182)

- **D1, the plan.** (a) U1 (i), U2 (ii), U3 (iv) with (viii), U4 (iii) + (v), close-out: 4 units,
  GPU ≈ 15–21.5 h (**recommended**: (i) is free and dates the window for (ii); (iv) is the
  causal test). (b) U1 only, then re-plan on its reading: one unit, plus a revised design. (c)
  (a) without U4: 3 units, which leaves "what it does" open.
- **D2, the carried continuations.** (a) (viii) in U3; (vii) stays carried, unscheduled
  (**recommended**: (viii) tests whether a low-rate restart gets a late transient as F⅒ did,
  which bears on the trigger; (vii) asks how F⅒'s run ends, which is "what it does" at a tenth).
  (b) both (+7.7 h). (c) neither.
- **D3, (vi) the LLC.** (a) Out of C8, to §7b T3/T4, and only if (ii) leaves the landscape open
  (**recommended**: it assumes a local minimum, and the window is ~50 updates in at the peak
  rate). (b) As a fifth unit.
- **D4, registration.** (a) Draft entries for U2's and U3's rules at their freezes, for you to
  register or decline before they compute (**recommended**: their outputs are wholly unseen, and
  P-SCH7/8 were registered this way). (b) Tier 1 throughout. In both, U1 stays tier 1: its logs
  are partly seen.
- **D5, F_w's horizon.** (a) To 128, and to 1000 only if it has no transient by 128 (+6.5 h;
  **recommended**: that is the one arm whose induction date answers "is the transient linked to
  induction"). (b) To 128 only.

## Worth challenging (for the PR)

1. **(i) reads the union of two event kinds**, with no bar on either alone. A grad-only or a
   loss-only reading could fit the edge-of-stability lean better. Rejected: two read families
   doubles the cells, and the kinds are reported beside.
2. **Background from the arm's own log** rather than from a no-transient twin. Only B_P has
   twins (B_O, B_K, BKO). The arms from init have none, since A0 is on a ramp. An own-log background
   mixes regimes (F's background runs to 1000, through induction).
3. **F reads for F¹ and F¹′.** F's onset is known only to C4's grid (a 16-update interval), so
   its window is 32 updates long. F¹ has the finer onset, but its log stops at 128.
4. **The constants are placed on A0's late log**, where noise may be smaller than early. A
   threshold set too low there gives false events early, which the background absorbs, and
   G-plant checks the real logs at the windows' own positions.
5. **U2's preconditioned read skips F⅒** (≈ 6 h to rebuild its v̂ at 784). The arm least like the
   others then has raw λ only.
6. **The Π bar treats the five conditions as independent.** F, F′ and F½ share an init.
