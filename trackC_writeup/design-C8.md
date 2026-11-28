<!-- trackC_writeup/design-C8.md -->
# Track C — design: C8, what triggers the transient, and what it does

**Status: frozen 2026-11-27 (Blocked 182 all (a), the user's). U1 ran the same day: (i) refused at
its gate (G-plant's grad power; `status-C.md` "C8 unit 1"; Blocked 183). U1's two gate deviations,
fixed before any log was opened, are in `c8_read.py`'s docstring.** Proposed 2026-11-27 (U0,
writing only; nothing computed, nothing trained). Opened by
the user 2026-11-27 ("U0, the literature scan and design ... candidate (i) first"). The row is
the user's, from 2026-11-24 at C4′ (d)+(f) U3b: "something ... seems to trigger the repulsive
stage ... deeply linked to induction, but still a little mysterious". Decisions D1–D5 are in STATE
Blocked 182. Revised after `/challenge-pr` on #387 (accept with changes; findings 1–6 taken
before any output): (i) gains batch twins and Π over four factors (1), a real-log G-null (2) and
the label "marked at ≥ half the peak" (5). U3 gains the manipulation check H-w (3). U2 reads each
re-run against its own onset (4). One path is fixed (6). Scan: `lit-C8.md`. Arithmetic:
`tools/math_checks/c8_thresholds.py`. (i)'s rule is
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

**The references.** The onsets above, read from `data/trackC/c4pdf/2026-11-25-u3c/reading.json`
(the main tree's `data/`, not HDD_1TB; `/challenge-pr` on #387, finding 6). F's and F′'s
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

**The batch-matched control** (`/challenge-pr` on #387, finding 1). Every arm on one order reads
the same batch at the same update (`design-C4pdf.md` "The arms"). So one hard batch can put an
event into several windows at once, whatever the arm's state. **An event at s does not count if
the arm's batch twin has an event of the same kind at s.** The twins: A0 for F, F½ and F⅒ (C4's
order), A0′ for F′ (order π), and B_O for B_P (A0's batches 256–511). A0 is not a twin for the
dynamics, since it is on a ramp. It is the right control for an event the batch makes. Twin
events are reported beside. The same exclusion applies inside the background.

**The constants, placed before any window of a with-transient arm is opened**, on A0's updates
2001–4000 (a control arm, smooth there): σ̂_L = 1.4826 · MAD(ΔL)/√2 and ĉ_g = 1.4826 ·
MAD(Δ ln g)/√2. Then δ_L = max(0.02, 6 σ̂_L) and κ = max(1.5, exp(6 ĉ_g)).

**The background**, per condition: the arm's log from 9 (265 on a branch) to min(end, 1000),
cut into non-overlapping blocks of the window's length, starting at the span's first update. A
block is dropped if it comes within 16 updates of the window. b_a is the share of the remaining
blocks that hold an event. At least 4 blocks are needed (each condition has 8–46,
`c8_thresholds.py`); with fewer, the condition is refused.

**The factors of Π** (finding 1). Conditions on one order whose windows share an update are one
factor, with the larger b. F and F½ share updates (at 3e-4, F½'s 45–64 lies inside F's 33–64),
so they are one factor. That leaves four: {F, F½}, F′, F⅒ and B_P. F⅒'s window (737–784) and
B_P's batches (264–287) share no update with the others. Π ≤ 0.05 then needs some b below
0.05^(1/4) ≈ 0.473 (`c8_thresholds.py` (3)).

**The gate (U1, before the read; revised after `/challenge-pr` on #387, finding 2: the first
G-null, on iid Gaussian noise at 6σ, could not fail):**

| gate | what | bar |
|---|---|---|
| G-null | **the full rule R_i on the control arms' real logs**: A0 stands in for F, F½ and F⅒ at their windows, A0′ for F′, B_K for B_P (with BKO as its twin check and B_O as its batch twin). It is run at the real windows and at every joint shift of all windows by a whole number of window lengths that keeps each inside its arm's span and clear of a branch's restart | "marked" or "marked at ≥ half the peak" at ≤ 5 % of placements; at the real windows, reported |
| G-plant | the same control logs at the real windows. A loss rise of 1.5 δ_L and a grad jump of κ^1.5 are planted at each update of the window in turn (the twin left unplanted) | detected at ≥ 90 % of positions in every window |
| G-edge | the same at 1 × δ_L and 1 × κ | reported (power at the threshold), no bar |
| G-synth | iid synthetic logs on each layout | a code test only, not a gate |
| H-ref | F's and F′'s rule-T references reproduce C4′ (c)'s L7 sets (T3–T5 on both) | exact |

If a gate fails, U1 reports a refusal. The constants may be re-placed once, and only before any
with-transient window has been opened. That re-placement is recorded.

**Rule R_i, at each gated τ:**

| label | when |
|---|---|
| **marked** | every one of the five conditions has a counted event in its window, Π over the four factors ≤ 0.05, and neither B_K nor BKO has one in B_P's window |
| **marked at ≥ half the peak** | F, F′, F½ and B_P each have a counted event, Π over their three factors ({F, F½}, F′, B_P) ≤ 0.05, B_K and BKO clean, and F⅒ has none |
| **silent** | none of the five has a counted event in its window |
| **mixed** | otherwise; each condition's label shown |

The second row was added before any output (`/challenge-pr` on #387, finding 5). F⅒ is the
condition least like the others: it is a tenth of the peak, its transient comes after ~750
updates and is a sixth of the size. So one quiet F⅒ window should not send a pattern the other
four share to "mixed". The cells stay these five conditions and this one statistic. The joint
reading holds only where it holds at all three gated τ. Otherwise it is "mixed across τ". The
three-factor bar needs some b below 0.05^(1/3) ≈ 0.368 (`c8_thresholds.py` (3)).

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
- *marked at ≥ half the peak*: as *marked*, for the four peak-like conditions; F⅒'s trigger is
  then a separate question, and U2's raw λ on F⅒ is read beside, not with the rest.
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
- **The re-runs are new trajectories, so each is read against its own onset** (`/challenge-pr`
  on #387, finding 4: F, F¹ and F¹′ are bit-equal at 1 of 128 updates, and their losses differ by
  up to ~1 %). Records (`c4_read`) are written on every saved state of F², B_P² and B_K², and
  rule T gives each its own onset **before any λ is computed**. F²'s and B_P²'s windows are their
  own, not F¹'s or B_P's. If F² or B_P² has no transient, or B_K² has one, the rule's premise
  fails on that re-run and (ii) is refused there, with the cause named.
- **Rule shape**, constants (the band θ, the batch, the iteration count) pushed in U2 before
  output, with a gate on a planted quadratic of known λ: **"crosses at the onset"** if, on F² and
  B_P², the first saved state with λ_P ≥ θ · threshold lies in that re-run's own window, and B_K² stays below
  θ · threshold through 128. **"crosses, not at the onset"** if a crossing falls outside the
  window. **"never crosses"** otherwise. θ is a band, since Cohen et al. is full-batch (`lit-C8.md`
  Parked 2, read before the freeze).
- **Lean** (the ROADMAP's): crosses at the onset.

### U2 as frozen (2026-11-27, before any λ; Blocked 183 (a), the user's)

Code `c8_train.py` `0daf988e9333`, `c8_sharp.py`, rule `c8_sharp_rules.py` (hashes in the draft
below; pushed in `9015166` before any output). `lit-C8.md` Parked 2 read first (Cohen et al. v2:
Eq. (2)'s preconditioner; §5, minibatch Adam "never rises more than a bit beyond" the full-batch
threshold and "almost" reaches it at their largest batch). What the shape above left open, and
the changes to it, all fixed before any λ:

| item | fixed as | why |
|---|---|---|
| P | (1 − β₁^k)·diag(√(v/(1 − β₂^k)) + ε), k the optimiser step | Cohen et al. Eq. (2) = AdamW's step as PyTorch applies it; the shape's diag(√v̂ + ε) drops the (1 − β₁^k) |
| v = 0 coordinates | excluded from P⁻¹H | embedding rows of tokens no batch has held since the optimiser's start: Adam never moved them, but a held-out token there adds ~1e8·H_ii. 118 of the batch's 6629 input tokens are unseen after one batch, 1 after four, 0 after sixteen (C4's order); the count masked is recorded per state |
| iteration | **Lanczos**, not power iteration (10–40 steps, ≤ 1e-3 relative change twice) | on G-quad's planted negative-dominant case (gap 0.93) a shifted power iteration missed by 4 % at 40 steps; Lanczos uses the same H·v products. The top Ritz value errs low |
| batch | C4's held-out rows `HELD_OUT[0]` … + 32, the first **1024** predictions each, fp32, eager | the double backward at 2048 does not fit the 3080's free memory (7.6 GB; 1536 peaks at 5.5 GiB). A 32k-token minibatch Hessian's top eigenvalue errs high against the full loss's |
| θ | **0.5** (θ = 1 beside) | from Cohen et al. §5 and the clip and weight decay outside their threshold; no C8 λ seen |
| windows | (i)'s, (t⁻ − 16, t], a branch's from 256 + 8 | "that re-run's own window" |
| the label left open | per re-run "at" / "outside" / "never" (BK2 "crosses" / "below"); "at, at, below" → crosses at the onset; any "outside" or BK2 "crosses" → crosses, not at the onset; "never, never, below" → never crosses; else **mixed** | the shape's "otherwise" would have called "at" with "never" "never crosses" |
| refusals | the premise (F2 or BP2 no transient, BK2 one), or an unconverged state whose being a crossing would flip the label | |
| raw λ | on the originals only (F¹, F½, F⅒, B_P, B_O, B_K at U1's windows ± 2 saved states; A0's grid ≤ 128), beside | the shape's arms; not on the re-runs |
| G-hvp | on Pythia's **released** step128/256/512 (not C8 arms): fp32 Rayleigh quotient against fp64's ≤ 1e-3 along u = H³r; fp64 H·u against an fp64 central difference (h = 1e-7) ≤ 1e-3; symmetry ≤ 1e-3 | a first version on step143000 failed: fp32 is off there by ~7e-3 (sharp late attention; 1.5e-7 at step128). The instrument is not claimed at the end of training |

**Budget:** one H·v at the batch is 3.4 s, so ≤ 2.3 min a state; ≈ 59 preconditioned + ≈ 50 raw
states ≈ 2.5–4 h, against the 2.5 h above (the 1024-token context halves the per-row cost; 32
rows doubles it). **A risk seen before output:** F2 and BP2 start a fresh AdamW at the peak with
no warmup, so their first states may already sit over θ·38/η. Then (ii) reads "crosses, not at
the onset" from its first state and cannot date the window. That reading is kept if it comes.

### U2's draft entry (D4 (a); yours to register or decline before any λ)

`c8_sharp measure` refuses until `trackC_writeup/c8_entries.json` is committed as
`{"ii": "<id>"}` (an id in `claims/registry.json`) or `{"ii": "declined"}`.

- *id:* the next `P-SCH` number (`P-SCH9`); claim H-RESIST; experiment "C8"; relevance 0.6.
- *statement:* On pythia-70m, for the re-runs F2 (seed 0's init, flat 1e-3, grid D to 128), BP2
  (A0's step-256 weights, fresh AdamW, flat 1e-3, 256 + offsets to 384) and BK2 (as BP2 with A0's
  rebuilt moments kept), at every gated centred τ arm (9.61e-5, 1e-4, 3e-4): the first saved state
  whose preconditioned sharpness λ_max(P⁻¹H) reaches 0.5 × 38/η lies in F2's and in BP2's own
  window (t⁻ − 16, t] of rule T's onset, and BK2 stays below it through 384 (`c8_sharp_rules`
  R_ii "crosses at the onset" at every arm).
- *h0:* the crossing is not tied to the onset: it comes outside a window, on BK2, or never.
- *falsifier:* at some gated arm R_ii reads "crosses, not at the onset", "never crosses" or
  "mixed". A refusal (premise, unconverged) or H-rerun failing scores nothing.
- *instrument:* `c8_train.py`, `c8_sharp.py`, `c8_sharp_rules.py` at the hashes the commit names;
  records by `c4_read._records_of`; rule T `c4pdf_rules.transient`.
- *null_construction:* calibration, not a p-value: G-quad (planted spectra) and G-hvp.
- *notes:* 70m only; one re-run per arm; λ on a 32 × 1024-token minibatch, a lower bound by
  Lanczos; the threshold is full-batch theory; θ placed from the literature; the risk above.

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
- **H-w, a manipulation check, before F_w's rule T is read** (`/challenge-pr` on #387, finding 3).
  Warming the moments at the init's weights changes only how far the first updates move. Both arms
  meet the same clipped first gradients (`c4_train.py` 290–291). So H-w compares F_w's relative
  drift from init, ‖θ_s − θ_0‖/‖θ_0‖, at s = 1, 4, 16, 48 with F²'s at the same s. **It passes
  if, at s = 1 or 4, F_w's drift differs from F²'s by more than F¹ and F¹′ differ from each other
  there** (the replicate spread). If it fails, the warm-up did not change the early steps beyond
  noise, and F_w gives **no reading** (not "fresh moments do not matter"). D5's extension then
  does not run. B_K↑ gets the same check against B_K, and its overshoot has to reach B_P's drift
  at 16 within the spread between B_P and B_P².
- **Rule shape:** rule T on each that passes its check, at the three gated τ. The joint reading:

| F_w | B_K↑ | reads |
|---|---|---|
| none | transient | large early steps, not fresh moments as such |
| transient | transient | large early steps at either state |
| none | none | fresh moments themselves |
| transient | none | the init and the trained state differ: no joint claim |

- **D5:** F_w continued to 1000 if H-w passes and it has no transient by 128. Its induction date is then the
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
- **D5, F_w's horizon.** (a) To 128, and to 1000 only if H-w passes and it has no transient by 128 (+6.5 h;
  **recommended**: that is the one arm whose induction date answers "is the transient linked to
  induction"). (b) To 128 only.

## Worth challenging (for the PR)

1. **(i) reads the union of two event kinds**, with no bar on either alone. A grad-only or a
   loss-only reading could fit the edge-of-stability lean better. Rejected: two read families
   doubles the cells, and the kinds are reported beside.
2. **Background from the arm's own log**, with batch twins excluding shared-batch events
   (revised after `/challenge-pr` on #387, finding 1: the first push let A0 be no control "since
   it is on a ramp", which was about its dynamics, not its batches). The own-log background still
   mixes regimes: F's background runs to 1000, through induction.
3. **F reads for F¹ and F¹′.** F's onset is known only to C4's grid (a 16-update interval), so
   its window is 32 updates long. F¹ has the finer onset, but its log stops at 128.
4. **The constants are placed on A0's late log**, where noise may be smaller than early. A
   threshold set too low there gives false events early, which the background absorbs, and
   G-plant checks the real logs at the windows' own positions.
5. **U2's preconditioned read skips F⅒** (≈ 6 h to rebuild its v̂ at 784). The arm least like the
   others then has raw λ only.
6. **The Π bar's factors.** F and F½ are merged into one factor, since their windows share
   updates and batches (finding 1). F, F′, F½ and F⅒ still share an init, and Π treats the four
   factors as independent beyond that.
7. **The second label, "marked at ≥ half the peak"** (finding 5), adds a cell to (i)'s family.
   It was added before any output, so the alternative of a separate family was rejected.
