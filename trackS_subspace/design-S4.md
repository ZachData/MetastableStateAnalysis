<!-- trackS_subspace/design-S4.md -->
# Track S — design: S4, the toy induction model

**Frozen 2026-11-08** on Blocked 148, the user's: **D1 (a′), D2–D5 (a)**, as recommended
("Decisions" below), before any arm was read. Proposed the same day and revised after
`/challenge-pr` on #331 (findings 1–5 taken, "Revised after #331" below). Choices U1's code places
where this text leaves one are recorded in `status-S.md` "S4 U1" as fixed before output. Row: `docs/ROADMAP.md` §7a S4. Opened by Blocked 147 (b), the user's. The scan before
it: `trackS_subspace/lit-S4.md` (this unit). Tier 1: S4 registers nothing and scores nothing.

## What S4 answers (fixed before reading)

| # | question | from |
|---|---|---|
| Q1 | **Is there a break to move?** On an attention-only toy trained on text, does V (the count of layer transitions where E₁ falls) go from low to a plateau over training, as on Pythia, and does the centred count? No released arm has a step-0 checkpoint (the toys start at step 1, H at 100), so "low" is read at the earliest checkpoint, with fresh inits of the same config beside (Instruments) | the gate for Q2; C2's "Established" 1–2 |
| Q2 | **The causal arm C2 lacks.** Where induction cannot form or is suppressed, does V's date or the centred date move against the arm where it forms? | ROADMAP §7a S4; C2's F2 ("break, then induction" not shown causal) |
| Q3 | **Without a warmup.** Do the toys' dates fall at a step, with a constant LR and no warmup, so the date cannot be the warmup's? | C2's F1 (the dates not separated from the warmup) |
| Q4 | **The known circuit.** With the previous-token and induction heads named, S1's atlas (shared write / read / key subspaces) on the toy: does the K-composition channel appear as a shared subspace, and when? | S1, S5; B2's C6 |
| Q5 | **The tied arm.** In a weight-tied (looped) model, the theory's setting: Q1–Q4 again, the planted negative sector (set after training, held fixed), and B7's partner persistence (depth ≥ 3) | ROADMAP S4's two "Received" paragraphs |

## The arms

| arm | model | induction | source | cost |
|---|---|---|---|---|
| **R2–R6** | `timaeus/L2` … `L6`: attention-only, untied, d 256, 8 × 32, Shortformer, vocab 5k; constant LR 1e-3, **no warmup**; 451 checkpoints each | forms (depth ≥ 2) | released (`lit-S4.md` §1) | forward passes only, 15.5 MB a checkpoint |
| **R1** | `timaeus/L1W256-dense`: 1 layer, checkpoints every 10 steps | **cannot form** (no second layer) | released; recipe unknown | as R2; read only if its recipe can be read from its files |
| **H** (D1) | HAPAX's `vanilla_1b` against `hapax_1b`: Pythia-1b's shape, the Pile, every 100 steps from 100 | vanilla forms; Hapax **suppressed** (1–2 of 128 heads above 80 % prefix-matching, against 8–9) | released | 2.0 GB a checkpoint; 21 a side, steps 100–2000 and 4000 (~84 GB, `HDD_1TB`) |
| **Tu, Tt** | our runs of L4's recipe: **u**ntied, and **t**ied (D3); each with the plain loss and with Hapax's mask (D2); 2 seeds | forms / suppressed / unknown in the tied arm (2601.09588) | trained here (the 3080) | unmeasured; U2 measures θ first |

`L1` is an empty repo, so R1 is the series' only 1-layer candidate. The L-series' data
(`timaeus/pile13m`) is private, so our runs read `timaeus/dsir-pile-10m` (tetrahedron's,
public) in the series' tokenizer. Tu therefore differs from R4 in its data, and Tu's known answer
is checked against R4 (U2).

## Instruments (reused, not rebuilt)

- **E₁ per layer** on unit-normed token states (resid_pre of each layer and the final
  resid_post), **V** = `trackA_force/a5_seeds.n_viol` at `REL_TOL`, and the **centred count**
  (`trackC_writeup/c2_aniso.centred`): C2's definitions, unchanged.
- **Prompts: C2's text, not C2's particles.** C2 read v1's 7 scored prompts cut at 512 Pythia
  tokens (`trackB_circuit/b0_gate.py`). S4 takes the text of each 512-token cut and retokenizes it
  per model. H keeps Pythia's tokenizer, so its particles are C2's. In the toys' 5k vocabulary
  the same text is 503–842 tokens, and repeats are far more common: in `hdbscan_code`, 236 of 530
  tokens are one single-space token (`/challenge-pr` on #331, finding 3). Shortformer puts no
  position in the residual stream, so those tokens start at one point. Per prompt and model, U1
  records the token count and the largest single-token share beside every count. A prompt whose
  largest share exceeds ¼ is read beside, not in the toys' V.
- **Dates:** A5's D4 (a), the first grid step with V ≥ half the median of V over a plateau
  window, for V and for the centred count. `trackA_force/a3.d4a` hard-codes Pythia's grid and
  window, so U1 writes the same rule as a pure function of (grid, V, window). That function is
  tested to equal `a3.d4a` on Pythia's grid before it reads any arm. **H's window is steps 1000,
  2000 and 4000**: H has Pythia's shape and tokens per step, and 4000 is in H's U1 read. **The
  toys' window is steps 525, 1050 and 2100** (the checkpoints nearest them). These are fixed
  steps, not a fraction of anything: under a constant LR, the fraction of a 75,000-step run that
  each represents means nothing. They are Pythia's three steps scaled by 75,000 / 143,000, which
  was chosen before any output and does not depend on any arm's induction date.
- **Step 0.** No released arm has one. The gate's re-inits and Q1's floor use fresh
  initialisations of each arm's config: TransformerLens's default init for the toys, and
  `GPTNeoXForCausalLM.from_config` for H. These are draws from each recipe's init law, not each
  run's own step 0, and every reading says so. The same rule dates **induction**: Olsson's prefix-matching score on repeated
  random tokens (the max over heads), with the previous-token score and the ICL score (loss at
  token 500 minus token 50) beside. The behavioural score is `p7_motifs/formation_curve.behavioural_induction_score`.
- **The gate** (calibration, `CLAUDE.md`): before any date is read on an arm, C2 unit 3b's
  preflight is run at two steps of that arm (one before and one after its induction date): plant A
  (a common vector) and plant B (token-independent spread) per transition, through both counts,
  with unit 3s's thresholds (re-init centred rate ≤ 5 %, plant B ≥ 0.4, plant A at 0.5 ≤ 0.1). An
  arm whose gate refuses has its V series recorded as description, and no date is read from it.
- **Q4's atlas:** S1's mean projector and R1′ (participation ratio against its Haar value),
  `trackS_subspace/s1.py`, with d 256 and r ≤ 32 in place of 410m's constants.

## Units

**U1, the released arms (no training).** R2–R6 on a subgrid of their 451 checkpoints (every step
1–26, then every second step of the log grid, ~120 per depth; the window's 525, 1050 and 2100
always included), and R1 if readable. H at steps 100–2000 and 4000 (D1). Order: (i) one checkpoint per arm opened and checked *populated* (E₁ finite, not
constant across layers; the prefix-matching score non-zero at the last step); (ii) the gate at two
steps per arm; (iii) every date. The `.pt` and `.pth` files are pickles: they are loaded with
`torch.load(weights_only=True)` under `python -I`, in their own directory.

Outcome rows (fixed here):

| reading | meaning |
|---|---|
| V = 0 at every step, or at its plateau from the first checkpoint | **no break to move** on that arm: Q2 and Q3 cannot be asked there |
| V dated; induction dated | the order and the gap in grid steps, per depth (description; one seed per depth) |
| V dated on R1 | a break where induction cannot form: Q2 answered "not induction's" for that recipe (a toy's answer) |
| H: both sides dated | each side's dates, the gap in grid steps, the per-prompt dates' spread beside; **no label** (one seed a side, no noise floor) |

**Then U2 is gated.** If U1 finds no break to move on any of R2–R6, U2 trains only Tt (for Q5) and
Tu is dropped. Otherwise:

**U2, the trained arms.** θ measured on L4's configuration (50 timed steps after 10 discarded).
Tu's known answer against R4: induction forms, its date within a factor 2 of R4's, and the loss
at R4's last step within 5 %. If it misses, the trained arms stop there and U2 reports the miss.
Then Tu and Tt, each plain and Hapax-masked, on 2 seeds. Checkpoints on the L-series' grid, plus
every step through the window that U1 dates on R4. ROADMAP's "every step saved" is about 1.2 TB an
arm at 75,000 steps. Runs end at the larger of 4 × R4's induction date and 4,000 (so the dating
window's 2100 is always inside), and at 75,000 at most. The
readings and gate are U1's.

**U3, the atlas, the circuit and the tied arm.** Q4 on R2, R4, Tu and Tt at their own dates
(before, at, and after induction, and the last step). The circuit is named by the scores (the
previous-token head, the induction head) and the K-composition score; then: is the previous-token
head's write space inside the induction head's key space shared beyond Haar? In Tt, the planted
negative sector (a degenerate negative pair in one head's OV, set after training and held fixed)
is read through E₁ and V against the unplanted model. B7's partner persistence runs on Tt and Tu
at depth 4, after the second-sink check. Close-out inside U3 if nothing is Parked.

## Exit and budget

**Exit:** each arm carries either dates for V, the centred count and induction (or "no break to
move", or "refused by its gate") and the gap between them; Q4's reading on at least one untied and
one tied toy with the circuit named; Q5's plant and B7's read; or, for each, a recorded reason it
cannot read. **Budget (D5):** 4 units: U0 (this), U1, U2, U3, with the close-out inside U3. At the
budget S4 closes at what it has.

## Fences (written before any output)

- **A toy's answer.** Vocab 5k, Shortformer, no MLP, constant LR, one seed per released depth: a
  date here says what such a model does, not what Pythia does.
- **H is the closest arm to C2 and is not C2's model.** It is 1B, not 410m. Hapax suppresses
  induction but does not remove it. Its schedule and data order are unread (U1 reads HAPAX's
  appendix and code first). It has no step 0, and one seed a side.
- **H and the reserved rung.** `p8_scale_ladder/design-8.md` reserves `pythia-1b`: "no induction
  measurement may be run on 1b or 1.4b until a prediction naming that model is in
  `claims/registry.json`". `vanilla_1b` is not `pythia-1b`'s checkpoints, but it is the same
  shape, data and hyperparameters. An induction date measured on it previews the reserved rung's,
  and `P-I7` ("a transformer not yet measured by this project") would lose it as a candidate
  (`/challenge-pr` on #331, finding 1). The reservation names induction only, so the energy
  counts on H are outside its letter. H's only possible noise floor (`pythia-1b`'s own checkpoints)
  stays closed while the reserve holds. D1 is where the user decides this.
- **R1's recipe is unknown.** A break on R1 reads "not induction's" only if its recipe is close
  enough to R2's (U1 states which fields match).
- **No warmup is not "a schedule separated".** The toys show whether a break exists without a
  warmup. They do not show that Pythia's break is not the warmup's (F1 stays as it is for 410m).

No closed-form claim is made here, so there is no `sympy` check. The Hapax mask's fraction on our
data is measured in U2 against the paper's 31.7 %.

## Decisions (Blocked 148, the user's)

**Decided 2026-11-08, the user's: D1 (a′), D2 (a), D3 (a), D4 (a), D5 (a).** So: H is read in U1
for the energy counts only, and no induction score of ours runs on `vanilla_1b` or `hapax_1b`
(H's induction dates are HAPAX's published ones); the trained arms use Hapax's mask with R1
beside; the tied arm is one block shared across 4 layers; the MLP arm is Parked to ROADMAP S4;
4 units, the close-out inside U3.

| | options | recommended |
|---|---|---|
| D1 HAPAX's pair (see "H and the reserved rung") | (a) in U1 beside the toys, induction measured on both sides; (a′) in U1 for the energy counts only, with induction's dates taken from HAPAX's published curves (prefix-matching, the loss drop), so no induction measurement of ours runs on a 1b-shaped model; (b) not in S4, carried to ROADMAP C5 as a C2 arm; (c) U1 is H only, the toys deferred | **(a′)**: it keeps the reserve to its letter and still asks Q2 in Pythia's shape, at about 84 GB of reads. The cost is induction dates read off published figures, coarser than ours. (a) previews the reserved rung, so it is for the user to take deliberately (design-8's own wording), not by default. (c) would drop Q3–Q5 |
| D2 "without repeats" | (a) Hapax's mask in the trained arms, R1 beside in U1; (b) Hapax's mask only; (c) a synthetic task with and without repeats by construction (ROADMAP's literal wording) | **(a)**: the mask keeps the inputs and removes only copying's gradient, and it matches H; (c) changes the data the energy is read on, so a date would move for two reasons |
| D3 the tied arm | (a) one attention block (with its LayerNorm) shared across 4 layers, embedding and unembedding untied, otherwise L4's recipe; (b) depth 2 (the classic circuit; B7 cannot read); (c) (a) and depth 6 | **(a)**: depth ≥ 3 for B7; R4 is its untied reference on the same recipe |
| D4 the MLP arm (ROADMAP's "and one with MLPs") | (a) dropped from S4, Parked to ROADMAP S4; (b) a fifth trained arm in U2 | **(a)**: the circuit is known in attention-only models, and no released MLP member of the series was found |
| D5 budget | (a) 4 units, close-out inside U3; (b) 3 units: U1, then U2 and U3 merged into one, Tt only | **(a)** |

## Revised after #331

`/challenge-pr` on #331: accept with changes; all five findings taken.

1. H would date induction on a model with `pythia-1b`'s recipe, which `design-8.md` reserves.
   That is now a fence, and D1 gains (a′) (energy only), now the recommendation.
2. H's dating window needs step 4000, and U1 read only steps 100–2000. 4000 is added (21 a side,
   about 84 GB).
3. "C2's instrument, unchanged" was not true of the toys' prompts. The 5k tokenizer changes the
   count and the repeats, and C2 cut at 512 tokens, not 1024. The prompts are now C2's 512-token
   text, retokenized, with the largest single-token share recorded and a ¼ bar.
4. Under a constant LR, "fractions of the run" are just fixed steps (525, 1050, 2100), and U2's
   stopping rule could end a run before 2100. The steps are now stated as fixed, and runs go to
   at least 4,000.
5. `a3.d4a` hard-codes Pythia's grid, and no released arm has a step 0. The rule is now written
   as a pure function tested against `a3.d4a`, and step 0 comes from fresh inits of each config,
   labelled as such.

## Worth challenging

- **Starting from released arms over training our own.** ROADMAP's row is "train a toy". The
  released series gives five depths for free, but on one seed per depth and private data, so
  the known-answer check for our own runs is against a model we cannot replay exactly.
- **v1's 7 prompts on a 5k-token model.** They keep C2's instrument, but retokenized they are
  different particles. A sample of the library's `val` is the alternative, at the cost of
  comparability with C2.
- **Folding H into S4 (D1 (a′))** widens a toy row toward C2's own question. The alternative is
  (b): keep S4 a toy and give H its own C5 row. Under (a′), induction's dates on H come from
  HAPAX's figures, read at their resolution.
- **The toys' plateau window is fixed at 525, 1050, 2100.** No principle maps Pythia's 1000–4000
  onto a no-warmup toy. If the toy's break comes after 2100, the rule dates it late or reads "no
  change". The alternative, a window after each arm's induction date, would tie V's
  date to the date it is compared with.
- **The gate's thresholds are 410m's.** Unit 3s's (5 %, 0.4, 0.1) are reused without re-deriving
  them on a 2–6-layer model, where V has only 2–6 transitions per prompt.
