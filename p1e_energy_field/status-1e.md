<!-- p1e_energy_field/status-1e.md -->
# Phase 1e — STATUS

<!-- phase-card -->
## Card

- **Question:** Read the residual stream as the theory's field rather than as clusters: where are the wells and crests of the energy at the measured β, and does each token's update ascend the field (attraction) or descend it (repulsion, the packing end)?
- **Inputs:** `pythia-410m`, the 18 Stage 0 steps, **8 long passages (1,032–2,041 tokens; hash `ba605f4e14b5`) primary**, the 7 v1 passages beside, unit LN1 rows, β = 3.5 [1.6, 5.6] — `p1e_energy_field/design-1e.md` "Inputs, fixed here"
- **Results:**
  - Opened 2026-10-06 with a literature scan; the design was frozen the same day as proposed, then its passages were changed to the long set before any output. The closed-form steps the design uses are checked (M6 corrected after review: in a cloud-centred frame the field also has a floor off the tokens' span) — `p1e_energy_field/design-1e.md` "The math", `p1e_energy_field/lit-1e.md`
  - Four new long passages were built under a rule committed before their sources were fetched, and all 8 long passages were extracted at the 18 steps on the GPU (activations only) — `p1e_energy_field/status-1e.md` "The 8 long passages"
  - By the frozen rule, each block's update ascends the field in every band at steps 8–256 and 16000–54000 (35 of 54 cells, 0.42 by chance); the label is the field's β = 0 mean term, and the cosine excess is small (≤ 0.16) — `p1e_energy_field/status-1e.md` "U2's block arm"
  - With each token's component along the shared update direction removed (mixed in every cell at init), the token-specific part descends the field at 32–2000 and ascends in L9–16 from 4000 and in L1–8 at 64–128 and from 16000; the early ascent is the shared update, and 143000's last-block descent is not token-specific — `p1e_energy_field/status-1e.md` "U2's block arm"
  - Splitting each block's update by a hooked pass, the MLP supplies most of the shared update and its shared part ascends too (24 of 31 windows by the rule fixed before output; the sink's value and the biases supply almost none); on the long passages attention's token-specific part never ascends the field in layers 1–22 and descends from step 256 on (v1 ascends at 8–32); the late token-specific ascent in L1–16 goes with the MLP's — `p1e_energy_field/status-1e.md` "U2's attention arm"
  - Against each head's own attention kernel instead of the idealised field, attention's token-specific move descends at steps 128–2000, in the sum and head by head; from 4000–8000 the sum over heads ascends while more heads still descend their own kernel than ascend it (10 of 13 windows; a majority of heads in only 2), so the late ascent is a property of the sum, not of the heads; the shared ascent is the MLP's up to 1000 and attention's mean pull from 2000 in the last layers (4 windows replicated on v1) — `p1e_energy_field/status-1e.md` "U2's per-head arm"
  - U2's reading is split by training stage (the user's decision): up to step 1000 the shared ascent is the MLP's and the token interactions descend their heads' kernels; from 2000 the shared ascent is attention's mean pull in the last layers, and from 4000 only the sum over heads follows its kernels, while more heads still descend their own than ascend it — `p1e_energy_field/status-1e.md` "U2's per-head arm"
  - At the measured β the field over the tokens has a single well in every trained layer band and step (one exception, the early layers at step 128); a few wells appear only at about twice that β and many by four times, so the probe's few wells were another frame's. Its density is more uneven than a matched Gaussian cloud's at every step, init included, even after correcting that comparison's own bias, and is mostly each token's closeness to the cloud's mean: punctuation at the centre, content words at the edge; by the end of training only the last layers are more uneven than at init — `p1e_energy_field/status-1e.md` "U1, the field at the tokens"
  - The saddle unit is closed at the measured β, since there are no wells there to find saddles between (the user confirmed); the phase's results are on one page — `p1e_energy_field/status-1e.md` "Blocked 30 decided"
  - Read beside that, at a β about three times the measured one, and in a frame with the cloud's mean removed. Both find wells: a few to hundreds early in training and one dominant well late at the higher β, and 2–4 late in the centred frame, the probe's count; at init the count depends on the passage length (15–17 on the long passages, up to one per token on the short ones). The probe's own frame (the residual centred without layer norm) agrees with it cell by cell, gives the same labels, and on the probe's own passages reproduces its 2–4 late wells. In both, the wells exist at init and sort by content. At the higher β, the comparison with a matched Gaussian is mostly forced, because the Gaussian has one well. Against init, training adds wells at steps 64–8000 and deepens them in the last layers, and both are gone by 16000. The rule's step-0 test compares labels only, so it calls almost none of this learned. In the centred frame, Phase 10's groups sit inside single wells well above chance, partly because both group tokens by proximity. A defect in the well finder (a float32 merge) was found and fixed on the way, and the measured-β result was audited clean — `p1e_energy_field/status-1e.md` "Blocked 30 (b) and (c)"
- **Superseded / wrong:**
  - #158's headline read the frozen labels as the token interactions ascending at 8–256 and descending ("repulsive") at 1000–8000 and 143000; the shuffle null cannot see an update every token shares, which carries both — `p1e_energy_field/status-1e.md` "Corrections received"
- **Registry:** none, because the phase is exploratory and unregistered; it is fenced off P-S1, P-γ1/P-γ2 and P-M1 — `p1e_energy_field/design-1e.md` "Fences"
- **Depends on:** 1d@4168cdd237, 10@c159a02c9b
- **Feeds:** 10
- **Open threads:**
  - Whether the late split between heads and their sum is heads cooperating or a few large heads (the saved means cannot split it) — `p1e_energy_field/status-1e.md` "U2's per-head arm"
  - The correlated against anti-correlated question at the token level needs a corpus for co-occurrence — `p1e_energy_field/design-1e.md` "Units"
  - Whether one well at the measured β is what the theory expects at this many tokens and dimensions is unread — `p1e_energy_field/status-1e.md` "U1, the field at the tokens"
  - Whether Phase 10's groups agreeing with the centred frame's wells is more than two proximity groupings agreeing is unread — `p1e_energy_field/status-1e.md` "Blocked 30 (b) and (c)"
- **After Phase 10:**
  - U4 from stored activations, once P-S1 is scored *(free)*; U3 closed at the measured β; U2 *(free for the block arm; forward pass per step and passage, GPU, for the attention and per-head arms)*
  - U5, after a corpus download and a co-occurrence count *(free)*
- **Reviewed:** 2026-10-09 · body `b21c9ed7ba`
<!-- /phase-card -->

The 2026-10-06 probe that prompted this phase ran under Phase 10's handoff
(`p10_cluster_function/handoff-10.md` Parked) on frames 1e does not use.

## The 8 long passages (2026-10-06)

**Why.** The user (2026-10-06): move towards only the larger prompts, since they carry more
information, and add passages now. 1d's long set had 4 passages, at step 0 and 143000 only; a
sign test over 4 passages cannot pass below p = 1/16. So 4 were added (the user chose the
sources) and all 8 were extracted at every step.

**Order (git shows it).** Freeze and U2's rule `9d746a2` → the new passages' rule `6de234e`
(`p1e_energy_field/long_prompts_1e.py` docstring, before any source was fetched) → the built texts
`3b6c64e` (tokenizer only) → the extractor `4f010a8` → the batch.

| key | tokens | source | units used |
|---|---|---|---|
| `wiki_paragraph_long` | 1840 | 1d (`p1d_cluster_ensemble/long_prompts/provenance.json`) | |
| `sullivan_ballou_long` | 1032 | 1d; the letter ends | |
| `hdbscan_code_long` | 2025 | 1d | |
| `latex_monograph_long` | 2036 | 1d | |
| `odyssey_butler_long` | 2008 | Gutenberg #1727, Book VII from its first paragraph; Butler's footnote markers dropped | 15 of 983 paragraphs |
| `horla_long` | 1960 | fr.wikisource rev 7351458 (validated; the 1887 text, Ollendorff 1895), from "8 mai." | 14 of 231 |
| `darwin_origin_long` | 1949 | Gutenberg #1228, ch. IV after its summary paragraph | 6 of 1160 |
| `hamlet_long` | 2041 | Gutenberg #1524, I.i from its first stage direction; lines kept | 270 of 6598 lines |

`sullivan_ballou_long` is half the others' length (the letter ends; 1d's rule 4). Sources,
sha256 and Gutenberg dates: `p1e_energy_field/long_prompts_1e/provenance.json`. One choice made
from the built text, before any model run: *Le Horla*'s Wikisource page leaves runs of spaces
where a page break falls inside a paragraph; they are collapsed to one space, as the page renders
them. None of the four is a v2 held-out source (`tests/test_p1e_long_prompts.py`).

**Extraction.** `p1e_energy_field/extract_long8.py`; `run.sh` beside the output
(`/run/media/system/HDD_1TB/mets_data/p1e_long8/2026-10-06/`): 8 × 18 = 144 runs, GPU (RTX 3080),
float32, eager, TF32 off, activations and norms only (no attention maps; ~190 MB and ~7 s per
run). Each manifest records the device and `long8_hash`. The first outputs were checked populated
before the batch (finite, rows unit to 3e-7, shape (25, n, 1024)).

**GPU against CPU** (*measured on all 8 pairs after `/challenge-pr` on #157, finding 4; it
corrects a one-pair reading that put the gap down to length*): this batch against 1d's stored CPU
long runs, 4 passages × steps 0 and 143000, max |Δ| of unit rows per quarter of the positions
(`p1e_energy_field/long8_targets.py`; output `targets_gpu_cpu.json` beside the runs). Tokens equal.

| step | L0–13 | L14–24 |
|---|---|---|
| 0 | 1.7–4.0e-7, every quarter | 1.8–3.6e-7, every quarter |
| 143000 | 0.8–2.2e-6 | **4.5e-5 to 1.8e-4**, already 4.5e-5–1.2e-4 in the first quarter |

The gap is the trained checkpoint's deep layers, not the late positions: positions in the first
quarter (the first 258–510) differ as much as the last. It is 6–25× the 512-token probe's 7.1e-6 at step
54000 (`docs/compute_profile.md` "The GPU"), from one pass shape against another, not from
position. So a long-passage cloud from this batch is **not** interchangeable with a CPU one at
the 1e-5 scale. 1e's primary readings use this batch only, which the GPU rule allows; v1 beside
is CPU, a continuity check (`design-1e.md` "Inputs").

**The batch (2026-10-06): 144 of 144 runs, every one checked populated** (shape (25, n, 1024) with
n the provenance count, finite, rows unit to 1e-4, norms > 0); all `cuda:0`, code `4f010a8`,
`long8_hash` `ba605f4e14b5` (pinned as `LONG8_HASH`; `load8` refuses a text that no longer hashes
to it); 6.1 s median per run, 24 GB on disk.

### Targets on the long passages (before any field was read)

Same producer and output. Massive tokens (`move_text.massive_positions`, union over the 18 steps)
are only position 0 or one more per passage, so U2's primary target set T1 + T2 keeps all but
1–2 positions. T1–T3 (R0's rule, beside) keeps:

| passage | n | T1–T3 | share kept per quarter |
|---|---|---|---|
| `wiki_paragraph_long` | 1840 | 698 (38 %) | 0.46, 0.36, 0.30, 0.40 |
| `sullivan_ballou_long` | 1032 | 417 (40 %) | 0.55, 0.39, 0.36, 0.31 |
| `hdbscan_code_long` | 2025 | 337 (17 %) | 0.42, 0.11, 0.07, 0.06 |
| `latex_monograph_long` | 2036 | 545 (27 %) | 0.41, 0.28, 0.21, 0.17 |
| `odyssey_butler_long` | 2008 | 677 (34 %) | 0.46, 0.30, 0.34, 0.24 |
| `horla_long` | 1960 | 683 (35 %) | 0.56, 0.34, 0.28, 0.21 |
| `darwin_origin_long` | 1949 | 680 (35 %) | 0.49, 0.29, 0.29, 0.32 |
| `hamlet_long` | 2041 | 653 (32 %) | 0.36, 0.27, 0.34, 0.31 |

(The reviewer's own count, by first occurrence alone, was 1–2 higher per passage: T2 drops the
massive tokens too.) Hence T1 + T2 primary for U2, T1–T3 beside (`design-1e.md` "U2's block arm:
the rule", row *targets*).

**Parked** (one line each: why / cost / the decision it could change):
- Why GPU and CPU differ by ~1e-4 at 143000's deep layers and only 2e-7 at step 0 / free from the
  stored runs, a GPU pass at 512 tokens of the same prefix would split pass shape from device /
  whether a cross-device comparison is ever allowed for long passages.

**How to re-run.** Texts: `python -m p1e_energy_field.long_prompts_1e` (reads the cached sources,
fetches them if absent; a changed source changes the hash). Runs: the `run.sh` above
(`--skip-existing` resumes).

## U2's block arm (2026-10-07): the frozen labels are mostly an update every token shares; the token-specific part descends the field at 32–2000 and ascends in L9–16 from 4000, L1–8 at 64–128 and from 16000

**Read "The shared update" below first.** The frozen rule's labels (this table) stand as its
output, but its shuffle null cannot see an update added to every token, and that update carries
the 8–256 ascent and the 143000 descent. Which reading counts is the user's (`STATE.md` Blocked 29).

**Input.** The 144 long runs above (`LONG8_HASH` `ba605f4e14b5`, GPU, code `4f010a8`) and v1's 7
passages at the 18 steps beside (Stage 0 CPU runs, R0's kept offsets from
`data/p10/reread_r0_2026-10-05/labels`); LN1 weights from each checkpoint's cached
`model.safetensors` (snapshot ids in `plan.json`). Rule: `design-1e.md` "U2's block arm: the rule",
unchanged. Producer `p1e_energy_field/u2_block.py` (`run`, `nll`), labels `u2_report.py`; output
`data/p1e/u2_block_2026-10-07/` (`report.txt`, `report.json`, `records/`, `nll.json`, `run.sh`).
First cell (step143000, `wiki_paragraph_long`) checked populated before the rest (552 of 552
cells, every block, 1,839 of 1,839 targets finite). All 270 runs done, no refusal.

**Labels, primary** (T1 + T2, causal field, β = 3.5; `X` = A − shuffle-null mean, band mean over
the 8 passages; 54 cells in L1–22):

| steps | L1–8 | L9–16 | L17–22 |
|---|---|---|---|
| 0–4 | mixed | **descends** at 0, 2 (−0.003, init; leans at 4) | mixed (leans ascends at 4) |
| 8–256 | ascends, +0.007 to +0.048 | ascends, +0.018 to +0.065 (leans at 256) | ascends, +0.018 to +0.158 |
| 512 | leans descends, −0.016 | leans descends (isolated) | ascends |
| 1000–8000 | **descends**, −0.008 to −0.018, β-robust | mixed at 1000–2000, ascends 4000–8000 | ascends, +0.027 to +0.031 |
| 16000–54000 | ascends | ascends | ascends |
| 143000 | ascends, +0.023 | ascends, +0.021 | **descends**, −0.027 (leans at β 1.6) |

| count over the 54 cells | ascends | descends | leans asc / des | mixed |
|---|---|---|---|---|
| observed | 35 | 7 | 2 / 3 | 7 |
| expected by chance | 0.42 (either) | | 3.4 (either) | |

- **Size.** `X` ≤ 0.16 everywhere: the block's move is mostly orthogonal to the field's force.
  "Ascends" is the sign of a small component shared by all 8 passages, not a mean-shift step.
- **143000, L17–22, by block:** blocks 17–18 ascend in 8 of 8 passages, 19 in 6, 20 in 3, and
  **blocks 21 and 22 descend in 8 of 8** (`X` −0.16, −0.12). At 54000 only block 22 does (2 of 8).
- **Mean move out (`Xt`)**: the same label in 71 of 72 primary-field cells, but this is *not*
  robustness: `Xt` removes the mean move after projection, which leaves a shared update in
  (`/challenge-pr` on #158, finding 1; the test `test_an_update_shared_by_every_token…`).
- **Sources.** Without the sink: the same labels, but the row cannot fail: in `φ_β` the sink is
  one source among hundreds, and says nothing of the model's own attention to it (finding 2).
  Full sum: nearly the same. **Local part** (`m_β − m_0`): descends at 8–16 in every band,
  ascends in L1–16 from 32; in L1–8 it is mixed from 2000.
- **β.** 1.6 and 5.6 give the same label in most cells (counts 33–36 ascends, 7 descends), and
  the **β = 0 field** (`mean0`, the plain causal mean, below) gives 30 ascends, 7 descends, the
  same windows: the label is the field's mean term, not the measured β (finding 4).
- **Step 0.** Only L9–16's faint descent at init (steps 0–4, step 2 sharing it); no label from
  step 8 on is also step 0's.
- **T1–T3 against T1 + T2:** 14 of 72 primary-field cells differ (6 at L0), every one by a
  degree (full, lean, mixed); none reverses a direction. Over all fields 149 of 864, 76 at L0.
- **v1 (beside, CPU, 7 passages):** the same windows (ascends 8–256 and 16000–54000, L1–8
  descends 1000–8000, L9–16 descends at init); at 143000 L17–22 is mixed (+0.001), not descends.
- **Per-cell ranks** carry little: the shuffle null's sd is ~1e-3 at 1,000–2,000 targets, so
  most cells sit at rank 0.001 one way or the other; the label is the cross-passage sign.
- **Position:** median Spearman of `a_i` with log offset within ±0.1 in L1–22 at the six steps
  read (0, 16, 128, 1000, 8000, 143000).
- **Memorisation (beside):** mean NLL at 143000 is 1.31 (`hdbscan_code_long`) to 3.26 nats
  (≈ ln 50304 = 10.8 at step 0); dropping the lowest leaves 143000's labels unchanged.
- **Not tested, a coincidence of windows:** L1–8's descent (512–8000) overlaps the OV spectrum's
  repulsive phase (1000–2000, `PROJECT.md` §3.12 A) and follows the energy break (128–512).
  The frame itself moves over training (LN1's gain and bias), so a change across steps can be it.

### The shared update (after `/challenge-pr` on #158, finding 1; beside, added after the output)

**Why.** If a block adds the same vector to every token's residual (the sink's value, an MLP
bias), each token's tangent projection of it lines up with the field's mean term, so the frozen
`X` reads it as ascending or descending, hundreds of null sds out; `Xt` does not remove it.
Readings (`u2_block.shared_cells`; the shared part estimated over the targets; causal field at
β 3.5 and the β = 0 field `mean0`):

| reading | what | at steps 0–8 (L1–22) |
|---|---|---|
| **`r1out`** (*after `/challenge-pr` on #159, finding 1*) | each token's residual update less its own component along the shared direction `ĉ = mean_t(x' − x)/|·|`, through LN1 | **mixed in every cell** (values ≤ 0.000) |
| `residout` | the residual update less its mean over the targets | ≈ 0.000, but L9–16 leans descends at 0–2 |
| `resid` | that mean update alone | L9–16 descends at 0–2 |
| `ambout` | #158's review's check (unit-frame move less its mean) | **ascends in every band**: not specific |

`r1out` is the token-specific reading: it removes a shared direction at any per-token weight (the
sink's value scaled by each token's attention to it), and unlike `residout` it cannot reverse a
small mover's move when step sizes are uneven (both tested, `tests/test_p1e_u2_block.py`; on our
synthetic `residout` read below `r1out` but did not flip, as it did on the review's).

| steps (long, T1 + T2) | `r1out` L1–8 | L9–16 | L17–22 | `resid` (shared alone) |
|---|---|---|---|---|
| 0–8 | mixed | mixed | mixed | ascends from 8, every band |
| 16 | mixed | mixed | descends, −0.002 | ascends |
| 32 | **descends**, −0.009 | **descends**, −0.029 | **descends**, −0.043 | ascends |
| 64–128 | ascends, +0.005 / +0.013 | descends at 64, leans asc at 128 (isolated) | **descends**, −0.049 to −0.060 | ascends |
| 256 | mixed | mixed | mixed | ascends |
| 512–1000 | **descends**, −0.026 / −0.024 | **descends**, −0.039 / −0.008 | leans / descends | L1–8 mixed, else ascends |
| 2000–8000 | descends to 4000, mixed 8000 | mixed, ascends from 4000 | mixed | **L1–8 descends**, else ascends |
| 16000–54000 | **ascends**, +0.009 to +0.029 | **ascends**, +0.013 to +0.028 | descends at 32000, leans at 16000 / 54000 | L1–8 leans, L9–16 ascends (mixed at 54000), L17–22 ascends |
| 143000 | ascends, +0.030 | ascends, +0.036 | mixed, −0.002 | L1–16 descend, L17–22 leans (isolated) |

| count over 54 cells | ascends | descends | leans asc / des | mixed |
|---|---|---|---|---|
| **`r1out`**, causal β 3.5 | 12 | 15 | 1 / 3 | 23 |
| `r1out`, β = 0 | 9 | 28 | 1 / 0 | 16 |
| `residout`, causal β 3.5 | 13 | 12 | 2 / 6 | 21 |
| `resid` (shared alone) | 31 | 7 | 3 / 3 | 10 |
| `ambout` (#158's review) | 51 | 0 | 3 / 0 | 0 |

- **The 8–256 ascent is mostly the shared update**: `resid` ascends there; `r1out` is mixed at
  8–16 and descends at 32 in every band (L1–8 alone ascends at 64–128).
- **143000's L17–22 descent is not token-specific** (`r1out` mixed, −0.002), but its shared
  reading is only a lean, isolated (`resid`), so "shared" is the weaker half of that claim.
- **L1–8's descent at 1000–8000 is both**: token-specific at 512–4000 (`r1out`), shared at
  2000–8000 (`resid`).
- **New with `r1out`:** L17–22's token-specific part descends late (32000; leans at 16000 and
  54000), which the frozen label hid under an ascending shared update.
- **`ambout` is not specific** (ascends at step 0 everywhere), so #158 review's two sign flips are
  not evidence.
- **The field's local structure matters for the token-specific part**: at β = 0 `r1out`
  descends in 28 cells, at β 3.5 in 15.
- **What removing the shared direction also removes:** a genuine pull of every token towards
  the cloud's mean is largely along one direction too, so `r1out` can remove real
  field-following; `resid` ascending is what the theory's mean-field attraction would also do.
  The attention arm splits the shared update into attention's and the MLP's, and attention's
  by key (position 0 against the rest; #159 review, finding 2).
- **v1 (beside), `r1out`:** the same windows: mixed at 0–8, descends in L9–22 at 32–2000
  (L9–16 mixed at 128), L1–8 at 512–8000; ascends in L1–16 from 8000 (L9–16) / 16000 (L1–8).

**Provenance.** Records name their producer as the git tree hash of `p1e_energy_field`
(`code` `tree:a03e3cd764`, survives rebases; #159 review, finding 3), with `device` and `mode`;
`run` refuses to resume over another producer's record. The shared rows ran on the GPU (CUDA
float64, 270 runs, ~25 min); the first cell matched the earlier CPU shared records (d4a1ad9,
kept as `records_shared_d4a1ad9/`) to 1e-16 on every overlapping row. The frozen `records/`
predate provenance: written by #158's `u2_block.py` (7869cbd) less one comment line, in one
CPU run on 2026-10-07 06:10–08:02.

### GPU reading (2026-10-07)

`p1e_energy_field/u2_torch.py` mirrors the numpy reader on a torch device; `u2_block run
--device cuda` uses it, and `u2_block agree` recomputes stored records and compares. On 6 runs
(long 143000 / 0 / 2000, v1 512, two shared), **CUDA float64 reproduces the CPU records: `X`,
`A` and the null mean to ≤ 7e-16, `Xt` to ≤ 9e-13, ranks and signs identical**, at ~20 s per
long run (shared mode 8–10 s): about 2.5× the 14-worker CPU pool, CPU left free. **Float32
fails** (`|ΔX|` up to 4e-4, 6 ranks moved, against L0's `X` ≈ 1e-3), so `cuda32` is not used.
Output `agree_cuda.json`, `agree_cuda32.json` beside the records; tests
`tests/test_p1e_u2_torch.py` (torch on the CPU in float64 equals numpy to 1e-10).

**How to re-run.** From the worktree root: `data/p1e/u2_block_2026-10-07/run.sh` (resumes;
`--first-only` for the first check), then `python -m p1e_energy_field.u2_block nll --runs <long8>
--out <dir>` and `... report --out <dir>`. CPU, float64, 14 workers: 143 s per long run alone,
~700 s each with 14 in parallel (memory bandwidth), 1 h 50 min for the 269 after the first.

**Parked** (why / cost / the decision it could change):
- ~~The last two blocks at 143000~~ *answered by the attention arm (below)*: neither the sink's
  value (share −0.01) nor the biases (0.04); attention's move and the MLP's shared part both
  descend there.
- L9–16's descent at init / free / whether init carries a structural sign; it is shared
  (`residout` ≈ 0 at init), so the frozen rows carry it and `residout` does not.

## U2's attention arm (2026-10-07): the MLP supplies most of the shared update, and its shared part ascends there too; on the long passages attention's token-specific part never ascends, and descends from step 256

**Input.** The 144 long runs (`LONG8_HASH` `ba605f4e14b5`), each re-run as a hooked forward pass
on the GPU (float32, eager, TF32 off), and v1's 7 at the 18 steps beside (R0's kept offsets).
Rule: `design-1e.md` "U2's attention arm: the rule" (`2dee9b9`, before any output). Producer
`p1e_energy_field/u2_attn.py`, report `u2_attn_report.py`; records name the producer as
`p1e_energy_field`'s tree `tree:2b38ef2cb2` (commit `4d35c6a`); output
`data/p1e/u2_attn_2026-10-07/` (`report.txt`, `report.json`, `records/`, `run.sh`).
**Checks, all 270 runs:** the pass against the stored unit rows ≤ 1.2e-7 (long, same device) and
≤ 7.9e-5 (v1, stored on the CPU; bound 1e-4); `x + attn + mlp = x'` exactly (0); the parts sum to
the block to ≤ 7.5e-8; the first cell's `block` row equals the block arm's record to 2.8e-10, and
the `block` row's labels reproduce the block arm's counts exactly (35 / 7 / 2 + 3 / 7). Test:
the hooked split equals the explicit sum over keys on a tiny random GPT-NeoX
(`tests/test_p1e_u2_attn.py`, smoke tier: `SMOKE_REAL_DEPS=1 pytest -m smoke`; passes on
transformers 4.44 and 4.57, `LESSONS.md` 4). Two earlier launches died on GPU memory (`LESSONS.md` 3); no
record from them was kept.

**Blocked 29, by the rule's reading** (fixed before output: in each (step, band) where the
block's `resid` ascends, the part with the largest share of the shared update whose own `resid`
ascends too):

*Reworded after `/challenge-pr` on #160, finding 1:* the rule picks the part supplying the
largest share of the shared update's **length** whose own `resid` ascends; it does not measure
who supplies the **ascent** (a cosine, blind to size). At 16000–32000 in L9–16 the MLP supplies
0.57–0.67 of the update and its own shared part descends, yet the block's still ascends. So
the table reads "the MLP supplies most of the shared update, and that part ascends too".

| carrier | windows (long, 31) | where | v1 (29) |
|---|---|---|---|
| `mlpx` (the MLP less its bias, token by token) → (b) not interaction | **24** | every band at 8–512, L9–16 at 1000, L17–22 at 16000–54000 | 24 |
| `keys` (attention to the other tokens) → (a) follows the field | 4 | L17–22 at 2000; L9–16 at 8000, 16000, 32000 | 2 |
| tied (shares within 0.1) | 3 | L17–22 at 1000, 4000, 8000 | 3 |

*After finding 2:* v1's tally matches in counts only. Its `keys` windows (L17–22 at 4000, L1–8
at 32000) share none with the long passages', and in 4 of the 6 the MLP supplies more of the
update than attention; attention is named there because the MLP's shared part does not ascend.
The attention-carried windows are not a replicated finding.

- **Shares of the block's shared update** (median over passages): the MLP 0.54–0.79 at every
  step but 1000–8000, where the two are close (attention 0.41–0.58, MLP 0.40–0.52); **the sink ≤ 0.05 and the biases
  ≤ 0.17 everywhere**, although attention to key 0 grows from ~0.004 (≤ 1000) to 0.09–0.48
  (16000) and 0.17–0.62 (143000): the sink's value adds almost nothing to the shared update.
- **Attention's shared part ascends too**, at a smaller share: `keys:resid` ascends in 37 of 54
  cells (the same windows as the block's `resid`, descends at 256–512 in L1–16). Up to step 512
  attention's output is nearly one vector for every token (sharedness 0.82–0.99): the mean pull.
- **The MLP's shared part** (`mlpx:resid`) ascends at 8–256 in every band (L9–22 to 1000) and
  descends in L1–16 from 2000 (as step 0 does, so not called learned there).

**Token-specific parts** (`r1out`, causal, β 3.5; long, T1 + T2):

| steps | attention (`attn`) L1–8 / L9–16 / L17–22 | MLP (`mlpx`) L1–8 / L9–16 / L17–22 |
|---|---|---|
| 0–8 | mixed / mixed / mixed | leans asc / leans des / mixed (step 0 the same) |
| 16–128 | mixed / mixed / leans descends at 64–128 | descends at 32 (all bands) and L9–16 at 64; L1–8 ascends 64–128; L17–22 descends 16–128 |
| 256–1000 | **descends** / **descends** / leans descends at 512–1000 | L1–8 descends 512–1000; L9–16 at 512; L17–22 at 1000 |
| 2000–8000 | **descends** / leans des at 2000, then mixed / leans des at 2000, then mixed | L1–8 descends at 2000, leans ascends after; L9–16 ascends from 4000; L17–22 mixed |
| 16000–143000 | **descends** / **descends** (lean at 143000) / **descends** (−0.013 to −0.031) | **ascends** in L1–16 (+0.016 to +0.049) / mixed |

| count over 54 cells | ascends | descends | leans asc / des | mixed |
|---|---|---|---|---|
| `attn:r1out` | **0** | 20 | **0** / 8 | 26 |
| `keys:r1out` | 0 | 20 | 0 / 10 | 24 |
| `mlpx:r1out` | 12 | 12 | 9 / 3 | 18 |
| `attn:frozen` (the frozen reading, shared part in) | 36 | 7 | 4 / 0 | 7 |
| chance | 0.42 (either) | | 3.4 (either) | |

- **On the long passages, attention's token-specific part never ascends the field in L1–22**
  (0 of 54 cells, no lean; mixed at 0–128; v1 differs at 8–32, below) and descends from 256 on: L1–8 at every step from 256, L9–16 at 256–2000
  and 16000–143000, L17–22 from 16000. It is attention to keys 1…i (`keys:r1out` the same); the
  sink's own move is small and mixed. At β = 0 the same (`mean0:keys:r1out` 24 descend, 0 ascend).
- **The block's late token-specific ascent in L1–16 (from 16000) goes with the MLP's**, not
  attention's. *After finding 4:* each part's `r1out` removes that part's own shared direction,
  not the block's, so the parts' readings do not add up to the block's; the signs are clear here,
  so the attribution stands as a reading, not a decomposition.
- **The last band at 143000** descends in both attention's move (`attn:frozen` −0.034) and the
  MLP's shared part (`mlpx:resid` −0.045); sink and biases carry none of it.
- **v1 (beside, CPU-stored, 7 passages):** the same Blocked 29 tally; `attn:r1out` descends in
  L1–16 at 256–1000, L17–22 at 512–1000 (lean at 512) and L1–8 to 143000, but leans or ascends at 8–32 in L1–16 (4
  ascend, 6 lean ascend of 54), which the long passages do not show.
- **Size.** `|X|` ≤ 0.06 for the token-specific rows: attention's token-specific move is mostly
  orthogonal to the field's force; "descends" is the sign of a small component shared by all 8
  passages. The field is the idealised `φ_β` (one head, `Q = K = V = I`); whether the real
  kernel says the same is the per-head arm.

**How to re-run.** From the worktree root: `data/p1e/u2_attn_2026-10-07/run.sh` (resumes;
`--first-only` for the first check; needs the block arm's records for that check). *After
finding 5:* the producer hash is `p1e_energy_field`'s whole tree, docs included, so resuming
these records needs a checkout of `4d35c6a`'s tree (later commits changed only docs and the
report); kept as is, since changing the hash would orphan the block arm's records too. Then
`python -m p1e_energy_field.u2_attn report --out <dir>` (reads the block arm's `report.json`
beside it for Blocked 29's table). GPU: ~55–60 s per long run, 3.1 GB; 270 runs ≈ 3 h.

**Parked** (why / cost / the decision it could change):
- Who supplies the shared **ascent**, not the update's length (finding 1) / the next GPU pass
  (the per-head arm) saves each part's mean update per block, then CPU only / Blocked 29: whether
  the shared ascent is the MLP's or attention's mean pull.
- v1's early attention ascent (8–32, L1–16) against the long passages' mixed / one GPU pass on
  the v1 texts' long-passage prefixes is free / whether early attention's token-specific sign
  depends on length or on the CPU-stored frame.
- Why the sink's value carries ≤ 0.05 of the shared update while heads give it up to 0.6 of
  their attention / free (its value norm, already in the pass) / whether "the sink" should
  stay a separate source in U2's field (design "the sink").

## U2's per-head arm (2026-10-07): against its heads' own kernels, attention's token-specific move descends at 128–2000, head by head; from 4000–8000 the sum over heads ascends while more heads still descend their own than ascend it

**Input.** The 144 long runs (`LONG8_HASH` `ba605f4e14b5`) and v1's 7 at the 18 steps (R0's kept
offsets), each re-run as a hooked forward pass on the GPU (float32, eager, TF32 off), every
layer's full attention map read in its own hook (CUDA float64). Rule: `design-1e.md` "U2's
per-head arm: the rule" (`8d6da34`, before any output; its populated check changed after the
first check refused, before any reading: heads that put all their weight on key 0 and the token
itself have no keys part or `V = I` field there). Producer `p1e_energy_field/u2_heads.py`, report
`u2_heads_report.py`; records name `p1e_energy_field`'s tree `tree:23c3822134` (commit `3d571fa`);
output `data/p1e/u2_heads_2026-10-07/` (`report.txt`, `report.json`, `records/`, `means/`,
`ascent/`, `run.sh`). **Checks, all 270 runs:** the pass against the stored rows ≤ 1.2e-7 (long),
within 1e-4 (v1); every attention row sums to 1 within 1e-6 with no weight above the diagonal;
the heads add up to attention's output within 4.5e-7 of the update's norm; the saved means' parts
sum to the block within 1e-6; the check row (`causal:keys:r1out`) equals the attention arm's
band values to 2.5e-16. Tests: `tests/test_p1e_u2_heads.py` (per-head parts and kernel fields
against explicit sums on a tiny random GPT-NeoX), `tests/test_p1e_u2_heads_pure.py` (Shapley).
~52 s per long run, 2.2 h for the 270.

**The reading** (fixed before output). In each trained window where the attention arm's
`keys:r1out` descends `φ_β` (30 of 54 on the long passages): attention's token-specific move
(`keys:r1out`) against `kernns`, the field its own heads' attention rows give with `V = I`:

| verdict | windows (long, 30) | where | v1 (24) |
|---|---|---|---|
| descends its own kernel: **attention repels what its heads attend to** | **11** | L17–22 at 64–128; L1–8 and L9–16 at 256–1000; L17–22 at 512–1000; L9–16 at 2000 | 11 |
| ascends its own kernel, **summed over heads only** (in 10 of the 13 more heads descend their own kernel than ascend it; *after `/challenge-pr` on #161, finding 1*) | **13** | L1–8 at 4000–143000; L9–16 at 16000–143000; L17–22 at 8000, 16000, 54000 | 11 |
| mixed: the real kernel does not sign it | 6 | L1–8 at 4 and 2000; L9–16 at 4000; L17–22 at 2000, 32000, 143000 | 2 |

| `kernns:keys:r1out` (long) | L1–8 | L9–16 | L17–22 |
|---|---|---|---|
| 0–64 | mixed | mixed | mixed (leans descends at 64) |
| 128–1000 | **descends** (lean at 128), −0.002 to −0.017 | **descends**, −0.008 to −0.035 | **descends** (lean at 512), −0.007 to −0.019 |
| 2000 | mixed | descends, −0.016 | mixed |
| 4000 | leans ascends | mixed | leans ascends |
| 8000–143000 | **ascends**, +0.003 to +0.009 | **ascends**, +0.020 to +0.029 | ascends at 8000; leans at 16000, 54000; mixed at 32000, 143000 |

Over 54 cells: 11 ascend, 11 descend, 4 / 3 lean, 25 mixed (chance 0.42 and 3.4); no label is
carried by step 0. `kern:attn:r1out` (the whole of attention, key 0 in as a source) differs in
verdict at 4 windows, all at the edges (4 L1–8 and 32000 L17–22 lean descends; 16000 and 54000
L17–22 mixed). **v1** (CPU-stored, 7 passages): the same shape, descends at 128–2000, ascends from 8000
in every band.

- **What the comparison isolates** (*after finding 4*). The move is the same in both readings, so
  where the sign differs between `φ_β` and `kernns` it is the field (the heads' `QK`) that differs.
  Where the move descends `kernns`, attention's output (its `OV` maps) moves tokens away from the
  `V = I` mean of what its heads attend to: in effect a test of `OV`'s sign on the attended mean,
  not only of which field is right.
- **Two regimes, split at 2000–4000.** Up to 2000, attention's token-specific move goes against
  both `φ_β` and its own heads' kernels, in the sum and head by head: the repulsive reading holds
  under the real kernel. From 4000–8000 the **sum** goes *with* the summed kernels and still
  against `φ_β`, **but more heads still descend their own than ascend it** (in 10 of the 13
  windows; *after `/challenge-pr` on #165, finding 2:* a majority of all heads descends in only 2
  of the 13, 5 counting leans, so "most heads" was wrong; in L1–8 at every step from 128 to 143000; only L9–16 at 32000, 54000, 143000
  lean the other way). So the late ascent is a property of the sum: either heads moving tokens
  towards where *other* heads attend (cross terms), or the shared pull leaking back, since `r1out`
  per head does not add up to `r1out` of the sum (#160's finding 4). This unit does not split them
  (Parked). Attention to key 0 rises over
  the same steps (head median in L9–22 ≤ 0.01 to 2000, 0.27–0.29 at 8000, 0.50–0.58 at 143000;
  L1–8 stays ≈ 0), but `kernns` has key 0 out, so what changes is which other keys the heads read.
- **Per head** (head h's keys part against its own field; a unit is one (block, head), labelled
  over passages). Step 0 is the baseline, since passages are not independent given a head: at
  step 0, 1–5 units per band are signed (chance 0.8–1.0) and 6–17 lean (chance 6–8). At 256–1000
  **most heads descend individually**: 49 (L1–8) and 86 (L9–16) of 128 at 256, 81–122 of 128 in
  L1–16 at 512–1000 (122 at 1000, L9–16), 67–79 of 96 in L17–22 at 512–1000. From 8000, while the
  sum ascends, more heads descend than ascend in 10 of the 13 windows (4000 L1–8 18 / 67;
  143000 L1–8 29 / 47; the exceptions L9–16 at 32000 53 / 49, 54000 56 / 41, 143000 62 / 38). The
  number of ascending heads rises from 2000 (L1–8: 0–4 at 256–1000, 15–29 from 2000).
- **The real field points nearly where `φ_β` points.** Median per-head `cos(g^h, g)` 0.69–0.96 in
  the windows (1.00 at 0–16, where heads attend near uniformly). The sign differs because the move
  is nearly orthogonal to both: `|X|` ≤ 0.035 on the long passages (≤ 0.053 on v1), and a small
  turn of the field flips the sign of a small projection. "Repels" and "wrong field" are both
  statements about that small component.
- **Sink-only heads.** At 143000 in L17–22, 10 of 96 units are short of 90 % of targets in some
  passage (all weight on key 0 and the token itself) and are kept out of the counts; L18 head 3
  is sink-only at 1,817 of 1,839 targets in `wiki_paragraph_long`. *After finding 2:* the report
  now checks the rule's explanation on every record, not only the first; 45 short cells are
  explained, 2 are not: v1's `latex_monograph` and `hdbscan_code` at 143000, L18 head 3, which puts
  a median 0.97–0.99 of its weight on a **newline** (position 10 / 34; a second sink), so its keys
  part is one vector for every token and `r1out` removes all of it (n = 0). Kept out of the counts
  with the rest; the rule's explanation did not foresee a sink other than key 0.
- **Whole heads, sink in** (`kern_h:head_h:frozen`, beside): 25–34 units ascend and 24–30 descend
  per band at step 0 already (the frozen reading carries each head's shared part); read against
  step 0, never chance.

**Blocked 29 by ascent** (fixed before output: where the block's `resid` ascends, the part with
the largest Shapley share of the shared move's projection on the force):

| carrier of the ascent | windows (long, 31) | where | v1 (29) |
|---|---|---|---|
| `mlpx` (the MLP less its bias) → (b) not interaction | **19** | every band at 16–64 and 256; L9–16 at 8; L9–22 at 128 and 512–1000 (at 256 in L1–16 and 512 in L9–16 attention's shared part *descends*: keys −0.62 to −0.37) | 21 |
| `keys` (attention to the other tokens) → (a) follows the field | **7** | L1–8 at 8; L17–22 at 2000–8000; L9–16 at 8000–32000 (the MLP's share negative there: −0.85 to −2.76) | 5 |
| tied (within 0.1) | 4 | L17–22 at 8, 16000, 32000, 54000 | 2 |
| not read: `F > 0` in half the passages or fewer (*after finding 3*) | 1 | L1–8 at 128 (2 of 8; v1 1 of 7) | 1 |

- **Replicated on v1:** `keys` at 8 L1–8 and L17–22 at 2000, 4000, 8000 (4 windows). v1 adds
  32000 L1–8; L9–16 at 8000–32000 are not v1 windows, so not replicated.
- **Against the attention arm's length reading** (24 `mlpx`, 4 `keys`, 3 tied): the same 4 `keys`
  windows plus 3 more. The verdict moves from "mostly the MLP" only after 2000: up to 1000 the
  MLP supplies the shared ascent (and at 256–512 attention's shared part works against it); from
  2000 in L17–22, and 8000–32000 in L9–16, attention's mean pull through the other tokens does.
- The sink's value supplies −0.13 to +0.07 (−0.19 on v1 at 32000 L1–8); the biases ≤ 0.20,
  except where the MLP's share is negative (0.46–0.85 at 8000–32000 L9–16; 1.22 on v1 at 32000
  L1–8).
- Shares are medians over passages of band-summed ratios, so they need not sum to 1; a share > 1
  means another part's contribution is negative. A raw projection with no null: beside the
  labels, not a label.

**How to re-run.** From the worktree root: `data/p1e/u2_heads_2026-10-07/run.sh` (resumes;
`--first-only` for the first check, which needs the attention arm's records), then
`OMP_NUM_THREADS=1 python -m p1e_energy_field.u2_heads ascent --out <dir> --block-out <block arm
dir> --workers 14` (CPU; multi-threaded BLAS in 12 workers was several times slower), then
`python -m p1e_energy_field.u2_heads report --out <dir>` (reads the attention and block arms'
`report.json` beside it). Resuming the records needs `3d571fa`'s `p1e_energy_field` tree.

**Blocked 29 decided (c) (user, 2026-10-08): U2's headline is split by training stage.** Both
readings change at 2000–4000, so neither (a) the frozen labels nor (b) `r1out` alone is the
headline. Up to step 1000 the shared ascent is the MLP's (an update every token gets, mostly not
interaction), and the token interactions descend their heads' own kernels (128–2000). From 2000
the shared ascent is attention's mean pull (L17–22, and L9–16 at 8000–32000), and from 4000 only
the sum over heads follows its kernels while more heads still descend their own than ascend it. Numbers: the
"Blocked 29 by ascent" table and the per-head reading above; the frozen labels and `r1out` stay
beside. This is the reading the 1e page opens with (`design-1e.md` "The 1e page").

**Parked** (why / cost / the decision it could change):
- The late split between heads and their sum (at 143000 L1–8, 47 heads descend their own field
  and 29 ascend, of 128; the sum ascends) / one GPU pass at a few steps (per-token cross terms; the saved means are not
  enough) / whether the late "follows its own kernel" is heads cooperating (one head moving
  tokens to where another attends) or a few large heads.
- Why the regime changes at 2000–4000, and why ascending heads rise after 2000, next to the
  weights-only OV spectrum's course (`PROJECT.md` §3.12 A) / free, from the records and §3.12 A /
  whether 1e's attention result and the OV turn are one event (`/challenge-pr` on #161, finding 1).
- A second sink (v1's L18 head 3 on a newline at 143000) / free: one pass per run, argmax key per
  head / whether "sink-only" should mean any single fixed key, not key 0.

## U1, the field at the tokens (2026-10-08): at the measured β the field has one well; its density is lumpier than a matched Gaussian everywhere, step 0 included, and is mostly the tokens' distance to the centroid

**Input.** The 144 long runs (`LONG8_HASH` `ba605f4e14b5`; T1 + T2 primary, T1–T3 beside) and
v1's 7 at the 18 steps (R0's kept offsets; c3x groups from R8x, Blocked 27 (a)), hidden states
0–23 in each layer's LN1 frame, unit rows; no forward pass. Rule: `design-1e.md` "U1: the rule"
(`f5709b7`, before any output; amended `9732341` after the first check, before any reading: the
Gaussian's density at all three β). Producer `p1e_energy_field/u1_field.py`, report
`u1_report.py`; records name `p1e_energy_field`'s tree at `9732341`; output
`data/p1e/u1_field_2026-10-08/` (`report.txt`, `labels.json`, `records/` with per-token `e_i`
and wells as `.npz`, `run.sh`, `first_check.json`). **Checks:** stored rows unit, manifests
carry the hash and step, targets and R8x's c3x rows index R0's labels (all 270 runs); first
cell populated; its GPU float32 wells equal a CPU float64 run's at L4 / L12 / L20 (agreement
1.0000), a check on one-well cells only (`/challenge-pr` on #164, finding 4), so repeated where
there are several (`calib/wells_check.json`: 9, 312 and 4 wells at step 128 L4 β 5.6 / 10 and
143000 L12 β 10, agreement 1.0 each); 0 unconverged trajectories in all 28,566 cells. Tests `tests/test_p1e_u1_field.py`
(planted vMF groups → their wells; LOO and causal density against loops; the Gaussian draw's
moments; AMI and purity at chance; the report's sign rule). ~35 s per long run, 85 min for the
270 (GPU float32 mean shift, CPU float64 finish).

**Wells: one.** At β 3.5, median over the 8 passages, the field over the tokens has **one well
holding every target in every trained band at every step** (`k = 1`, `k_eff = 1.00`), on the long
passages and on v1, except L1–8 at step 128 (2 wells, `k_eff` 1.18, largest 0.96). β 1.6: one
everywhere, L0 included. β 5.6: one, except L1–8 / L9–16 at 128 (`k_eff` 3.4 / 2.5). Only L0 (the
embedding) has a few wells at 3.5 on the long passages, 0–4000 (`k` 4, `k_eff` 1.8), one by
143000. The matched Gaussians have one well too, so label (2) reads `Xw = 0` (the rule's "mixed")
in 52 of 54 trained cells (L1–8: more wells at 128, leans at 64); label (4) is **not read**
except L1–8 at 64–128 ("content"; *weak, `/challenge-pr` on #164 finding 3:* its side wells hold
2–4 % of the tokens, and a passage enters a band on any one layer with two wells), and the c3x
purity on v1 is not read at all. The placed merge tolerance (1e-3) changes no count: at 1e-4 and
1e-2 every cell's well count is the same (0 of 10,368 long and 9,072 v1 cells; the report's
beside medians missed these counts until the key names were fixed after #164 opened). **The
sweep** (L4 / L12 / L20 × 18 steps × 8 passages = 432 cells per β; no β between 5.6 and 10 or 10
and 20): one well up to β 5.6 at every step but 128. **At β 10 a few wells is common:** 196 of
432 cells have 2–20 (111 one, 125 more than 20; e.g. L12 2–6 at step 0, L20 2–26 at 512–1000,
L12 1–14 at 143000), median `k_eff` 1–8 (79 at L4, step 128). At 20, 382 of 432 have more than 20
(L20 at 143000 still `k_eff` 1.3); at 50–100, near one per token. *Corrected after `/challenge-pr`
on #164, finding 1: the first text said "no range of a few wells".* So a few wells exist at
β ≈ 10, 1.8× the measured interval's top (2.9× its 3.5), not at the measured β. **The probe's
"2–4 wells" were the cloud-centred frame's** (`p10_cluster_function/handoff-10.md` Parked), not
β's.

**Density: lumpier than its Gaussian, and mostly the centroid.** Label (1), `sd(e) − sd(e_G)`:

| β | trained cells lumpier / leans / mixed / smoother (of 54) | step 0 |
|---|---|---|
| 1.6 | 43 / 4 / 5 / 2 | lumpier L1–8, leans L9–16, mixed L17–23 |
| 3.5 | 51 / 0 / 1 / 2 | lumpier in every band |
| 5.6 | 50 / 2 / 0 / 2 | lumpier in every band |

β-robust; T1–T3 the same (53 of 54 at 3.5); v1 41 of 54 (smoother in 4). **Step 0 carries it, so
it is not called learned** (the rule). Smoother only at step 16, L9–16 and L17–23.

**(1′), calibrated** (*`/challenge-pr` on #164, finding 2; rule `a809dc1` before it was computed;
`p1e_energy_field/u1_calib.py`, output `calib/` beside the records, `calib/report.txt`*). Putting
the matched Gaussian on the sphere biases `Xe` up once training concentrates the covariance: a
structureless cloud (a null draw scored against its own Gaussians) reads `bias` ≈ 0 at steps
0–16 and up to +0.04 / +0.03 / +0.10 (L1–8 / L9–16 / L17–23, β 3.5) at 143000, about half the raw
`Xe` late in training. Against it, `Xe′ = Xe − bias` still reads **lumpier** in 42 + 5 leans /
49 / 50 of 54 trained cells at β 1.6 / 3.5 / 5.6 (v1: 54 / 54 / 53 + 1), and step 0 still carries
it (bias ≈ 0 there), so the label and "not learned" stand. **What the bias changes is the size
after init** (`Xe′` median, β 3.5, L1–8 / L9–16 / L17–23): step 0 +0.10 / +0.04 / +0.02; a peak at
32–256 up to +0.15 / +0.09 / +0.13; 143000 +0.06 / +0.03 / +0.09. So by the end L1–16 are *less*
lumpy than at init and only L17–23 more; the raw `Xe`'s "143000 +0.11 / +0.05 / +0.19" was half
bias. Checks: the device's bias against CPU float64 on the first cell, max |Δ| 6.8e-9; test
`test_calibrated_lumpiness_is_zero_on_a_structureless_cloud` (the bias predicts a structureless
draw's score). **The density is the β → 0
term:** R² of `e_i` on `⟨u_i, ū⟩` (`beside` `r2_mean`) is 0.65–1.00 in every trained band at
every step, except L1–8 at 64–256 (0.39, 0.33, 0.56; the two wells are at 128). *Corrected after
`/challenge-pr` on #165, finding 3: the first text read six steps only (0.66–1.00 there, except
128) and missed L1–8's 64 and 256.* **Dense** tokens are
punctuation and whitespace, **void** are word starts and continuations (L9–16, pooled over
passages and layers: punctuation is 0.66 / 0.51 / 0.41 of the densest decile at 0 / 1000 / 143000
against 0.19 of all), at init too.

**Density and position** (label (3), Spearman with log offset): mostly mixed (47 of 54 at β 3.5;
46, 50 at 1.6, 5.6); **denser later** only in L17–23 at 16 and 143000 (ρ median +0.09, +0.20;
β-robust) and leaning at 1000–2000 in L9–23. R² on log offset ≤ 0.07 in every trained band. T1–T3
reads more "denser later" (11, plus 13 leans), since it keeps the front of each passage. The
causal density (log mean over j < i) is also near position-free (median ρ −0.25 to +0.29).

**Reading.** At the measured β, `φ_β` in β's frame is a single basin over the tokens: there are no
wells to call clusters, and U3's "saddles between U1's wells" has nothing to read at 1.6–5.6. What
the field does carry is a density that is mostly each token's pull towards the cloud's mean (the
same β = 0 term U2's frozen labels turned out to be, "U2's block arm"), with punctuation at the
centre and content words at the edge, already at init. For Phase 10 (Blocked 27, "Relation to
the other threads" in the design): a well is not a better unit than c3x at this β. Blocked 30 decided (a): "Blocked 30 decided"
below.

**How to re-run.** From the worktree root: `data/p1e/u1_field_2026-10-08/run.sh` (resumes;
`--first-only` for the first check), then `python -m p1e_energy_field.u1_field report --out
<dir>`. Resuming the records needs `9732341`'s `p1e_energy_field` tree.

**Parked** (why / cost / the decision it could change):
- Wells at β ≈ 10, the one range with a few / free, from the stored runs (the sweep's code, all
  layers) / whether a well is a unit anywhere near β, and U3 there (Blocked 30 (b)).
- The two wells in L1–8 at step 128, the one trained exception / free, from the saved labels /
  what they are, beside U2's 32–256 window.
- Whether one well at β 3.5 is what the theory predicts at this n (~2,000) and d (1,024)
  (`/challenge-pr` on #164, finding 5) / a reading of 2312.10794's single-cluster thresholds,
  not done / would make Blocked 30 (a) the theory's expectation rather than a null result.

## Blocked 30 decided (2026-10-08): U3 closed at the measured β; the 1e page

**Decision** ((a), as recommended, recorded from the session prompt, which carried the
alternatives "(a) [or (b)/(c)]"; **the user to confirm** on #165, as Blocked 27 (a) was): U3 ("saddles between U1's wells") is closed as
"no wells at the measured β", U1's result standing as its answer. Reading wells at β ≈ 10 (b) or
in the cloud-centred frame (c) would choose a β or a frame in order to find wells, the placed-bar
problem 1e opened to avoid (`design-1e.md` "Why a new phase"); both stay in U1's Parked list.
U4 stays fenced until P-S1 is scored; U5 stays parked (no corpus). No unit of 1e is scheduled.

**The page:** https://claude.ai/artifact/8ZqeVhbPrtTBxkDWGm4Hts (private until the user shares
it). U2's labels as (step, band) grids for six readings (whole block, the shared update alone,
token-specific block, MLP and attention parts, attention against its heads' own kernels), split
at 1000 / 2000 as the user's Blocked 29 (c); per-head descend / ascend shares under the sum's
label; U1's sweep (median `k_eff` against β at L4 / L12 / L20 for steps 0, 128, 1000, 143000)
and the calibrated lumpiness `Xe′` at β 3.5. **Input:** long passages (`ba605f4e14b5`), T1 + T2,
from `data/p1e/u2_block_2026-10-07/report.json` (`t12|causal|3.5`, `t12|causal:resid|3.5`,
`t12|causal:r1out|3.5`), `u2_attn_2026-10-07/report.json` (`t12|causal:mlp:r1out|3.5`,
`t12|causal:keys:r1out|3.5`), `u2_heads_2026-10-07/report.json` (`t12|kernns:keys:r1out|0.0`,
per head `kernns_h:keys_h:r1out`), `u1_field_2026-10-08/labels.json` (`sweep_k_eff`) and
`calib/labels.json` (β 3.5). The R² and punctuation figures are quoted from "U1" above, not
recomputed. **Builder** `p1e_energy_field/page.py` (template `viz/index.html`): no
computation of its own; refuses a missing cell, and refuses if a grid's label counts over the
trained bands differ from its report's printed counts (all six match, e.g. the whole block 35
ascend of 54, the kernel row 11 / 11 / 4 + 3 leans / 25). The published page is its output.
Not checked: no browser or JS engine on this machine, so the page was published without a
rendered look; the chart ramps were not run through the palette validator (no `node`).

**How to re-run:** `python -m p1e_energy_field.page --data <main>/data/p1e --out <dir>`, then
publish `<dir>/index.html` to the URL above.

## Blocked 30 (b) and (c), beside (a) (2026-10-09): both find wells, and step 0 has them too; centred, a few (2–4 late) that c3x groups agree with; at β 10 training adds wells over the init's at 64–8000 and loses them from 16000

*User, 2026-10-08, after #165 merged: (a) confirmed, and "work on all three", so (b) and (c) are
read **beside** (a), not in place of it.* Rules fixed before output: `design-1e.md` "U1 beside"
(`fecd694`) and "U3 at β 10" (`5f2959b`, amended before any reading in `775a04c`, `3b55242` and
`9d7d514`). **A defect, found before any reading** (`2608db3`, the amendment under "U1 beside"):
U1's GPU phase merged rows by a float32 dot whose error reaches its 1e-6 tolerance. (b), (c), (c′)
and U3 are on the fixed producer (`p1e_energy_field` tree `752d284`); (a) is #164's records on the
old producer, audited with the fixed one (first row below). The first builds sit unread in
`data/p1e/superseded_merge32/`. Each run is read only after its audit (`u1_audit`, below) passes.

| run | output (`data/p1e/`) | audit: cells, differing from CPU float64, min agreement | state |
|---|---|---|---|
| (a) U1 at 3.5 (#165, old producer) | `u1_field_2026-10-08/audit.json` | 810, **0**, 1.0000 (879,228 target rows; no `k`, `k_G` or `Xw` change) | stands; correction below |
| (b) U1, β 10 (7, 14 beside) | `u1_beta10_2026-10-08/` | 810, 2, 0.9990 (3 rows; one singleton missed, `k` 10 against 11 at 8000 `odyssey_butler_long` L20; `Xw` ≤ 0.006) | read |
| U3, β 10 | `u3_beta10_2026-10-08/` | its wells equal (b)'s in every cell (1.0000); 0 negative persistences | read |
| (c) U1, centred | `u1_centred_2026-10-08/` | 810, 1, 0.9995 (1 row; `Xw` ≤ 0.004) | read |
| (c′) U1, raw centred | `u1_rawcentred_2026-10-08/` | 810, 3, 0.9995 (3 rows; `k` changes in 3 cells, `k_G` in none; `Xw` ≤ 0.004) | read in its own PR (`/challenge-pr` on #167, finding 5), below |

**(a)'s sweep, checked with (b)'s fixed cells:** at β 10, L4 / 12 / 20, 10 of 432 cells change
(each by 1–7 wells, all in cells with more than 20 wells), and the bins "111 one, 196 with 2–20,
125 more than 20" are unchanged. (a)'s β 1.6 and 5.6 wells and its β 20–100 sweep were not audited.

### (b) U1 at β 10 (long passages, T1 + T2; labels at β 10)

| label (trained cells of 54) | β 7 | β 10 | β 14 | step 0 |
|---|---|---|---|---|
| (1) lumpier / other | 51 / 3 | 50 + 1 lean / 3 | 50 / 4 | lumpier in every band |
| (2) more wells than its Gaussian | (not read: the Gaussian's wells at 10 only) | **37 + 6 leans** / 11 mixed | — | more in L1–8, L9–16; leans more in L17–23 |
| (3) density and position | mixed 50 | mixed 52 | mixed 52 | mixed |
| (4) what the wells are | — | **content 43**, not read 8 | — | content in every band |

Chance is 0.42 all-8 and 3.4 leans per 54. **Every label that departs from chance is carried by
step 0, so none is called learned** (the rule).

*After `/challenge-pr` on #167:*
- **(2) is mostly forced at β 10 (finding 1).** In 2,504 of 3,128 trained cells all 4 matched
  Gaussians have a single well. There `Xw = log k_eff ≥ 0` by construction, so (2) can almost
  never read *fewer*. Here "more wells" means the data has more than one well where its Gaussian
  has one, not a comparison of counts. In (c) only 70 cells are like this, so (c)'s (2) is a real
  comparison.
- **Training adds wells, which the rule cannot show (finding 2).** The step-0 rule compares labels,
  not values. Paired by passage, trained `Xw` exceeds step 0's in 7–8 of 8 passages in L9–16 and
  L17–23 at every step 64–8000, and in L1–8 at 64, 256 and 1000–4000. It falls below step 0's from
  16000 (L1–8 0 of 8 at 16000 and 143000; L9–23 2–4 of 8). So training adds wells over the
  init's at 64–8000, then removes them.

The wells (median over passages, β 10):

| step | L1–8 `k` / `k_eff` / largest | L9–16 | L17–23 | L0 |
|---|---|---|---|---|
| 0 | 61 / 17.2 / 0.60 | 3 / 1.5 / 0.90 | 2 / 1.2 / 0.97 | 666 / 232 / 0.07 |
| 128 | 265 / 110 / 0.22 | 18 / 5.9 / 0.49 | 7 / 2.8 / 0.68 | same |
| 1000 | 47 / 11.4 / 0.48 | 29 / 5.7 / 0.57 | 22 / 3.5 / 0.64 | same |
| 2000 | 112 / 26 / 0.39 | 139 / 22 / 0.45 | 27 / 3.4 / 0.72 | same |
| 8000 | 41 / 4.4 / 0.74 | 64 / 9.4 / 0.57 | 13 / 1.6 / 0.92 | 646 / 227 / 0.07 |
| 16000 | 3 / 1.2 / 0.97 | 5 / 1.3 / 0.95 | 2 / 1.0 / 1.00 | 574 / 216 / 0.07 |
| 143000 | 13 / 2.2 / 0.89 | 3 / 1.1 / 0.98 | 2 / 1.1 / 0.97 | 551 / 201 / 0.07 |

So a few to tens of wells at 64–8000, and from 16000 one well holding 89–100 % of targets with a
handful of side wells. The merge tolerance changes the count in 148 of 10,368 long cells (at 1e-4
or 1e-2), more than at 3.5 (0). **c3x purity on v1 (beside; the rule fixed no sign rule for it):**
over trained cells with c3x groups and `k2 ≥ 2` (655), R0's c3x groups sit in one well more than
under permuted well labels: median excess +0.09 / +0.12 / +0.11 (L1–8 / L9–16 / L17–23), and
`p ≤ 0.05` in 59–69 % of cells. Step 0 has no c3x groups, so there is no init comparison. T1–T3
beside: lumpier 54 / 54; wells read *fewer* than the Gaussian at 2000–8000 (the repeats co-locate).

### U3 at β 10 (long passages, L4 / 12 / 20, β 10)

Label (5), `Xp = log(1 + P) − mean_draws log(1 + P_G)`, `P` the cell's summed persistence:

| step | L1–8 (L4) | L9–16 (L12) | L17–23 (L20) |
|---|---|---|---|
| 0 | deeper +3.41 | deeper +1.55 | mixed +0.45 |
| 2–32 | deeper | deeper at 2–4, then mixed | mixed |
| 64–1000 | deeper (leans at 128) | deeper (leans at 512) | **deeper** |
| 2000–4000 | deeper | mixed | **deeper** |
| 8000 | deeper | deeper | leans deeper |
| 16000–54000 | mixed (≈ 0) | mixed / leans | mixed (≈ 0) |
| 143000 | leans deeper | mixed | mixed (0.00) |

Trained cells: deeper 28, leans 5, mixed 21, **shallower 0** (chance 0.42 / 3.4). *But (finding
1 on #167):* the Gaussians have zero persistence in 336 of 408 trained cells, where
`Xp ≥ 0` by construction, so "shallower 0" is largely forced. L4 and L12 carry "deeper" at step
0, so those are not called learned. **L20's at 64–8000 is not carried by step 0, so by the rule it
is learned.** *Finding 2:* that rests on step 0 reading "mixed", because at L20 four passages have
one well at init (`Xp` = 0) and four have +0.9 to +1.0. The trained medians (+0.8 to +2.5) do
exceed the init's, which agrees with the paired `Xw` reading above. It goes again from 16000,
where the Gaussians and mostly the data too have one well (`P` ≈ 0 both sides). Medians: `P` 29 / 4 / 0.7 nats at step 0 (L4 / 12 / 20), up to
113 at L4 and 81 at L12 at 128–2000, and 0–4 at 143000; the deepest single death (`p_max`) is
1.2–2.8 nats where a cell has deaths. **`Xp` follows `Xw`**: per (trained step, passage, layer)
cell, Spearman 0.86 and the same sign in 98.5 % of 408 cells. A deeper landscape here is mostly
more wells, each 1–3 nats deep, not a few deep ones. Part of that agreement is the forced sign
above: both are ≥ 0 wherever the Gaussian has one well.

**Edge heights (corrected after `/challenge-pr` on #167, finding 3).** The graph's edge height
samples each edge at 3 points (its ends and the midpoint), so it is not the edge's minimum. A
65-point geodesic finds a lower point on most saddle edges: 34 of 37 (1000 `hamlet_long` L12) and
290 of 311 (128 `wiki_paragraph_long` L4), up to 0.56 nats, raising `P` by 5 % and 24 %; the
review found 415 of 519, up to 0.9. So the graph's pass heights are not lower bounds, and
`design-1e.md`'s "every pass height … is a lower bound" does not hold for them. The band is also
sampled at its 24 images. No label changes sign (finding 1).

**NEB (beside):** 1,943 bands, 1,888 converged. The band's pass sits above the graph's by a median
of +0.74 nats (quartiles +0.03, +1.59). The Gaussians' trees are graph-only, so `Xp` compares the
graph with the graph; the size of each side's error is not measured. 156 bands (8 %) ended below
the graph's pass (to −4.8). **Crests:** the token classes at saddle edges show no consistent
enrichment against all targets (median ratios 0.2–1.5 by class and step group, noisy at few
deaths). Not read further.

### (c) U1 in the cloud-centred frame (`unit(u_i − ū)`; long passages, T1 + T2; labels at β 3.5)

| label (trained cells of 54) | β 1.6 | β 3.5 | β 5.6 | step 0 (β 3.5) |
|---|---|---|---|---|
| (1) lumpier | 48 + 2 leans | 53 + 1 lean | 54 | lumpier in every band (+0.27–0.29) |
| (2) more wells than its Gaussian | (not read) | **50 + 4 leans** | — | more in every band (+1.45–1.56) |
| (3) density and position | mixed 52 | mixed 52 | mixed 53 | mixed |
| (4) what the wells are | — | **content 53**, mixed 1 | — | content in every band |

**Every label is carried by step 0, so none is called learned.** The wells (median over passages):

| step | L1–8 `k` / `k_eff` / largest | L9–16 | L17–23 | L0 |
|---|---|---|---|---|
| 0 | 17 / 9.0 / 0.25 | 16 / 8.5 / 0.24 | 15 / 8.4 / 0.26 | 20 / 10.3 / 0.22 |
| 128 | 9 / 5.1 / 0.46 | 3 / 2.8 / 0.48 | 3 / 2.5 / 0.49 | same |
| 1000 | 9 / 5.8 / 0.37 | 4 / 3.3 / 0.40 | 3 / 2.9 / 0.45 | same |
| 8000 | 4 / 3.1 / 0.45 | 4 / 3.5 / 0.41 | 4 / 3.3 / 0.43 | 18 / 9.7 / 0.24 |
| 143000 | 3 / 2.5 / 0.52 | 3 / 2.6 / 0.54 | 2 / 1.9 / 0.69 | 6 / 3.6 / 0.55 |

So with the cloud's mean removed, the field at the measured β has **a few wells (2–4 by 143000)**:
the count the probe saw, though the probe's own frame is (c′), unread (finding 5 on #167). It has more at init (15–17, `k_eff` ≈ 8.5), and training reduces
them. Their excess over the matched Gaussian (`Xw`) also shrinks, from +1.5 at init to +0.1–0.4 at
143000, while staying positive. AMI against token class is 0.10–0.15 at init and 0.16–0.35 in
training, except L17–23 at 143000 (0.04, the one "mixed"), with position ≈ 0 throughout. The merge tolerance changes 4 of 10,368 long cells. **c3x
purity on v1 (beside; no sign rule):** R0's c3x groups sit in one well well above the permuted
labels, median excess +0.34 / +0.38 / +0.35 (L1–8 / L9–16 / L17–23) and `p ≤ 0.05` in 90–93 % of
1,477 cells. *Caveat:* c3x groups and these wells both partition the same rows by proximity, so
purity above a permutation is partly expected for any two such partitions. It says the two agree,
not that the well is the better unit. β is not measured in this frame (3.5 carried over), and M6
holds: mean shift from the targets stays in their span and does not see the frame's flat floor.

### (c′) U1 in the raw-centred frame (`unit(x_i − x̄)`, no LN1; the probe's frame): it agrees with (c), and on the probe's own cells it reproduces the probe's 2–4 wells at 143000; the init count depends on the token count

*Read 2026-10-09, after its audit passed (table above). Report `u1_rawcentred_2026-10-08/report.txt`,
run from the tree of #167's head (`735f7d3`); records on `752d284`.*

| label (trained cells of 54) | β 1.6 | β 3.5 | β 5.6 | step 0 (β 3.5) |
|---|---|---|---|---|
| (1) lumpier | 43 + 6 leans (smoother 2 leans, L9–23 at 32) | 54 | 54 | lumpier in every band (+0.27–0.28) |
| (2) more wells than its Gaussian | (not read) | **49 + 5 leans** | — | more in every band (+1.43–1.53) |
| (3) density and position | mixed 52 | mixed 52 | mixed 54 | mixed |
| (4) what the wells are | — | **content 53**, mixed 1 (L17–23 at 143000, as (c)) | — | content in every band |

**Every label is carried by step 0, so none is called learned.** The wells (median over passages):

| step | L1–8 `k` / `k_eff` / largest | L9–16 | L17–23 | L0 |
|---|---|---|---|---|
| 0 | 17 / 8.9 / 0.25 | 16 / 8.6 / 0.25 | 15 / 8.3 / 0.27 | 21 / 10.9 / 0.19 |
| 128 | 9 / 4.5 / 0.53 | 4 / 2.9 / 0.49 | 3 / 2.6 / 0.47 | same |
| 1000 | 9 / 5.2 / 0.41 | 4 / 3.5 / 0.40 | 4 / 3.1 / 0.45 | 20 / 10.1 / 0.19 |
| 8000 | 3 / 2.3 / 0.68 | 3 / 2.7 / 0.54 | 3 / 2.9 / 0.44 | 12 / 5.0 / 0.51 |
| 143000 | 3 / 2.0 / 0.67 | 3 / 2.4 / 0.59 | 2 / 2.2 / 0.55 | 4 / 2.3 / 0.72 |

- **The probe's own cells** (v1 `homer_iliad` and `wiki_paragraph`, L4 / 12 / 20, `r0`; in the
  records; `/challenge-pr` on #168, finding 1): at 143000, 2–4 wells at β 3.5 in all 6 (`k_eff`
  1.9–2.7). At β 5.6 they shatter in L4 and L12 (151–254 wells) but not in L20 (3). At step 0,
  22–271 wells, 4 of 6 above 200 (most tokens their own well). So the probe's 2–4 late wells
  hold under U1's rule, in its frame and on its cells.
- **The init count depends on the token count; the late count does not.** On the long passages
  (~1,000–2,000 targets) init has 15–17 wells (`k_eff` ≈ 8.5); on v1 (~210–270) it has up to one
  per token. At 143000 both have 2–4. So "15–17 at init falling to 2–4", here and in (c), is a
  long-passage number, not a property of the init alone. `Xw` falls from +1.4–1.5 at init to
  +0.12–0.46 at 143000 and stays positive.
- **(c′) agrees with (c), cell by cell** (long, `t12`, β 3.5, 3,456 cells, the same target
  positions in both): the same `k` in 58 % (20 % when each cell is paired with another passage's
  at the same step and layer), within 1 in 89 % (49 %), median |Δ`k_eff`| 0.21. The token
  partitions agree (AMI median 0.82, quartiles 0.63 / 0.90), but 9 % of cells are below 0.4 (10 %
  of trained cells, none at step 0). What this isolates: centring and the unit norm already undo
  LN1's per-token shift and scale, so (c) against (c′) tests mainly LN1's per-feature weights
  (and the order of normalising and centring). Those change the wells in a tenth of cells, not
  the reading. The one visible difference in the medians is L0 (the embedding, which trains):
  21 → 12 → 4 wells at 0 / 8000 / 143000, against (c)'s 20 → 18 → 6.
- AMI against token class 0.09–0.11 at init, 0.15–0.38 in training, except L17–23 at 143000
  (0.05, the "mixed"); against position ≤ 0.04.
- **c3x purity on v1** (beside, no sign rule; 1,480 cells): median excess +0.30 / +0.34 / +0.35
  (L1–8 / L9–16 / L17–23), `p ≤ 0.05` in 88–91 %. Same caveat as (c): two proximity partitions.
  It has no step-0 floor: at step 0 no cell has both c3x groups and two or more multi-token
  wells, so purity is not read as learned (as in (b) and (c)).
- Merge tolerance changes 3 of 10,368 long cells. β is not measured in this frame (3.5 carried
  over); (1′)'s calibration is not computed (the rule).

The wells table, the probe's cells, AMI, purity and the comparison come from
`p1e_energy_field.u1_beside_read` (see "How to re-run"). On (c) it gives the
published numbers exactly, except (c)'s 128 L17–23 `k_eff`: the value is 2.55, now printed 2.5
where it said 2.6.

**Reading.** Read beside (a), both choices find wells, and neither finds learned ones. At β 10
(placed, 1.8× the measured interval's top), β's frame has tens of wells early in training and a
dominant well with side wells late. In the centred frame at the measured β there are a few wells,
2–4 by the end of training. In both, the wells exist at init and sort by content at init. They
outnumber the matched Gaussian's at init too, though at β 10 that comparison is mostly forced
(the Gaussian has one well). What training changes is how many wells there are, which the step-0
rule cannot label. The centred frame goes from ~17 to 2–4. At β 10 training adds wells over the
init's at 64–8000 (paired, 7–8 of 8 passages in L9–23), then falls to one dominant well from 16000,
with depth between wells in the last layers at 64–8000 (U3, L20) gone by then too. For Phase 10 (Blocked 27): in the centred frame c3x groups agree with the wells (purity
+0.35), and that agreement is the one sign here that a well and a c3x group pick out the same
tokens. Whether it is more than two proximity partitions agreeing is not read.

**How to re-run.** From a checkout whose `p1e_energy_field` tree is `752d284` (`2608db3`, or
`../Mets-u3`): each output's `run.sh` (resumes; it `cd`s to `../Mets-u3`), then
`python -m p1e_energy_field.u1_audit --out <dir>` (resumes; exits 1 below the gate) and
`... u1_field report` / `... u3_saddles report --out <dir>`. The reports refuse unless the audit
covers every record at ≥ 0.999 (from this PR's review fix; U3's report reads (b)'s audit), and
re-running them reproduces the stored `report.txt` exactly ((a)'s gains only its frame line).
`data/p1e/audit.sh <dir>` runs the audit. (c′) finished and its audit passed (2026-10-09).
`../Mets-u3` is no longer needed for a run. The (b) / (c) / (c′) wells tables, AMI, c3x purity
and the (c)-against-(c′) comparison come from `python -m p1e_energy_field.u1_beside_read --data
<main>/data/p1e <dir> [--against <dir>]`. It reads only, and its output for (c) matches the
tables above.

**Parked** (why / cost / decision it could change):
- Whether c3x purity in the centred frame beats any proximity partition / free, a k-means or
  random-ball partition of the same `k` as the null in place of permuted labels / whether the
  centred well is a candidate unit for Phase 10 (Blocked 27).
- (a)'s β 1.6 / 5.6 wells and β 20–100 sweep, not audited / ~1 h, an audit flag / (a)'s beside
  numbers only (its labels are at 3.5, audited clean).
- The NEB bands that end below the graph / free, from the records / whether 24 images are too few
  where a pass crosses a token's bump.

## Corrections received

- 2026-10-07 (#158's follow-up, `/challenge-pr` on #158 finding 1): #158's headline ("ascends
  at 8–256 and 16000–54000; L1–8 descends at 1000–8000; the last two blocks descend at 143000",
  and `STATE.md`'s "repulsive") read the frozen labels as token interactions. An update every
  token shares carries the early ascent and the 143000 descent; the token-specific part
  (`r1out`, after #159's review) is the table under "The shared update".
- 2026-10-08 (#165, `/challenge-pr` on #165 findings 1–3): `r1out`'s summary now names L1–8's
  ascent at 64–128 beside L9–16 from 4000 and L1–8 from 16000 (card, heading, `STATE.md`); #161's
  "most heads still descend their own" is "more heads descend than ascend" (a majority in 2 of
  13 windows); #164's R² range is read at all 18 steps (0.65–1.00, except L1–8 at 64–256).
- 2026-10-09 (this PR; `design-1e.md` "U1 beside", amendment of 2026-10-08): U1's well finder
  merged rows by a float32 dot whose error reaches its tolerance (fixed, `2608db3`). #164 / #165's
  U1 at β 3.5 is audited clean (810 cells, 0 differ). Its β 10 sweep loses 1–7 wells in 10 of 432
  cells with more than 20, and no quoted count changes. Its β 1.6 / 5.6 wells and β 20–100 sweep were
  not audited ("Blocked 30 (b) and (c)").
