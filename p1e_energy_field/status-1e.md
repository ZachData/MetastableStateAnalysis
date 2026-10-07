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
  - With the update every token shares removed (a null that reads 0 at init), the token-specific part descends the field at 32–2000 and ascends in L1–16 only from 16000; the early ascent and the descent at 143000's last blocks are the shared update — `p1e_energy_field/status-1e.md` "U2's block arm"
- **Superseded / wrong:**
  - #158's headline read the frozen labels as the token interactions ascending at 8–256 and descending ("repulsive") at 1000–8000 and 143000; the shuffle null cannot see an update every token shares, which carries both — `p1e_energy_field/status-1e.md` "Corrections received"
- **Registry:** none, because the phase is exploratory and unregistered; it is fenced off P-S1, P-γ1/P-γ2 and P-M1 — `p1e_energy_field/design-1e.md` "Fences"
- **Depends on:** 1d@4168cdd237, 10@c159a02c9b
- **Feeds:** none
- **Open threads:**
  - Whether an update every token shares counts as following the field is the user's decision, and it sets which reading the attention and per-head arms carry; the attention arm splits the shared part into attention and MLP — `STATE.md` Blocked 29
  - The correlated against anti-correlated question at the token level needs a corpus for co-occurrence — `p1e_energy_field/design-1e.md` "Units"
- **After Phase 10:**
  - U1, U3, U4 from stored activations *(free)*; U2 *(free for the block arm; forward pass per step and passage, GPU, for the attention and per-head arms)*
  - U5, after a corpus download and a co-occurrence count *(free)*
- **Reviewed:** 2026-10-07 · body `5e1eaa8a4a`
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

## U2's block arm (2026-10-07): the frozen labels are mostly an update every token shares; the token-specific part descends the field at 32–2000 and ascends in L1–16 from 16000

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
Readings (`u2_block.shared_cells`; mean over the targets, causal field at β 3.5 and the β = 0
field `mean0`): **`residout`**, the residual update less its mean over the targets, through LN1
and onto the sphere (a shift shared in the residual is removed exactly; reads 0.000 at init in
every band, so the row is specific); **`resid`**, that mean update alone; **`ambout`**, the
review's check (the unit-frame move less its mean, before projection).

| steps (long, T1 + T2) | `residout` L1–8 | L9–16 | L17–22 | `resid` (shared alone) |
|---|---|---|---|---|
| 0–16 | ≈ 0, mixed | ≈ 0, mixed | ≈ 0 (leans descends at 16) | ascends from 8, every band |
| 32 | **descends** | **descends**, −0.028 | **descends**, −0.042 | ascends |
| 64–128 | ascends, +0.006 / +0.013 | descends at 64, leans asc at 128 | **descends**, −0.041 to −0.053 | ascends |
| 256 | mixed | mixed | mixed | ascends |
| 512–1000 | **descends**, −0.024 / −0.026 | **descends**, −0.038 / −0.008 | mixed / lean | L1–8 mixed, else ascends |
| 2000–8000 | descends to 4000, mixed 8000 | mixed, ascends from 4000 | mixed, ascends at 4000 | **L1–8 descends**, else ascends |
| 16000–143000 | **ascends**, +0.009 to +0.030 | **ascends**, +0.013 to +0.034 | leans descends / mixed | 143000: L1–16 descend, L17–22 leans |

| count over 54 cells | ascends | descends | leans asc / des | mixed |
|---|---|---|---|---|
| `residout`, causal β 3.5 | 13 | 12 | 2 / 6 | 21 |
| `residout`, β = 0 | 9 | 25 | 2 / 5 | 13 |
| `resid` (shared alone) | 31 | 7 | 3 / 3 | 10 |
| `ambout` (the review's) | 51 | 0 | 3 / 0 | 0 |

- **The 8–256 ascent is mostly the shared update**: `resid` ascends there, `residout` is ≈ 0 at
  8–16 and descends at 32 (L1–8 alone ascends at 64–128). **143000's L17–22 descent is the shared update** (`residout` −0.001,
  mixed; `resid` leans descends). **L1–8's descent at 1000–8000 is both**: token-specific at
  512–4000 (`residout` descends), shared at 2000–8000 (`resid` descends).
- **The review's `ambout` is not specific**: it reads ascends at step 0 in every band (the `0`
  marks in `report.txt`), since renormalising leaves part of a shared shift in it. Its two
  sign flips (#158's comment) are therefore not evidence; `residout` is the check.
- **The field's local structure matters for the token-specific part**: at β = 0 `residout`
  descends in 25 cells, at β 3.5 in 12 (`report.txt`).
- **What removing the mean also removes:** a genuine pull of every token towards the cloud's
  mean is largely shared too, so `residout` can remove real field-following; `resid` ascending
  is what the theory's mean-field attraction would also do. The attention arm (hooks) splits
  the shared update into attention's and the MLP's, which this arm cannot.
- **v1 (beside):** `residout` the same windows: ≈ 0 at 0–16, descends in L9–22 at 32–256
  (L9–16 mixed at 128), in L1–8 at 512–8000 and L9–16 at 512–2000, ascends in L1–16 from
  8000 (L9–16) / 16000 (L1–8).

**Provenance.** The shared records carry their producer (`code` d4a1ad9, rebased unchanged as
0cd92be on `claude/p1e-u2-shared`; `device`, `mode`), and `run` now refuses to resume over a
record from other code. The frozen `records/` predate this: they were written by #158's
`u2_block.py` (7869cbd) less one comment line, all in one run on 2026-10-07 06:10–08:02.
Run: `run.sh --mode shared` (CPU, 14 workers, ~250 s each in the pool, 47 min).

### GPU reading (2026-10-07)

`p1e_energy_field/u2_torch.py` mirrors the numpy reader on a torch device; `u2_block run
--device cuda` uses it, and `u2_block agree` recomputes stored records and compares. On 6 runs
(long 143000 / 0 / 2000, v1 512, two shared), **CUDA float64 reproduces the CPU records to
≤ 9e-13 in every field, ranks and signs identical**, at ~20 s per long run (shared mode 8 s):
about 2.5× the 14-worker CPU pool, CPU left free. **Float32 fails** (`|ΔX|` up to 4e-4, 6
ranks moved, against L0's `X` ≈ 1e-3), so `cuda32` is not used. Output `agree_cuda.json`,
`agree_cuda32.json` beside the records; tests `tests/test_p1e_u2_torch.py` (torch on the CPU
in float64 equals numpy to 1e-10).

**How to re-run.** From the worktree root: `data/p1e/u2_block_2026-10-07/run.sh` (resumes;
`--first-only` for the first check), then `python -m p1e_energy_field.u2_block nll --runs <long8>
--out <dir>` and `... report --out <dir>`. CPU, float64, 14 workers: 143 s per long run alone,
~700 s each with 14 in parallel (memory bandwidth), 1 h 50 min for the 269 after the first.

**Parked** (why / cost / the decision it could change):
- The last two blocks at 143000 / the attention arm (GPU) / whether their shared update is the
  sink's value or the MLP's bias (it is not token interaction: `residout` is mixed there).
- L9–16's descent at init / free / whether init carries a structural sign; it is shared
  (`residout` ≈ 0 at init), so the frozen rows carry it and `residout` does not.

## Corrections received

- 2026-10-07 (#158's follow-up, `/challenge-pr` on #158 finding 1): #158's headline ("ascends
  at 8–256 and 16000–54000; L1–8 descends at 1000–8000; the last two blocks descend at 143000",
  and `STATE.md`'s "repulsive") read the frozen labels as token interactions. An update every
  token shares carries the early ascent and the 143000 descent; the token-specific part
  (`residout`) is the table under "The shared update".
