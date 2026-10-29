<!-- trackT_timaeus/status-T.md -->
# Track T — STATUS

<!-- phase-card -->
## Card

- **Question:** Do Timaeus's released checkpoints (Dodecahedron, a fourth training schedule) and the local learning coefficient give C2's energy break a second schedule and a degeneracy curve to sit beside, and which released model and settings make the LLC a calibrated instrument here?
- **Inputs:** none run; T1 read Hugging Face metadata (cards, config files, branch and file listings, LFS hashes) and papers through a summarising fetch, listed in `trackT_timaeus/status-T.md` "T1, the inventory (2026-10-29)"; T2 unit 1 ran `dodecahedron-32m` and `-67m` at steps 1 and 9536, 3 re-inits each, and `pythia-410m` steps 0–4000, the 7 v1 prompts (`976b62d677bc`), code `c6d56ab1085e`, listed in `trackT_timaeus/status-T.md` "T2 unit 1"
- **Results:**
  - Opened 2026-10-29 (Blocked 117 (a), the user's: T1 and T2 first) with T1's questions, grades and a one-unit budget fixed before reading — `trackT_timaeus/design-T1.md`
  - T1 closed at its budget: Dodecahedron's 40B tokens fits a global batch of 8 × 524k (per GPU), not 524k, and under it T2's token window overlaps both its step and its LR-dose windows, so T2 separates LR dose from step either way and tests tokens only if the batch is 524k; `-v2` is a rename and `-archived` the same weights, but `-pilot-dense` is a different run; 2402.02364's LLC settings are read, its model is most likely `tetrahedron-3m-og`; no released model deeper than 2 layers has a published whole-model LLC curve (`L4` has one head's, 2507.21449); `gpt-2-small` is a GPT-NeoX of Pythia-160m's shape; the Pythia negative stands for the two papers read — `trackT_timaeus/status-T.md` "T1, the inventory (2026-10-29)"
  - T2 unit 1 (design frozen on D1–D4 (a)): the producer reproduces A5's seed-0 records on `pythia-410m` exactly and its planted effect and null pass, and Dodecahedron loads as trained with the state convention of GPT-NeoX; but every centred arm fails its gate on both models (plant A, which centring should remove, turns 0.21–0.51 of the free units), so the centred count is not the statistic it was on Pythia and GPT-2 and the design goes back to the user (Blocked 120) — `trackT_timaeus/status-T.md` "T2 unit 1"
- **Superseded / wrong:**
  - The ROADMAP's T2 windows (from #282) read the card's per-GPU batch as the global one and used seed 0's break bracket, not A5's nine-seed (128, 512]; re-derived in T1 — `trackT_timaeus/status-T.md` "T1, the inventory (2026-10-29)"
- **Registry:** none, because Track T is tier 1
- **Depends on:** none
- **Feeds:** none
- **Open threads:**
  - Dodecahedron's global batch is inferred from the card's arithmetic, not read from a training log; the user asks Timaeus (Blocked 118 (b)), and tokens is scored in T2 only on `design-T2.md` D4's condition — `trackT_timaeus/status-T.md` "T1, the inventory (2026-10-29)"
  - T2's centred count fails its gate on Dodecahedron; whether T2 reads E₁ alone, a re-built centred count or stops is Blocked 120 — `trackT_timaeus/status-T.md` "T2 unit 1"
  - Which seed of `tetrahedron-3m` the paper's curve is from, and why `-og`'s checkpoint spacing differs from the `icl` sweep's — `trackT_timaeus/status-T.md` "T1, the inventory (2026-10-29)"
- **After Phase 10:**
  - The `L0`–`L6` attention-only depth series at constant LR with no warmup, as a schedule without the warmup confound *(forward pass, small models)*
  - `gpt-2-small` as a Pythia-160m-shaped run checkpointed every 100 steps *(forward pass; its data and schedule unread)*
- **Reviewed:** 2026-10-29 · body `60ccca4273`
<!-- /phase-card -->

## Units

| unit | state | where |
|---|---|---|
| T1. the inventory | **done 2026-10-29** (one unit, at budget) | `design-T1.md`; "T1, the inventory" below |
| T2. V and E_β on Dodecahedron | design **frozen 2026-10-29** (Blocked 119: D1–D4 (a)); **unit 1 done 2026-10-29** (1 of 3): producer, known answer and planted gate pass; **the centred count's arms fail their gate on both models**, so the design goes back to the user (Blocked 120) before the batch | `design-T2.md`; "T2 unit 1" below |
| T3. LLC positive control | candidate | `docs/ROADMAP.md` §7b |
| T4. LLC on C2's checkpoints | candidate, after T3 | `docs/ROADMAP.md` §7b |

## T2 unit 1: the producer, the calibration gate, the preflight (2026-10-29)

Design frozen first (`ab6826c`, pushed before any Dodecahedron weight was fetched). **Input:** the
7 v1 prompts (`prompts_hash` `976b62d677bc`), ≤ 512 tokens, float32, eager, TF32 off, RTX 3080;
`timaeus/dodecahedron-32m` and `-67m` at steps 1 and 9536 only (`model.safetensors`, stored
float32; `HDD_1TB/mets_data/hf`, 0.76 GB); `pythia-410m` steps 0–4000 from `<main>/data/hf`.
Code `c6d56ab1085e` (`trackT_timaeus/t2.py`); outputs `data/trackT/t2u1/2026-10-29-{gate,known,
preflight}/` (main tree): `gate.json` `53c56522e514`, `known_answer.json` `5f3cd65ac02c`,
`preflight_plants.json` `38bb1b793ccd` (32m), `e41098a01bee` (67m).

| check | result |
|---|---|
| known answer (410m through this producer's path) | **pass**: e1 equals A5's seed-0 records exactly (max relative deviation 0.0, tokens equal); V 0 through 128, 19 at 256, 56 at 512, plateau 57, half-rise 512 |
| planted effect (crossing at each band's geometric middle) | **pass**, 4 scored schedules × 2 counts: inst. reads 16 / 35, cum. 65 / 141, frac. 193 / 359, step 263 / 574; each consistent set is the design's outcome-table row (fraction and step not split, as the design says); `joint` keeps the planted schedule. With tokens scored (tested, not in force) tok524k passes too, its centred crossing (2024) inside the early window, which reads 2002 beside the late 2340 (both tok524k's) |
| planted null (20,000 draws, crossings uniform in log step over (1, 4369], 4 cells independent) | "neither" 98.2 %; each scored schedule survives by chance 0.50–0.54 %; every "consistent with …" label ≤ 0.49 % (not a p) |
| the models load as trained | step 1's mean loss 9.77 / 9.80 nats (ln 16384 = 9.70), step 9536's 3.66 / 3.46 (placed bar < 6) |
| the state convention | 17 / 21 states; the first equals the embeddings and the last the final norm of the last block's output, exactly (max difference 0.0), as GPT-NeoX and GPT-2 |
| tokenizer | one `tokenizer.json` (`4edef5ce3cb6`) at steps 1 and 9536 on both; saved by tokenizers ≥ 0.20 (merges as pairs), read by the env's 0.19 with each pair joined (no piece holds a space; the producer refuses otherwise); no BOS, as Pythia |
| the re-inits as the init | HF's Llama init at 0.02: every weight's SD within 0.989–1.012 (32m) and 0.994–1.007 (67m) of step 1's |
| `tools.design_preflight` (both counts, both models) | `floor` only (E 9.18, 10.3 < 20), as every C2 arm: C2 and T2 compute no E. Reports `981d68aecc0e` (32m u), `5351ef132934` (32m c), `460a89a4c486` (67m u), `504e6f9c98f6` (67m c) |

**The centred arms (D2 (a)), gated as 3b: every arm fails, on both models** (the bars: re-init
rate ≤ 5 %, plant B share ≥ 0.4, plant A share ≤ 0.1; B's dose 0.25 on both):

| model | arm | re-init rate | B share | **A share** |
|---|---|---|---|---|
| 32m (112 trained units, 336 re-init) | null-matched 5.60e-4 | 6.2 % | 0.98 | **0.38** |
| | 1e-4 | 8.9 % | 0.98 | **0.21** |
| | 3e-4 | 8.3 % | 0.98 | **0.23** |
| | B-matched 8.82e-3 | 0.3 % | 0.47 | **0.43** |
| 67m (140, 420) | null-matched | **refused**: −3.05e-4 ≤ 0 | — | — |
| | 1e-4 | 5.5 % | 1.00 | **0.43** |
| | 3e-4 | 4.5 % | 1.00 | **0.49** |
| | B-matched 4.05e-3 | 0.2 % | 0.78 | **0.51** |

Plant A, a common vector at 0.5 of the median token norm that centring should remove, turns
0.21–0.51 of the centred count's free units on Dodecahedron (on 410m, 70m, 160m and GPT-2 small it
passed). So the centred count is not the token-centred statistic it was on those models, and
`TAU_ARMS` is not frozen: `run` refuses. By the design ("Any arm failing sends the design back"),
**Blocked 120**, the user's, before any scored record.

**The uncentred count at D2's 1e-3** (never gated, as A5, 3b, 3c; the report above cited): re-init
rate 6.2 % (32m), 8.1 % (67m), trained rate at 9536 0.75, 0.71, plant B share 0.46, 0.78, plant A
0.04, 0.07. Its re-init rate is above the 5 % the centred arms are held to (3c's GPT-2 re-inits
read 0); the EARLY check (V at steps 1–2 ≥ half the plateau → unreadable) is what guards it.

**Defects fixed in this unit:** the null-matched derivation now refuses a zero re-init uncentred
rate (3c's Parked 0, carried here as the next unit deriving a centred arm; in `t2.py`, not in the
frozen `c2_seeds.py`, so C4 and C5 still need their own) and a tolerance ≤ 0 (67m's, found here).

**Not established:** why plant A leaks (per-token normalisation before centring on a residual
stream with a wide norm spread is a guess, not tested); anything about a date on Dodecahedron (no
grid step but 1 and 9536 has been run).

**Parked:** (1) Why plant A leaks on Llama / the token-norm spread of step 9536's states and plant A
read through `e1r` (raw states centred before normalising, already in every record) / a preflight
re-read, minutes / Blocked 120 (b) would need it first. (2) The uncentred count's 6–8 % re-init
rate / a gate for it would be a new rule on a count A5, 3b and 3c read ungated / whether a reading
on Dodecahedron's E₁ is held to the centred count's null bar.

**Re-run:** `python -m trackT_timaeus.t2 gate --out <dir>`; `METS_DATA=<main>/data
HF_HOME=<main>/data/hf python -m trackT_timaeus.t2 known --out <dir>` (2 min); `HF_HOME=<HDD_1TB>/
mets_data/hf python -m trackT_timaeus.t2 fetch --steps 1 9536`, then `… preflight --model {32m,67m}
--out <dir>` (1–2 min each), then `python -m tools.design_preflight <dir>/<m>/preflight_input_{u,c}.json
--out …`. Tests: `tests/test_trackT_t2.py` (pure, 15); `python3 tools/math_checks/t2_grid.py`
(T1–T6; T6 checks the producer's constants and `predicted`).

## T2, the design (proposed 2026-10-29; frozen 2026-10-29)

`design-T2.md`, opened by Blocked 118 (b), the user's. Nothing downloaded or run; the bands and
outcome table come from `tools/math_checks/t2_grid.py` (6 of 6 pass; re-run:
`python3 tools/math_checks/t2_grid.py`). What it fixes and what it found on paper:

| item | value |
|---|---|
| grid | the 53 released steps 1–9536 (same on `-32m` and `-67m`); 9536 the preflight's trained step |
| step against cumulative LR | separated under either batch: E₁ 35–121 against 165–574 (shared 141); centred 65–225 against 308–1072 (shared 263) |
| fraction against step | not split: 0.7 × step; they part only at 420–574 (E₁), 917–1072 (centred) |
| tokens at 4.19M | overlaps cumulative and step (the step band one octave down); beside, not scored |
| tokens at 524k | scored only if a reply confirming it is quoted here before the first scored record (D4 (a), proposed) |
| plateau (D3 (a), proposed; revised after `/challenge-pr` on #284) | early 1253–3738 unless tokens is scored, then late 5108–8158; the union of both plateaus' dates. V rises late on all 9 A5 seeds (1000–4000 to 8000–16000, by 3–20), so a late window leans dates later |
| width-scaled dose (beside) | cumulative with LR × d_model / 1024 lands on step's band at 141–205 (E₁): a step reading excludes only the raw dose |
| downloads | about 6.7 GB (`-32m`) and 14.3 GB (`-67m`) for the 53 steps |

D1–D4 decided all (a) (Blocked 119, 2026-10-29). Open for the user: the message to Timaeus
(drafted in `design-T2.md`), and Blocked 120 ("T2 unit 1" above).

## T1, the inventory (2026-10-29)

Sources: Hugging Face API (`/api/models`, `/refs`, `/paths-info`, `/commits`, raw cards and
yamls; 1,943 `timaeus/*` repos, 1,797 of them `rl-lm-pythia*` fine-tunes, not read further);
GitHub `timaeus-research/icl`, `timaeus-research/paper-rllcs-2024`; arXiv HTML of 2402.02364v3
(App. A.4.1, F.1.1, Table 1), 2507.21449 (App. E), 2606.22389v2, the abstract page of
2402.03698; the AF post "Timaeus in 2024" (2025-02-20). The arXiv HTML was read through a
summarising fetch that quotes the named sections, not as a full reading (not [R]); T3's design
re-reads A.4.1 in full before it freezes. No weights downloaded. No `.bin` or `.pth` pickle was
opened. Grades as `design-T1.md`, with "confirmed" from the papers resting on that fetch.

### Q1. A released model with a published LLC curve, deeper than 2 layers — **no, leaning**

| model | published LLC | depth | source |
|---|---|---|---|
| 2402.02364's LM (Q2) | whole-model, along training | 2 (and 1) | the paper (confirmed) |
| `timaeus/L4` = 2507.21449's model | **per head, one head shown** (layer 3, head 2), Fig. 4; sampler benchmark; γ, nβ, chains not stated | 4, attention-only | App. E matches `L4`'s yamls field for field (layers, d_model 256, 8 × 32, ctx 1024, vocab 5000, Shortformer, AdamW 1e-3, wd 0.05, 75,000 steps, batch 32, DSIR Pile); the paper names no repo, so leaning |
| anything else | none found | — | search, and the 2024 review (no curve) |

So the cheapest overlay (ROADMAP §7b, E_β over a published curve, no LLC estimated) has no
deeper model to use. `L4`'s one-head curve is not a whole-model control.

### Q2. 2402.02364's model and settings — **settings confirmed; model leaning `tetrahedron-3m-og`**

- **Settings** (App. A.4.1, Table 1, two-layer): SGLD, 20 chains × 200 steps, 1 draw a step,
  ε = 0.001, γ = 100, nβ = 21.7, minibatch m = 100, data lines 10m–11m of the DSIR-filtered Pile
  (10m–10.1m for validation); burn-in not stated for the LM. One-layer: ε = 0.003, γ = 300.
  nβ is m / log m (100 / ln 100 = 21.7), not n / log n (A.4.1). Table 1 also lists μ = 13m,
  "dataset size for gradient minibatches", beside A.4.1's 1M-line SGLD set; which set the
  minibatches are drawn from is not stated, and T3's design settles it before it freezes
  (`/challenge-pr` on #283, finding 2).
- **Model** (F.1.1): 2 layers attention-only, d_model 256, 8 heads, ctx 1024, Shortformer,
  vocab 5000, 3,355,016 parameters. The paper links only `github.com/timaeus-research/icl`.
- **Repo**: `icl`'s `sweeps/language/training-runs/tetrahedron-3m.yaml` is a 2-layer, 50,000-step
  run, seeds 0–4, data `timaeus/dsir-pile-10m` (public); its config defaults match F.1.1 (d_head
  32, batch 100, AdamW 1e-3, wd 0.05). `timaeus/tetrahedron-3m-og`: 501 checkpoints 0–50,000
  every 100, `losses.txt`, uploaded 2023-12-23, no card. Not confirmed: the sweep's checkpoint
  rule (log 100 + linear 2,000) does not give `-og`'s every-100 spacing, and which seed the
  published curve used is not stated.

### Q3. Dodecahedron's provenance — **mostly confirmed; the global batch leaning**

| item | answer | grade |
|---|---|---|
| `-v2` | the API redirects `dodecahedron-32m-v2` and `-67m-v2` (307) to `-32m` and `-67m`: a rename | confirmed |
| `-archived` | same weights: LFS sha256 equal at steps 1, 103, 491, 9536 (32m), one folder per step | confirmed at 4 of 53 |
| steps | `-32m` and `-67m` both carry the card's 53 steps (1 … 9536) | confirmed |
| `-32m` `sgd-relaxed` branch | "SGD-relaxed w0 fixed point (relaxation of step9536; fp32 patterning base)", 2026-06-16 | confirmed (commit) |
| `-pilot-dense` | **a different run**: weights differ from `-32m` at steps 1, 2, 103, 141; every step 1–160 released, `main` = step 500, no card; same architecture and tokenizer size | confirmed different; its schedule unknown |
| architecture | Llama, RMSNorm, RoPE θ 1e4, SiLU MLP, GQA (32m: 16 layers, 384, 6 heads / 2 KV; 67m: 20, 512, 8 / 4), tied embeddings, vocab 16,384 | confirmed (`config.json`) |
| LR | 3.5e-3, 1,000 warmup, cosine; AdamW β 0.9 / 0.95, wd 0.01, clip 1.0; bf16 | card only, leaning |
| tokens a step | the card's "64 micro × 4 accum × 2048" is 524,288; with its 8 × H100, 8 × 524,288 = 4,194,304, and 9,536 × 4,194,304 = 40.0B, the card's total (524k alone gives 5.0B) | **leaning 4.19M**: arithmetic on the card, no training log |

**What this does to T2's mapping.** The break window is A5's registered (128, 512] over 9
seeds (7 at 512, 2 at 256: `trackA_force/status-A.md` "A5 units 2 and 3"), not seed 0's
256–512 (corrected by `/challenge-pr` on #283, finding 1). Pythia: 2,097,152 tokens a step, LR
3e-4 over a 1,430-step warmup; Dodecahedron's warmup linear from 0. This table is the one home
of these crossing windows (the checkpoints *inside* each); the grid dates each schedule
*predicts*, which add the checkpoint just past a window's top (step's 574, cumulative's 141) and
so overlap at 141, are `design-T2.md`'s outcome table (`/challenge-pr` on #284, finding 3).

| reading | Dodecahedron steps | checkpoints inside |
|---|---|---|
| cumulative LR | (31, 125] | 35, 40, 47, 55, 65, 75, 88, 103, 121 |
| step | (128, 512] | 141, 165, 193, 225, 263, 308, 359, 420, 491 |
| tokens at 4.19M a step (leaning) | (64, 256] | 65 … 121 and 141 … 225: **overlaps both rows above** |
| tokens at 524k a step (the card's figure) | (512, 2048] | 574, 671, 785, 917, 1072, 1253, 1465, 1713, 2002 |

**What T2 can separate.** Cumulative LR from step under either batch (the windows abut at
125 / 128). Tokens only under 524k a step: at 4.19M, Dodecahedron takes twice Pythia's tokens
a step, so the token window is the step window one octave down and covers half of each of the
other two. So the batch decides whether T2 tests tokens at all (Blocked 118).
`-pilot-dense` covers steps 1–160 at every step, finer in the LR window, but it is a different
run with no stated schedule.

### Q4. `timaeus/gpt-2-small` — **architecture confirmed; data, schedule and LLC unknown**

GPT-NeoX (`config.json`): 12 layers, 768, 12 heads, rotary 0.25, parallel residual, vocab 50,304:
**Pythia-160m's shape, not GPT-2's**; initialised from `gpt-2-small-init-seed-5`; a copy of
`georgeyw/gpt-2-small` (2024-03-14). Checkpoints every 100, steps 100–65,000 (650);
`-log-spacing` 430 checkpoints to step 800,000. Data, LR and batch sit in `training_args.bin`
(a pickle, not opened). No card; no published LLC on it found.

### Q5. No published LLC along training on any Pythia model — **stands, leaning**

2402.03698 (withdrawn, merged into 2308.12108): deep linear networks only. 2606.22389v2: 3-layer
MLPs on MNIST, probes started at each checkpoint with a Gaussian tether, stated as not stationary
LLC estimates; code `github.com/yingjiacai/SIVE`. 2507.21449: a 4-layer attention-only model (Q1),
not Pythia. The 2024 review: LLC "scaled to models with billions of parameters", no model or
curve named. Still a search result: unindexed or unpublished work is not ruled out.

### Parked (why / cost / which decision it could change)

1. The `L0`–`L6` series (attention-only, d_model 256, `pile13m`, constant LR 1e-3, **no warmup**,
   75,000 steps, checkpoints dense early then every 300; the `H*`, `dm*` repos vary heads and
   width) / forward passes on 3M models / a break here could not be the warmup, which C2's F1
   cannot separate on Pythia. Carried to ROADMAP §7b.
2. `gpt-2-small` / its schedule needs the pickle read in isolation (`python -I`, own directory),
   then forward passes at 160m / a Pythia-architecture run on a different schedule. Carried to
   ROADMAP §7b.
3. `L4`'s one-head LLC curve as a partial positive control for a refined LLC / T3's cost / only
   if T4 goes per-head. Carried to ROADMAP §7b (T3).

## Corrections received
