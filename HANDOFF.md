<!-- HANDOFF.md -->
# HANDOFF — session of 2026-09-01

Written to be pasted into, or read at the start of, a fresh session. It records
what changed, what was measured, what is still open, and where everything
lives. Every number below is measured on this machine.

Branch: `claude/rescaler-cache-identity-test`, three commits past the previous
session's `fe2ae6e`. Nothing is merged and no PR is open.

**A sweep is running.** See §6 before starting anything that needs the CPU or
the disk.

---

## 1. The machine

| | |
|---|---|
| Repo | `/mnt/mets` (fuse share — **not** where large output should go) |
| Bulk volume | `/mnt/vm_storage` (~247 GB free after the author cleared space; the running sweep needs ~84) |
| venv | `/mnt/mets/.venv` — torch 2.13.0+cpu, transformers 4.57.6, numpy 2.5.2, scipy 1.18.1 |
| CPU / RAM | 10 cores, 19 GB |

```bash
export HF_HOME=/mnt/vm_storage/hf_cache
export METS_RESULTS_DIR=/mnt/vm_storage/mets_results
export HF_HUB_OFFLINE=1        # every 410M registry revision is mirrored locally
export HF_HUB_DISABLE_XET=1
```

**All 33 of `PYTHIA_410M_STEPS` are mirrored into `HF_HOME` as
`model.safetensors`** (~1.62 GB each), so nothing streams. Take safetensors and
NOT `pytorch_model.bin` when adding revisions: the hub carries both, they are
1.62 GB against 0.91 GB, and mixing them across a grid varies precision between
checkpoints silently.

`transformers` is pinned `<5` deliberately; on 5.x GPT-NeoX moved rotary
parameters into `config.rope_parameters` and `core/rope.py`'s `rotary_pct`
default then fires, silently reporting `rotary_ndims=64` where pythia-410m
rotates 16. Lifting the pin requires fixing `core/rope.py` first.

`/mnt/mets/results/` still holds 315 GB of pre-refactor runs. Nothing was
deleted; reclaiming it is the author's call.

---

## 2. What was committed

| commit | what |
|---|---|
| `f7e95bc` | Every motif join confined to the context its positions belong to |
| `9655631` | The pairing null refuses when it permutes a constant, before the floats decide |
| `7196e24` | P-I1's own sweep grid, its `relay_owner` and its dominant prompt, registered |

Gate went 2183 → **2203 passed / 5 skipped / 30 deselected**. Tier 3 unchanged
at 569 passed / 2 skipped. Recorded in `docs/CI_BASELINE.md`.

---

## 3. Defects found and fixed, with how they were found

Each is mutation-checked: revert the fix and the named test fails.

### 3.1 Three motifs joined rows by position without grouping first

`target` and `source` are **per-prompt token indices**, and `run_7` writes all
8 battery prompts into one `InteractionTable`. Position 7 of one prompt and
position 7 of another are different particles. None of the three errored.

* **`find_relays`** composed a tag written in one prompt with a match found in
  another. The previous session measured 23,050,007 relays against the
  2,560,483 a per-prompt join gives — **9.0×**, entirely spurious. Now
  verified after the fix on the real step-54000 table:
  `find_relays(t)` over the whole battery returns **2,560,483**, equal to the
  per-prompt sum, and reproduces all eight per-prompt counts.
* **`hub_mask`** pooled in-degrees, and the leave-one-out baseline they are
  compared against, across prompts. Measured: **305,233 pooled against 437,508
  grouped**. The direction is *dilution* — pooling buried 30% of the real
  per-prompt hubs under a baseline the other prompts inflated. **That is not
  the direction predicted before measuring it**; the first docstring claimed
  manufacture, and was corrected to what the data says. Both directions are
  reachable and the oracle test plants the manufacture case, since it is the
  one a test can pin.
* **`mutual_mask`** could read 5←4 in one prompt and 4←5 in another as each
  other's reverse. On the same table the collision does not fire: **105,752
  either way**. Latent, fixed for correctness, with the collision planted in a
  test rather than waited for.

`RelayInstance` now carries `prompt_key` with **no default**, because
`tag_position` cannot be resolved back to a particle without it —
`relay_target_flags` was matching on position alone and flagged the particle at
that index in every prompt of the battery.

`hub_mask`'s grouping is one sort rather than one full-length boolean scan per
group; it needed that once the context entered the key.

### 3.2 The pairing null returned a p-value computed from rounding

`paired_colocation_arm`'s statistic is `-mean|ca - cb[p]|`, a sum over a
**permuted multiset**. If either side's change centroids are all equal, every
pairing reproduces the observation exactly: the null *is* the observation and
the attainable floor is 1.000. `EVALUABILITY.md`'s twenty-first lesson, reached
by this arm's own construction.

The guard was `float(stats.max()) == float(stats.min())`, checked after the
fact. Exact equality is the wrong test — floating-point summation order splits
a mathematically constant statistic across two adjacent doubles, so the guard
does not fire.

**This was live on the finished sweep.** Every head's relay change centroid is
`3.8664179398591134`, *exactly* equal, with no noise to break the tie, because
the series is zero at 11 of 12 checkpoints. Across 2001 sampled pairings the
statistic spanned **4.4e-16** and took two distinct values; p came out 0.764,
0.764, 0.778, 0.778, 0.784, 0.786 over six permutation seeds. The number was
the seed and the rounding, nothing else.

The check is now structural — either side constant, tested before the statistic
exists — and names the grid as the cause. The after-the-fact test survives as a
tolerance backstop at the floating-point scale of a mean over `n_units` terms,
~1e-13 here, ten orders below any effect the statistic can carry.

It is shared with CLAIM-B's arms, so the fix is not P-I1's alone.

---

## 4. The four decisions — registered

Put to the author with the costs measured, and registered on 2026-09-01. The
first was reframed by §3.2 before it was asked.

1. **P-I1's grid** — `REGISTERED_P_I1_SWEEP`, in
   `core/changepoint_colocation.py` beside CLAIM-B's. Nineteen steps:
   `0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000,
   32000, 54000, 143000`. A **superset** of the CLAIM-B sweep, so the twelve
   tables on disk are reused. `adjudicate_p_i1` now refuses every other grid,
   the division `adjudicate_claim_b` already makes; `p_value_p_i1` still
   computes on any.
2. **`P_I1_RELAY_OWNER = "matcher"`** — the stage-2 head, because the arm pairs
   this series against the behavioural induction score, which is mean attention
   on induction pairs and therefore the matcher's own behaviour.
3. **Endpoints** — steps 0 and 143000 added, so `P_I1_ENDPOINT_STEPS` are in
   the grid and `endpoint_flags` reports on the steps the registry named.
4. **`P_I1_DOMINANT_PROMPT = "repeated_tokens"`** — kept, since it is a
   registered battery member, with the excluding-it series carried beside the
   including-it one. Reported, never scored — the standing `endpoint_flags`
   has.

### Why the grid was the binding constraint, and the other three were not

The step function is real: no edge reaches the 0.5 motif threshold at any of
the eleven checkpoints from step 1 to step 1000, so every head's relay series
is exactly zero there and jumps once. Every head's change centroid is then the
midpoint of the single interval (1000, 54000].

**All three `relay_owner` choices give ONE distinct centroid** — 68, 80 and 87
heads alike. **Excluding `repeated_tokens` does not change it** — eleven zeros
either way. The units contribute one location, not 80, and no number of heads
or of prompts fixes that. Only intervals inside the transition do.

The five log-spaced fills take the interval midpoints inside (1000, 54000) from
**1 to 6**. 52 published checkpoints lie in the gap, so the count was a cost
decision and not an availability one.

---

## 5. What the data said (unchanged from the previous session)

Per-prompt relays at step 54000, now confirmed by the whole-table call:
`repeated_tokens` **1,551,930** (61%), `latex_monograph` 310,346,
`homer_iliad` 166,132, `sullivan_ballou` 151,710, `wiki_paragraph` 146,601,
`camus_letranger` 81,876, `hdbscan_code` 78,024, `paper_excerpt` 73,864.

| step | max `attractive_frac` | median | edges ≥ 0.5 |
|---|---|---|---|
| 1 | 0.4861 | 0.3717 | 0 |
| 1000 | 0.4766 | 0.1029 | 0 |
| 54000 | **0.9364** | 0.2726 | **2,015,626** |

`dim(U_pos)/d` — what an isotropic force would score — is 0.501 at step 1,
0.351 at step 1000, 0.453 at step 54000.

`relay_owner` costs at step 54000: `tag_writer` 68 heads / 2,560,483 relays,
`matcher` 80 / 2,560,483, `both` 87 / 5,120,966 (each relay counted twice).
The head **sets** differ; these are not rescalings.

---

## 6. The sweep that is running

Relaunched 14:51 on 2026-09-01, `nohup bash /mnt/vm_storage/mets_runs/sweep.sh`,
log at the session scratchpad's `sweep.log`. Seven new checkpoints —
0, 2000, 4000, 8000, 16000, 32000, 143000 — at ~35 min and ~12 GB each. It is
resumable and skips any step whose `interaction_table.npz` exists, so
re-running it is safe.

`sweep.sh` **reads its step list from `REGISTERED_P_I1_SWEEP`** rather than
restating it, so it cannot name a grid the gate refuses. It also **reuses a
complete phase-1 directory** for a model if one exists, since phase 1 is ~20
minutes and re-running it because a later stage failed is the expensive way to
be wrong.

### Two ways the first launch failed, both now guarded

Worth reading before touching the script, because neither announced itself.

1. **HuggingFace's Xet CDN (`us.aws.cdn.hf.co`) was unreachable** while
   `huggingface.co` answered in 0.5 s, so any check that did not hit the CDN
   specifically looked healthy. The seven new revisions were not cached; the
   twelve from the previous session were. `run_1` printed the download failure,
   then printed `Results in:` and exited **0**, so the script read a directory
   containing no prompts as a success and ran phase 2 and phase 7 on it. Fixed
   by mirroring every revision and running with `HF_HUB_OFFLINE=1`, so a reach
   for the network fails immediately instead of stalling.
2. **The guard added for (1) was itself wrong**, and cost three checkpoints at
   ~20 minutes each. It counted prompt directories with
   `ls -d ${P1}/${M}_*`, which also matches the per-prompt PNGs and so reports
   64 where it means 8. It now asks for the eight directories by name — which is
   what the phase-7 argument loop had always done. The three discarded phase-1
   runs were complete and were recovered by the reuse path rather than re-run.

The lesson worth carrying: a guard that rejects good input is not the safe
direction of a guard being wrong. Both failures here were silent in the sense
that mattered — one produced an empty artifact, the other threw away a good
one — and neither would have been visible in the final numbers.

When it finishes, re-run the curve analysis over all nineteen steps. The
previous session's `/mnt/vm_storage/mets_runs/curve.py` calls
`find_relays(t.filter(prompt_key=p))` per prompt — that workaround is no longer
needed (§3.1) and the whole-table call gives the same answer, but leaving it in
costs only time.

**The first thing to check when the tables land** is whether the degeneracy
clears: the six interval midpoints inside (1000, 54000) should give more than
one distinct change centroid across heads. If they do not — if every head still
jumps in the same interval — the transition is sharper than the fill resolves
and §4's decision 1 comes back with a denser grid.

---

## 7. The one remaining blocker

**The relay-count null does not exist.** `formation_gate` requires the series
to be the excess above the N1/N2 offset-null envelope, and
`core/qk_offset_null.py` computes N1/N2 for the **QK antisymmetry statistic**
(`a_frac`, from weights and offsets) — not for relay counts.
`formation_curve.assert_gate_ready` refuses the series, correctly: handing raw
counts to `p_value_p_i1` would report a p-value against a null the series never
cleared.

This is a null-construction decision, and `claims/EVALUABILITY.md` prescribes
the order: **compute the attainable floor, name what the statistic degenerates
on, check what the measurement grid contributes, and only then build the
control.** The first three steps changed the design before any control existed
on `P-AB1`, which is that document's case for the order.

Two of those steps are done for the *arm*, and both are in §3.2 and §4: the
statistic degenerates on a series whose change mass falls in one interval, and
the grid contributed the entire failure.

**Step 2 is now done for the relay count itself, and it constrains the null.**
Across the 8 battery prompts at step 54000, the raw relay count against the
prompt's own induction-pair supply (`n_induction` from
`core.battery_structure.analyze_prompt`) runs **r = +0.9958** — 99% of the
cross-prompt variance in the count is the prompt's combinatorics, not the
model's circuitry. Excluding `repeated_tokens` it is +0.8908. Nothing else is
close: n_tokens −0.39, n_same_content −0.36, n_distinct_tokens −0.79.

| prompt | tokens | n_induction | relays | relays/induction |
|---|---|---|---|---|
| `repeated_tokens` | 265 | **34,191** | 1,551,930 | 45.4 |
| `latex_monograph` | 446 | 2,873 | 310,346 | 108.0 |
| `homer_iliad` | 562 | 2,518 | 166,132 | 66.0 |
| `sullivan_ballou` | 482 | 2,038 | 151,710 | 74.4 |
| `wiki_paragraph` | 467 | 1,598 | 146,601 | 91.7 |
| `hdbscan_code` | 300 | 1,247 | 78,024 | 62.6 |
| `camus_letranger` | 465 | 963 | 81,876 | 85.0 |
| `paper_excerpt` | 286 | 554 | 73,864 | 133.3 |

This also settles why `repeated_tokens` carries 61% of the battery: ". . . ."
× 265 holds **34,191** induction pairs against the next prompt's 2,873, twelve
times as many, because every repeated token pairs with every other. Its share
is a fact about the prompt, not about the checkpoint — which is worth carrying
into §4's decision 4, since it means the excluding-it series is not a
robustness check but a different question.

**The constraint that falls out:** a relay-count null that does not hold
`n_induction` fixed per prompt is testing whether the prompt has induction
pairs, which is known before the model runs. That is `EVALUABILITY.md`'s "a
null that randomised more than the claim is about". What survives normalising
is not nothing — relays per induction pair still spans 45 to 133, a factor of
three — and that residue is where a formation signal would have to live.

What is still not done is step 1 (the attainable floor) and the control itself.
The obvious shape is a degree-preserving rewiring within each (context, layer,
head) that keeps each head's edge count and attractive fraction and randomises
which particles the edges connect — and it must now also preserve the induction
structure. That is a design choice of the class this repository puts to the
author, not one to register from the code.

**Do not start it from the control.** That is the order `EVALUABILITY.md` names
and the one nine previous passes were corrected by.

---

## 8. The artifact format, and what its compressibility says

**Edge tables are now written compressed** (`InteractionTable.save`,
`ParticleTable.save`). Measured on the step-54000 sweep table, 19,077,120
edges: **5.49 GB → 0.35 GB, 15×**. Across the 19-step grid that is 104 GB → 7
GB. `np.load` reads both encodings, so every table written before this keeps
loading unchanged; `tools/recompress_tables.py` rewrites the existing ones.

The rest of the repository already compressed — `p1_mstate_tracking/p1_io.py`,
`core/frame_card.py` — so this was the two tables that did not, not a new
policy.

### Why it compresses that hard, which is the part worth thinking about

The per-column ratios are a direct measure of how much of each column is
structure rather than signal:

| column | ratio | why |
|---|---|---|
| `layer` | 693× | 24 values, perfectly uniform |
| `checkpoint_step` | 686× | **one** value per table |
| `real_frac`, `imag_frac` | 686× | **entirely NaN** — see below |
| `model` | 295× | one value, stored as `<U21` per row |
| `pair_type` | 174× | 4 values, stored as `<U12` per row |
| `target` | 64× | token positions, 0–561 |
| `source` | 11× | token positions, more varied |
| `attractive_frac`, `repulsive_frac` | 4.5× | real measurements |
| `offset` | 7.9× | 512 values |
| **`weight`** | **1.9×** | post-softmax attention — nearly all information |

**A 5.49 GB table carries roughly 350 MB of measurement.** The other 94% is
identity columns repeated once per row plus an unpopulated channel. That is
not a defect — it is what a columnar format does with key columns — but it is
worth knowing before sizing storage or an instance.

### `real_frac` and `imag_frac` are NaN in every row of every table

100.0% NaN, 19,077,120 of 19,077,120, on the step-54000 table. **This is
deliberate and correctly recorded**, not a silent gap: `p7_motifs/run_7.py`'s
module docstring item 3 says the rotational channel is not wired there, `U_S`
/ `U_A` stay `None`, and the manifest records `rotational_channel: "absent"`
so a table missing the channel cannot be mistaken for one measured to have
none. `status-7.md` finding 2 is why NaN rather than 0.0.

What it costs, and what is worth looking into: **0.31 GB per table, 5.8 GB
across the grid, storing NaN.** Supplying the channel needs Phase 2b's
`extract_schur_blocks` output per checkpoint, which is a separate run;
`p7_io.rotational_channel_from_blocks` is the seam. Two questions follow, and
neither has been answered:

1. Does any registered prediction need the rotational channel? If one does,
   every table on disk is missing an input it requires, and the manifest flag
   is the only thing that says so.
2. If none does, the two columns are schema that no producer fills and no
   consumer reads, and `OPTIONAL_VALUE_COLUMNS` is carrying them on the
   strength of a plan rather than a use.

### The categorical option, not taken

Compression fixes disk and does nothing for RAM — a compressed table still
expands to 5.49 GB on load. Storing `model` / `prompt_key` / `pair_type` as
int8 category codes would take the in-memory table to **1.89 GB**, which is
what would let a 16 GB instance stop being tight. It changes the schema and
every consumer reading those columns as strings, so it is a deliberate piece
of work with its own tests, not a one-line change. Not done.

---

## 9. Untouched, and named so it is not mistaken for done

`core/precision_policy.py`'s **P2** (Pythia ships fp16; an fp16-epsilon
perturbation splits a genuinely real eigenvalue pair into a complex one) and
**item 13** (the forward pass runs under bf16 autocast, so activations carry
that noise floor). The previous session's float64 change addresses neither.

---

## 10. Reproducing anything

```bash
cd /mnt/mets && source .venv/bin/activate
export HF_HOME=/mnt/vm_storage/hf_cache METS_RESULTS_DIR=/mnt/vm_storage/mets_results

./scripts/check.sh gate     # 2203 passed / 5 skipped / 30 deselected
./scripts/check.sh all      # adds tier 3: 569 passed / 2 skipped

bash /mnt/vm_storage/mets_runs/sweep.sh      # resumable; skips completed steps
PYTHONPATH=/mnt/mets python /mnt/vm_storage/mets_runs/curve.py
```

Two records carry file hashes and must be rewritten whenever
`core/changepoint_colocation.py` or `p7_motifs/formation_gate.py` changes;
both re-derive their numbers, and neither moved this session:

```bash
python3 -m tools.dry_run_claim_b_p_i1 --write      # ~1 min
python3 -m tools.claim_b_grid_feasibility --write  # ~4 min
```

`pythonpath = .` in `pytest.ini` applies to pytest only — a plain
`python script.py` needs `PYTHONPATH=/mnt/mets`.
