<!-- HANDOFF.md -->
# HANDOFF — session of 2026-08-31

Written to be pasted into, or read at the start of, a fresh session. It records
what changed, what was measured, what is still open, and where everything
lives. Every number below is measured on this machine; where a first reading
was wrong it says so, because two of them were.

Branch: `claude/rescaler-cache-identity-test`, five commits ahead of
`a7ce1c2`. Nothing is merged and no PR is open.

---

## 1. The machine

| | |
|---|---|
| Repo | `/mnt/mets` (fuse share, ~55 GB free — **not** where large output should go) |
| Bulk volume | `/mnt/vm_storage` (~150 GB free; symlinked as `~/vm_storage`) |
| venv | `/mnt/mets/.venv` — torch 2.13.0+cpu, transformers 4.57.6, numpy 2.5.2, scipy 1.18.1 |
| CPU / RAM | 10 cores, 19 GB |

Two environment variables matter and neither is set by default:

```bash
export HF_HOME=/mnt/vm_storage/hf_cache          # 19 GB of checkpoints already cached
export METS_RESULTS_DIR=/mnt/vm_storage/mets_results   # added this session; default is ./results
```

`transformers` is pinned `<5` deliberately. On 5.x, GPT-NeoX moved rotary
parameters into `config.rope_parameters` and `core/rope.py`'s
`rotary_pct` default then fires, silently reporting `rotary_ndims=64` where
pythia-410m rotates 16. Lifting the pin requires fixing `core/rope.py` first.

**`/mnt/mets/results/` holds 315 GB of pre-refactor runs.** Nothing was
deleted. Reclaiming it is the author's call. All new output goes to
`/mnt/vm_storage/mets_results`.

---

## 2. What was committed

| commit | what |
|---|---|
| `aa437ec` | Registry gained `CLAIM-B`'s registered sweep — `pythia-410m-step54000` had no `MODEL_CONFIGS` entry, so the repo could not build the grid its own gate requires |
| `017f749` | Two gates that could not fire (see §3) |
| `2cabaec` | `p7_motifs/run_7.py` — the Phase 7 driver, build step 8's first half |
| `3558b70` | `p7_motifs/formation_curve.py` — build step 8's second half |
| `dd6b0de` | Phase 2 decomposed in float64 |

Gate went 2115 → **2183 passed / 5 skipped / 30 deselected**; tier 3 569
passed / 2 skipped, unchanged; tier 0 clean. Recorded in `docs/CI_BASELINE.md`.

---

## 3. Defects found, with how they were found

Each was invisible to the existing tests, and each is now mutation-checked
(revert the fix, the new test fails by name).

1. **`tokenize_prompt` returned two strings, not token ids.** The test was
   `isinstance(enc, dict)`, and transformers' `BatchEncoding` extends
   `UserDict`, not `dict` — so `list(enc)` iterated the mapping's KEYS and
   every prompt tokenized to `["input_ids", "attention_mask"]`. The stubbed
   session's fake returns a plain dict, so no test could take the branch
   production took.
2. **`check_prompt_admissible` read keys nothing writes.** It consulted
   `degeneracy` / `degeneracy_modes`; `analyze_prompt` emits `flags` and
   `verdict`. The one refusal between a degenerate prompt and a motif rate
   returned `None` for every real report. Its three tests all passed a
   hand-made shape the repo never produces.
3. **Phase 2 decomposed in single precision.** `_extract_*` reads weights with
   torch's `.float()`, so `scipy.linalg.schur` returned vectors orthogonal
   only to `‖ZᵀZ − I‖ = 1.5e-5`, and `P = ZZᵀ` was non-idempotent at ~6e-6
   against `PROJECTOR_TOL = 1e-6`. `_as_basis` refused step 2 of the sweep and
   accepted steps 1 and 4 **on equally non-idempotent projectors** — `np.allclose`
   carries `rtol=1e-5` beside the atol, so which checkpoints failed was
   arbitrary. In float64: 3.2e-14 and 3.6e-09.
4. **A bug in `formation_curve.py`, caught before it shipped.** The head axis
   was intersected over the *sparse* relay maps, which drops every head with
   no relay at any checkpoint — precisely the heads that go on to form. On the
   real sweep that axis is empty. It now comes from the dense behavioural
   series.

### Two readings that were wrong and were corrected

- "float32 *storage* causes the projector failure" — no: an exact projector
  cast to float32 gives 5.7e-7, an order of magnitude *better* than what was on
  disk. The loss was in the computation.
- "first real forward pass in the project's history" — no: `results/` holds
  Phase 2 runs from 2026-08-03 and 2026-08-13 over 27 checkpoints. What had
  never run was **Phase 7**; no `interaction_table.npz` had ever existed.

---

## 4. The sweep — DONE

12/12 checkpoints of `REGISTERED_CLAIM_B_SWEEP`, 8 prompts each, zero
refusals. 11:08 → 18:36 on 2026-08-31, ~35 min per checkpoint, flat.

| | |
|---|---|
| Edges | 19,077,120 per checkpoint, **229M total** |
| Tables | `/mnt/vm_storage/mets_runs/p7/step{N}/interaction_table.npz` (~5 GB each) |
| Phase 1 / 2 dirs | recorded per checkpoint in `p1_dir.txt` / `p2_dir.txt` beside each table |
| Sweep script | `/mnt/vm_storage/mets_runs/sweep.sh` (resumable; skips a step whose table exists) |
| Curve analysis | `/mnt/vm_storage/mets_runs/curve.py` → `formation_curve_raw.json` |

Prompts: the 8 admissible ones. `short_heterogeneous` is refused by
`check_prompt_admissible` (20 tokens, 1 induction pair → `insufficient`).

---

## 5. What the data says

### The formation curve is a step function

```
step      1  2  4  8  16  32  64  128  256  512  1000        54000
relays    0  0  0  0   0   0   0    0    0    0     0    2,560,483
```

Excluding `repeated_tokens`: eleven zeros, then 1,008,553.

Not a bug. **No edge in eleven checkpoints reaches the 0.5 motif threshold.**

| step | max `attractive_frac` | median | edges ≥ 0.5 |
|---|---|---|---|
| 1 | 0.4861 | 0.3717 | 0 |
| 512 | 0.4665 | 0.2566 | 0 |
| 1000 | 0.4766 | 0.1029 | 0 |
| 54000 | **0.9364** | 0.2726 | **2,015,626** |

`dim(U_pos)/d` — what an isotropic force would score — is **0.501** at step 1
(the untrained OV spectrum splits ~50/50 like a random matrix), 0.351 at step
1000, 0.453 at step 54000. Early forces sit tightly *below* the isotropic
baseline; by step 54000 the max is far above it. The step is real.

### But the grid cannot locate it

The registered sweep was chosen (§6r) to maximise retention of **CLAIM-B's**
anchor window. For **P-I1** the same grid puts **11 of 12 points in the flat
zero**, resolving the transition only to *somewhere in (1000, 54000]* — a
53,000-step interval with no point inside it. This is §6r's own lesson
arriving on the other claim.

**P-I1's registered falsifier endpoints are not in the grid at all.**
`P_I1_ENDPOINT_STEPS = (0, 143000)`; the sweep runs 1 → 54000. Both absent, so
`endpoint_flags` reports on a first step that is not 0 and a last that is not
143000.

### `relay_owner` — the three choices, measured at step 54000

| choice | heads with relays | total relays |
|---|---|---|
| `tag_writer` | 68 | 2,560,483 |
| `matcher` | 80 | 2,560,483 |
| `both` | 87 | 5,120,966 |

Not rescalings of one another — the head sets differ.

### One prompt dominates

Per-prompt relays at step 54000: `repeated_tokens` **1,551,930** (61% of all),
`latex_monograph` 310,346, `homer_iliad` 166,132, `sullivan_ballou` 151,710,
`wiki_paragraph` 146,601, `camus_letranger` 81,876, `hdbscan_code` 78,024,
`paper_excerpt` 73,864.

`repeated_tokens` is ". . . ." × 265. It reads `usable` because
`n_distinct > 1` under this tokenizer, so the `uniform` flag never fires. Any
battery-averaged statistic is substantially a statement about that one prompt.
That is a registered criterion, so it was reported, not changed.

---

## 6. Open — the author's decisions

None of these were taken, because all four are pre-registered criteria.

1. **`relay_owner`** — `relay_strength` is keyed by
   `(layer_1, head_1, layer_2, head_2)` but `P_I1_UNIT` is the head. The
   collapse is a definition. Costs measured above.
2. **Checkpoints between 1000 and 54000**, to locate the transition. Roughly
   35 min and ~12 GB per added checkpoint.
3. **Whether P-I1 needs its own grid** including 0 and 143000, its registered
   falsifier endpoints.
4. **Whether `repeated_tokens` belongs in the battery** for this statistic.
   Excluding it at curve-assembly needs no re-run.

---

## 7. Blockers

**Hard blocker: the relay-count null does not exist.** `formation_gate`
requires the series to be the excess above the N1/N2 offset-null envelope.
`core/qk_offset_null.py` computes N1/N2 for the **QK antisymmetry statistic**,
not for relay counts. `formation_curve.assert_gate_ready` therefore refuses
the series, correctly — handing raw counts to `p_value_p_i1` would report a
p-value against a null the series never cleared. Constructing that null is a
null-construction decision; `claims/EVALUABILITY.md` prescribes the order
(floor → what the statistic degenerates on → what the grid contributes → only
then the control).

**Known defect, not yet fixed.** `find_relays` joins a `prev_token` edge to a
`match` edge **by particle position**, and positions are per-prompt token
indices. `run_7` writes all 8 prompts into one table, so calling `find_relays`
on a `run_7` artifact composes a tag written in one prompt with a match in
another. Measured at step 54000: **23,050,007 relays concatenated vs 2,560,483
summed per prompt — 9.0× inflation.** Nothing errors. All numbers in §5 are
the correct per-prompt ones. The fix belongs in `motif_alphabet.find_relays`
(group by `prompt_key`) rather than as a caller's obligation.

**Untouched, and named so it is not mistaken for done.**
`core/precision_policy.py`'s P2 (Pythia ships fp16; an fp16-epsilon
perturbation splits a genuinely real eigenvalue pair into a complex one) and
item 13 (the forward pass runs under bf16 autocast, so activations carry that
noise floor). The float64 change addresses neither.

---

## 8. Suggested next step

Fix `find_relays`'s cross-prompt join first — it is a real defect, it is
small, and every future relay number depends on it. Then put decisions 1–4 to
the author with the costs already measured above. The relay-count null is the
only thing standing between the finished sweep and P-I1's first p-value, and
it should be built in `EVALUABILITY.md`'s prescribed order rather than
started from the control.

### Reproducing anything

```bash
cd /mnt/mets && source .venv/bin/activate
export HF_HOME=/mnt/vm_storage/hf_cache METS_RESULTS_DIR=/mnt/vm_storage/mets_results

./scripts/check.sh gate     # 2183 passed / 5 skipped / 30 deselected
./scripts/check.sh all      # adds tier 3: 569 passed / 2 skipped

# one checkpoint, end to end
bash /mnt/vm_storage/mets_runs/sweep.sh      # resumable; skips completed steps
PYTHONPATH=/mnt/mets python /mnt/vm_storage/mets_runs/curve.py
```

`pythonpath = .` in `pytest.ini` applies to pytest only — a plain
`python script.py` needs `PYTHONPATH=/mnt/mets`.
