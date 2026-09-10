<!-- p7e_consolidation/status-7e.md -->
# Phase 7e — STATUS

**Last verified:** 2026-09-10 (opened this day).
**Overall:** the phase's gate measurement is **already run and it answered**.
Consolidation of the aligned core is viable; `L11H14` cannot be folded in. No
surgery has been performed yet. Nothing is registered and nothing can be —
pythia-410m is spent under `check_registry` rule 3, **and** every measurement
after the first surgery would be on a model this project modified, which is a
second and independent bar. Restore checks exact (`0.0e+00`) on every run below.

Read `design-7e.md` for why the phase exists and what would falsify it, and
`PROJECT.md` §3.12-V for the numbers in their 7d context.

## What is answered

### The capacity question — **energy says no, usefulness says yes**

`ambient_budget.py`, steps 16000 and 143000. One head's OV is **rank ≤ 64**.

| | 16000 | 143000 |
|---|---|---|
| ambient participation ratio | 22.0 | 8.4 |
| joint effect, own basis, 90 % | 186 dims | 141 dims |
| joint effect, **ambient ordering**, 90 % | **355** | **255** |
| energy inside ambient top-64 | 0.732 | 0.780 |

**This refuted the phase's own opening argument.** The design predicted the
effect would concentrate in the ambient trunk because the ambient PR is ~22; it
does not — 90 % needs **355** ambient directions, so the set writes substantially
into directions the residual stream barely uses. **Private low-variance
bandwidth, not the shared trunk.** That also explains how members can be
near-orthogonal to one another while all being induction heads.

Only **73–78 %** of the joint energy fits one head's budget, so the phase fails
the energy criterion — and survives, because **energy is not usefulness**.

### Useful rank — **the set has two classes, not a gradient**

`useful_rank.py`, step 16000, 16 sequences, matched-norm random controls. `r*`
is the smallest OV rank recovering 90 % of that head's own causal effect.

| member | effect `dNLL` | **`r*`** | control `r*` |
|---|---|---|---|
| `L7H8` | +1.107 | **1** | 48 |
| `L12H5` | +0.431 | **1** | 48 |
| `L8H9` | +0.127 | **2** | 48 |
| `L5H2` | +2.227 | **12** | 24 |
| `L8H6` | +0.219 | **24** | 48 |
| `L11H14` | +0.187 | **64** | *never* |

**`L7H8` — the head §3.11–§3.12 was built on — recovers 97 % of its causal
effect from a single direction.** The five low-rank members' useful ranks sum to
**40, inside one head's 64-dim budget**, and since those five are also 7d's
mutually *aligned* ones (δ-cos 0.7–0.9) their union is smaller still.
**Consolidation of the aligned core is viable.**

### `L11H14` is anti-ordered, measured directly

`useful_rank.py --bottom` keeps the **smallest** `r` directions instead of the
largest — the worst rank-`r` approximation by gain, run precisely for that.

| | `r*` | top-1 | bottom-32 | bottom-48 |
|---|---|---|---|---|
| `L7H8` | 1 | **+0.971** | +0.041 | +0.095 |
| `L5H2` | 12 | +0.123 | +0.109 | +0.616 |
| `L11H14` | 64 | **−0.096** | **+0.717** | **+0.846** |

For `L11H14` the ordering is **bottom-`r` > matched-random > top-`r` at every
rank**, and at `r = 1` recovery is **negative**: keeping its top singular
direction is *worse than deleting the head*. The Eckart-Young optimal
approximation — provably the best approximation of the **operator** — is the
**worst** of the three at preserving the **function**.

`L7H8` is the textbook contrast, so **the inversion is specific, not an
instrument failure**. `L5H2` sits with `L7H8`: gain-ordered, just less
concentrated. **`L11H14` differs in kind, not degree**, and this is the sixth
independent axis separating it (see `design-7e.md`).

**Its orthogonality is load-bearing: the `L11H14`-last test of `design-7e.md` is
answered without surgery.**

## The hold this places on an instrument elsewhere

**Any SVD-ordered rank truncation misleads on `L11H14`-like heads** — including
`tools/run/induction_rank_sweep.py`'s entire `r*` construction, whose `svd` basis
is one of its two. **Do not quote an `svd`-basis `r*` for a head that has not
been checked with `--bottom`.** The `schur` basis orders by **eigenvalue** rather
than gain and carries a **sign**, so it may not inherit the defect. That
comparison is **weights-only, costs no forward passes, and is the cheapest open
action in the phase.**

## What is open

1. **`schur` vs `svd` on `L11H14`** — free, and it decides whether an existing
   instrument is compromised. Do this first.
2. **OV-core principal angles between members' top-`r*` subspaces.** Converts
   "40 ≤ 64, probably fine" into an actual union number. 7d's alignment is
   between *effects in the residual stream*; this is between *OV directions in
   the 64-dim core*, and they are not the same thing. Weights-only.
3. **The core consolidation itself** — five members into one survivor. Rank is
   **necessary, not sufficient**: the survivor must reproduce those effects
   through its **own** QK pattern at its **own** layer, which no rank arithmetic
   captures. `design-7e.md` states the objective, the two principled orders, and
   the four required checks (probe NLL, **natural-text loss**, a flattened
   interaction matrix, restore-exactness).
4. **`r*` across checkpoints.** Step 16000 only so far, and 7d showed the
   geometry moves sharply between 5000 and 143000. `r*` is grid-resolved:
   `r* = 12` means "between 8 and 12".
5. **Why `L11H14` is full-rank and anti-ordered** — now the most interesting
   single question here.

## Reproducing

```bash
cd /run/media/system/WDS_500/Mets && source .venv/bin/activate
export HF_HOME=$PWD/data/hf HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=4

python -u p7e_consolidation/ambient_budget.py                     # ~10 min
python -u p7e_consolidation/useful_rank.py --top 6 --seqs 16 \
        --chunk 2 --controls 3 --step 16000                       # ~35 min
python -u p7e_consolidation/useful_rank.py --heads L11H14,L7H8,L5H2 \
        --seqs 16 --chunk 2 --controls 1 --bottom \
        --out data/analysis/useful_rank_bottom.json               # ~25 min
```

**`--chunk 2` is not optional at 16 sequences on this box.** These runners were
killed for memory twice at `--chunk 4` while an unrelated ~5 GB job was
resident. Outputs are git-ignored and written incrementally.
