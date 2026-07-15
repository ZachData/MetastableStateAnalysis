# Smoke tier — status and how to extend it

## What's here now

- `conftest.py`: `SMOKE_REAL_DEPS` env gate on stub installation, plus
  `_register_smoke_models`, `tiny_phase1_dir`, `tiny_phase2_dir` fixtures.
- `pytest.ini`: `smoke` marker registered.
- `test_phase1_smoke.py`, `test_phase2_smoke.py`: real, runnable (pending the
  open question below), asserting on real files.

Run: `SMOKE_REAL_DEPS=1 pytest -m smoke -v`. Needs network once, to pull
`hf-internal-testing/tiny-random-gpt2` (a few hundred KB); cached after.

## Confirmed vs. inferred

Confirmed by reading the actual files in this project:
- `hf-internal-testing/tiny-random-gpt2` and
  `hf-internal-testing/tiny-random-GPTNeoXForCausalLM` exist on the HF Hub
  (checked directly, not assumed).
- Phase 1's run-directory stem (`{model_name.replace('/','_')}_{prompt_key}`)
  and phase 2's `_find_run_dir`/`_run_stem` use the exact same format —
  verified in `run_1.py` lines 231/275/424 and `run_2.py` lines 405/409/417.
  So the tiny model's `/` (from the HF org prefix) round-trips correctly;
  this was worth checking rather than assuming, since every existing
  `MODEL_CONFIGS` key ("gpt2", "bert-base-uncased", ...) has no slash and
  would not have exercised this path before.
- Phase 1 does write `layer_metrics.json` (not just `.csv`) — confirmed in
  `io_utils.py`, since phase 2 gates on that exact file existing.

Not confirmed, because `core/models.py`'s real implementation wasn't in the
provided files (only its test-stub double, which has no `load_model`):
- Whether `load_model(model_name)` does the plain
  `cfg["model_class"].from_pretrained(model_name)` these fixtures assume.
  Every existing registry key is a valid HF repo id for its `model_class`,
  which makes this a reasonable inference, not a verified fact. If
  `load_model` does something per-architecture-specific beyond that, the
  `_register_smoke_models` fixture needs a matching adjustment.
- The exact filename `reporting.save_verdict` writes — `test_phase2_smoke.py`
  globs for `*verdict*` rather than an exact name for this reason; tighten
  it once the real function is visible.

First time either test actually runs is also the first time these get
resolved for real — that's the point of the smoke tier.

## Recommendation: skip phase 3 and `low_rank_ae.py` for now

The v2 plan freezes both as "candidate for deletion; git history is the
archive," with no real work happening on them "in the meantime." A smoke
test is real work — something to write, keep passing, and maintain. Writing
one for code slated for deletion cuts against the freeze's own rationale.
Suggest explicitly excluding these two from the "one per phase" smoke
requirement rather than silently skipping them — worth confirming rather
than assuming either way.

## Extending to the remaining phases

Entrypoint signatures already found by reading each `run_N.py` (not
guessed):

| Phase | Entrypoint | Needs live model? | Needs prior phase's output? |
|---|---|---|---|
| 1b (`p1b_hemisphere`) | `run_1b.run_all(...)` | yes | no |
| 2i (`p2b_imaginary`) | functions in `run_2i.py`, no single `run_all` seen — `load_ov_data`/`load_activations` read from disk | no (offline) | phase 1 + phase 2 dirs |
| 2c (`p2c_churchland`) | `run_2c.run_c1/run_c2/run_c3(...)`, `load_model_and_tokenizer` | yes | phase 1 dir (partially) |
| 4 (`p4_mstate_features`) | `run_4.run_track1(...)` | unconfirmed | phase 1 + phase 2 + phase 3 dirs |
| 5 (`p5_single_mstate_analysis`) | functions imported into `run_5.py` (`select_cluster`, `cluster_profile`, ...), no bare `run_all` seen | no (offline) | phase 1 + phase 2 dirs, `build_global_projectors` from `p6_subspace` |
| 5b (`p5b_manifold_steering`) | `run_5b.main(argv=None)` | unconfirmed | unconfirmed |
| 6 (`p6_subspace`) | functions imported into `run_6.py`, no bare `run_all` seen | no (offline) | phase 1 + phase 2 dirs |

Pattern to follow, in this order (matches the dependency column above):

1. `1b` next — same shape as phase 1/2 already built: real model, own
   output, no cross-phase lookup. Copy the `tiny_phase1_dir` fixture
   pattern directly.
2. `2i`, `5`, `6` after — offline consumers of phase 1 + phase 2 dirs
   already produced by the fixtures that exist now. No new model-loading
   question, just wiring `tiny_phase1_dir`/`tiny_phase2_dir` into whatever
   each module's actual offline entrypoint turns out to need — the
   functions above are named from imports, not verified call signatures,
   so a quick read of each before wiring is worth it.
2. `2c`, `4`, `5b` last — each has an open "needs live model?" or "needs
   phase 3?" question that phase 3 being frozen makes moot for `4`
   specifically (its phase-3-dependent code path presumably gets skipped
   or stubbed, not exercised) and unconfirmed for the other two.

Every one of these still needs the same treatment given to phase 2 here:
read the actual `run_N.py` function body (not just its signature) before
writing the fixture, the same way `_find_run_dir`'s stem format and the
`layer_metrics.json` filename were checked here rather than assumed.
