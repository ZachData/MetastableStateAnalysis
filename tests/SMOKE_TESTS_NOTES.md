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

## The first real run (2026-08-30)

The tier had never been executed outside CI. Running it settled the open
questions above and found four defects, none of which any other tier could
see. Result now: **36 passed, 0 failed**, ~19 s, on Fedora / Python 3.14.7
with torch 2.13.0+cpu and transformers 4.57.6.

Resolving the "not confirmed" list first: `load_model` does do the plain
`cfg["model_class"].from_pretrained(model_name)` these fixtures assume — with
`attn_implementation="eager"` added, see below — so `_register_smoke_models`
needed no adjustment, and the `*verdict*` glob matched.

What it found:

* **`core/rope.py:112` silently reported the wrong rotary geometry.**
  `float(getattr(cfg, "rotary_pct", 1.0))` — transformers 5 moved GPTNeoX's
  rotary geometry into `config.rope_parameters`, so under 5.x the default
  fires and Pythia's card reports `rotary_ndims = 64` where the model rotates
  16. The docstring four lines above it says "Never assume rotary_pct == 1.0
  ... assuming full rotary silently changes every downstream number." Pinned
  `transformers<5` rather than taught `rope_parameters`; lifting the pin
  requires fixing `core/rope.py` first.
* **`output_attentions=True` returns a tuple of `None`.** Modern transformers
  defaults these architectures to `sdpa`, which does not materialise the
  attention matrix. Not `None` — a tuple whose every element is `None`, so a
  guard of the form `if out.attentions is not None` passes and the indexing
  after it raises. `core/models.py`'s docstring predicted this precisely
  ("when that shim is removed ... Phase 1's entire sinkhorn/Fiedler/entropy
  family would go quiet without raising"); 4.57 removed the shim. `load_model`
  already pinned eager, but `core/lm_loading.py` (both paths, the checkpoint
  sweep among them), `p2d_io.py` and this tier's own fixture did not. All now
  go through `core.models.from_pretrained_eager`.
* **`GPTNeoXAttention` no longer exposes `num_attention_heads`.** Three sites
  in `p2_eigenspectra/weights.py` and one in `head_ablation.py` read it off
  the module directly. `core/pythia_weights` had already solved this with
  `_attn_geometry`'s module → config → weight-shape walk; those four call
  sites had simply never been switched over. They are now.
* **`_is_torch_tensor` was truthy for everything under the test stub.**
  `torch.is_tensor(x)` on the conftest's MagicMock torch returns a MagicMock,
  so `_to_numpy` took its torch branch on plain ndarrays and died on
  `.detach()`. This made `split_qkv_from_layer`'s comment — "_to_numpy is
  idempotent on ndarrays, so calling it again inside split_qkv_gptneox is
  harmless" — false in exactly the isolated tier, and it stayed hidden only
  because no caller had yet passed an ndarray through both. Now an
  `isinstance` check against a real class, the same shape of fix as
  `TestTorchStubIsScipySafe`.

Two findings that are recorded rather than fixed:

**`hidden_states` and forward hooks part company under transformers 5.**
`test_core_intervention_smoke.py` asserts that zeroing block 0's output changes
`activations[1]`, and its comment argues why. That is **correct** on 4.57 and
false on 5.16.1, where a forward hook replacing a block's output leaves that
block's own `hidden_states` entry at the pre-hook value while changing every
later one. Verified with raw transformers and a plain `register_forward_hook`,
no project code involved. The test is right for the pinned version and was left
alone; it is a third reason the pin is the right call.

**`run_model_with_hook` has no production caller.** Audited on the back of the
above: every reference outside `core/intervention.py` is a docstring or a test.
The p7 steering and patching gates take `activations: np.ndarray` and never
load a model or touch `hidden_states` — they are all `pure` tier. So the
off-by-one has no blast radius today, and the hazard is entirely prospective:
it lands the first time a real intervention run happens, which is the pilot.

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
