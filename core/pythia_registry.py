"""
core/pythia_registry.py — MODEL_CONFIGS entries for the Pythia checkpoint
schedule.

Almost every entry is a real published checkpoint. The one exception is
`pythia-1.4b-random`, built by `build_pythia_random_baseline`.

History worth keeping, because the wrong fix was applied once already: an
earlier entry named a "norm_matched" scheme that `core.models` did not
implement, so it raised on load — outside `run_full`'s handler, killing the
whole sweep. The response was to delete the entry and treat the published
step-0 checkpoint as the untrained-weights object. That silently collapsed
two objects the project's own documents say cannot be collapsed:

  - **step 0** — the developmental origin. GPT-NeoX's *own* init, at its own
    variance scaling. A point on the training trajectory, and the subject of
    PREDICTIONS.md claim (a).
  - **`pythia-1.4b-random`** — the final checkpoint's weights, structure
    destroyed, per-parameter Frobenius norm preserved. Not on the training
    trajectory at all. The continuity control for Blog 1's trained-vs-random
    contrast, and the subject of PREDICTIONS.md claim (c), which carries a
    hard stop.

`design-5c.md` (§"consequences", item 2) states the construction as
"norm-matched, not fresh-init" and explains why the two can't be merged;
`p1_mstate_tracking/visualization/checkpoints.py::family_baselines` resolves
them as separate slots and gives them separate plot styles (`STEP0_STYLE`
vs `RANDOM_BASELINE_STYLE`); `checkpoint_scalars.compute_distance_from_random`
returns `{}` for any family missing the `-random` slot. Deleting the entry
therefore did not remove the dependency, it just made every
distance-from-random figure render empty without saying why.

The scheme is implemented now (`core.models.randomize_weights`,
scheme="norm_matched"). The crash-scope defect is separate and still open —
`run_2.py` calls `randomize_weights` and `analyze_weights` outside the
per-model `try`, so any exception there still costs the queued sweep. See
`p2_eigenspectra/ISSUES_p2.md` item A3; do not treat this docstring as
evidence that path is safe.

`pythia-1.4b-random` deliberately carries no `checkpoint_step` and does not
match `checkpoints._STEP_RE`, so it is excluded from the step axis and can
only be drawn via the baseline slot.

Weight revision vs tokenizer revision
-------------------------------------
`revision` pins the weights to a training step. `tokenizer_revision` is
separate and defaults to None (main) because Pythia's tokenizer is byte
identical across every checkpoint branch — it is the same GPT-NeoX BPE
vocabulary at step 0 and step 143,000. Pinning it bought nothing and cost
two things: a redundant tokenizer fetch on each of the 37 checkpoints in a
full sweep, and a hard failure on any branch whose tokenizer files are
absent, which would surface as "model {name} failed to load" with no
indication that the weights were fine.
"""

from transformers import GPTNeoXModel, AutoTokenizer


# ---------------------------------------------------------------------------
# Checkpoint schedules
# ---------------------------------------------------------------------------

# Step A — 410M pilot. Log-spaced through the first 512 steps (where the
# plan expects the sharpest structural change), then a coarse linear sweep.
PYTHIA_410M_PILOT_STEPS = sorted(set(
    [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    + list(range(1000, 20001, 2000))
    + [40000, 60000, 80000, 100000, 120000, 143000]
))

# Step B — anchored 1.4B schedule (plan table, provisional rationale column).
# The 2-3 reserved adaptive slots are deliberately NOT included here — they
# get appended once item 8 locates the sharpest inter-checkpoint change;
# adding placeholder steps now would misrepresent them as decided.
PYTHIA_1_4B_ANCHOR_STEPS = [0, 8, 256, 512, 1000, 2000, 8000, 16000, 64000, 143000]

# Expensive-tier anchors (plan: "four checkpoints, 1.4B only"), held
# provisional until the cheap tier locates the transitions — see plan text
# for the one insertion explicitly kept open between 1,000 and 16,000.
PYTHIA_1_4B_EXPENSIVE_STEPS = [0, 1000, 16000, 143000]

# Every step EleutherAI actually published, for validating the above.
PYTHIA_ALL_STEPS = sorted(set(
    [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    + list(range(1000, 143001, 1000))
))

PYTHIA_410M_REPO = "EleutherAI/pythia-410m"
PYTHIA_1_4B_REPO = "EleutherAI/pythia-1.4b"


def _revision_for_step(step: int) -> str:
    return f"step{step}"


def _pythia_entry(hf_repo: str, step: int) -> dict:
    return {
        "model_class":        GPTNeoXModel,
        "tokenizer_class":    AutoTokenizer,
        "is_albert":          False,
        "random_init":        False,
        "hf_repo":            hf_repo,
        "revision":           _revision_for_step(step),
        # Explicit None, not an omitted key: load_model falls back to
        # `revision` when the key is absent, which is right for any future
        # model whose tokenizer genuinely varies by revision. Pythia's does
        # not, so it says so.
        "tokenizer_revision": None,
        "checkpoint_step":    step,
    }


# The trained checkpoint the random control is matched against. This is the
# *final* checkpoint, not step 0: the control asks "what does a structureless
# model at the trained model's scale do", so the scale it borrows has to be
# the trained one. Matching to step 0 would make it a second copy of the
# developmental-origin object.
PYTHIA_RANDOM_MATCH_STEP = 143000


def build_pythia_random_baseline(hf_repo: str = PYTHIA_1_4B_REPO,
                                 name: str = "pythia-1.4b-random") -> dict:
    """
    The norm-matched random control for the 1.4B family.

    Loads the final trained checkpoint, then `randomize_weights` overwrites
    every parameter with a Gaussian draw rescaled to that parameter's trained
    Frobenius norm. `run_full` applies this after `load_model` when
    `random_init` is set.

    `checkpoint_step` is absent on purpose. This object is not a point on the
    training trajectory, and giving it a step would place it on the step axis
    of every checkpoint figure.

    The seed must match the seed used for the corresponding Phase 1 run, or
    the OV decomposition will not correspond to the activations it is being
    cross-referenced against. `run_1`/`run_2` both thread `--random-init-seed`
    for this reason; the multi-seed aggregate path
    (`checkpoint_scalars.compute_distance_from_random`'s `random_agg`) expects
    several such runs under the same name.
    """
    return {
        "model_class":        GPTNeoXModel,
        "tokenizer_class":    AutoTokenizer,
        "is_albert":          False,
        "random_init":        True,
        "random_init_scheme": "norm_matched",
        "hf_repo":            hf_repo,
        "revision":           _revision_for_step(PYTHIA_RANDOM_MATCH_STEP),
        "tokenizer_revision": None,
    }


def build_pythia_model_configs() -> dict:
    """
    MODEL_CONFIGS-format entries for every Pythia-410M pilot checkpoint and
    every Pythia-1.4B anchor checkpoint.

    Keys: "pythia-410m-step{N}", "pythia-1.4b-step{N}", and the single
    non-checkpoint entry "pythia-1.4b-random".
    """
    cfgs = {}

    for step in PYTHIA_410M_PILOT_STEPS:
        cfgs[f"pythia-410m-step{step}"] = _pythia_entry(PYTHIA_410M_REPO, step)

    for step in PYTHIA_1_4B_ANCHOR_STEPS:
        cfgs[f"pythia-1.4b-step{step}"] = _pythia_entry(PYTHIA_1_4B_REPO, step)

    # The second baseline object. Named to satisfy
    # `family_baselines("pythia-1.4b", ...)["random"]`, which matches
    # "{base}-random" exactly and refuses cross-size substitution — so a
    # 410M family still correctly reports no random control rather than
    # borrowing this one.
    cfgs["pythia-1.4b-random"] = build_pythia_random_baseline()

    return cfgs