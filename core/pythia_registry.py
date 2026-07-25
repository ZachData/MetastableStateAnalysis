"""
core/pythia_registry.py — MODEL_CONFIGS entries for the Pythia checkpoint
schedule.

Every entry is a real published checkpoint. There is deliberately no
randomized baseline here: the entry that once existed named a
"norm_matched" scheme that core.models.randomize_weights does not
implement and never did, so it raised on load — outside run_all's handler,
killing the sweep. The published step-0 checkpoint is the untrained-weights
object now. See PREDICTIONS.md prediction (c), which needs restating
against step 0.

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


def build_pythia_model_configs() -> dict:
    """
    MODEL_CONFIGS-format entries for every Pythia-410M pilot checkpoint and
    every Pythia-1.4B anchor checkpoint.

    Keys: "pythia-410m-step{N}", "pythia-1.4b-step{N}".
    """
    cfgs = {}

    for step in PYTHIA_410M_PILOT_STEPS:
        cfgs[f"pythia-410m-step{step}"] = _pythia_entry(PYTHIA_410M_REPO, step)

    for step in PYTHIA_1_4B_ANCHOR_STEPS:
        cfgs[f"pythia-1.4b-step{step}"] = _pythia_entry(PYTHIA_1_4B_REPO, step)

    return cfgs