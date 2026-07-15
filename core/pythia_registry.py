"""
core/pythia_registry.py — Pythia (GPT-NeoX) checkpoint registry.

New module, additive to core/config.py (transition plan v2, item 5: Pythia
model support). Kept separate so the existing MODEL_CONFIGS build-out in
core/config.py doesn't need to be hand-edited per checkpoint — one import
and one merge line does it:

    from core.pythia_registry import build_pythia_model_configs
    MODEL_CONFIGS.update(build_pythia_model_configs())

Every Pythia entry is a *checkpoint*, not a separate model in the
GPT-2/BERT/ALBERT sense: same HF repo id, different `revision`. This is
why the registry entries below carry both `hf_repo` and `revision` keys
that the existing GPT-2/BERT/ALBERT entries don't have.

load_model needs one small addition to use them — it isn't made here
because core/models.py's current body wasn't available to edit directly
(only its test stub was). The change is:

    repo_id  = cfg.get("hf_repo", model_name)   # fall back: existing behavior
    revision = cfg.get("revision")               # None for every non-Pythia entry
    model     = cfg["model_class"].from_pretrained(repo_id, revision=revision)
    tokenizer = cfg["tokenizer_class"].from_pretrained(repo_id, revision=revision)

`revision=None` is accepted by `from_pretrained` (means "main branch"), so
this is a no-op for every existing registry key — nothing about non-Pythia
loading changes.
"""

from transformers import GPTNeoXModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Checkpoint schedules (plan v2, "Checkpoint schedule — pilot first, anchors
# second, adaptive slots reserved")
# ---------------------------------------------------------------------------

# Pythia's own published checkpointing convention: log-spaced through step
# 512, then every 1000 steps to 143000 (final). This is the full set;
# PYTHIA_410M_PILOT_STEPS and PYTHIA_1_4B_ANCHOR_STEPS below are subsets of it.
PYTHIA_LOG_SPACED_STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
PYTHIA_LINEAR_STEPS     = list(range(1000, 143001, 1000))
PYTHIA_ALL_STEPS        = PYTHIA_LOG_SPACED_STEPS + PYTHIA_LINEAR_STEPS

# Step A — dense pilot on Pythia-410M. Plan: "20-30 checkpoints ... all the
# log-spaced early steps plus a spread through late training." Provisional —
# item 8 (pilot sweep) is what actually locates the transitions; this list
# is the input to that step, not its output.
PYTHIA_410M_PILOT_STEPS = sorted(set(
    PYTHIA_LOG_SPACED_STEPS
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

PYTHIA_410M_REPO = "EleutherAI/pythia-410m"
PYTHIA_1_4B_REPO = "EleutherAI/pythia-1.4b"


def _revision_for_step(step: int) -> str:
    return f"step{step}"


def _pythia_entry(hf_repo: str, step: int) -> dict:
    return {
        "model_class":     GPTNeoXModel,
        "tokenizer_class": AutoTokenizer,
        "is_albert":       False,
        "random_init":     False,
        "hf_repo":         hf_repo,
        "revision":        _revision_for_step(step),
        "checkpoint_step": step,
    }


def build_pythia_model_configs() -> dict:
    """
    MODEL_CONFIGS-format entries for every Pythia-410M pilot checkpoint,
    every Pythia-1.4B anchor checkpoint, and the norm-matched random
    baseline.

    Keys: "pythia-410m-step{N}", "pythia-1.4b-step{N}", "pythia-1.4b-random".
    """
    cfgs = {}

    for step in PYTHIA_410M_PILOT_STEPS:
        cfgs[f"pythia-410m-step{step}"] = _pythia_entry(PYTHIA_410M_REPO, step)

    for step in PYTHIA_1_4B_ANCHOR_STEPS:
        cfgs[f"pythia-1.4b-step{step}"] = _pythia_entry(PYTHIA_1_4B_REPO, step)

    # Two-baseline policy (plan, "Two random baselines, not one"):
    # norm-matched randomization of the FINAL checkpoint — the continuity
    # control for Blog 1's trained-vs-random contrast — kept as a distinct
    # object from true step-0 init (already covered by the
    # "pythia-1.4b-step0" entry above).
    #
    # Assumes "norm_matched" is a valid scheme string in the existing
    # randomize_weights (mirrors how gpt2-large-random must already be
    # registered, per Blog 1's "random weights norm-matched to trained
    # values" description) — not verified directly, since
    # randomize_weights's body wasn't available to check. Worth confirming
    # before the replication gate (item 6) runs.
    final_step    = PYTHIA_1_4B_ANCHOR_STEPS[-1]
    random_entry  = _pythia_entry(PYTHIA_1_4B_REPO, final_step)
    random_entry["random_init"]        = True
    random_entry["random_init_scheme"] = "norm_matched"
    cfgs["pythia-1.4b-random"] = random_entry

    return cfgs
