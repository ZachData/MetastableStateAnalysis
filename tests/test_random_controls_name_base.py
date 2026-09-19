"""
tests/test_random_controls_name_base.py — every `random_init` entry in
MODEL_CONFIGS must name the architecture it re-initialises.

`core.models.load_model` resolves the Hub repo as
`cfg.get("hf_repo", cfg.get("pretrained_name", model_name))`. A random
control whose entry carries neither key therefore asks the Hub for a repo
named after its own registry key — `gpt2-large-random` — which does not
exist, and the whole model is skipped inside `run_1`'s per-model handler
with a misleading protobuf error. That is how CLAIM-C's reference-random
arm turned out never to have loaded (2026-09-17, `PROJECT.md` §3.36).

Smoke tier: `core.config` imports the real transformers classes, and the
pure tier stubs it with `MODEL_CONFIGS = {}`, which would make this vacuous.

    SMOKE_REAL_DEPS=1 pytest -m smoke tests/test_random_controls_name_base.py -v
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.smoke


def _repo_id(name: str, cfg: dict) -> str:
    # Mirrors core.models.load_model's resolution exactly.
    return cfg.get("hf_repo", cfg.get("pretrained_name", name))


def test_every_random_control_resolves_to_a_base_architecture():
    from core.config import MODEL_CONFIGS

    random_entries = {k: v for k, v in MODEL_CONFIGS.items()
                      if v.get("random_init", False)}
    assert random_entries, "the registry lost all its random controls"

    for name, cfg in random_entries.items():
        repo = _repo_id(name, cfg)
        assert repo != name, (
            f"{name}: no hf_repo/pretrained_name, so load_model would ask "
            f"the Hub for a repo called {name!r}")
        # The base must be a trained entry of the same class, so the control
        # re-initialises exactly the architecture it is a control for.
        trained = [k for k, v in MODEL_CONFIGS.items()
                   if not v.get("random_init", False)
                   and _repo_id(k, v) == repo
                   and v["model_class"] is cfg["model_class"]]
        assert trained, (
            f"{name}: base repo {repo!r} is not the repo of any trained "
            f"{cfg['model_class'].__name__} entry in the registry")


def test_random_controls_map_agrees_with_base_repos():
    """`RANDOM_CONTROLS` (trained → control) must agree with the controls'
    own base repos, or `--random-baseline` pairs a control with the wrong
    trained model."""
    from core.config import MODEL_CONFIGS, RANDOM_CONTROLS

    for trained, control in RANDOM_CONTROLS.items():
        assert _repo_id(control, MODEL_CONFIGS[control]) == \
            _repo_id(trained, MODEL_CONFIGS[trained]), (trained, control)
