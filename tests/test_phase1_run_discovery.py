"""Regression tests for p1_io.find_phase1_run_dir under checkpoint-suffixed
model names. Pythia registry keys differ only by a numeric step suffix, so
prefix matching resolves pythia-410m-step1 to a step1000/step128/step16 run
and returns it silently — the wrong checkpoint, not a missing one."""
import pytest
from pathlib import Path

from p1_mstate_tracking.p1_io import find_phase1_run_dir

PROMPT = "wiki_paragraph"


def _make(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "layer_metrics.json").write_text("{}")
    return d


@pytest.mark.parametrize("wanted,decoys", [
    ("pythia-410m-step1",   ["pythia-410m-step1000", "pythia-410m-step128",
                             "pythia-410m-step16"]),
    ("pythia-410m-step2",   ["pythia-410m-step256", "pythia-410m-step2000"]),
    ("pythia-1.4b-step8",   ["pythia-1.4b-step8000"]),
    ("pythia-1.4b-step16",  ["pythia-1.4b-step16000"]),
    ("pythia-1.4b-step64",  ["pythia-1.4b-step64000"]),
    ("gpt2",                ["gpt2-medium", "gpt2-large", "gpt2-xl"]),
    ("gpt2-large",          ["gpt2-large-random"]),
])
def test_exact_checkpoint_wins_over_prefix_neighbours(tmp_path, wanted, decoys):
    for name in decoys:                      # decoys written first (older)
        _make(tmp_path, f"{name}_{PROMPT}")
    _make(tmp_path, f"{wanted}_{PROMPT}")
    assert find_phase1_run_dir(tmp_path, wanted, PROMPT).name == f"{wanted}_{PROMPT}"


@pytest.mark.parametrize("wanted,decoys", [
    ("pythia-410m-step1", ["pythia-410m-step1000"]),
    ("gpt2",              ["gpt2-medium"]),
])
def test_prefix_neighbour_alone_warns(tmp_path, wanted, decoys):
    """The requested run is absent. Returning a neighbour is the pre-v2
    fallback and is allowed, but must not be silent."""
    for name in decoys:
        _make(tmp_path, f"{name}_{PROMPT}")
    with pytest.warns(UserWarning, match="no exact run directory"):
        find_phase1_run_dir(tmp_path, wanted, PROMPT)


def test_albert_iteration_tag_still_resolves(tmp_path):
    _make(tmp_path, f"albert-xlarge-v2_48iter_{PROMPT}")
    d = find_phase1_run_dir(tmp_path, "albert-xlarge-v2", PROMPT)
    assert d is not None and "48iter" in d.name


def test_sublayer_stream_not_returned_for_base_model(tmp_path):
    """@attn/@ffn dirs are supplementary streams; a base-model lookup that
    silently lands on one would analyse the wrong residual."""
    _make(tmp_path, f"gpt2-large_attn_{PROMPT}")
    _make(tmp_path, f"gpt2-large_{PROMPT}")
    assert find_phase1_run_dir(tmp_path, "gpt2-large", PROMPT).name == f"gpt2-large_{PROMPT}"