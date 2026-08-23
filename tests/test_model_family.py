"""Tests for core/model_family.py — the single architecture-detection idiom.

The bug this replaces: analysis_p1._is_causal_model used startswith() over
("gpt2","pythia","gpt-neox","gptneox") while plots.analyze_value_eigenspectrum
used substring containment. They agreed on registry keys and disagreed on
repo-prefixed names, so the smoke checkpoint took the GPT-NeoX branch for V
extraction and the non-causal branch for sinkhorn in the same run.
"""
import pytest

from core.model_family import (
    model_family, is_causal_model, is_albert, is_bert, is_gpt2, is_gptneox,
)

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pyproject.toml [tool.pytest.ini_options].markers.
pytestmark = pytest.mark.pure


@pytest.mark.parametrize("name,family", [
    ("albert-base-v2",                                    "albert"),
    ("albert-base-v2-random",                             "albert"),
    ("albert-xlarge-v2@48iter",                           "albert"),
    ("bert-base-uncased",                                 "bert"),
    ("bert-large-uncased",                                "bert"),
    ("gpt2",                                              "gpt2"),
    ("gpt2-xl",                                           "gpt2"),
    ("gpt2-large-random@attn",                            "gpt2"),
    ("pythia-410m-step0",                                 "gptneox"),
    ("pythia-1.4b-step143000",                            "gptneox"),
    ("EleutherAI/pythia-1.4b",                            "gptneox"),
    ("hf-internal-testing/tiny-random-gpt2",              "gpt2"),
    ("hf-internal-testing/tiny-random-GPTNeoXForCausalLM","gptneox"),
    ("some-unregistered-model",                           None),
])
def test_family_resolution(name, family):
    assert model_family(name) == family


def test_albert_beats_bert():
    """'albert' contains 'bert'; marker order must resolve it to albert."""
    assert is_albert("albert-base-v2")
    assert not is_bert("albert-base-v2")


@pytest.mark.parametrize("name", [
    "gpt2", "gpt2-medium", "pythia-410m-step512",
    "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
])
def test_causal(name):
    assert is_causal_model(name)


@pytest.mark.parametrize("name", [
    "bert-base-uncased", "albert-xlarge-v2", "some-unregistered-model",
])
def test_not_causal(name):
    assert not is_causal_model(name)


def test_smoke_checkpoint_agrees_across_both_call_sites():
    """The regression itself: causal detection and V-branch selection must
    resolve the same name to the same family. Before core/model_family.py
    this checkpoint was causal=False and gptneox=True simultaneously."""
    name = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
    assert is_gptneox(name) and is_causal_model(name)


def test_case_and_separator_insensitive():
    assert model_family("GPT_NeoX-Base") == "gptneox"
    assert model_family("GPT2-Large") == "gpt2"