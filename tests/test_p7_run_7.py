"""
tests/test_p7_run_7.py — the Phase 7 driver.

Every test here builds a Phase 1 run directory and a Phase 2 weights
directory from scratch and drives `run_7.main` over them. No model is
loaded and no network is touched: the tokenizer is substituted at
`run_7.load_tokenizer`, which exists as a named function for exactly this
reason, and the OV circuits, projectors, activations and attention tensors
are small synthetic arrays in the shapes Phase 2 and Phase 1 write.

The subject is the refusals as much as the output. This driver stands
between four artifacts that agree about nothing structurally — an
activation array on the L2 sphere, an attention tensor indexed by layer, a
per-layer OV decomposition and a projector file — and the failure mode that
matters is not a crash but a table of plausible numbers built from a
mismatched join. Each refusal below is checked by constructing the
mismatch, not by asserting the message.
"""

import json

import numpy as np
import pytest

from core.artifacts import get_spec
from core.interactions import InteractionTable
from p7_motifs import run_7
from p7_motifs.run_7 import (
    RunRefused,
    _parse_prompt_arg,
    layer_input_index,
    raw_activations,
)

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pyproject.toml [tool.pytest.ini_options].markers.
pytestmark = pytest.mark.pure

MODEL = "pythia-410m-step1000"
D_MODEL, N_LAYERS, N_HEADS = 6, 2, 2


class FakeTokenizer:
    """Word-level, stable ids, plain-dict encoding — the same shape
    tests/test_battery_structure.py's fake uses."""

    bos_token_id = None
    name_or_path = "fake"

    def __init__(self):
        self._vocab = {}

    def __call__(self, text):
        return {"input_ids": [self._vocab.setdefault(w, len(self._vocab))
                              for w in text.split()]}


#: A text with genuine induction structure under a word-level tokenizer:
#: repeated bigrams, several distinct offsets, and non-induction repeats to
#: populate the same-content null.
USABLE_TEXT = (
    "alpha beta gamma delta alpha beta epsilon zeta alpha beta gamma eta "
    "theta iota alpha beta kappa lambda gamma delta mu nu alpha beta xi "
    "omicron gamma delta pi rho alpha beta sigma tau gamma delta upsilon"
)
DEGENERATE_TEXT = ". . . . . . . . . . . . . . . . . . . ."


def _write_phase1(tmp_path, key, n_tokens, *, keep_embedding=True,
                  n_states=None, with_norms=True, n_layers=N_LAYERS,
                  n_heads=N_HEADS, n_attn_tokens=None, seed=0):
    rng = np.random.default_rng(seed)
    run = tmp_path / "p1" / f"{MODEL}_{key}"
    run.mkdir(parents=True, exist_ok=True)

    if n_states is None:
        n_states = n_layers + 1 if keep_embedding else n_layers
    A = rng.standard_normal((n_states, n_tokens, D_MODEL))
    norms = np.linalg.norm(A, axis=-1)
    payload = {"activations": A / norms[..., None]}
    if with_norms:
        payload["norms"] = norms
    np.savez_compressed(run / "activations.npz", **payload)

    json.dump({"hidden_state_0_is_embedding": bool(keep_embedding),
               "final_hidden_state_is_post_ln": False},
              open(run / "geometry.json", "w"))

    w = n_attn_tokens or n_tokens
    att = np.tril(np.abs(rng.standard_normal((n_layers, n_heads, w, w))))
    att /= att.sum(-1, keepdims=True)
    np.savez_compressed(run / "attentions.npz", attentions=att)
    return run


def _write_phase2(tmp_path, *, n_layers=N_LAYERS, n_heads=N_HEADS, seed=1):
    rng = np.random.default_rng(seed)
    p2 = tmp_path / "p2"
    p2.mkdir(parents=True, exist_ok=True)
    lnames = [f"layer_{i}" for i in range(n_layers)]
    json.dump({"is_per_layer": True, "layers": lnames, "model": MODEL},
              open(p2 / f"ov_summary_{MODEL}.json", "w"))
    np.savez_compressed(
        p2 / f"ov_weights_{MODEL}.npz",
        **{f"ov_head{h}_{ln}": rng.standard_normal((D_MODEL, D_MODEL)) / D_MODEL
           for ln in lnames for h in range(n_heads)})
    proj = {}
    for ln in lnames:
        Q, _ = np.linalg.qr(rng.standard_normal((D_MODEL, D_MODEL)))
        Zp, Zn = Q[:, :D_MODEL // 2], Q[:, D_MODEL // 2:]
        proj[f"schur_attract_{ln}"] = Zp @ Zp.T
        proj[f"schur_repulse_{ln}"] = Zn @ Zn.T
    np.savez_compressed(p2 / f"ov_projectors_{MODEL}.npz", **proj)
    return p2


@pytest.fixture
def battery(monkeypatch):
    """The driver reads prompt text from core.config.PROMPTS; substituting
    it keeps these tests independent of the committed battery, which is the
    author's and changes for reasons that have nothing to do with this
    module."""
    # `import core.config as cfg`, not `import core.config`: the stubbed
    # session injects the module into sys.modules directly, so the `core`
    # package carries no `config` attribute for the dotted form to reach.
    import core.config as cfg

    prompts = {"usable_prompt": USABLE_TEXT, "degenerate_prompt": DEGENERATE_TEXT}
    monkeypatch.setattr(cfg, "PROMPTS", prompts, raising=False)
    # MODEL_CONFIGS is {} under the stub, so the registry lookup has to be
    # supplied here for the checkpoint_step threading to be observable.
    monkeypatch.setattr(cfg, "MODEL_CONFIGS",
                        {MODEL: {"checkpoint_step": 1000,
                                 "hf_repo": "EleutherAI/pythia-410m"}},
                        raising=False)
    monkeypatch.setattr(run_7, "load_tokenizer", lambda model: FakeTokenizer())
    return prompts


def _n_tokens(text):
    return len(FakeTokenizer()(text)["input_ids"])


def _run(tmp_path, p2, runs, *extra):
    argv = ["--p2-dir", str(p2), "--model", MODEL, "--sign-channel", "schur",
            "--out", str(tmp_path / "out")]
    for key, path in runs:
        argv += ["--prompt", f"{key}={path}"]
    return run_7.main(argv + list(extra))


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_writes_a_table_that_satisfies_its_registered_contract(
            self, tmp_path, battery):
        """The artifact contract was written in 2026-08-22 before any
        producer existed, precisely so the first producer could be checked
        against it rather than the other way round."""
        n = _n_tokens(USABLE_TEXT)
        run = _write_phase1(tmp_path, "usable_prompt", n)
        p2 = _write_phase2(tmp_path)

        assert _run(tmp_path, p2, [("usable_prompt", run)]) == 0

        t = InteractionTable.load(tmp_path / "out" / "interaction_table.npz")
        spec = get_spec("phase7", "interaction_table")
        assert not [k for k in spec.required_keys if k not in t.columns]
        assert len(t) > 0

    def test_every_layer_and_head_is_represented(self, tmp_path, battery):
        n = _n_tokens(USABLE_TEXT)
        run = _write_phase1(tmp_path, "usable_prompt", n)
        p2 = _write_phase2(tmp_path)
        _run(tmp_path, p2, [("usable_prompt", run)])

        t = InteractionTable.load(tmp_path / "out" / "interaction_table.npz")
        assert sorted(set(t.columns["layer"].tolist())) == list(range(N_LAYERS))
        assert sorted(set(t.columns["head"].tolist())) == list(range(N_HEADS))

    def test_pair_types_are_populated_from_token_identity(self, tmp_path, battery):
        """A table where every edge is `neither` is what a broken tokenizer
        produces, and it is indistinguishable from a real result at the
        shape level — P-I1 and P-I3 both read induction pairs off this
        column."""
        n = _n_tokens(USABLE_TEXT)
        run = _write_phase1(tmp_path, "usable_prompt", n)
        p2 = _write_phase2(tmp_path)
        _run(tmp_path, p2, [("usable_prompt", run)])

        t = InteractionTable.load(tmp_path / "out" / "interaction_table.npz")
        kinds = set(np.unique(t.columns["pair_type"]).tolist())
        assert "induction" in kinds
        assert kinds - {"induction", "strict", "same_content", "neither"} == set()

    def test_checkpoint_step_is_threaded_from_the_registry_onto_every_edge(
            self, tmp_path, battery):
        n = _n_tokens(USABLE_TEXT)
        run = _write_phase1(tmp_path, "usable_prompt", n)
        p2 = _write_phase2(tmp_path)
        _run(tmp_path, p2, [("usable_prompt", run)])

        t = InteractionTable.load(tmp_path / "out" / "interaction_table.npz")
        assert set(t.columns["checkpoint_step"].tolist()) == {1000}

    def test_retention_records_the_cutoff(self, tmp_path, battery):
        """An absent edge is not a zero-force edge, so the thinning has to
        be recoverable from the file itself."""
        n = _n_tokens(USABLE_TEXT)
        run = _write_phase1(tmp_path, "usable_prompt", n)
        p2 = _write_phase2(tmp_path)
        _run(tmp_path, p2, [("usable_prompt", run)], "--top-k-per-target", "3")

        t = InteractionTable.load(tmp_path / "out" / "interaction_table.npz")
        assert t.retention is not None
        assert int(t.retention["k"]) == 3


# ---------------------------------------------------------------------------
# The rotational channel: absent, and recorded as absent
# ---------------------------------------------------------------------------

class TestRotationalChannelAbsence:

    def test_real_and_imag_are_nan_not_zero(self, tmp_path, battery):
        """status-7.md finding 2: "no projector supplied" and "no component
        in that channel" must not collapse. A 0.0 here would read as a
        measured absence of rotation."""
        n = _n_tokens(USABLE_TEXT)
        run = _write_phase1(tmp_path, "usable_prompt", n)
        p2 = _write_phase2(tmp_path)
        _run(tmp_path, p2, [("usable_prompt", run)])

        t = InteractionTable.load(tmp_path / "out" / "interaction_table.npz")
        for col in ("real_frac", "imag_frac"):
            assert np.isnan(np.asarray(t.columns[col], dtype=float)).all()

    def test_the_manifest_says_the_channel_was_never_supplied(
            self, tmp_path, battery):
        n = _n_tokens(USABLE_TEXT)
        run = _write_phase1(tmp_path, "usable_prompt", n)
        p2 = _write_phase2(tmp_path)
        _run(tmp_path, p2, [("usable_prompt", run)])

        man = json.load(open(tmp_path / "out" / "manifest.json"))
        blob = json.dumps(man)
        assert "rotational_channel" in blob and "absent" in blob


# ---------------------------------------------------------------------------
# Admissibility
# ---------------------------------------------------------------------------

class TestPromptAdmissibility:

    def test_a_degenerate_prompt_is_skipped_and_named(self, tmp_path, battery):
        n_ok, n_bad = _n_tokens(USABLE_TEXT), _n_tokens(DEGENERATE_TEXT)
        ok = _write_phase1(tmp_path, "usable_prompt", n_ok)
        bad = _write_phase1(tmp_path, "degenerate_prompt", n_bad, seed=5)
        p2 = _write_phase2(tmp_path)

        assert _run(tmp_path, p2, [("usable_prompt", ok),
                                   ("degenerate_prompt", bad)]) == 0

        t = InteractionTable.load(tmp_path / "out" / "interaction_table.npz")
        assert set(np.unique(t.columns["prompt_key"]).tolist()) == {"usable_prompt"}
        man = json.load(open(tmp_path / "out" / "manifest.json"))
        assert "degenerate_prompt" in json.dumps(man)

    def test_a_run_with_no_usable_prompt_fails(self, tmp_path, battery):
        """A null result from a battery that carried no structure is a
        tokenizer artifact, not a finding."""
        n_bad = _n_tokens(DEGENERATE_TEXT)
        bad = _write_phase1(tmp_path, "degenerate_prompt", n_bad)
        p2 = _write_phase2(tmp_path)
        assert _run(tmp_path, p2, [("degenerate_prompt", bad)]) == 1
        assert not (tmp_path / "out" / "interaction_table.npz").exists()

    def test_an_unknown_battery_key_is_refused(self, tmp_path, battery):
        n = _n_tokens(USABLE_TEXT)
        run = _write_phase1(tmp_path, "usable_prompt", n)
        p2 = _write_phase2(tmp_path)
        assert _run(tmp_path, p2, [("not_a_prompt", run)]) == 1


# ---------------------------------------------------------------------------
# Frame resolution
# ---------------------------------------------------------------------------

class TestRawActivations:

    def test_norms_are_multiplied_back(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((3, 5, D_MODEL))
        norms = np.linalg.norm(A, axis=-1)
        out = raw_activations({"activations": A / norms[..., None], "norms": norms})
        assert np.allclose(out, A)

    def test_a_run_without_norms_is_refused(self):
        """The raw stream is unrecoverable for these artifacts. A unit-norm
        stand-in makes every force magnitude wrong by a per-token factor
        and nothing about the shapes would say so."""
        with pytest.raises(RunRefused, match="norms"):
            raw_activations({"activations": np.zeros((2, 3, D_MODEL)),
                             "norms": None, "run_dir": "somewhere"})

    def test_the_driver_refuses_a_run_without_norms(self, tmp_path, battery):
        n = _n_tokens(USABLE_TEXT)
        run = _write_phase1(tmp_path, "usable_prompt", n, with_norms=False)
        p2 = _write_phase2(tmp_path)
        assert _run(tmp_path, p2, [("usable_prompt", run)]) == 1


class TestLayerInputIndex:

    def test_embedding_kept_indexes_directly(self):
        for layer in range(N_LAYERS):
            assert layer_input_index(layer, N_LAYERS + 1, N_LAYERS, False) == layer

    def test_embedding_stripped_shifts_by_one(self):
        assert layer_input_index(1, N_LAYERS, N_LAYERS, True) == 0

    def test_layer_zero_is_unavailable_when_the_embedding_was_stripped(self):
        """Refused rather than started at layer 1: a formation curve
        missing its first layer and one that has it are not the same
        measurement."""
        with pytest.raises(RunRefused, match="embedding"):
            layer_input_index(0, N_LAYERS, N_LAYERS, True)

    def test_a_depth_disagreement_is_refused(self):
        with pytest.raises(RunRefused, match="stored state"):
            layer_input_index(5, N_LAYERS + 1, N_LAYERS, False)

    def test_the_driver_refuses_a_stripped_run_at_layer_zero(
            self, tmp_path, battery):
        n = _n_tokens(USABLE_TEXT)
        run = _write_phase1(tmp_path, "usable_prompt", n, keep_embedding=False)
        p2 = _write_phase2(tmp_path)
        assert _run(tmp_path, p2, [("usable_prompt", run)]) == 1

    def test_an_unrecorded_convention_stops_the_run(self, tmp_path, battery):
        """run_2d's rule, one phase over: the convention is read from the
        artifact or the run stops. Exit 2 marks it as a precondition the
        caller can fix by re-extracting, not a bad argument."""
        n = _n_tokens(USABLE_TEXT)
        run = _write_phase1(tmp_path, "usable_prompt", n)
        json.dump({}, open(run / "geometry.json", "w"))
        p2 = _write_phase2(tmp_path)
        assert _run(tmp_path, p2, [("usable_prompt", run)]) == 2


# ---------------------------------------------------------------------------
# Joins between artifacts that do not know about each other
# ---------------------------------------------------------------------------

class TestJoinRefusals:

    def test_a_layer_count_disagreement_is_refused(self, tmp_path, battery):
        n = _n_tokens(USABLE_TEXT)
        run = _write_phase1(tmp_path, "usable_prompt", n, n_layers=1,
                            n_states=N_LAYERS + 1)
        p2 = _write_phase2(tmp_path, n_layers=N_LAYERS)
        assert _run(tmp_path, p2, [("usable_prompt", run)]) == 1

    def test_a_head_count_disagreement_is_refused(self, tmp_path, battery):
        n = _n_tokens(USABLE_TEXT)
        run = _write_phase1(tmp_path, "usable_prompt", n, n_heads=N_HEADS + 1)
        p2 = _write_phase2(tmp_path, n_heads=N_HEADS)
        assert _run(tmp_path, p2, [("usable_prompt", run)]) == 1

    def test_a_token_count_disagreement_is_refused(self, tmp_path, battery):
        """Two artifacts from different tokenizations. The attention matrix
        would still index — it is simply the wrong prompt's."""
        n = _n_tokens(USABLE_TEXT)
        run = _write_phase1(tmp_path, "usable_prompt", n, n_attn_tokens=n + 3)
        p2 = _write_phase2(tmp_path)
        assert _run(tmp_path, p2, [("usable_prompt", run)]) == 1

    def test_a_missing_attention_tensor_is_refused(self, tmp_path, battery):
        n = _n_tokens(USABLE_TEXT)
        run = _write_phase1(tmp_path, "usable_prompt", n)
        (run / "attentions.npz").unlink()
        p2 = _write_phase2(tmp_path)
        assert _run(tmp_path, p2, [("usable_prompt", run)]) == 1

    def test_a_missing_phase2_decomposition_is_refused(self, tmp_path, battery):
        n = _n_tokens(USABLE_TEXT)
        run = _write_phase1(tmp_path, "usable_prompt", n)
        p2 = _write_phase2(tmp_path)
        (p2 / f"ov_weights_{MODEL}.npz").unlink()
        assert _run(tmp_path, p2, [("usable_prompt", run)]) == 1


class TestPromptArgument:

    def test_key_and_directory_must_both_be_given(self):
        import argparse
        for bad in ("results/somewhere", "=dir", "key="):
            with pytest.raises(argparse.ArgumentTypeError):
                _parse_prompt_arg(bad)

    def test_a_well_formed_pair_parses(self):
        key, path = _parse_prompt_arg("homer_iliad=results/run")
        assert key == "homer_iliad" and str(path) == "results/run"
