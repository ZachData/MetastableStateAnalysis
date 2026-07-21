"""
tests/test_phase6_io.py

Covers the three structural mismatches between Phase 1 on-disk output
and Phase 6's expectations:

  1. core.io.load_phase1_run (missing module)
       - activations.npz loaded as (n_layers, n_tokens, d) ndarray
       - hdbscan_labels.json dict → list[ndarray] conversion
       - tokens.txt parsed to list[str]
       - optional attentions.npz loaded when present; None when absent
       - sparse label dict (missing layers) filled with noise-label arrays
       - corrupt labels file doesn't crash; returns None

  2. labels_per_layer type contract
       - list indexed by int, not string-keyed dict
       - length matches activations.shape[0]
       - each element is ndarray of length n_tokens

  3. build_context phase1 directory resolution
       - hyphen-named dirs found when stem is underscore form
       - prompt_key glob works across flat (non-nested) layout
       - falls back to any subdir when prompt_key has no match
       - no match → ctx keys absent (not KeyError)

Run:
    pytest tests/test_phase6_io.py -v
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt

sys.path.insert(0, str(Path(__file__).parents[1]))

from p1_mstate_tracking.p1_io import load_phase1_run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_phase1_run(
    base: Path,
    dirname: str,
    n_layers: int = 4,
    n_tokens: int = 8,
    d: int = 16,
    n_heads: int = 2,
    tokens: list[str] | None = None,
    write_attentions: bool = False,
    label_layers: list[int] | None = None,   # which layers get label arrays; None = all
    corrupt_labels: bool = False,
) -> Path:
    run_dir = base / dirname
    run_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)

    # activations.npz
    acts = rng.standard_normal((n_layers, n_tokens, d)).astype(np.float32)
    np.savez(run_dir / "activations.npz", activations=acts)

    # tokens.txt
    tok = tokens or [f"tok_{i}" for i in range(n_tokens)]
    with open(run_dir / "tokens.txt", "w") as f:
        for i, t in enumerate(tok):
            f.write(f"{i}\t{t}\n")

    # hdbscan_labels.json
    if corrupt_labels:
        (run_dir / "hdbscan_labels.json").write_text("{invalid json")
    else:
        layers_to_write = label_layers if label_layers is not None else list(range(n_layers))
        lab = {
            str(L): rng.integers(0, 2, size=n_tokens).tolist()
            for L in layers_to_write
        }
        with open(run_dir / "hdbscan_labels.json", "w") as f:
            json.dump(lab, f)

    # events.json
    events = {
        "merge_layers": [2],
        "energy_violations": {"1.0": [2, 3]},
    }
    with open(run_dir / "events.json", "w") as f:
        json.dump(events, f)

    # trajectory.json
    traj = {
        "plateau_layers": [0, 1],
        "trajectories": [{"id": 0, "chain": [[0, 0], [1, 0]]}],
    }
    with open(run_dir / "trajectory.json", "w") as f:
        json.dump(traj, f)

    # geometry.json
    geo = {"prompt": dirname.split("_")[-1], "model": "test-model",
           "n_layers": n_layers, "n_tokens": n_tokens, "d_model": d}
    with open(run_dir / "geometry.json", "w") as f:
        json.dump(geo, f)

    # attentions.npz (optional)
    if write_attentions:
        attn = rng.random((n_layers, n_heads, n_tokens, n_tokens)).astype(np.float32)
        np.savez(run_dir / "attentions.npz", attentions=attn)

    return run_dir


# ============================================================================
# 1 — load_phase1_run: basic contract
# ============================================================================

class TestLoadPhase1RunBasic(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp)
        self.run_dir = _make_phase1_run(Path(self._tmp), "albert-xlarge-v2_wiki")

    def test_activations_shape(self):
        """activations key returns (n_layers, n_tokens, d) ndarray."""
        p1 = load_phase1_run(self.run_dir)
        acts = p1["activations"]
        self.assertIsNotNone(acts)
        self.assertEqual(acts.ndim, 3)
        self.assertEqual(acts.shape, (4, 8, 16))

    def test_tokens_is_list_of_strings(self):
        """tokens key returns list[str] of length n_tokens."""
        p1 = load_phase1_run(self.run_dir)
        toks = p1["tokens"]
        self.assertIsInstance(toks, list)
        self.assertEqual(len(toks), 8)
        self.assertTrue(all(isinstance(t, str) for t in toks))

    def test_events_is_list(self):
        """events key returns a list (may be empty if events.json absent)."""
        p1 = load_phase1_run(self.run_dir)
        self.assertIsInstance(p1.get("events", []), list)

    def test_trajectories_is_list(self):
        p1 = load_phase1_run(self.run_dir)
        self.assertIsInstance(p1.get("trajectories", []), list)

    def test_attentions_none_when_file_absent(self):
        """attentions must be None when attentions.npz doesn't exist."""
        p1 = load_phase1_run(self.run_dir)
        self.assertIsNone(p1.get("attentions"))

    def test_attentions_loaded_when_file_present(self):
        run_dir = _make_phase1_run(
            Path(self._tmp), "albert-xlarge-v2_wiki_with_attn",
            write_attentions=True,
        )
        p1 = load_phase1_run(run_dir)
        attn = p1.get("attentions")
        self.assertIsNotNone(attn)
        self.assertEqual(attn.shape, (4, 2, 8, 8))


# ============================================================================
# 2 — hdbscan_labels type contract (the primary structural mismatch)
# ============================================================================

class TestHdbscanLabelsContract(unittest.TestCase):
    """
    Phase 1 writes hdbscan_labels.json as {str(layer_idx): [int, ...]}.
    load_phase1_run must convert this to list[ndarray] so that
    ctx["labels_per_layer"][L] (integer index) works in all Track B/D
    sub-experiments.
    """

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp)
        self.run_dir = _make_phase1_run(Path(self._tmp), "model_prompt")

    def test_labels_per_layer_is_list(self):
        """Must be a list, not a string-keyed dict."""
        p1 = load_phase1_run(self.run_dir)
        labels = p1.get("hdbscan_labels")
        self.assertIsNotNone(labels)
        self.assertIsInstance(labels, list)

    def test_list_length_matches_n_layers(self):
        p1 = load_phase1_run(self.run_dir)
        self.assertEqual(len(p1["hdbscan_labels"]), p1["activations"].shape[0])

    def test_each_element_is_ndarray(self):
        p1 = load_phase1_run(self.run_dir)
        for i, arr in enumerate(p1["hdbscan_labels"]):
            self.assertIsInstance(arr, np.ndarray, msg=f"layer {i} not ndarray")

    def test_each_element_length_matches_n_tokens(self):
        p1 = load_phase1_run(self.run_dir)
        n_tok = p1["activations"].shape[1]
        for i, arr in enumerate(p1["hdbscan_labels"]):
            self.assertEqual(len(arr), n_tok, msg=f"layer {i} wrong length")

    def test_integer_indexing_works(self):
        """The critical contract: ctx['labels_per_layer'][0] must not raise."""
        p1 = load_phase1_run(self.run_dir)
        labels = p1["hdbscan_labels"]
        # Integer indexing must succeed (would fail on a string-keyed dict)
        try:
            _ = labels[0]
            _ = labels[-1]
        except (TypeError, KeyError) as e:
            self.fail(f"Integer indexing failed: {e}")

    def test_sparse_labels_gap_filled_with_noise(self):
        """
        If hdbscan_labels.json is sparse (e.g. only layers 0 and 3 present
        out of 4), missing layers must be filled with noise-label arrays
        (all -1), not None — so downstream code never gets a None element.
        """
        run_dir = _make_phase1_run(
            Path(self._tmp), "sparse_run",
            n_layers=4, n_tokens=8,
            label_layers=[0, 3],      # layers 1 and 2 absent
        )
        p1 = load_phase1_run(run_dir)
        labels = p1["hdbscan_labels"]
        self.assertEqual(len(labels), 4)
        for i in (1, 2):
            arr = labels[i]
            self.assertIsInstance(arr, np.ndarray)
            npt.assert_array_equal(arr, np.full(8, -1, dtype=np.int32))

    def test_corrupt_labels_returns_none_not_exception(self):
        """Corrupt JSON must not crash; hdbscan_labels falls back to None."""
        run_dir = _make_phase1_run(
            Path(self._tmp), "corrupt_labels", corrupt_labels=True
        )
        try:
            p1 = load_phase1_run(run_dir)
        except Exception as e:
            self.fail(f"load_phase1_run raised on corrupt labels: {e}")
        # Either None or a list of noise arrays is acceptable
        labels = p1.get("hdbscan_labels")
        if labels is not None:
            self.assertIsInstance(labels, list)

    def test_absent_labels_file_returns_none(self):
        """No hdbscan_labels.json → None, not an exception."""
        run_dir = _make_phase1_run(Path(self._tmp), "no_labels")
        (run_dir / "hdbscan_labels.json").unlink(missing_ok=True)
        p1 = load_phase1_run(run_dir)
        self.assertIsNone(p1.get("hdbscan_labels"))


# ============================================================================
# 3 — build_context phase1 directory resolution
# ============================================================================

class TestBuildContextPathResolution(unittest.TestCase):
    """
    run_6.build_context currently does:
      phase1_dir / stem  →  phase1_dir/albert_xlarge_v2/
    But Phase 1 writes flat dirs like:
      phase1_dir/albert-xlarge-v2_48iter_wiki_paragraph/
    This is a two-level vs one-level layout mismatch, compounded by
    hyphen vs underscore naming.

    We test the expected resolution behaviour so that any fix to
    build_context or the path helper can be validated here.
    """

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp)
        self.p1_dir = Path(self._tmp)

    def _make_run(self, dirname, prompt_key="wiki_paragraph"):
        return _make_phase1_run(self.p1_dir, dirname)

    # ------------------------------------------------------------------
    # Helper that should be extracted from build_context into core/io.py
    # ------------------------------------------------------------------
    def _find_phase1_run_dir(self, phase1_dir: Path, model_name: str, prompt_key: str):
        """
        Expected resolution logic (to be implemented in p1_mstate_tracking.p1_io or run_6.py):
          1. Try phase1_dir / stem / *prompt_key*   (legacy nested layout)
          2. Try phase1_dir / *{model_stem}*{prompt_key}*  (flat layout, hyphen names)
          3. Try phase1_dir / *{model_hyphen}*{prompt_key}*
          4. Fall back to most-recently modified matching dir
        """
        from p1_mstate_tracking.p1_io import find_phase1_run_dir
        return find_phase1_run_dir(phase1_dir, model_name, prompt_key)

    def test_hyphen_dir_found_for_underscore_stem(self):
        """albert-xlarge-v2_48iter_wiki_paragraph found for albert-xlarge-v2 + wiki_paragraph."""
        self._make_run("albert-xlarge-v2_48iter_wiki_paragraph")
        d = self._find_phase1_run_dir(self.p1_dir, "albert-xlarge-v2", "wiki_paragraph")
        self.assertIsNotNone(d, "Should find hyphen-named dir")
        self.assertTrue(d.exists())

    def test_prompt_key_selects_correct_dir(self):
        """Two dirs for same model, different prompts → correct one selected."""
        self._make_run("albert-xlarge-v2_wiki_paragraph")
        self._make_run("albert-xlarge-v2_sullivan_ballou")
        d = self._find_phase1_run_dir(self.p1_dir, "albert-xlarge-v2", "sullivan_ballou")
        self.assertIn("sullivan_ballou", d.name)

    def test_missing_model_returns_none(self):
        """No matching dirs → None, not KeyError or FileNotFoundError."""
        d = self._find_phase1_run_dir(self.p1_dir, "nonexistent-model", "wiki_paragraph")
        self.assertIsNone(d)

    def test_no_prompt_match_falls_back_to_any_model_dir(self):
        """Prompt key not in any dir name → fall back to any dir for that model."""
        self._make_run("albert-xlarge-v2_some_prompt")
        d = self._find_phase1_run_dir(self.p1_dir, "albert-xlarge-v2", "missing_prompt")
        # Should return something rather than None, since model exists
        self.assertIsNotNone(d)


# ============================================================================
# 4 — Integration: ctx["labels_per_layer"] shape matches activations
# ============================================================================

class TestCtxLabelActivationAlignment(unittest.TestCase):
    """
    Simulate what build_context does and verify the shapes are consistent.
    This catches the case where labels_per_layer is returned correctly by
    load_phase1_run but then mishandled by build_context.
    """

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp)

    def _make_ctx_from_p1(self, n_layers=6, n_tokens=10, d=32):
        run_dir = _make_phase1_run(
            Path(self._tmp), "model_prompt",
            n_layers=n_layers, n_tokens=n_tokens, d=d,
        )
        p1 = load_phase1_run(run_dir)
        acts = p1["activations"]
        ctx = {}
        ctx["activations_per_layer"] = [acts[L] for L in range(acts.shape[0])]
        ctx["labels_per_layer"]      = p1.get("hdbscan_labels")
        return ctx, n_layers, n_tokens

    def test_labels_length_matches_activations_length(self):
        ctx, n_layers, _ = self._make_ctx_from_p1()
        self.assertEqual(
            len(ctx["activations_per_layer"]),
            len(ctx["labels_per_layer"]),
        )

    def test_labels_shape_matches_token_count(self):
        ctx, n_layers, n_tokens = self._make_ctx_from_p1()
        for L in range(n_layers):
            acts_tok = ctx["activations_per_layer"][L].shape[0]
            lab_tok  = ctx["labels_per_layer"][L].shape[0]
            self.assertEqual(acts_tok, lab_tok, msg=f"Mismatch at layer {L}")

    def test_label_arrays_are_integer_dtype(self):
        ctx, n_layers, _ = self._make_ctx_from_p1()
        for L in range(n_layers):
            arr = ctx["labels_per_layer"][L]
            self.assertTrue(
                np.issubdtype(arr.dtype, np.integer),
                msg=f"Layer {L}: dtype {arr.dtype} not integer",
            )


if __name__ == "__main__":
    unittest.main()
