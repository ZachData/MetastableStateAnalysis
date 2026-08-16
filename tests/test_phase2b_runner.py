"""
tests/test_phase2b_runner.py — the sweep.

The two classes that matter are `TestErrorsAreNotSwallowed` (a bare
`try/except Exception` per prompt is how Block 4 shipped raising `NameError`
on every run of every model while the summary still wrote) and
`TestWeightsWorkHappensOncePerCheckpoint` (the rescalers are
prompt-independent and were being rebuilt per prompt).
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from p2b_imaginary import p2b_io, run_2b


STEPS = [0, 8, 512, 143000]
PROMPTS = ["wiki_paragraph", "short_heterogeneous"]
GATE = 2.0
D = 12
N_LAYERS_OV = 4
N_LAYERS_ACT = 5
N_TOKENS = 10
N_HEADS = 4


def _write_ov(weights_dir: Path, stem: str, scale=0.05, n_heads=N_HEADS):
    """
    An `ov_weights_{stem}.npz` in the shape `weights.py` really writes:
    `ov_total_{layer}` AND `ov_head{h}_{layer}`, with the total the sum of
    the heads. The per-head arrays were always in the file; nothing in Phase
    2b read them until `run_head_circuits`.
    """
    rng = np.random.default_rng(abs(hash(stem)) % 2**31)
    arrays = {}
    d_head = D // n_heads
    for i in range(N_LAYERS_OV):
        heads = []
        for h in range(n_heads):
            W_O = rng.normal(size=(D, d_head))
            W_V = rng.normal(size=(d_head, D))
            OV_h = scale * (W_O @ W_V) / np.sqrt(D)
            arrays[f"ov_head{h}_layer_{i}"] = OV_h
            heads.append(OV_h)
        arrays[f"ov_total_layer_{i}"] = sum(heads)
    np.savez_compressed(weights_dir / f"ov_weights_{stem}.npz", **arrays)


def _write_ov_without_heads(weights_dir: Path, stem: str, scale=0.05):
    """A weights file written before `weights.py` saved per-head arrays."""
    rng = np.random.default_rng(abs(hash(stem)) % 2**31)
    np.savez_compressed(
        weights_dir / f"ov_weights_{stem}.npz",
        **{f"ov_total_layer_{i}": scale * rng.normal(size=(D, D))
           for i in range(N_LAYERS_OV)},
    )


def _write_p1_run(phase1_dir: Path, stem: str, prompt: str):
    run = phase1_dir / f"{stem}_{prompt}"
    run.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(run.name)) % 2**31)
    X = rng.normal(size=(N_LAYERS_ACT, N_TOKENS, D))
    X = X / np.linalg.norm(X, axis=-1, keepdims=True)
    np.savez_compressed(run / "activations.npz", activations=X,
                        norms=np.ones((N_LAYERS_ACT, N_TOKENS), dtype=np.float32))
    with open(run / "geometry.json", "w") as f:
        json.dump({"model": stem, "prompt": prompt, "n_tokens": N_TOKENS,
                   "d_model": D,
                   "layers": [{"layer": i, "effective_rank": 6.0,
                               "effective_rank_normed": 5.0}
                              for i in range(N_LAYERS_ACT)]}, f)
    with open(run / "energies.json", "w") as f:
        json.dump({"layers": [{"layer": i, "energies": {"1.0": 1.0 + 0.1 * i}}
                              for i in range(N_LAYERS_ACT)],
                   "violation_layers": {"1.0": [2]}}, f)
    return run


class _Sweep(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.weights = self.tmp / "weights"
        self.phase1 = self.tmp / "phase1"
        self.out = self.tmp / "out"
        for d in (self.weights, self.phase1, self.out):
            d.mkdir()
        for step in STEPS:
            stem = f"pythia-410m-step{step}"
            _write_ov(self.weights, stem)
            for p in PROMPTS:
                _write_p1_run(self.phase1, stem, p)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def sweep(self, **kw):
        kw.setdefault("gate_threshold", GATE)
        kw.setdefault("base", "pythia-410m")
        return run_2b.run_sweep(self.weights, self.phase1, self.out, **kw)


# ---------------------------------------------------------------------------
# The checkpoint axis
# ---------------------------------------------------------------------------

class TestCheckpointAxisIsTheOuterLoop(_Sweep):

    def test_every_checkpoint_appears_once_in_training_order(self):
        c = self.sweep()
        self.assertEqual(c["steps"], sorted(STEPS))
        self.assertEqual(c["n_checkpoints"], len(STEPS))

    def test_step_is_carried_on_every_artifact(self):
        self.sweep()
        for step in STEPS:
            stem = f"pythia-410m-step{step}"
            with open(self.out / stem / "block1a_rotational_spectrum.json") as f:
                self.assertEqual(json.load(f)["checkpoint_step"], step)
            with open(self.out / stem / PROMPTS[0] /
                      "block1b_rescaled_comparison.json") as f:
                self.assertEqual(json.load(f)["checkpoint_step"], step)

    def test_manifest_written_per_checkpoint(self):
        self.sweep()
        for step in STEPS:
            m = json.loads((self.out / f"pythia-410m-step{step}" /
                            "manifest.json").read_text())
            self.assertEqual(m["checkpoint_step"], step)
            self.assertEqual(m["hf_revision"], f"step{step}")
            self.assertEqual(m["phase"], "2b")

    def test_steps_filter(self):
        c = self.sweep(steps=[8, 512])
        self.assertEqual(c["steps"], [8, 512])

    def test_families_do_not_mix(self):
        _write_ov(self.weights, "pythia-1.4b-step512")
        c = self.sweep()
        self.assertNotIn("pythia-1.4b-step512", c["results"])


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

class TestWeightsWorkHappensOncePerCheckpoint(_Sweep):

    def test_rescalers_built_once_per_checkpoint_not_per_prompt(self):
        """
        `expm` on the OV matrices is prompt-independent. `run_2i.py` rebuilt
        it inside every (model, prompt) pair: on Study B that is 27 x 9 x 3 x
        24 exponentials of a 1024x1024 matrix where 27 x 3 x 24 suffice.
        """
        from p2b_imaginary import rotational_rescaled as rr
        with mock.patch.object(rr, "build_rescalers",
                               wraps=rr.build_rescalers) as spy:
            self.sweep(blocks=["1b"])
        # 3 frames (V, S, A) per checkpoint, not per prompt.
        self.assertEqual(spy.call_count, 3 * len(STEPS))

    def test_block_1a_runs_once_per_checkpoint(self):
        with mock.patch.object(run_2b.schur_block, "analyze_rotational_spectrum",
                               wraps=run_2b.schur_block.analyze_rotational_spectrum
                               ) as spy:
            self.sweep(blocks=["1a"])
        self.assertEqual(spy.call_count, len(STEPS))

    def test_cost_estimate_reflects_the_cache(self):
        with_cache = run_2b.estimate_cost(27, 9, 24, 1024, ["1b"])
        self.assertEqual(with_cache["expm_calls"], 27 * 3 * 24)
        self.assertLess(with_cache["expm_calls"], 27 * 9 * 3 * 24)

    def test_nulls_dominate_the_estimate_when_enabled(self):
        plain = run_2b.estimate_cost(27, 9, 24, 1024, ["1a"])
        nulls = run_2b.estimate_cost(27, 9, 24, 1024, ["1a"], with_nulls=True)
        self.assertGreater(nulls["schur_calls"], 10 * plain["schur_calls"])


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class TestErrorsAreNotSwallowed(_Sweep):

    def test_a_failing_block_raises_by_default(self):
        """
        Block 4 shipped raising `NameError` on every prompt of every run
        because a bare `try/except Exception` recorded it and the summary
        still wrote.
        """
        with mock.patch.object(run_2b, "run_block_1b",
                               side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                self.sweep(blocks=["1b"])

    def test_continue_on_error_records_and_reports_loudly(self):
        with mock.patch.object(run_2b, "run_block_1b",
                               side_effect=RuntimeError("boom")):
            c = self.sweep(blocks=["1b"], continue_on_error=True)
        self.assertEqual(c["n_failed"], len(STEPS) * len(PROMPTS))
        head = run_2b.sweep_summary_lines(c)[:4]
        self.assertTrue(any("FAILURES" in l and "INCOMPLETE" in l for l in head))

    def test_clean_sweep_reports_zero_failures(self):
        c = self.sweep()
        self.assertEqual(c["n_failed"], 0)
        head = run_2b.sweep_summary_lines(c)[:4]
        self.assertTrue(any("FAILURES: 0" in l for l in head))
        self.assertFalse(any("INCOMPLETE" in l for l in head))

    def test_a_checkpoint_with_no_weights_is_reported_not_dropped(self):
        """
        A checkpoint Phase 2 never wrote weights for does not appear in the
        `ov_weights_*.npz` glob, so without this it is simply ABSENT: 26 rows
        instead of 27, with nothing saying which one.
        """
        (self.weights / "ov_weights_pythia-410m-step8.npz").unlink()
        c = self.sweep(expected_steps=STEPS)
        self.assertEqual(c["missing_checkpoints"], [8])
        self.assertEqual(c["results"]["pythia-410m-step8"]["status"],
                         "no_ov_weights")
        head = run_2b.sweep_summary_lines(c)[:5]
        self.assertTrue(any("MISSING CHECKPOINTS" in l and "[8]" in l
                            for l in head))

    def test_no_missing_checkpoints_on_a_complete_sweep(self):
        c = self.sweep(expected_steps=STEPS)
        self.assertEqual(c["missing_checkpoints"], [])
        head = run_2b.sweep_summary_lines(c)[:5]
        self.assertTrue(any("MISSING CHECKPOINTS: none" in l for l in head))

    def test_explicit_steps_filter_also_reports_absences(self):
        (self.weights / "ov_weights_pythia-410m-step512.npz").unlink()
        c = self.sweep(steps=[8, 512])
        self.assertEqual(c["missing_checkpoints"], [512])

    def test_legacy_output_dir_is_refused(self):
        (self.out / "phase2i_results.json").write_text("{}")
        with self.assertRaises(RuntimeError):
            self.sweep()

    def test_unknown_block_is_rejected_with_a_pointer(self):
        with self.assertRaises(SystemExit) as cm:
            run_2b.main(["--weights-dir", str(self.weights),
                         "--phase1-dir", str(self.phase1),
                         "--output-dir", str(self.out),
                         "--blocks", "1a,3"])
        self.assertIn("PLAN_2b.md", str(cm.exception))


# ---------------------------------------------------------------------------
# Blocks are independent
# ---------------------------------------------------------------------------

class TestBlocksAreNotNested(_Sweep):

    def test_1a_alone(self):
        self.sweep(blocks=["1a"])
        stem = f"pythia-410m-step{STEPS[0]}"
        self.assertTrue((self.out / stem /
                         "block1a_rotational_spectrum.json").exists())
        self.assertFalse((self.out / stem / PROMPTS[0]).exists())

    def test_1b_alone(self):
        self.sweep(blocks=["1b"])
        stem = f"pythia-410m-step{STEPS[0]}"
        self.assertFalse((self.out / stem /
                          "block1a_rotational_spectrum.json").exists())
        self.assertTrue((self.out / stem / PROMPTS[0] /
                         "block1b_rescaled_comparison.json").exists())

    def test_1b_does_not_depend_on_the_1a_verdict(self):
        """
        `run_2i.py` placed Blocks 3 and 4 after `if not run_block2: return`,
        so on the (constant) `rotation_neutral` verdict neither was reachable.
        """
        a = self.sweep(blocks=["1b"])
        shutil.rmtree(self.out)
        self.out.mkdir()
        b = self.sweep(blocks=["1a", "1b"])
        for stem in a["results"]:
            self.assertEqual(
                {p: js["interpretation"]["overall"]
                 for p, js in a["results"][stem]["block1b"].items()},
                {p: js["interpretation"]["overall"]
                 for p, js in b["results"][stem]["block1b"].items()},
            )


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

class TestArtifacts(_Sweep):

    def test_subresults_validate_against_the_registry(self):
        from core.artifacts import validate_artifact
        self.sweep()
        stem = f"pythia-410m-step{STEPS[0]}"
        for d, name in ((self.out / stem, "block1a_rotational_spectrum"),
                        (self.out / stem / PROMPTS[0],
                         "block1b_rescaled_comparison")):
            self.assertTrue(validate_artifact(d, "phase2b", name)["ok"])

    def test_combined_file_uses_the_new_name(self):
        self.sweep()
        self.assertTrue((self.out / "phase2b_results.json").exists())
        self.assertFalse((self.out / "phase2i_results.json").exists())

    def test_combined_json_is_finite(self):
        self.sweep()
        s = (self.out / "phase2b_results.json").read_text()
        self.assertNotIn("NaN", s)
        self.assertNotIn("Infinity", s)

    def test_frame_ledger_on_every_1b_artifact(self):
        self.sweep(blocks=["1b"])
        js = json.loads((self.out / f"pythia-410m-step{STEPS[0]}" /
                         PROMPTS[0] /
                         "block1b_rescaled_comparison.json").read_text())
        self.assertEqual(js["frame"]["kind"], "l2_sphere")
        self.assertEqual(js["frame"]["model_rev"], f"pythia-410m-step{STEPS[0]}")

    def test_phase1_cross_check_is_written(self):
        """
        Phase 2b gates on normed rank and Phase 1 on raw, so the counts are
        expected to differ. Recording the difference keeps it from sitting
        invisibly inside every elimination rate.
        """
        self.sweep(blocks=["1b"])
        js = json.loads((self.out / f"pythia-410m-step{STEPS[0]}" /
                         PROMPTS[0] /
                         "block1b_rescaled_comparison.json").read_text())
        cc = js["phase1_cross_check"]["1.0"]
        self.assertIn("n_p2b", cc)
        self.assertEqual(cc["n_phase1"], 1)


# ---------------------------------------------------------------------------
# The emissions that were computed and dropped
# ---------------------------------------------------------------------------

class TestPlanesSidecar(_Sweep):
    """
    Block 1a's per-plane spectrum, written beside the subresult rather than
    inside it. A sidecar purely for size — at d = 1024 a layer holds up to
    512 planes and `phase2b_results.json` embeds every checkpoint's Block 1a
    JSON — but the split has to be visible from the JSON, or a reader cannot
    tell "this run had no planes" from "this run predates the emission".
    """

    def test_written_next_to_the_subresult(self):
        self.sweep(blocks=["1a"])
        for step in STEPS:
            stem = f"pythia-410m-step{step}"
            self.assertTrue((self.out / stem / "planes.npz").exists(),
                            f"no planes sidecar for {stem}")

    def test_holds_every_layer(self):
        self.sweep(blocks=["1a"])
        stem = f"pythia-410m-step{STEPS[0]}"
        arrays = p2b_io.load_sidecar(self.out / stem, "planes")
        self.assertEqual(len(arrays["layer_names"]), N_LAYERS_OV)
        for layer in arrays["layer_names"]:
            for field in ("rho", "theta", "sign", "idx"):
                self.assertIn(f"{layer}__{field}", arrays)

    def test_the_json_says_whether_the_sidecar_exists(self):
        self.sweep(blocks=["1a"])
        stem = f"pythia-410m-step{STEPS[0]}"
        js = json.loads((self.out / stem /
                         "block1a_rotational_spectrum.json").read_text())
        self.assertTrue(js["has_plane_arrays"])
        self.assertNotIn("plane_arrays", js["per_layer"][0])
        self.assertIn("plane_quantiles", js["per_layer"][0])

    def test_no_planes_writes_no_sidecar_and_says_so(self):
        self.sweep(blocks=["1a"], with_planes=False)
        stem = f"pythia-410m-step{STEPS[0]}"
        self.assertFalse((self.out / stem / "planes.npz").exists())
        js = json.loads((self.out / stem /
                         "block1a_rotational_spectrum.json").read_text())
        self.assertFalse(js["has_plane_arrays"])

    def test_the_sidecar_filename_matches_the_registry(self):
        """
        One writer and one reader disagreeing about a filename is the drift
        `core/artifacts.py` exists to stop, so `sidecar_path` raises rather
        than preferring either side.
        """
        from core.artifacts import get_spec
        self.assertEqual(p2b_io.sidecar_path(self.out, "planes").name,
                         get_spec("phase2b", "planes").filename)


class TestPrecisionSurface(_Sweep):
    """
    Precision-policy item P2 — whether "84-97% complex" survives the fp16
    round-trip the checkpoints actually went through, swept over the relative
    tolerance rather than taken at the shipped 0.01. The module that computes
    it was written against this block and was never called from it, so the
    item has been a caveat in prose with no number since it was raised.

    Off by default: it costs ~10 dense eigendecompositions per layer.
    """

    def test_absent_unless_asked_for(self):
        self.sweep(blocks=["1a"])
        js = json.loads((self.out / f"pythia-410m-step{STEPS[0]}" /
                         "block1a_rotational_spectrum.json").read_text())
        self.assertNotIn("precision", js)

    def test_written_when_asked_for(self):
        self.sweep(blocks=["1a"], with_precision=True)
        js = json.loads((self.out / f"pythia-410m-step{STEPS[0]}" /
                         "block1a_rotational_spectrum.json").read_text())
        p = js["precision"]
        self.assertEqual(p["n_layers"], N_LAYERS_OV)
        self.assertIn(p["overall_verdict"],
                      ("stable", "threshold_sensitive", "precision_sensitive",
                       "both"))
        surface = p["per_layer"]["layer_0"]["surface"]
        self.assertEqual(len(surface["baseline"]), len(surface["tols"]))

    def test_the_verdict_does_not_re_assert_a_withdrawn_finding(self):
        """
        `precision_verdict`'s note is written into every artifact this
        emits. It used to say "Phase 2b's rotation_neutral result does not
        consult this tolerance; elim_signed = 1.0 stands regardless" — a
        withdrawn finding re-asserted in the output of a module whose own
        docstring retracts it.
        """
        self.sweep(blocks=["1a"], with_precision=True)
        js = json.loads((self.out / f"pythia-410m-step{STEPS[0]}" /
                         "block1a_rotational_spectrum.json").read_text())
        verdict = js["precision"]["per_layer"]["layer_0"]["verdict"]
        self.assertEqual(verdict["scope"], "block1a_complex_fraction")
        self.assertNotIn("stands regardless", verdict["note"])
        self.assertIn("withdrawn", verdict["note"])
        self.assertNotIn("causal_conclusion_affected", verdict)

    def test_estimate_cost_counts_it(self):
        cheap = run_2b.estimate_cost(4, 2, 24, 1024, ["1a"])
        dear = run_2b.estimate_cost(4, 2, 24, 1024, ["1a"],
                                    with_precision=True)
        self.assertEqual(cheap["eig_calls"], 0)
        self.assertGreater(dear["eig_calls"], 0)
        self.assertGreater(dear["estimated_seconds"],
                           cheap["estimated_seconds"])

    def test_a_non_finite_value_is_written_as_null_not_refused(self):
        """
        `complex_fraction_surface` reports z / percentile as NaN for a
        deterministic perturbation — one draw, so there is no distribution to
        score against. That is the right in-memory value and not a writable
        one: `json.dump(..., allow_nan=False)` raised from deep inside the
        encoder, naming neither the block nor the key.
        """
        self.sweep(blocks=["1a"], with_precision=True)
        text = (self.out / f"pythia-410m-step{STEPS[0]}" /
                "block1a_rotational_spectrum.json").read_text()
        self.assertNotIn("NaN", text)
        sigma = json.loads(text)["precision"]["per_layer"]["layer_0"][
            "surface"]["sigma"][0]
        self.assertIsNone(sigma["z_score"])   # null, not 0.0 and not absent


class TestNullsShareOneSample(_Sweep):

    def test_every_statistic_is_scored_against_the_same_draws(self):
        """
        The null used to be drawn per statistic, which multiplied its cost by
        len(NULL_STATISTICS) and scored each statistic against a different
        random matrix — so a difference between two z-scores mixed a real
        difference with two independent sampling draws.
        """
        from p2b_imaginary.rotational_schur import NULL_STATISTICS
        self.sweep(blocks=["1a"], with_nulls=True, n_null_draws=4)
        js = json.loads((self.out / f"pythia-410m-step{STEPS[0]}" /
                         "block1a_rotational_spectrum.json").read_text())
        nulls = js["per_layer"][0]["nulls"]
        self.assertEqual(set(nulls), set(NULL_STATISTICS))
        for res in nulls.values():
            self.assertTrue(res["shared_null_sample"])
            self.assertEqual(res["n_null"], 4)

    def test_the_dynamical_statistic_is_covered(self):
        from p2b_imaginary.rotational_schur import NULL_STATISTICS
        self.assertIn("frac_repulsive_real_part", NULL_STATISTICS)


class TestHeadCircuits(_Sweep):
    """
    `head_circuits.py` landed with its own tests and was never called from
    the runner, so no artifact carried `summed_vs_per_head`. The per-head
    arrays were in the OV npz the whole time; this loader simply did not read
    them.

    What the block is for: `ov_total = sum_h ov_per_head` is the effective
    operator only if every head shares an attention pattern, which the model
    does not satisfy. The published 84-97.5% is a statistic of that sum, so
    whether it describes any HEAD is a separate question, and
    `head_agreement` is its answer.
    """

    def test_the_loader_reads_the_per_head_arrays(self):
        stem = f"pythia-410m-step{STEPS[0]}"
        ov = p2b_io.load_ov_data(self.weights, stem)
        self.assertEqual(len(ov["ov_per_head"]), N_LAYERS_OV)
        self.assertEqual(len(ov["ov_per_head"][0]), N_HEADS)
        # And the sum really is the total, which is what makes the summed /
        # per-head comparison a comparison of the same operator.
        np.testing.assert_allclose(sum(ov["ov_per_head"][0]),
                                   ov["ov_total"][0], atol=1e-12)

    def test_heads_are_ordered_numerically_not_lexicographically(self):
        """
        A lexicographic sort puts head 10 before head 2 and produces a
        plausible-looking per-head table in the wrong order — the same
        failure the layer-key sort is explicit about.
        """
        stem = "pythia-410m-step99"
        _write_ov(self.weights, stem, n_heads=12)
        ov = p2b_io.load_ov_data(self.weights, stem)
        with np.load(self.weights / f"ov_weights_{stem}.npz") as data:
            for h in range(12):
                np.testing.assert_array_equal(
                    ov["ov_per_head"][0][h], data[f"ov_head{h}_layer_0"])

    def test_subresult_is_written_and_validates(self):
        from core.artifacts import validate_artifact
        self.sweep(blocks=["1a"])
        for step in STEPS:
            d = self.out / f"pythia-410m-step{step}"
            self.assertTrue((d / "block1a_head_circuits.json").exists())
            self.assertTrue(
                validate_artifact(d, "phase2b", "block1a_head_circuits")["ok"])

    def test_it_reports_both_objects_and_the_gap(self):
        self.sweep(blocks=["1a"])
        js = json.loads((self.out / f"pythia-410m-step{STEPS[0]}" /
                         "block1a_head_circuits.json").read_text())
        layer = js["per_layer"][0]
        self.assertIn("complex_energy_fraction", layer["summed"])
        self.assertIn("complex_energy_fraction_mean", layer["per_head"])
        self.assertAlmostEqual(
            layer["gap"],
            layer["summed"]["complex_energy_fraction"]
            - layer["per_head"]["complex_energy_fraction_mean"], places=9)
        self.assertEqual(layer["n_heads"], N_HEADS)
        # The summed object is labelled as the counterfactual it is.
        self.assertIn("shares an attention pattern", layer["summed"]["caveat"])

    def test_a_weights_file_without_heads_is_a_status_not_a_crash(self):
        """
        A weights file written before `weights.py` saved per-head arrays is a
        normal input. It must report why the block is absent rather than
        raising or silently producing nothing.
        """
        stem = "pythia-410m-step77"
        _write_ov_without_heads(self.weights, stem)
        c = self.sweep(blocks=["1a"], steps=[77])
        rec = c["results"][stem]
        self.assertEqual(rec.get("head_circuits_status"), "no_per_head_weights")
        self.assertNotIn("head_circuits", rec)
        self.assertFalse((self.out / stem / "block1a_head_circuits.json").exists())

    def test_can_be_switched_off(self):
        self.sweep(blocks=["1a"], with_heads=False)
        d = self.out / f"pythia-410m-step{STEPS[0]}"
        self.assertFalse((d / "block1a_head_circuits.json").exists())


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary(_Sweep):

    def test_1a_trajectory_table_has_a_row_per_checkpoint(self):
        lines = run_2b.sweep_summary_lines(self.sweep(blocks=["1a"]))
        body = [l for l in lines if l.strip().startswith(tuple(
            str(s) for s in STEPS))]
        self.assertEqual(len(body), len(STEPS))

    def test_1a_table_reports_both_conventions(self):
        lines = run_2b.sweep_summary_lines(self.sweep(blocks=["1a"]))
        header = next(l for l in lines if "cplx_frac" in l)
        self.assertIn("legacy", header)

    def test_1b_table_states_the_invariance_control_is_excluded(self):
        lines = run_2b.sweep_summary_lines(self.sweep(blocks=["1b"]))
        self.assertTrue(any("invariance control" in l for l in lines))
        self.assertTrue(any("orthogonal" in l for l in lines))

    def test_verdict_tally_present(self):
        lines = run_2b.sweep_summary_lines(self.sweep(blocks=["1b"]))
        self.assertTrue(any("Verdict tally" in l for l in lines))

    def test_summary_file_written(self):
        self.sweep()
        self.assertTrue((self.out / "phase2b_summary.txt").exists())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCli(_Sweep):

    def test_dry_run_lists_checkpoints_and_costs_without_writing(self):
        rc = run_2b.main(["--weights-dir", str(self.weights),
                          "--phase1-dir", str(self.phase1),
                          "--output-dir", str(self.out),
                          "--base", "pythia-410m", "--dry-run"])
        self.assertEqual(rc, 0)
        self.assertFalse((self.out / "phase2b_results.json").exists())

    def test_default_betas_is_study_b_beta_only(self):
        args = run_2b.build_parser().parse_args(
            ["--weights-dir", "w", "--phase1-dir", "p", "--output-dir", "o"])
        self.assertEqual([float(b) for b in args.betas.split(",")], [1.0])

    def test_exit_code_nonzero_on_failures(self):
        with mock.patch.object(run_2b, "run_block_1b",
                               side_effect=RuntimeError("boom")):
            rc = run_2b.main(["--weights-dir", str(self.weights),
                              "--phase1-dir", str(self.phase1),
                              "--output-dir", str(self.out),
                              "--base", "pythia-410m",
                              "--blocks", "1b",
                              "--gate-threshold", str(GATE),
                              "--continue-on-error"])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
