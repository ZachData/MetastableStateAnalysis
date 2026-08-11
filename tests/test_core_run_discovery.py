"""
tests/test_core_run_discovery.py — core/run_discovery.py.

Pure path/string logic, no torch, no numpy. Runnable anywhere.

The load-bearing test is TestB1Regression: it reconstructs the exact
directory set the Pythia-410M pilot produces and asserts that resolving
step 1 returns step 1 and nothing else. The old substring matcher returned
ten checkpoints for that query.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from core.run_discovery import (
    RunRef,
    DuplicateRunError,
    parse_checkpoint_name,
    read_run_ref,
    discover_runs,
    index_by_prompt_and_step,
    sweep_for_prompt,
    resolve_anchor,
    sweep_report_lines,
)

# The pilot's real schedule (core/pythia_registry.py PYTHIA_410M_PILOT_STEPS).
PILOT_STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 3000, 5000,
               7000, 9000, 11000, 13000, 15000, 17000, 19000, 40000, 60000,
               80000, 100000, 120000, 143000]

PROMPTS = ["sullivan_ballou", "wiki_paragraph", "short_heterogeneous",
           "repeated_tokens"]


def _mkrun(root: Path, model: str, prompt: str,
           manifest: bool = False, geometry: bool = True,
           step=None, revision=None) -> Path:
    d = root / f"{model}_{prompt}"
    d.mkdir(parents=True, exist_ok=True)
    if geometry:
        (d / "geometry.json").write_text(json.dumps({"prompt": prompt}))
    if manifest:
        man = {"model": model, "prompt_key": prompt}
        if step is not None:
            man["checkpoint_step"] = step
        if revision is not None:
            man["hf_revision"] = revision
        (d / "manifest.json").write_text(json.dumps(man))
    return d


class _Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


# ---------------------------------------------------------------------------

class TestParseCheckpointName(unittest.TestCase):

    def test_step_variant(self):
        self.assertEqual(parse_checkpoint_name("pythia-410m-step2000"),
                         ("pythia-410m", 2000, False))

    def test_step_zero_is_a_step_not_a_none(self):
        base, step, rand = parse_checkpoint_name("pythia-410m-step0")
        self.assertEqual(step, 0)
        self.assertIsNot(step, None)
        self.assertFalse(rand)

    def test_random_control_is_distinct_from_step0(self):
        self.assertEqual(parse_checkpoint_name("pythia-1.4b-random"),
                         ("pythia-1.4b", None, True))

    def test_plain_model(self):
        self.assertEqual(parse_checkpoint_name("gpt2-large"),
                         ("gpt2-large", None, False))

    def test_dotted_base_survives(self):
        self.assertEqual(parse_checkpoint_name("pythia-1.4b-step143000"),
                         ("pythia-1.4b", 143000, False))

    def test_step_suffix_must_be_terminal(self):
        # 'step12' in the middle is not the grammar.
        self.assertEqual(parse_checkpoint_name("pythia-step12-foo"),
                         ("pythia-step12-foo", None, False))


class TestReadRunRef(_Tmp):

    def test_geometry_source(self):
        d = _mkrun(self.tmp, "pythia-410m-step512", "wiki_paragraph")
        ref = read_run_ref(d)
        self.assertEqual(ref.model, "pythia-410m-step512")
        self.assertEqual(ref.base, "pythia-410m")
        self.assertEqual(ref.step, 512)
        self.assertEqual(ref.prompt_key, "wiki_paragraph")
        self.assertEqual(ref.source, "geometry")

    def test_manifest_wins_and_carries_revision(self):
        d = _mkrun(self.tmp, "pythia-410m-step512", "wiki_paragraph",
                   manifest=True, step=512, revision="step512")
        ref = read_run_ref(d)
        self.assertEqual(ref.source, "manifest")
        self.assertEqual(ref.hf_revision, "step512")

    def test_dirname_fallback_when_no_json(self):
        d = _mkrun(self.tmp, "pythia-410m-step64", "sullivan_ballou",
                   geometry=False)
        ref = read_run_ref(d)
        self.assertEqual(ref.source, "dirname")
        self.assertEqual(ref.step, 64)
        self.assertEqual(ref.prompt_key, "sullivan_ballou")

    def test_dirname_fallback_with_known_keys_handles_underscored_model(self):
        d = self.tmp / "org_my-model-step8_wiki_paragraph"
        d.mkdir()
        ref = read_run_ref(d, known_prompt_keys=PROMPTS)
        self.assertEqual(ref.model, "org_my-model-step8")
        self.assertEqual(ref.prompt_key, "wiki_paragraph")
        self.assertEqual(ref.step, 8)

    def test_manifest_step_disagreeing_with_name_raises(self):
        d = _mkrun(self.tmp, "pythia-410m-step512", "wiki_paragraph",
                   manifest=True, step=1000)
        with self.assertRaises(DuplicateRunError):
            read_run_ref(d)

    def test_unidentifiable_dir_returns_none(self):
        d = self.tmp / "scratch"
        d.mkdir()
        self.assertIsNone(read_run_ref(d))

    def test_corrupt_geometry_falls_through_not_raises(self):
        d = self.tmp / "pythia-410m-step32_wiki_paragraph"
        d.mkdir()
        (d / "geometry.json").write_text("{not json")
        ref = read_run_ref(d)
        self.assertIsNotNone(ref)
        self.assertEqual(ref.source, "dirname")
        self.assertEqual(ref.step, 32)


class TestB1Regression(_Tmp):
    """The bug this module exists for."""

    def setUp(self):
        super().setUp()
        for step in PILOT_STEPS:
            for pk in PROMPTS:
                _mkrun(self.tmp, f"pythia-410m-step{step}", pk)
        _mkrun(self.tmp, "pythia-410m-random", "wiki_paragraph")

    def test_step1_resolves_to_step1_alone(self):
        # Old behaviour: 'pythia-410m-step1' substring-matched 12 of the
        # pilot's 27 checkpoints — every step whose digits begin with 1:
        # 1, 16, 128, 1000, 11000, 13000, 15000, 17000, 19000, 100000,
        # 120000, 143000.
        refs = discover_runs(self.tmp, base="pythia-410m-step1")
        self.assertEqual({r.step for r in refs}, {1})
        self.assertEqual(len(refs), len(PROMPTS))

    def test_naive_substring_would_have_matched_ten(self):
        # Documents the failure being regressed against, so the test states
        # the bug rather than only the fix.
        names = [f"pythia-410m-step{s}" for s in PILOT_STEPS]
        naive = [n for n in names if "pythia-410m-step1" in n]
        self.assertEqual(len(naive), 12)
        self.assertEqual(
            naive,
            [f"pythia-410m-step{s}" for s in
             (1, 16, 128, 1000, 11000, 13000, 15000, 17000,
              19000, 100000, 120000, 143000)],
        )
        anchored = [n for n in names
                    if parse_checkpoint_name(n)[1] == 1]
        self.assertEqual(anchored, ["pythia-410m-step1"])

    def test_base_query_returns_the_whole_sweep(self):
        refs = discover_runs(self.tmp, base="pythia-410m")
        checkpoints = [r for r in refs if not r.is_random]
        self.assertEqual(len(checkpoints), len(PILOT_STEPS) * len(PROMPTS))
        self.assertEqual({r.step for r in checkpoints}, set(PILOT_STEPS))

    def test_sweep_for_prompt_is_ordered_and_complete(self):
        refs = discover_runs(self.tmp, base="pythia-410m")
        sweep = sweep_for_prompt(refs, "wiki_paragraph")
        self.assertEqual([r.step for r in sweep], sorted(PILOT_STEPS))

    def test_random_control_excluded_from_sweep_but_present_in_index(self):
        refs = discover_runs(self.tmp, base="pythia-410m")
        sweep = sweep_for_prompt(refs, "wiki_paragraph")
        self.assertTrue(all(not r.is_random for r in sweep))
        index = index_by_prompt_and_step(refs)
        self.assertIn(None, index["wiki_paragraph"])
        self.assertTrue(index["wiki_paragraph"][None].is_random)

    def test_step0_and_random_are_separate_slots(self):
        refs = discover_runs(self.tmp, base="pythia-410m")
        index = index_by_prompt_and_step(refs)["wiki_paragraph"]
        self.assertIn(0, index)
        self.assertIn(None, index)
        self.assertFalse(index[0].is_random)
        self.assertNotEqual(index[0].run_dir, index[None].run_dir)

    def test_include_random_false_drops_only_the_control(self):
        refs = discover_runs(self.tmp, base="pythia-410m", include_random=False)
        self.assertTrue(all(not r.is_random for r in refs))
        self.assertEqual(len(refs), len(PILOT_STEPS) * len(PROMPTS))

    def test_step_filter(self):
        refs = discover_runs(self.tmp, base="pythia-410m",
                             steps=[16, 512, 3000, 143000],
                             include_random=False)
        self.assertEqual({r.step for r in refs}, {16, 512, 3000, 143000})

    def test_prompt_filter(self):
        refs = discover_runs(self.tmp, base="pythia-410m",
                             prompt_keys=["sullivan_ballou"])
        self.assertEqual({r.prompt_key for r in refs}, {"sullivan_ballou"})

    def test_ordering_is_deterministic(self):
        a = discover_runs(self.tmp, base="pythia-410m")
        b = discover_runs(self.tmp, base="pythia-410m")
        self.assertEqual([r.run_dir for r in a], [r.run_dir for r in b])
        self.assertEqual([r.run_dir for r in a],
                         sorted([r.run_dir for r in a],
                                key=lambda p: (a[[x.run_dir for x in a].index(p)].prompt_key,
                                               -1 if a[[x.run_dir for x in a].index(p)].step is None
                                               else a[[x.run_dir for x in a].index(p)].step,
                                               a[[x.run_dir for x in a].index(p)].model)))

    def test_resolve_anchor(self):
        refs = discover_runs(self.tmp, base="pythia-410m")
        ref = resolve_anchor(refs, "wiki_paragraph", 143000)
        self.assertEqual(ref.step, 143000)

    def test_resolve_missing_anchor_lists_available(self):
        refs = discover_runs(self.tmp, base="pythia-410m")
        with self.assertRaises(KeyError) as cm:
            resolve_anchor(refs, "wiki_paragraph", 999)
        self.assertIn("143000", str(cm.exception))


class TestCrossFamilyIsolation(_Tmp):

    def setUp(self):
        super().setUp()
        _mkrun(self.tmp, "pythia-410m-step512", "wiki_paragraph")
        _mkrun(self.tmp, "pythia-1.4b-step512", "wiki_paragraph")
        _mkrun(self.tmp, "pythia-1.4b-random", "wiki_paragraph")
        _mkrun(self.tmp, "gpt2-large", "wiki_paragraph")

    def test_410m_query_excludes_14b(self):
        refs = discover_runs(self.tmp, base="pythia-410m")
        self.assertEqual([r.model for r in refs], ["pythia-410m-step512"])

    def test_14b_random_belongs_to_14b_only(self):
        refs = discover_runs(self.tmp, base="pythia-1.4b")
        self.assertEqual({r.model for r in refs},
                         {"pythia-1.4b-step512", "pythia-1.4b-random"})

    def test_no_base_filter_returns_everything_identifiable(self):
        refs = discover_runs(self.tmp)
        self.assertEqual(len(refs), 4)


class TestDuplicateDetection(_Tmp):

    def test_two_dirs_same_prompt_and_step_raises(self):
        _mkrun(self.tmp, "pythia-410m-step512", "wiki_paragraph")
        # A re-run written under a differently-named dir but same identity.
        d = self.tmp / "rerun_pythia-410m-step512_wiki_paragraph"
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps({
            "model": "pythia-410m-step512",
            "prompt_key": "wiki_paragraph",
            "checkpoint_step": 512,
        }))
        refs = discover_runs(self.tmp, base="pythia-410m")
        self.assertEqual(len(refs), 2)
        with self.assertRaises(DuplicateRunError):
            index_by_prompt_and_step(refs)

    def test_same_dir_twice_is_not_a_duplicate(self):
        d = _mkrun(self.tmp, "pythia-410m-step512", "wiki_paragraph")
        ref = read_run_ref(d)
        index_by_prompt_and_step([ref, ref])   # must not raise


class TestReport(_Tmp):

    def test_provenance_and_missing_revision_are_surfaced(self):
        _mkrun(self.tmp, "pythia-410m-step0", "wiki_paragraph",
               manifest=True, step=0, revision="step0")
        _mkrun(self.tmp, "pythia-410m-step512", "wiki_paragraph")
        lines = sweep_report_lines(discover_runs(self.tmp, base="pythia-410m"))
        blob = "\n".join(lines)
        self.assertIn("manifest=1", blob)
        self.assertIn("geometry=1", blob)
        self.assertIn("hf_revision", blob)

    def test_empty(self):
        self.assertEqual(sweep_report_lines([]), ["(no runs discovered)"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
