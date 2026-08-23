"""
tests/test_p2b_viz_smoke.py — p2b_imaginary/visualization.

Two tiers, matching tests/SMOKE_TESTS_NOTES.md's split and
tests/test_p1b_viz_smoke.py's shape. Neither tier needs torch or a model:
this package reads artifacts only, and the fixture writes them by calling
the phase's own `run_block_1a` / `run_block_1b`.

  pure  — the loader grammar and the class vocabularies. These catch the
          failures that actually bite here: a verdict the phase can emit
          that has no colour (and so renders as a refusal gray — a real
          finding displayed as "not a measurement"), a beta that survives
          JSON as a string on one side of a record and a float on the
          other, a refusal mapped to 0.0 anywhere, and the step-axis mirror
          in `style.py` drifting from the Phase 1 module it mirrors.

  smoke — the CLI end to end against the synthetic Phase 2b directory,
          asserting every class produces figures, that a `--blocks 1a`
          sweep degrades to printed skips rather than errors, and that a
          pre-rewrite directory is refused.

Analysis logic is imported from `p2b_imaginary`, never restated here, so
the tests assert delegation rather than agreement between two copies.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from p2b_imaginary.visualization._fixture import (
    FIXTURE_PROMPTS, FIXTURE_STEPS, N_OV_LAYERS, build_fixture,
)
from p2b_imaginary.visualization.loaders import (
    GAPS, depth_matrix, describe_sweep, elim_row, layer_field, load_sweep,
    reference_beta,
)
from p2b_imaginary.visualization.style import (
    FRAME_COLORS, REFUSAL_COLOR, STATUS_COLORS, STATUS_MARKERS,
    VERDICT_COLORS, VERDICT_ORDER, signed_norm, step_x,
)

# Tier: deps -- needs the heavy tier importable (torch / transformers /
# scikit-learn / matplotlib). No model download, no run artifacts.
# Measured, not assumed; see pyproject.toml markers.
pytestmark = pytest.mark.deps


# ─────────────────────────────────────────────────────────────────────────────
# pure — class vocabularies
# ─────────────────────────────────────────────────────────────────────────────

def test_every_verdict_the_phase_can_emit_has_a_colour():
    """
    A verdict in `VERDICTS` but missing from the palette renders as refusal
    gray — a MEASUREMENT verdict silently demoted to "not a finding", which
    is the exact confusion this phase was reopened over.
    """
    from p2b_imaginary.rotational_rescaled import VERDICTS

    assert set(VERDICTS) <= set(VERDICT_COLORS)
    assert set(VERDICTS) <= set(VERDICT_ORDER), \
        "VERDICT_ORDER drives the legend; a verdict missing from it is absent"
    assert "missing" in VERDICT_COLORS, \
        "a record with no interpretation must still have a cell colour"


def test_every_frame_the_phase_can_emit_has_a_colour_and_the_control_is_gray():
    from p2b_imaginary.rotational_rescaled import FRAME_KEYS

    assert set(FRAME_KEYS) <= set(FRAME_COLORS)
    # The invariance control must not be given a data colour: it is an
    # algebraic identity, and colouring it like a measurement is how
    # `elim_rotation = 0.0` came to be read as one.
    assert FRAME_COLORS["remove_rotation"] == REFUSAL_COLOR


def test_every_elimination_refusal_status_has_a_marker():
    """
    `elimination_rate` names four refusals plus 'ok'. Each needs a marker,
    because they share one colour by design — they are four ways of having
    no number rather than four numbers.
    """
    statuses = ("ok", "no_violations_to_eliminate",
                "different_transitions_scored", "no_transitions_scored",
                "different_counting_rule")
    for s in statuses:
        assert s in STATUS_COLORS and s in STATUS_MARKERS
    refusal_colors = {STATUS_COLORS[s] for s in statuses if s != "ok"}
    assert refusal_colors == {REFUSAL_COLOR}
    assert len({STATUS_MARKERS[s] for s in statuses}) == len(statuses)


def test_signed_norm_is_centred_on_zero_not_on_the_data():
    """
    The elimination rate is unclipped by design (Phase 2 verification item
    V2). A table of mostly-positive rates would put matplotlib's neutral
    colour at some positive rate — drawing "no effect" where there is one.
    """
    lopsided = np.array([0.9, 0.8, 0.75, -0.1])
    norm = signed_norm(lopsided)
    assert norm.vcenter == 0.0
    assert norm.vmin == -norm.vmax
    assert norm(0.0) == pytest.approx(0.5)


def test_step_axis_mirror_agrees_with_phase_1s_module():
    """
    `style.py` mirrors Phase 1's step-axis helpers when that module cannot
    be imported (it reaches sklearn through `.series`). A fallback that can
    silently disagree is worse than no fallback, so it is pinned here —
    skipped, not failed, when the real module is unavailable.
    """
    ck = pytest.importorskip("p1_mstate_tracking.visualization.checkpoints")

    steps = [0, 1, 8, 512, 3000, 143000]
    assert np.allclose(step_x(steps), ck._step_x(steps))
    theirs, ours = ck.step_norm(steps), None
    from p2b_imaginary.visualization.style import step_norm as ours_fn
    ours = ours_fn(steps)
    assert (theirs.vmin, theirs.vmax) == (ours.vmin, ours.vmax)


def test_gap_registry_names_where_each_fix_belongs():
    """
    Every gap carries the function that drops it. The catalogue's promise is
    that a reader can go straight to the code; a gap with no address is a
    complaint rather than a report.
    """
    for gap in GAPS:
        assert gap["id"] and gap["key"] and gap["what"]
        assert "." in gap["where"], f"{gap['id']} has no module.function"


# ─────────────────────────────────────────────────────────────────────────────
# pure — loaders
# ─────────────────────────────────────────────────────────────────────────────

def test_layer_field_maps_json_null_to_nan_and_keeps_length():
    """
    NaN is a REAL value in Block 1a: a layer with no 2x2 blocks has NaN for
    every angle statistic by construction, and `p2b_io.json_default` writes
    that as JSON null. Dropping the row rather than NaN-ing it would shift
    every depth profile in the package by one.
    """
    per_layer = [{"theta_mean": 0.4}, {"theta_mean": None},
                 {"theta_mean": 1.2}, {}]
    v = layer_field(per_layer, "theta_mean")
    assert v.size == 4
    assert v[0] == 0.4 and v[2] == 1.2
    assert np.isnan(v[1]) and np.isnan(v[3])


def test_layer_field_missing_field_is_all_nan_not_empty():
    v = layer_field([{"a": 1.0}, {"a": 2.0}], "does_not_exist")
    assert v.size == 2 and np.all(np.isnan(v))


def test_reference_beta_resolves_a_float_to_the_string_key_it_is_stored_under():
    """
    `interpretation.reference_beta` is a float; `frames[*].counts` and
    `comparison` are keyed by `str(beta)`. Assuming either convention at a
    call site is the artifact-contract bug class `core/artifacts.py` exists
    to kill.
    """
    js = {"interpretation": {"reference_beta": 1.0},
          "comparison": {"1.0": {"elim_full": {"rate": None}}}}
    assert reference_beta(js) == "1.0"

    # And the numeric fallback, for a writer that rounded differently.
    js = {"interpretation": {"reference_beta": 0.1},
          "comparison": {"0.10": {}}}
    assert reference_beta(js) == "0.10"


def test_elim_row_keeps_a_refusal_as_none():
    """
    The one invariant this whole package turns on. `elimination_rate`
    returns None with a status for four refusals precisely because the
    pre-rewrite code returned 0.0 for all of them; anything here mapping
    None to 0.0 reproduces that defect.
    """
    js = {"interpretation": {"reference_beta": 1.0},
          "comparison": {"1.0": {
              "elim_full": {"rate": None,
                            "status": "no_violations_to_eliminate"},
              "elim_signed": {"rate": -0.5, "status": "ok"}}}}
    row = elim_row(js)
    assert row["elim_full"]["rate"] is None
    assert row["elim_signed"]["rate"] == -0.5


# ─────────────────────────────────────────────────────────────────────────────
# smoke — the fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    return build_fixture(tmp_path_factory.mktemp("p2bout"))


def test_fixture_is_discoverable(fixture_dir):
    sweep = load_sweep(fixture_dir)
    assert sweep is not None
    assert sweep.source == "combined"
    assert sweep.base == "pythia-410m"
    # Six real checkpoints plus the one Phase 2 wrote no OV weights for.
    assert len(sweep.checkpoints) == len(FIXTURE_STEPS) + 1
    assert len(sweep.with_1a) == len(FIXTURE_STEPS)
    assert sweep.steps == sorted(FIXTURE_STEPS)
    assert sweep.prompts == sorted(FIXTURE_PROMPTS)
    assert sweep.has_trajectory


def test_fixture_is_deterministic(tmp_path):
    """
    A fixture seeded from `hash()` on a str changes on every process, and so
    would every figure drawn against it. Two builds must be byte-identical.
    """
    a = build_fixture(tmp_path / "a")
    b = build_fixture(tmp_path / "b")
    ja = json.loads((a / "phase2b_results.json").read_text())
    jb = json.loads((b / "phase2b_results.json").read_text())
    for j in (ja, jb):
        j.pop("wall_time_seconds", None)
    assert ja["results"] == jb["results"]


def test_the_missing_checkpoint_is_present_as_a_status_not_an_absence(
        fixture_dir):
    """
    `run_sweep`'s `expected_steps` exists so a checkpoint Phase 2 never
    wrote appears in the table with a status rather than vanishing from a
    glob. The loader must carry that through, and the reason must say what
    happened.
    """
    sweep = load_sweep(fixture_dir)
    missing = [c for c in sweep.checkpoints if c.status == "no_ov_weights"]
    assert len(missing) == 1
    assert missing[0].step in sweep.missing_checkpoints
    assert any("no OV weights" in m for m in missing[0].missing)


def test_a_failed_prompt_is_visible_as_a_failure(fixture_dir):
    """
    `--continue-on-error` writes `{"status": "failed"}` for the prompt and
    records the traceback. That record must survive to the figures as a
    failure — an absent cell and a failed cell mean different things.
    """
    sweep = load_sweep(fixture_dir)
    failed = [(c, p) for c in sweep.checkpoints
              for p, js in c.block1b.items() if js.get("status") == "failed"]
    assert len(failed) == 1
    ck, _ = failed[0]
    assert ck.failures and ck.block1b_scored().keys() != ck.block1b.keys()


def test_fixture_exercises_both_a_refusal_and_a_measurement(fixture_dir):
    """
    A fixture where every comparison refuses would leave the measurement
    half of the verdict palette undrawn, and vice versa. Both paths must be
    present, and at least one rate must be NEGATIVE — the unclipped reading
    Phase 2's `max(0, …)` destroys and F2 exists to draw.
    """
    sweep = load_sweep(fixture_dir)
    rates, statuses = [], set()
    for ck in sweep.with_1b:
        for js in ck.block1b_scored().values():
            for res in elim_row(js).values():
                statuses.add(res["status"])
                if res["rate"] is not None:
                    rates.append(res["rate"])
    assert "ok" in statuses
    assert statuses - {"ok"}, "no refusal anywhere in the fixture"
    assert any(r < 0 for r in rates), "no negative (unclipped) rate to draw"


def test_fixture_exercises_truncation(fixture_dir):
    """
    Truncation is the mechanism that makes `elim_signed = 1.0` free, and
    F3/F4/V5 draw nothing without one. `e^{−A}` is orthogonal, so the
    invariance control must NEVER be the frame that truncated.
    """
    sweep = load_sweep(fixture_dir)
    truncated = [(key, fr) for ck in sweep.with_1b
                 for js in ck.block1b_scored().values()
                 for key, fr in js["frames"].items() if fr["truncated"]]
    assert truncated, "no truncated frame in the fixture"
    assert all(reason for _, fr in truncated
               for reason in [fr["truncation_reason"]])
    assert not any(key == "remove_rotation" for key, _ in truncated), \
        "e^{-A} is orthogonal and cannot overflow — a truncated control is a bug"


def test_the_invariance_control_holds_in_every_record(fixture_dir):
    """
    The identity `status-2b` withdrew the headline over. If this ever fails
    it is a numerical failure of `expm` or the accumulation, not a finding
    about rotation — and F5 would be drawing a broken control.
    """
    sweep = load_sweep(fixture_dir)
    for ck in sweep.with_1b:
        for js in ck.block1b_scored().values():
            assert js["invariance"]["status"] == "identity_holds"


def test_depth_matrix_is_ordered_by_step_and_shaped_by_depth(fixture_dir):
    sweep = load_sweep(fixture_dir)
    steps, mat = depth_matrix(sweep.checkpoints, "complex_energy_fraction")
    assert steps == sorted(steps) == sorted(FIXTURE_STEPS)
    assert mat.shape == (len(FIXTURE_STEPS), N_OV_LAYERS)
    assert np.isfinite(mat).all()


def test_gaps_are_detected_from_the_artifact(fixture_dir):
    """
    Four gaps are unconditional properties of today's serializers, so a
    fixture written by the real code has them open — and the day one is
    closed in `p2b_imaginary/` the detection must notice without a change
    here.
    """
    sweep = load_sweep(fixture_dir)
    open_ids = {g["id"] for g in sweep.gaps}
    assert {"G1", "G2", "G3", "G7"} <= open_ids
    assert sweep.has_gap("G1")
    assert "comparison_to_json" in sweep.gap_reason("G1")

    # Closing one is visible immediately: plant the key the detector looks
    # for and the gap goes away.
    ck = sweep.with_1b[0]
    js = next(iter(ck.block1b_scored().values()))
    js["frames"]["original"]["per_layer"] = {"energies": {}}
    assert not sweep.has_gap("G1")


def test_step_and_prompt_filters_apply_to_the_cross_checkpoint_view(
        fixture_dir):
    """
    Filtering the per-checkpoint figures and not the cross-checkpoint ones
    would be worse than not filtering: the trajectory would silently span
    checkpoints the caller excluded.
    """
    sweep = load_sweep(fixture_dir, steps=[0, 8],
                       prompts=[FIXTURE_PROMPTS[0]])
    assert sweep.steps == [0, 8]
    assert sweep.prompts == [FIXTURE_PROMPTS[0]]
    assert set(sweep.combined_view["results"]) == {"pythia-410m-step0",
                                                   "pythia-410m-step8"}


def test_describe_sweep_reports_the_open_gaps(fixture_dir):
    text = describe_sweep(load_sweep(fixture_dir))
    assert "open data gaps" in text
    assert "G3" in text and "p2b_imaginary/" in text


def test_a_legacy_directory_is_refused_by_the_phases_own_check(tmp_path):
    """
    Pre-rewrite artifacts were scored with an absolute 1e-6 threshold and a
    3.0 rank gate, and their `elim_rotation` column is an identity. The
    refusal is `p2b_io.refuse_legacy_run_dir`, imported — a second copy here
    would be a second place for it to go stale.
    """
    (tmp_path / "phase2i_results.json").write_text("{}")
    with pytest.raises(RuntimeError, match="pre-rewrite"):
        load_sweep(tmp_path)


def test_an_interrupted_sweep_is_read_from_its_subdirectories(fixture_dir,
                                                              tmp_path):
    """
    A sweep killed partway has written every subresult it reached and no
    combined file — and that is exactly the directory someone wants to look
    at. Reconstruction is explicitly partial and says so.
    """
    import shutil

    partial = tmp_path / "partial"
    shutil.copytree(fixture_dir, partial)
    (partial / "phase2b_results.json").unlink()

    sweep = load_sweep(partial)
    assert sweep is not None
    assert sweep.source == "subdirectories"
    assert len(sweep.with_1a) == len(FIXTURE_STEPS)
    assert sweep.betas == [1.0]
    # Nothing on disk records what was ASKED for, so this must not claim to
    # know which checkpoints are missing.
    assert sweep.missing_checkpoints == []


# ─────────────────────────────────────────────────────────────────────────────
# smoke — figures
# ─────────────────────────────────────────────────────────────────────────────

def test_generate_all_produces_every_class(fixture_dir, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from p2b_imaginary.visualization.pipeline import CLASSES, generate_all

    produced = generate_all(fixture_dir, tmp_path)

    for cls in CLASSES:
        assert produced.get(cls), f"{cls} produced no figures"
    for paths in produced.values():
        for p in paths:
            assert p.exists() and p.stat().st_size > 0

    assert (tmp_path / "_cross").is_dir()
    assert (tmp_path / "pythia-410m-step0").is_dir()
    assert (tmp_path / "pythia-410m-step0" / FIXTURE_PROMPTS[0]).is_dir()


def test_unknown_class_is_rejected_rather_than_silently_skipped(fixture_dir,
                                                               tmp_path):
    from p2b_imaginary.visualization.pipeline import generate_all
    with pytest.raises(ValueError, match="unknown figure class"):
        generate_all(fixture_dir, tmp_path, classes=["spectrun"])


def test_a_blocks_1a_sweep_degrades_to_skips(fixture_dir, tmp_path, capsys):
    """
    `--blocks 1a` is the cheapest run the phase supports and the one that
    answers its first open question, so a directory with no Block 1b at all
    is a normal input — not an error, and not a silent half-empty output.
    """
    import matplotlib
    matplotlib.use("Agg")
    from p2b_imaginary.visualization.pipeline import generate_all

    sweep = load_sweep(fixture_dir)
    for ck in sweep.checkpoints:
        ck.data.pop("block1b", None)

    produced = generate_all(fixture_dir, tmp_path / "figs", sweep=sweep)
    assert produced["spectrum"], "spectrum needs only Block 1a"
    assert produced["trajectory"], "the training axis needs only Block 1a"
    assert not produced["frames"]
    assert not produced["verdicts"]

    out = capsys.readouterr().out
    assert "frames: skipping" in out and "verdicts: skipping" in out


def test_a_single_checkpoint_draws_what_it_can_and_skips_the_rest(
        fixture_dir, tmp_path, capsys):
    """
    One checkpoint is a point, and a point drawn as a trajectory is how a
    single measurement acquires a slope. Coverage and the transitions map
    still draw — "this sweep can address none of the dated events" is the
    most useful thing they can say.
    """
    import matplotlib
    matplotlib.use("Agg")
    from p2b_imaginary.visualization.pipeline import generate_all

    produced = generate_all(fixture_dir, tmp_path / "figs",
                            classes=["trajectory", "report"], steps=[512])
    names = {p.stem for p in produced["trajectory"] + produced["report"]}
    assert names == {"sweep_coverage", "known_transitions_map"}
    assert "trajectory: skipping" in capsys.readouterr().out


def test_nulls_class_skips_when_the_sweep_ran_without_them(fixture_dir,
                                                          tmp_path, capsys):
    import matplotlib
    matplotlib.use("Agg")
    from p2b_imaginary.visualization.pipeline import generate_all

    sweep = load_sweep(fixture_dir)
    for ck in sweep.checkpoints:
        for rec in ck.per_layer:
            rec.pop("nulls", None)

    produced = generate_all(fixture_dir, tmp_path / "figs", sweep=sweep,
                            classes=["nulls"])
    assert produced["nulls"] == []
    assert "--with-nulls" in capsys.readouterr().out


def test_figure_names_match_the_catalogue():
    """
    FIGURES-2b.md is the catalogue and each module declares the names it
    draws. A figure renamed in one and not the other is a broken reference
    in the document that reviewers navigate by.
    """
    from p2b_imaginary.visualization import (
        curiosities, frames, nulls, report_fig, spectrum, trajectory,
        verdicts,
    )

    catalogue = (Path(__file__).parent.parent / "p2b_imaginary" /
                 "visualization" / "FIGURES-2b.md").read_text()
    for module in (spectrum, frames, trajectory, report_fig, verdicts, nulls,
                   curiosities):
        for name in module.FIGURES:
            assert f"`{name}`" in catalogue, \
                f"{name} is drawn by {module.__name__} and is not in FIGURES-2b.md"
