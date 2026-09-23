"""
`tools/run/stage0_chunk.py` — the time-boxed Stage 0 driver.

Pure: a tmp_path stands in for `METS_RESULTS_DIR`. What these pin down is the
definition of "done" (the thing that makes the driver resumable without a
ledger) and that a chunk never plans past its budget. Whether the estimates are
right is measured by the sweep itself, not faked here.
"""
import json

import pytest

from tools.run import stage0_chunk as s0

pytestmark = pytest.mark.pure

PIN = "a" * 40


def _run(root, step, prompt, *, sha=PIN, battery=s0.BATTERY_HASH_V2,
         labels=None, pair=None, manifest=True, exp="exp", n_tokens=403, wall=200.0,
         pair_file=True):
    exp_dir = root / exp
    run_dir = exp_dir / f"pythia-410m-step{step}_{prompt}"
    run_dir.mkdir(parents=True)
    (run_dir / "tokens.txt").write_text("\n".join(["t"] * n_tokens))
    (run_dir / "hdbscan_labels.json").write_text(
        json.dumps({"0": [0, 1]} if labels is None else labels))
    if pair_file:
        pa_path = exp_dir / "pair_agreement.json"
        pa = json.loads(pa_path.read_text()) if pa_path.exists() else {}
        pa[prompt] = {"n_semantic": 5, "n_artifact": 1, "n_noise": 2} if pair is None else pair
        pa_path.write_text(json.dumps(pa))
    if manifest:
        (run_dir / "manifest.json").write_text(json.dumps({
            "model": f"pythia-410m-step{step}", "checkpoint_step": step,
            "prompt_key": prompt, "prompt_battery_hash": battery, "git_sha": sha,
            "wall_time_seconds": wall, "n_tokens": n_tokens}))
    return run_dir


def test_complete_pinned_run_is_done(tmp_path):
    _run(tmp_path, 64, "wiki_byzantium")
    assert set(s0.scan_done(tmp_path, PIN)) == {(64, "wiki_byzantium")}


@pytest.mark.parametrize("kw", [
    {"sha": "b" * 40},                          # another commit
    {"battery": "1e47918ef77a"},                # the v1 WDS directories
    {"labels": {}},                             # the HDBSCAN outage's `{}`
    {"pair": {"n_semantic": 0, "n_artifact": 0, "n_noise": 0}},  # zeroed pair_agreement
    {"manifest": False},                        # killed mid-prompt
    {"pair_file": False},                       # prompt finished, invocation killed
])
def test_incomplete_or_foreign_runs_are_not_done(tmp_path, kw):
    _run(tmp_path, 64, "wiki_byzantium", **kw)
    assert s0.scan_done(tmp_path, PIN) == {}


def test_todo_follows_checkpoint_order_and_drops_finished(tmp_path):
    for p in ("a", "b"):
        _run(tmp_path, s0.CHECKPOINTS[0], p)
    _run(tmp_path, s0.CHECKPOINTS[1], "a", exp="exp2")
    rem = s0.todo(["a", "b"], s0.scan_done(tmp_path, PIN))
    assert rem[0] == (s0.CHECKPOINTS[1], ["b"])
    assert [s for s, _ in rem] == list(s0.CHECKPOINTS[1:])


def test_checkpoints_are_the_wds_grid():
    assert sorted(s0.CHECKPOINTS) == [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000,
                                      2000, 4000, 8000, 16000, 32000, 54000, 143000]


def test_fit_scale_keeps_probe_until_enough_runs():
    few = [s0.Done(0, "p", None, 999.0, 403)] * (s0.MIN_REFIT - 1)
    assert s0.fit_scale(few) == s0.PROBE_SECONDS
    many = [s0.Done(0, "p", None, 100.0, 403)] * s0.MIN_REFIT
    assert s0.fit_scale(many) == pytest.approx(100.0)


K = s0.MAX_PROMPTS_PER_INVOCATION
PER = s0.estimate_seconds(403, s0.PROBE_SECONDS)
PER_CKPT = (20 // K) * s0.LOAD_SECONDS + 20 * PER


def test_invocations_are_capped_so_a_kill_loses_little():
    prompts = [f"p{i}" for i in range(20)]
    n = {p: 403 for p in prompts}
    rem = [(s, list(prompts)) for s in s0.CHECKPOINTS]
    chunk = s0.plan_chunk(rem, n, s0.PROBE_SECONDS, 1e9)
    assert max(len(p) for _, p in chunk) == K
    assert [p for s, ps in chunk if s == s0.CHECKPOINTS[0] for p in ps] == prompts


def test_plan_never_exceeds_budget_and_splits_a_checkpoint():
    prompts = [f"p{i}" for i in range(20)]
    n = {p: 403 for p in prompts}
    budget = PER_CKPT + s0.LOAD_SECONDS + 3.5 * PER   # one checkpoint and 3 of the next
    rem = [(s, list(prompts)) for s in s0.CHECKPOINTS]
    chunk = s0.plan_chunk(rem, n, s0.PROBE_SECONDS, budget)
    assert [len(p) for _, p in chunk] == [K] * (20 // K) + [3]
    assert chunk[-1] == (s0.CHECKPOINTS[1], prompts[:3])
    assert s0.chunk_seconds(chunk, n, s0.PROBE_SECONDS) <= budget


def test_chunks_left_covers_everything():
    prompts = [f"p{i}" for i in range(20)]
    n = {p: 403 for p in prompts}
    rem = [(s, list(prompts)) for s in s0.CHECKPOINTS]
    assert s0.n_chunks_left(rem, n, s0.PROBE_SECONDS, PER_CKPT * 7 + 1) == 3   # 7 + 7 + 5


def test_first_checkpoint_is_the_probed_one():
    # the 09-22 probe passed at step143000, so a first-invocation failure there
    # is the toolchain's; step0 is random weights
    assert s0.CHECKPOINTS[0] == 143000


def test_killed_invocation_leaves_orphans_that_the_index_excludes(tmp_path):
    good = _run(tmp_path, 64, "a")
    orphan = _run(tmp_path, 64, "b", exp="killed", pair_file=False)
    done = s0.scan_done(tmp_path, PIN)
    assert s0.orphans(tmp_path, PIN, done) == [orphan]
    index = json.loads(s0.write_index(tmp_path, PIN, done).read_text())
    assert index["runs"] == {"64|a": str(good)} and index["pin"] == PIN


def test_token_count_prefers_the_manifest(tmp_path):
    d = _run(tmp_path, 64, "a", n_tokens=403)
    (d / "tokens.txt").write_text("t\n" * 403)   # trailing newline: 404 fields
    assert s0.scan_done(tmp_path, PIN)[(64, "a")].n_tokens == 403


def test_prompt_larger_than_budget_is_refused():
    with pytest.raises(ValueError):
        s0.n_chunks_left([(0, ["p"])], {"p": 403}, s0.PROBE_SECONDS, 1.0)
