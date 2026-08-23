"""
tests/test_p6_i1_adjudication.py — P6-I1 threaded end to end (POPPER_PLAN.md B6-first).

P6-I1 is the first prediction in this project wired all the way from a science
module's p-value into the falsification ledger. These tests exercise that whole
path on synthetic data — `run_induction_ov` → `compare_induction_vs_semantic`'s
Mann-Whitney U → `core.adjudication` → a record on disk → `verify_ledger`
replaying it.

The synthetic construction is not meant to be realistic. It is meant to make
the *direction* of the result known in advance, so a broken calibrator or a
mis-signed test shows up as a wrong e-value rather than as a plausible number.
Two arms are built for the same reason `UPDATE_PLAN.md` §5.6 gives ("an anchor
that only tests the identity case tests almost nothing"): one where induction
heads genuinely write more into the imaginary channel, one where they do not.

The most important assertion in the file is the *negative* one:
`run_induction_ov` must not adjudicate unless asked. Adjudication is opt-in
because this function is exercised by fixtures, and `adjudicate` refuses to
overwrite an existing record — so one accidental fixture run would permanently
occupy P6-I1's slot in the real ledger with a synthetic p-value.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pyproject.toml [tool.pytest.ini_options].markers.
pytestmark = pytest.mark.pure

from core.adjudication import (
    AdjudicationRefused,
    adjudicate,
    load_adjudications,
    registry_entry,
    verify_ledger,
)
from core.evalues import calibrate
from p6_subspace.induction_ov import compare_induction_vs_semantic, run_induction_ov

N_TOKENS = 12
D = 16
N_HEADS = 8
RANK = 8


def _orthonormal(d: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, k)))
    return Q[:, :k]


def _projectors(d: int = D, seed: int = 0) -> dict:
    """Complementary P_A / P_S projectors, so align_rot + align_real ≈ 1."""
    U = _orthonormal(d, d, seed)
    U_A, U_S = U[:, : d // 2], U[:, d // 2:]
    return {"per_layer": [{"P_A": U_A @ U_A.T, "P_S": U_S @ U_S.T}]}


def _induction_attention(is_induction: bool) -> np.ndarray:
    """
    Row-stochastic attention that either does or does not implement induction.

    `induction_score` requires query > key, query - key >= 2, key >= 1, and
    token_ids[query-1] == token_ids[key-1]. The token_ids below repeat with
    period 4, so (q, k) = (q, q-4) satisfies the identity match.

    The background level matters more than it looks. A flat background of 0.01
    normalizes to 1/N_TOKENS = 0.083 per entry, which is *above* the 0.05
    induction threshold — so every head reads as an induction head and the
    semantic arm comes out empty. The background here is therefore small
    enough that a non-induction head's weight on an induction pair lands near
    1e-4, well under the 0.01 the semantic-arm fallback requires.
    """
    A = np.full((N_TOKENS, N_TOKENS), 1e-4)
    for q in range(N_TOKENS):
        if is_induction and q >= 5:
            A[q, q - 4] = 1.0
        else:
            A[q, q] = 1.0          # attend to self: no induction structure
    return A / A.sum(axis=1, keepdims=True)


def _ctx(induction_writes_into: str = "A", seed: int = 0, **extra) -> dict:
    """
    Build a p6 context in which heads 0-3 are induction heads and 4-7 are not.

    `induction_writes_into` selects which subspace the induction heads' OV
    write mass occupies; the semantic heads always get the other one.
    "A" is P6-I1's predicted direction, "S" is the contrary arm.

    Note this is a choice of *subspace*, not a sign. `ov_write_alignment`
    measures the fraction of the write matrix's top singular vectors landing
    in each projector's range, which is invariant under negation — so scaling
    the A-bias by a negative number produces an identical alignment and does
    not reverse anything. Getting a genuinely contrary arm requires swapping
    which projector the bias is built from.
    """
    if induction_writes_into not in ("A", "S"):
        raise ValueError("induction_writes_into must be 'A' or 'S'")
    rng = np.random.default_rng(seed)
    proj = _projectors(seed=seed)
    P_A = proj["per_layer"][0]["P_A"]
    P_S = proj["per_layer"][0]["P_S"]
    P_ind = P_A if induction_writes_into == "A" else P_S
    P_sem = P_S if induction_writes_into == "A" else P_A

    attn, writes = [], []
    for h in range(N_HEADS):
        is_induction = h < N_HEADS // 2
        attn.append(_induction_attention(is_induction))
        # Small isotropic base plus a strong bias into the chosen subspace, so
        # the top singular vectors sit where the arm intends.
        base = rng.standard_normal((D, D)) * 0.05
        toward = P_ind if is_induction else P_sem
        writes.append(base + 3.0 * (toward @ rng.standard_normal((D, D))))

    X = rng.standard_normal((N_TOKENS, D))
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    ctx = {
        "attn_matrices": attn,
        "head_write_matrices": writes,
        "token_ids": np.array([i % 4 for i in range(N_TOKENS)]),
        "token_activations": X,
        "projectors": proj,
        "layer_idx": 0,
        "layer_name": "synthetic",
        "induction_threshold": 0.05,
        # No head_classify_result: the fallback picks low-induction heads as
        # the semantic arm, which is what heads 4-7 are by construction.
    }
    ctx.update(extra)
    return ctx


# ---------------------------------------------------------------------------
# The negative assertion, first because it matters most
# ---------------------------------------------------------------------------

class TestAdjudicationIsOptIn:

    def test_no_adjudication_without_the_flag(self, tmp_path):
        res = run_induction_ov(_ctx("A", adjudications_dir=tmp_path))
        assert res.payload["p6_i1_adjudication"] is None
        assert not list(tmp_path.glob("*.json"))

    def test_fixture_run_cannot_touch_the_real_ledger(self, tmp_path):
        """
        With the flag set, the record goes where ctx says — never to
        claims/adjudications/ by accident.
        """
        before = {r["prediction_id"] for r in load_adjudications()}
        run_induction_ov(_ctx("A", adjudicate=True, adjudications_dir=tmp_path))
        after = {r["prediction_id"] for r in load_adjudications()}
        assert before == after
        assert (tmp_path / "P6-I1.json").exists()


# ---------------------------------------------------------------------------
# The path itself
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_record_written_with_the_registry_claim(self, tmp_path):
        res = run_induction_ov(_ctx("A", adjudicate=True, adjudications_dir=tmp_path))
        adj = res.payload["p6_i1_adjudication"]
        assert adj is not None
        assert adj["prediction_id"] == "P6-I1"
        assert adj["claim"] == registry_entry("P6-I1").claim == "H-OPERATOR"
        assert "mannwhitneyu" in adj["test_name"]

    def test_e_value_is_the_calibrated_p(self, tmp_path):
        res = run_induction_ov(_ctx("A", adjudicate=True, adjudications_dir=tmp_path))
        adj = res.payload["p6_i1_adjudication"]
        p = res.payload["p6_i1"]["mwu_pvalue"]
        assert adj["p_value"] == pytest.approx(p)
        assert adj["e_value"] == pytest.approx(calibrate(p, adj["kappa"]))

    def test_supporting_direction_gives_evidence_against_the_null(self, tmp_path):
        """
        Induction heads writing more into A => small p => e > 1, i.e. the
        running product moves *up*.
        """
        res = run_induction_ov(_ctx("A", adjudicate=True, adjudications_dir=tmp_path))
        p6i1 = res.payload["p6_i1"]
        assert p6i1["delta_align_rot"] > 0
        assert res.payload["p6_i1_adjudication"]["e_value"] > 1.0

    def test_contrary_direction_counts_against_the_prediction(self, tmp_path):
        """
        The non-symmetric arm. With the gap reversed the MWU's one-sided p
        approaches 1, and a p near 1 must give e = kappa < 1 — a
        non-falsification is evidence *against* the hypothesis, not neutral.
        This is the property that keeps the process honest, and a calibrator
        that clipped or floored its input would pass every test above and fail
        this one.
        """
        res = run_induction_ov(_ctx("S", adjudicate=True, adjudications_dir=tmp_path))
        p6i1 = res.payload["p6_i1"]
        assert p6i1["mwu_pvalue"] > 0.5
        assert res.payload["p6_i1_adjudication"]["e_value"] < 1.0

    def test_record_replays_under_verify(self, tmp_path):
        run_induction_ov(_ctx("A", adjudicate=True, adjudications_dir=tmp_path))
        assert verify_ledger(tmp_path) == []

    def test_summary_reports_the_e_value(self, tmp_path):
        res = run_induction_ov(_ctx("A", adjudicate=True, adjudications_dir=tmp_path))
        text = "\n".join(res.summary_lines)
        assert "Adjudication" in text
        assert "e-value" in text
        assert "cumulative E for H-OPERATOR" in text

    def test_missing_artifact_hashes_flagged_in_the_record(self, tmp_path):
        res = run_induction_ov(_ctx("A", adjudicate=True, adjudications_dir=tmp_path))
        assert "NO ARTIFACT HASHES" in res.payload["p6_i1_adjudication"]["notes"]

    def test_artifact_hashes_pass_through(self, tmp_path):
        res = run_induction_ov(_ctx("A", adjudicate=True, adjudications_dir=tmp_path,
                                    artifact_hashes=["a" * 64]))
        adj = res.payload["p6_i1_adjudication"]
        assert adj["artifact_hashes"] == ["a" * 64]
        assert "NO ARTIFACT HASHES" not in adj["notes"]


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------

class TestDegenerate:

    def test_untestable_arm_is_not_a_refusal(self, tmp_path, capsys):
        """
        `compare_induction_vs_semantic` returns mwu_pvalue=None when an arm has
        fewer than two heads. That is "the test could not run", which must
        write no record and report no refusal — reporting one would put a
        policy violation in the log where a data limitation belongs.
        """
        ctx = _ctx("A", adjudicate=True, adjudications_dir=tmp_path)
        ctx["induction_threshold"] = 10.0        # nothing clears it
        res = run_induction_ov(ctx)
        assert res.payload["p6_i1"]["mwu_pvalue"] is None
        assert res.payload["p6_i1_adjudication"] is None
        assert not list(tmp_path.glob("*.json"))
        assert "refused" not in capsys.readouterr().err

    def test_second_run_refuses_rather_than_overwriting(self, tmp_path, capsys):
        run_induction_ov(_ctx("A", adjudicate=True, adjudications_dir=tmp_path))
        first = (tmp_path / "P6-I1.json").read_text()
        res = run_induction_ov(_ctx("A", seed=7, adjudicate=True,
                                    adjudications_dir=tmp_path))
        assert res.payload["p6_i1_adjudication"] is None
        assert "already been adjudicated" in capsys.readouterr().err
        assert (tmp_path / "P6-I1.json").read_text() == first

    def test_a_refused_run_says_so_in_its_summary(self, tmp_path):
        run_induction_ov(_ctx("A", adjudicate=True, adjudications_dir=tmp_path))
        res = run_induction_ov(_ctx("A", seed=7, adjudicate=True,
                                    adjudications_dir=tmp_path))
        text = "\n".join(res.summary_lines)
        assert "NOT recorded" in text
        assert "not a failed prediction" in text
