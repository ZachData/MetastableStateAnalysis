"""
tests/test_run_behavioural.py — the parts of P-I1's B arm that are new here.

`tools/run/behavioural.py` reuses `induction_candidates` and `tokenize_prompt`
(tested in `core`) and `behavioural_induction_score`'s reading of the attention
tensor (tested in `test_p7_formation_curve.py`). What is new and file-format
dependent is two things: parsing `tokens.txt` back to token strings for the
tokenisation tripwire, and the pair-count-weighted cross-prompt pool. Both are
checked here; the sibling `curve.py` has no test for the same reason its logic
lives in `core`.
"""

import numpy as np
import pytest

from tools.run.behavioural import (
    BehaviouralArmRefused,
    N_HEADS,
    N_LAYERS,
    _tokens_txt,
    prompt_pair_sums,
)

pytestmark = pytest.mark.pure


class TestTokensTxtParse:
    def test_reads_the_writer_format(self, tmp_path):
        # p1_io.py writes f"{i:3d}  {tok}\n" — 3-wide index, two spaces, token.
        (tmp_path / "tokens.txt").write_text(
            "  0  But\n"
            "  1  Ġthen\n"
            " 10  ,\n"
            "100  Ġthe\n"
        )
        assert _tokens_txt(tmp_path) == ["But", "Ġthen", ",", "Ġthe"]

    def test_keeps_internal_and_trailing_spaces_in_a_token(self, tmp_path):
        (tmp_path / "tokens.txt").write_text("  0  a b \n")
        assert _tokens_txt(tmp_path) == ["a b "]


def _attn(n=6):
    a = np.zeros((N_LAYERS, N_HEADS, n, n), dtype=np.float32)
    return a


class TestPromptPairSums:
    def test_sum_not_mean_over_pairs(self):
        a = _attn()
        a[2, 3, 4, 1] = 0.4
        a[2, 3, 5, 2] = 0.6
        s, npair = prompt_pair_sums(a, [(4, 1), (5, 2)])
        assert npair == 2
        assert s[2, 3] == pytest.approx(1.0)          # 0.4 + 0.6, not 0.5
        assert s[0, 0] == 0.0

    def test_empty_pairs_is_zeros_and_zero_count(self):
        s, npair = prompt_pair_sums(_attn(), [])
        assert npair == 0
        assert s.shape == (N_LAYERS, N_HEADS)
        assert not s.any()

    def test_a_pair_outside_the_matrix_is_refused(self):
        with pytest.raises(BehaviouralArmRefused):
            prompt_pair_sums(_attn(n=6), [(9, 1)])

    def test_wrong_head_shape_is_refused(self):
        with pytest.raises(BehaviouralArmRefused):
            prompt_pair_sums(np.zeros((N_LAYERS, N_HEADS + 1, 4, 4)), [(2, 0)])

    def test_pool_is_pair_count_weighted(self):
        # Two "prompts": one contributes 3 pairs summing to 0.9 at (L1,H1),
        # the other 1 pair of 0.5. The pooled mean is 1.4 / 4 = 0.35, not the
        # unweighted mean of the per-prompt means (0.3 + 0.5) / 2 = 0.4.
        a1 = _attn()
        a1[1, 1, 2, 0] = a1[1, 1, 3, 0] = a1[1, 1, 4, 1] = 0.3
        a2 = _attn()
        a2[1, 1, 5, 2] = 0.5
        s1, c1 = prompt_pair_sums(a1, [(2, 0), (3, 0), (4, 1)])
        s2, c2 = prompt_pair_sums(a2, [(5, 2)])
        pooled = (s1 + s2) / (c1 + c2)
        assert pooled[1, 1] == pytest.approx(0.35)
