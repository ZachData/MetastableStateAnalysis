"""
`p1e_energy_field/u1_field.py` — the parts that need scikit-learn (AMI): AMI and the c3x purity
at chance under permutation, the opening's well (`design-1e.md` "U1: the rule", tests).
"""
import numpy as np
import pytest

from p1e_energy_field import u1_field as u1

pytestmark = pytest.mark.deps


def test_ami_and_purity_at_chance_under_permutation():
    from sklearn.metrics import adjusted_mutual_info_score as ami
    rng = np.random.default_rng(4)
    w = rng.integers(0, 3, 600)
    pos = np.arange(1, 601)
    vals = [ami(u1.position_bins(pos), rng.permutation(w)) for _ in range(50)]
    assert abs(np.mean(vals)) < 0.01
    d = u1.describe(np.repeat([0, 1, 2], 200), pos, rng.permutation(np.repeat(list("abc"), 200)),
                    [np.arange(0, 10), np.arange(300, 310)], rng)
    assert d["ami_pos"] > 0.5 and abs(d["ami_cls"]) < 0.02
    assert d["purity"] == 1.0 and d["purity_p"] < 0.01
    assert d["open_share"] == 1.0 and abs(d["open_well_share"] - 1 / 3) < 1e-12
