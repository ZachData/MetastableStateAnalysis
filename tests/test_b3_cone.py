"""B3 unit 2's parts (`trackB_circuit/design-B3.md` "L"): inputs and partners, the log ratio and
its interval, the onset, the confinement share, L1's null and false pass, G4's draws (the hooks: `test_b3_cone_smoke.py`)."""

import numpy as np
import pytest

from trackB_circuit import b3_cone as bc

pytestmark = pytest.mark.pure


def test_inputs_are_seeded_repeated_and_controls_share_the_first_half():
    a, c = bc.tokens(3), bc.tokens(3, control=True)
    assert a == bc.tokens(3) and len(a) == bc.N
    assert a[:bc.HALF] == a[bc.HALF:]                     # repeated
    assert c[:bc.HALF] == a[:bc.HALF] and c[bc.HALF:] != a[bc.HALF:]
    assert min(a) >= bc.VOCAB[0] and max(a) < bc.VOCAB[1]
    J = bc.positions(3)
    assert len(set(J)) == bc.N_J and J.min() >= bc.J_LO and J.max() < bc.J_HI
    assert (J + bc.PARTNER0 + max(bc.MATCH)).max() < bc.N
    assert np.allclose(np.linalg.norm(bc.directions(3), axis=1), 1)


def test_partner_is_the_query_that_reads_key_j_at_offset_plus_one():
    # +1 induction: query HALF + j′ attends key j′ + 1, so key j is read by query HALF + j − 1
    j = 20
    assert bc.HALF + (j - 1) == j + bc.PARTNER
    assert bc.HALF + j == j + bc.PARTNER0                  # offset 0 (`L7H8`)


def test_log_ratio_reads_partner_over_median_of_six_neighbours():
    S, nJ, T = 2, 3, bc.N
    sig = np.ones((S, nJ, T))
    J = np.array([[10, 20, 30], [11, 21, 31]])
    for s in range(S):
        for a in range(nJ):
            sig[s, a, J[s, a] + bc.PARTNER] = np.e
            sig[s, a, J[s, a] + bc.PARTNER + 1] = 100.0   # i + 1 is not in the matched set
    v = bc.log_ratio(sig, J, bc.PARTNER)
    assert np.allclose(v, 1.0)
    assert bc.ell(v) == pytest.approx(1.0)


def test_boot_ci_brackets_and_resamples_sequences():
    g = np.random.default_rng(0)
    v = g.normal(0.5, 0.1, (32, 16))
    lo, hi = bc.boot_ci(v, 1, n=2000)
    assert lo < np.median(v) < hi and lo > 0
    z = g.normal(0, 1, (32, 16))
    lo, hi = bc.boot_ci(z, 2, n=2000)
    assert lo < 0 < hi


def test_onset_needs_two_steps_above():
    steps = [0, 2, 4, 8]
    assert bc.onset(steps, [-1, 0.1, -0.1, 0.2]) is None
    assert bc.onset(steps, [-1, 0.1, 0.1, -0.2]) == 2


def test_confinement_is_partner_share_from_j_plus_2():
    sig = np.zeros((1, 1, bc.N))
    J = np.array([[10]])
    sig[0, 0, 11] = 50                                    # j + 1 left out
    sig[0, 0, 10 + bc.PARTNER] = 3
    sig[0, 0, 40] = 1
    assert bc.confinement(sig, J)[0, 0] == pytest.approx(0.75)


def test_null_sets_layer_matched_without_members():
    ns = bc.null_sets()
    assert ns.shape == (bc.N_NULL, 3)
    assert [set(ns[:, k] // bc.N_HEADS) for k in range(3)] == [{7}, {8}, {11}]
    mem = {bc.head_index(m) for m in bc.MEMBERS}
    assert not mem & set(ns.ravel())
    assert (ns == bc.null_sets()).all()


def test_loo_false_pass_on_distinct_values():
    assert bc.loo_false_pass(np.arange(200.0)) == pytest.approx(2 / 200)   # 199/199 and 198/199; 197/199 < 0.99
    assert bc.rank_of(5.0, np.arange(10.0)) == 0.5


def test_g4_draws_layer8_non_members_and_d_range():
    d = bc.g4_draws()
    assert len(d) == bc.G4_SEEDS
    assert all(h // bc.N_HEADS == 8 and h != bc.head_index("L8H6") for h, _ in d)
    assert all(bc.G4_D[0] <= x <= bc.G4_D[1] for _, x in d)
    assert max(x for _, x in d) + 4 < bc.CUT_G4


def test_floored_replaces_and_counts():
    s, b = bc.floored(np.array([[0.0, 1e-3, 1.0]]), np.array([1e-3]), 0.5)
    assert np.allclose(s, [[2e-3, 2e-3, 2.0]]) and b.tolist() == [[True, True, False]]


def test_l6_configs_single_heads_of_layer_6_and_all_together():
    from trackB_circuit import b3_cone_l6 as l6
    cf = l6.configs()
    assert cf[0] == {"name": "none"} and len(cf) == 1 + bc.N_HEADS + 1
    singles = cf[1:-1]
    assert [c["name"] for c in singles] == [f"L6H{h}" for h in range(bc.N_HEADS)]
    assert all(len(c["ablate"]) == 1 and c["ablate"][0] // bc.N_HEADS == 6 for c in singles)
    assert sorted(cf[-1]["ablate"]) == [6 * bc.N_HEADS + h for h in range(bc.N_HEADS)]
    assert l6.LAYER > bc.SITE and l6.READ_MID == l6.LAYER + 1
