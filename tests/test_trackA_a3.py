"""A3's locator (`trackA_force/a3.py`, design-A3.md §4): pure parts, synthetic."""
import numpy as np
import pytest

from trackA_force import a3

pytestmark = pytest.mark.deps  # a1a's imports
from trackA_force.a1a import STEPS


def _step_at(i, amp=1.0, sign=1.0):
    x = np.zeros(len(STEPS)); x[i:] = sign * amp
    return x


def test_half_rise_places_first_step_past_half():
    i = STEPS.index(512)
    r = a3.place(_step_at(i), "up", sigma=0.01)
    assert r["reading"] == "located" and r["step"] == 512 and r["low_power"] is False


def test_down_row_is_signed():
    r = a3.place(_step_at(STEPS.index(64), sign=-1.0), "down", sigma=0.01)
    assert r["step"] == 64
    assert a3.place(_step_at(STEPS.index(64)), "down", sigma=0.01)["reading"] == "no change"


def test_floor_and_low_power():
    x = _step_at(STEPS.index(256))
    assert a3.place(x, "up", sigma=1 / 3.9)["reading"] == "no change"
    r = a3.place(x, "up", sigma=1 / 5.0)
    assert r["reading"] == "located" and r["low_power"] is True


def test_reference_ignores_steps_after_4000():
    x = _step_at(STEPS.index(8000), amp=10.0)
    x[STEPS.index(128):STEPS.index(8000)] = 1.0
    assert a3.place(x, "up", sigma=0.01)["step"] == 128


def test_turning_point_is_the_minimum():
    x = np.zeros(len(STEPS))
    x[STEPS.index(64):] = -1.0; x[STEPS.index(128)] = -2.0; x[STEPS.index(1000):] = -3.0
    r = a3.place(x, "down", sigma=0.01, kind="turn")
    assert r["step"] == 1000  # the minimum over <= 4000 is the late fall
    x[STEPS.index(1000):] = -0.5
    assert a3.place(x, "down", sigma=0.01, kind="turn")["step"] == 128


def test_lr_anchor_is_1000_and_d4a():
    e10 = np.array([a3.lr(s) for s in STEPS])
    assert a3.place(e10, "up", None, floor=None)["step"] == 1000
    v = np.zeros(len(STEPS)); v[STEPS.index(256)] = 10; v[STEPS.index(512):] = 57
    assert a3.d4a(v)["step"] == 512


def test_synthetic_spread_matches_the_rows_own():
    """The gate's noise has the real row's per-step spread, in the statistic's units."""
    rng = np.random.default_rng(1)
    n, S = 35, len(STEPS)
    tau = np.geomspace(1e-6, 1e-2, S); tau[0] = 0
    w = rng.integers(400, 3000, size=(n, 1)).astype(float)
    score = rng.standard_normal((n, S)) * tau + 0.004
    pooled = np.stack([score * w, np.broadcast_to(w, (n, S))], axis=-1)
    for units, name in ((score[..., None], "mean"), (pooled, "pooled")):
        real = a3.unit_scores(units, name)[..., 0]
        want = np.std(real - real[:, :1], axis=0, ddof=1)
        syn = a3._synthetic(units, "up", name, rng, 4000)
        got = np.std(a3.unit_scores(syn.reshape((-1,) + syn.shape[2:]), name)
                     .reshape(4000, n, S), axis=(0, 1))
        assert np.allclose(got[1:], want[1:], rtol=0.05)


def test_gate_with_constant_noise_passes_like_the_calibration(monkeypatch):
    """With one noise size at every step the gate passes, as a3_locator.py's iid model does."""
    monkeypatch.setattr(a3, "N_GATE", 200)
    monkeypatch.setattr(a3, "N_BOOT", 200)
    rng = np.random.default_rng(2)
    u = rng.standard_normal((7, len(STEPS), 1))
    g = a3.gate_row("E4", u, rng)
    assert g["null_no_change"] >= 0.95
    assert min(e["placed_right"] for e in g["effect"]) >= 0.85


def test_bootstrap_se_of_mean_delta():
    rng = np.random.default_rng(0)
    u = np.zeros((400, len(STEPS), 1)); u[:, 5, 0] = rng.standard_normal(400)
    se = a3.boot_se(u, a3.stat_mean, 5, rng)
    assert abs(se - 1 / np.sqrt(400)) < 0.01


def test_boot_se_all_matches_one_step():
    u = np.random.default_rng(3).standard_normal((50, len(STEPS), 1))
    a = a3.boot_se_all(u, a3.stat_mean, np.random.default_rng(4), n_boot=4000)
    b = a3.boot_se(u, a3.stat_mean, 6, np.random.default_rng(5), n_boot=4000)
    assert a[0] == 0 and abs(a[6] - b) / b < 0.06


def test_planted_shape_shifts_each_statistic_exactly():
    """Why one bootstrap per noise draw serves every amplitude (gate_row's docstring)."""
    rng = np.random.default_rng(6)
    n, S = 35, len(STEPS)
    w = rng.integers(400, 3000, size=(n, 1)).astype(float)
    z = rng.standard_normal((n, S)); sh = np.linspace(0, 2, S)
    for name in ("mean", "median", "pooled", "medmed"):
        if name == "medmed":                       # 4 types, layers resampled together
            zt = rng.standard_normal((n, S, 4))
            st = a3.STATS[name]
            assert np.allclose(st(zt + sh[:, None]), st(zt) + sh)
            continue
        if name == "pooled":
            mk = lambda y: np.stack([y * w, np.broadcast_to(w, (n, S))], axis=-1)
        else:
            mk = lambda y: y[..., None]
        st = a3.STATS[name]
        assert np.allclose(st(mk(z + sh)), st(mk(z)) + sh)


def test_censored_and_rederived_low_power():
    x = _step_at(STEPS.index(4000))
    r = a3.place(x, "up", sigma=0.01)
    assert a3.censored(r) is True
    r2 = a3.place(_step_at(STEPS.index(256)), "up", sigma=0.1)    # ref / σ = 10
    assert a3.censored(r2) is False
    gate = {"effect": [{"to_index": STEPS.index(256), "power": 12, "sigma_ref": 0.1}]}
    assert a3.low_power(r2, gate) is True
    gate["effect"][0]["power"] = 6
    assert a3.low_power(r2, gate) is False
    gate["effect"][0]["sigma_ref"] = 0.5          # power is in the gap's σ̄_ref, not the row's σ
    assert a3.low_power(r2, gate) is True
    assert a3.low_power({"reading": "no change"}, gate) is None


def test_gate_floor_is_rederived_and_checked(monkeypatch):
    monkeypatch.setattr(a3, "N_GATE", 1000)
    monkeypatch.setattr(a3, "N_BOOT", 200)
    rng = np.random.default_rng(7)
    g = a3.gate_row("E4", rng.standard_normal((7, len(STEPS), 1)), rng)
    assert 1.5 < g["floor"] < 6 and g["null_no_change"] >= 0.95   # the upper bound's margin
    assert all(e["power"] is not None and e["power"] <= 18 for e in g["effect"])
    assert g["effect"][0]["curve"]["2"] < g["effect"][0]["curve"]["9"]   # power rises


def test_floor_is_at_least_four(monkeypatch):
    """Blocked 72 (iii) (a): the floor used is max(4, the row's own bound)."""
    monkeypatch.setattr(a3, "N_GATE", 300)
    monkeypatch.setattr(a3, "N_BOOT", 100)
    rng = np.random.default_rng(8)
    g = a3.gate_row("E2", rng.standard_normal((384, len(STEPS), 1)), rng)
    assert g["own_floor"] < 4 and g["floor"] == 4.0


def test_medmed_row_gates(monkeypatch):
    monkeypatch.setattr(a3, "N_GATE", 200)
    monkeypatch.setattr(a3, "N_BOOT", 100)
    rng = np.random.default_rng(9)
    u = rng.standard_normal((24, len(STEPS), 4)); u[:, 0, :] = 0
    syn = a3._synthetic(u, "up", "medmed", rng, 5)
    assert syn.shape == (5, 24, len(STEPS), 4)
    g = a3.gate_row("E7", u, rng)
    assert g["floor"] >= 4 and g["null_no_change"] >= 0.9


def test_powerlaw_alpha_recovers_pareto():
    from trackA_force.a3_weights import powerlaw_alpha
    rng = np.random.default_rng(10)
    for a in (2.0, 3.0, 4.5):
        x = (1 - rng.random(2000)) ** (-1 / (a - 1))
        assert abs(powerlaw_alpha(x)["alpha"] - a) < 0.25


def test_n_clusters():
    from trackA_force.a3_sim import n_clusters
    e = np.eye(4)
    U = np.stack([e[0], e[0], e[1], (e[1] + 0.01 * e[2]) / np.linalg.norm(e[1] + 0.01 * e[2]), e[3]])
    assert n_clusters(U) == 3


def test_constant_row_is_refused_by_name():
    u = np.ones((7, len(STEPS), 1))
    assert a3.constant_row(u, "mean") is True
    u[0, 3, 0] = 2.0
    assert a3.constant_row(u, "mean") is False


def test_strict_pairs_key_follows_an_earlier_copy_of_the_query():
    """E3 at +1 (Blocked 72 (ii) (a)): key − 1 holds the query's token; the frozen E3's key is
    aligned with the query (key − 1 holds the query's predecessor)."""
    from core.battery_structure import induction_candidates
    ids = [10, 11, 12, 13, 10, 11, 12, 13]
    strict = set(induction_candidates(ids, strict=True))
    assert (4, 1) in strict and (5, 2) in strict          # query 10 at 4 -> key 11 at 1
    assert all(ids[k - 1] == ids[q] for q, k in strict)
    off0 = set(induction_candidates(ids))
    assert (5, 1) in off0 and all(ids[k - 1] == ids[q - 1] for q, k in off0)


def test_dip_has_its_minimum_one_step_past_the_gap():
    for i in a3.WIN[1:-1]:
        d = a3._dip(i)
        assert int(np.argmin(d[list(a3.WIN)][1:])) + 1 == i + 1
        x = 10 * d
        r = a3.place(x, "turn", sigma=0.1, kind="turn")
        assert r["index"] == i + 1 and r["reading"] == "located"


def test_turn_row_gates_on_its_dip(monkeypatch):
    """E6's gate (design-A3 §4, unit 4): the null and the planted dip at the row's own noise."""
    monkeypatch.setattr(a3, "N_GATE", 300)
    monkeypatch.setattr(a3, "N_BOOT", 100)
    rng = np.random.default_rng(11)
    g = a3.gate_row("E6u", rng.standard_normal((64, len(STEPS), 1)), rng)
    assert g["floor"] >= 4 and g["null_no_change"] >= 0.93
    assert min(e["placed_right"] for e in g["effect"]) >= 0.85


def test_turn_ratio_is_the_fall_into_the_minimum():
    x = np.zeros(len(STEPS)); x[STEPS.index(64)] = -3.0; x[STEPS.index(128):] = -1.0
    sig = np.ones(len(a3.WIN))
    assert a3._ratio(x, sig, "turn", "turn") == 3.0
    assert a3._ratio(-x, sig, "turn", "turn") == -np.inf      # no fall: always "no change"


def test_pack_blocks_starts_at_documents_and_shares_none():
    from trackA_force.a3_ngram import pack_blocks
    docs = [(i, [i + 1] * n + [0]) for i, n in enumerate([3, 9, 2, 2, 5, 1, 7, 4, 4, 4])]
    c = pack_blocks(iter(docs), n_blocks=3, block=4, n_table=6)
    assert c["block_docs"] == [[0], [1], [2, 3]]
    assert c["blocks"].tolist() == [[1, 1, 1, 0], [2] * 4, [3, 3, 0, 4]]
    assert c["table_docs"] == [4] and c["table"].tolist() == [5, 5, 5, 5, 5, 0]


def test_ngram_overlap_runs():
    from trackA_force.a3_ngram import ngrams
    assert ngrams([1, 2, 3, 4], n=3) == {(1, 2, 3), (2, 3, 4)}
    assert ngrams([1, 2], n=3) == set()


def test_witten_bell_rows_and_cross_terms_match_dense_kl():
    """The sparse cross term gives KL(q ‖ p) as computed densely, for both predictors."""
    import torch
    from trackA_force import a3_ngram as ng
    rng = np.random.default_rng(12)
    V = 40
    table = rng.integers(0, 30, size=3000)            # ids 30..39 unseen: q_u = 0 there
    t = ng.tables(table, V=V)
    assert abs(t["q_u"].sum() - 1) < 1e-12
    for v in range(V):
        assert abs(ng.q_b_row(t, v).sum() - 1) < 1e-12
    assert np.array_equal(ng.q_b_row(t, 35), t["q_u"])     # unseen context: q_u
    ctx = np.array([0, 5, 35, 5, 29])
    logp = torch.log_softmax(torch.from_numpy(rng.standard_normal((len(ctx), V))), dim=-1)
    cu, cb = ng.cross_terms(logp, ctx, t, torch.from_numpy(t["q_u"]))
    P = logp.numpy()
    for k, v in enumerate(ctx):
        qb = ng.q_b_row(t, int(v))
        want_u = ng.neg_entropy(t["q_u"]) - t["q_u"] @ P[k]
        want_b = ng.neg_entropy(qb) - qb @ P[k]
        assert abs((ng.neg_entropy(t["q_u"]) - float(cu[k])) - want_u) < 1e-10
        assert abs((ng.neg_entropy(qb) - float(cb[k])) - want_b) < 1e-10
        assert want_b >= 0 and want_u >= 0


def _ngram_dir(tmp_path, loss_end=2.2, step0_off=0.0, bad_hash=False):
    """A fake a3_ngram record set: 18 steps, a small table, its manifest."""
    import hashlib, json, math
    from trackA_force import a3_ngram as ng
    rng = np.random.default_rng(13)
    blocks = rng.integers(0, 100, size=(4, 8)); table = rng.integers(0, 100, size=500)
    np.savez(tmp_path / "corpus.npz", blocks=blocks, table=table)
    sha = lambda a: hashlib.sha256(a.tobytes()).hexdigest()[:12]
    (tmp_path / "corpus.json").write_text(json.dumps(
        {"blocks_sha": "0" * 12 if bad_hash else sha(blocks), "table_sha": sha(table)}))
    flat = ng.neg_entropy(ng.tables(table)["q_u"]) + math.log(ng.VOCAB)
    for s in STEPS:
        ku = flat * (1 + step0_off) if s == 0 else 1.0
        (tmp_path / f"step{s}.json").write_text(json.dumps(
            {"step": s, "code": "c", "device": "cpu", "kl_u": [ku] * 4, "kl_b": [2.0] * 4,
             "loss": [loss_end if s == 143000 else 5.0] * 4}))
    return tmp_path


def test_rows_ngram_reads_and_refuses(tmp_path):
    """The checks placed before output (design-A3 §4, unit 4) refuse; #236 finding 4."""
    r = a3.rows_ngram(_ngram_dir(tmp_path))
    assert r["E6u"].shape == (4, len(STEPS), 1)
    for kw in ({"loss_end": 5.0}, {"step0_off": 0.2}, {"bad_hash": True}):
        d = tmp_path / str(len(kw) + hash(str(kw)) % 1000); d.mkdir()
        with pytest.raises(a3.A3Refused):
            a3.rows_ngram(_ngram_dir(d, **kw))


def test_overlap_check_equals_the_set_check():
    """Unit 5's fast check is unit 4's ``ngrams(ids) & bad``, including on hash collisions'
    confirmation (design-A3 §4 "Placed before any unit 5 output")."""
    from trackA_force.a3_ngram import ngrams, overlap_check
    rng = np.random.default_rng(14)
    prompts = [rng.integers(0, 50, size=40).tolist() for _ in range(3)]
    bad = set().union(*(ngrams(p, n=5) for p in prompts))
    hit = overlap_check(bad, n=5)
    for k in range(300):
        d = rng.integers(0, 50, size=int(rng.integers(0, 60))).tolist()
        if k % 3 == 0 and len(d) > 10:                     # plant a run from a prompt
            p = prompts[k % 3]; a = int(rng.integers(0, 35))
            d[3:8] = p[a:a + 5]
        assert hit(d) == bool(ngrams(d, n=5) & bad)


def test_tables_chunked_counts_every_pair_once():
    from trackA_force import a3_ngram as ng
    rng = np.random.default_rng(15)
    table = rng.integers(0, 30, size=1001).astype(np.int32)
    one, many = ng.tables(table, V=40), ng.tables(table, V=40, chunk=7)
    assert (one["B"] != many["B"]).nnz == 0 and one["B"].sum() == len(table) - 1
    assert np.array_equal(one["c"], many["c"]) and np.array_equal(one["T"], many["T"])


def _rows(**idx):
    """Synthetic located rows by grid index; a value None is "no change", "c" censored."""
    rows = {}
    for k, i in idx.items():
        if i is None:
            rows[k] = {"reading": "no change", "index": None}
        elif i == "c":
            rows[k] = {"reading": "located", "index": 11, "censored": True}
        else:
            rows[k] = {"reading": "located", "index": i, "censored": False}
    return rows


def test_pairs_aligned_misaligned_unread():
    """design-A3 §3 and §4 "Placed before any unit 5 output"."""
    rows = _rows(E1=9, E2=9, E3=10, E5=3, E7="c", E11=None)
    assert a3.read_pair(rows, "E1", "E3")["verdict"] == "aligned"
    p = a3.read_pair(rows, "E1", "E5")
    assert p["verdict"] == "misaligned" and p["first"] == "E5"
    assert a3.read_pair(rows, "E1", "E7")["verdict"] == "unread"        # censored, 73 (ii)
    assert a3.read_pair(rows, "E1", "E11")["verdict"] == "unread"
    assert "a reproduction" in " ".join(a3.read_pair(rows, "E1", "E2")["notes"])
    assert "inside warmup (E10)" in a3.read_pair(rows, "E1", "E3")["notes"]


def test_two_dates_read_only_where_both_agree():
    """Blocked 74 (a) for E6b, 73 (iii) for E8: one verdict from both dates, else unread."""
    agree = _rows(E1=9, E6b=8, E6b_full=9)
    assert a3.read_pair(agree, "E1", "E6b")["verdict"] == "aligned"
    split = _rows(E1=9, E6b=8, E6b_full=11)                 # aligned at 256, misaligned at 2000
    assert a3.read_pair(split, "E1", "E6b")["verdict"] == "unread (table)"
    both_mis = _rows(E1=9, E6b=5, E6b_full=6)
    p = a3.read_pair(both_mis, "E1", "E6b")
    assert p["verdict"] == "misaligned" and p["first"] == "E6b"
    flips = _rows(E1=9, E6b=7, E6b_full=11)                 # misaligned in opposite orders
    assert a3.read_pair(flips, "E1", "E6b")["verdict"] == "unread (table)"
    assert a3.read_pair(_rows(E1=9, E6b=8, E6b_full=None), "E1", "E6b")["verdict"] == "unread"
    e8 = _rows(E1=9, E8=10, E8_peak=9)
    assert a3.read_pair(e8, "E1", "E8")["verdict"] == "aligned"
    assert a3.read_pair(_rows(E1=8, E8=10, E8_peak=9), "E1", "E8")["verdict"] == \
        "unread (two dates)"


def test_unigram_first_needs_both_dates():
    assert a3.unigram_first(_rows(E6u=7, E6b=8, E6b_full=9)) is True
    assert a3.unigram_first(_rows(E6u=7, E6b=8, E6b_full=7)) is False
    assert a3.unigram_first(_rows(E6u=7, E6b=8, E6b_full=None)) is None


def test_order_groups_ties_and_skips_censored():
    o = a3.order(_rows(E1=9, E2=9, E5=3, E7="c", E6b=8, E6b_full=9))
    assert [e["rows"] for e in o] == [["E5"], ["E6b"], ["E1", "E2"]]
    assert o[1]["steps"] == [STEPS[8], STEPS[9]]


def test_from_peak_dates_the_fall_after_a_rise():
    x = np.zeros(len(STEPS)); x[:] = 10.0
    x[8] = 17.0                                              # a peak at 256
    x[9:13] = [10.0, 5.5, 3.0, 4.0]                         # least at 2000: not censored
    r = a3.from_peak(x, "down")
    assert r["peak_step"] == 256 and r["step"] == 512 and not r["censored"]
