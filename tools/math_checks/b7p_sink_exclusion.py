"""B7′'s sink-excluded read: what dropping a prompt's second-sink keys does to a row, why B7's N0
could not remove such a key, that the detection rule finds nothing under B7's planted nulls at the
real design points, and N0 re-estimated per bootstrap draw (`docs/ROADMAP.md` §7, B7′;
`trackB_circuit/design-B7p.md` "The math").

WHY THIS EXISTS
---------------
B7 read mid-depth persistence that was mostly rows kept on one early `.` or `\\n` (a second sink;
`trackB_circuit/status-B.md` "B7 on one page"). B7′ drops each prompt's second-sink keys as position
0 is dropped, under a rule fixed before output, and re-gates with a planted second sink. This check
fixes, before any row is read, what the drop does to a planted row, and what the rule and the
re-estimated N0 do on the planted nulls.

WHAT IT SHOWS
-------------
1. **Dropping keys keeps the order of the rest.** Renormalising a row over keys outside a set S
   divides every kept entry by the same 1 − s, so the argmax among kept keys is unchanged and a
   row condensed on a kept key stays condensed on it (its share a / (1 − s) ≥ a).
2. **A planted sink row, before and after the drop.** A row putting 0.7 of its non-sink mass on a
   sink key k, 0.2 on a content partner π ≠ k and 0.1 uniformly over keys 1..i is condensed on k
   (share 0.7 + 0.1 / i > 1/2), and after k is dropped condensed on π (share
   (0.2 + 0.1/i) / (0.3 − 0.1/i) > 1/2 for every i ≥ 2): the hidden partner is recovered. A row
   putting 0.8 on k and 0.2 uniformly has, after the drop, equal shares 1 / (i − 1) on every key,
   so it is condensed only at i = 2 (never at i ≥ 16): the drop makes no partner up.
3. **Why B7's N0 could not remove a fixed key.** Rows of one prompt on one key k have offsets
   i − k, distinct within a 32-position bin; the offsets of two different rows agree only when
   their positions do. So a bin of frozen sink rows has P = 1 and its c_i comes only from other
   prompts' rows at the same offset: at the real n's, with one sink key per prompt drawn in
   10..54, N0 ≤ 0.05 while P = 1, and the cell reads "persistence beyond position".
4. **The detection rule finds nothing under B7's planted nulls at the real design points.** A
   key is a sink of (prompt, layer) when it holds ≥ 1/4 of that layer's condensed rows (16 heads,
   ≥ 50 rows). With partners uniform on 1..i − 2 (G2, G7), one map in every head (G1) or
   previous-token (G4), the largest share of any key over 400 draws per prompt is about 0.05 at
   full rows (key 1, which every row can reach; the assert is < 0.1), and at 50 rows thinned at random it reaches 1/4 in none of 2000 draws.
5. **N0 per bootstrap draw (the U-statistic).** With prompt multiplicities m_q from a draw,
   Σ_{j ≠ j′} m_j m_j′ x_j y_j′ = (Σ m x)(Σ m y) − Σ m² x y, so N0 over pairs of different
   *original* rows is a closed form in the per-prompt offset histograms. Counting a row's copies
   as different rows (weight m (m − 1)) brings back unit 1b's self-pair bias in every draw that
   repeats a prompt: at 50 rows of frozen partners it puts N0 several times its reshuffled value,
   while over different original rows it stays within 15 % of it (or 4 standard errors of the
   simulation's difference, 1500 draws).

WHAT IT DOES NOT PROVE
----------------------
- That 410m's second sinks are rows like check 2's (the share on the sink and what is under it are
  read, not assumed); a sink row with mass on 2–3 content keys is out after the drop, as B7's
  diffuse rows are.
- That ≥ 1/4 of a layer's condensed rows separates sinks from content in 410m: it is placed in the
  gap of B7's sink table (on the key 0.50–0.94 of a prompt's mid-depth rows; on any other key ≤ 0.06),
  which was seen before this rule was written, so it is not a blind threshold. A content key that
  a quarter of a layer's condensed rows point at is dropped as a sink, whatever its token.
- That the drop leaves "content" rows: what remains is rows whose top non-sink key has > 1/2 of
  the non-sink mass; positional rows remain (N0 takes them out, as in B7).
- Check 5's expectation is for the counts; c_i restricted to offsets valid at i is a ratio
  (B7's check 7 caveat), bounded here by simulation only.
"""
import numpy as np
import sympy as sp

# 1. Dropping a set S: kept entries scale by 1/(1 − s), order and condensation kept.
a, s = sp.symbols("a s", nonnegative=True)
assert sp.simplify(a / (1 - s) - a - a * s / (1 - s)) == 0          # a/(1−s) = a + a s/(1−s) ≥ a

# 2. Planted rows. Hidden partner: before the drop condensed on k, after it on π.
i = sp.symbols("i", positive=True, integer=True)
before = sp.Rational(7, 10) + sp.Rational(1, 10) / i
after = (sp.Rational(2, 10) + sp.Rational(1, 10) / i) / (sp.Rational(3, 10) - sp.Rational(1, 10) / i)
assert sp.simplify(before - sp.Rational(1, 2) - (sp.Rational(1, 5) + sp.Rational(1, 10) / i)) == 0   # > 0
assert sp.simplify(after - (2 * i + 1) / (3 * i - 1)) == 0
assert sp.simplify(after - sp.Rational(1, 2) - (i + 3) / (2 * (3 * i - 1))) == 0              # > 0 for i ≥ 1
# Pure sink row: after the drop every kept key has 0.2/i of 0.2 (i − 1)/i, share 1/(i − 1).
pure = (sp.Rational(2, 10) / i) / (sp.Rational(2, 10) * (i - 1) / i)
assert sp.simplify(pure - 1 / (i - 1)) == 0
assert all(1 / (v - 1) <= 0.5 for v in range(3, 600)) and 1 / (16 - 1) < 0.5

# 3. B7's N0 on a bin of frozen sink rows: one key per prompt, P = 1, N0 small.
rng = np.random.default_rng(0)
NS, H, BIN, ROW = (465, 242, 512, 446, 286, 482, 467), 16, 32, 16


def n0_bins(pos, pid, oa, ob, w=None, copies=False):
    """Mean c_i over rows, c_i over ordered pairs of different rows of i's bin with offsets ≤ i − 1
    whose offsets agree; `w` the rows' prompt multiplicities (a draw), rows weighted by w in the
    mean. `copies` counts a row's own copies as different rows (weight w (w − 1))."""
    w = np.ones(len(pos)) if w is None else w
    num_all, den_all = 0.0, 0.0
    for b0 in range(ROW, pos.max() + 1, BIN):
        r = np.nonzero((pos >= b0) & (pos < b0 + BIN))[0]
        if r.size == 0:
            continue
        for k in r:
            v = r[(np.maximum(oa[r], ob[r]) <= pos[k] - 1)]
            if v.size < 2:
                continue
            wa, ag = w[v], (oa[v][:, None] == ob[v][None, :])
            W = wa[:, None] * wa[None, :]
            if copies:
                W[np.diag_indices_from(W)] = wa * (wa - 1)
            else:
                W[np.diag_indices_from(W)] = 0.0
            if W.sum() > 0:
                num_all += w[k] * (W * ag).sum() / W.sum()
                den_all += w[k]
    return num_all / den_all


pos = np.concatenate([np.arange(ROW, n) for n in NS])
pid = np.concatenate([np.full(n - ROW, q) for q, n in enumerate(NS)])
ks = rng.integers(10, 55, len(NS))
off = pos - ks[pid]                                       # every row on its prompt's sink key
n0_sink = n0_bins(pos, pid, off, off)
assert n0_sink < 0.05, n0_sink                            # P = 1 on these rows; P − N0 > 0.95

# 4. The detection rule under B7's nulls: the largest key share of a (prompt, layer)'s condensed rows.
TAU, MINR = 0.25, 50


def max_share(keys):
    return np.bincount(keys).max() / keys.size


full, thin = [], []
for n in NS:
    ii = np.arange(ROW, n)
    for _ in range(400):
        g2 = np.concatenate([rng.integers(1, ii - 1) for _ in range(H)])          # G2 / G7
        g1 = np.tile(rng.integers(1, ii - 1), H)                                  # G1
        full += [max_share(g2), max_share(g1), max_share(np.tile(ii - 1, H))]    # G4
    for _ in range(2000 // len(NS) + 1):
        g2 = np.concatenate([rng.integers(1, ii - 1) for _ in range(H)])
        thin.append(max_share(rng.choice(g2, MINR, replace=False)))
assert max(full) < 0.1, max(full)                         # a fifth of the rule or less
assert sum(t >= TAU for t in thin) == 0, max(thin)

# 5. N0 per draw: the identity, then copies against different original rows at 50 rows.
m = sp.symbols("m0:4", positive=True)
x, y = sp.symbols("x0:4"), sp.symbols("y0:4")
lhs = sum(m[j] * m[k] * x[j] * y[k] for j in range(4) for k in range(4) if j != k)
rhs = sum(m[j] * x[j] for j in range(4)) * sum(m[k] * y[k] for k in range(4)) - sum(m[j] ** 2 * x[j] * y[j] for j in range(4))
assert sp.expand(lhs - rhs) == 0

res = {"frozen": [], "reshuffled": [], "frozen_copies": []}
for _ in range(1500):                                     # at 300 the two means differ by ~1 s.e.
    sel = rng.choice(pos.size, MINR, replace=False)
    p_, q_ = pos[sel], pid[sel]
    oa = p_ - rng.integers(1, p_ - 1)
    w = rng.multinomial(7, np.full(7, 1 / 7))[q_].astype(float)
    if w.sum() == 0:
        continue
    res["frozen"].append(n0_bins(p_, q_, oa, oa, w))
    res["frozen_copies"].append(n0_bins(p_, q_, oa, oa, w, copies=True))
    res["reshuffled"].append(n0_bins(p_, q_, oa, p_ - rng.integers(1, p_ - 1), w))
f, fc, rs = (np.nanmean(res[k]) for k in ("frozen", "frozen_copies", "reshuffled"))
se = np.hypot(*(np.nanstd(res[k]) / np.sqrt(len(res[k])) for k in ("frozen", "reshuffled")))
assert abs(f - rs) < max(0.15 * rs, 4 * se), (f, rs, se)
assert fc > 3 * rs, (fc, rs)

print("b7p_sink_exclusion: 5 checks pass (a drop keeps the kept keys' order; a planted 0.7 / 0.2 / 0.1 "
      "row is condensed on the sink before and on its hidden partner after, a 0.8 / 0.2 row on nothing; "
      f"B7's N0 on frozen sink rows {n0_sink:.4f} with P = 1; under G1, G2, G4 the largest key share "
      f"{max(full):.4f} at full rows, {max(thin):.3f} at 50 (rule 1/4); N0 per draw at 50 frozen rows "
      f"{f:.4f} over different original rows, {fc:.4f} with copies, reshuffled {rs:.4f})")
