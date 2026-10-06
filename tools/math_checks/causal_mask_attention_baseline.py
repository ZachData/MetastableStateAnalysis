"""The structural baselines a causally-masked attention matrix has BEFORE any
content enters, and what they imply for three quantities this project reports.

WHY THIS EXISTS
---------------
`p10_cluster_function/attention-10.md` audits the attention flip -- trained
models routing ~1.6x layer-average attention to unclustered tokens and ~0.5x to
clustered ones. The statistic is

    received[key] = sum over heads, sum over queries of attn[h, query, key]
    reported as   received[population].mean() / received.mean()

with the diagonal zeroed and NOTHING ELSE divided out. Under a causal mask a
token at position j is visible to only n - j queries, so `received` has a
built-in tilt toward early positions before content enters at all. This derives
that tilt in closed form so the audit compares against a number rather than an
intuition.

It also derives the row-side baselines for attention entropy and for the
partition function Z_beta,i, because the same mask acts on them in the OPPOSITE
direction -- which bears on an identification `math-1.md` Sec 1A.6 asserts and
nobody has measured.

WHAT IT DOES NOT PROVE
----------------------
Everything here is the CONTENT-FREE baseline: attention uniform within the
causal triangle, and (for Z) all pairwise inner products equal. That is the
null, not the model. It says what the statistic reports when the network does
nothing, which is exactly what an enrichment ratio needs and exactly what the
flip's measurement lacks. It says nothing about whether the observed deviation
from the baseline is large, and nothing about any particular checkpoint.

The Z results additionally assume the concentration regime (Theorem 6.9's
common inner product). Outside it they are a leading-order reading, not an
identity.

Run: python3 tools/math_checks/causal_mask_attention_baseline.py
"""
import sympy as sp

results = []


def record(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if detail:
        print(f"      {detail}")
    results.append(ok)


i, j, k, n = sp.symbols("i j k n", integer=True, positive=True)

# ---------------------------------------------------------------------------
# 1. Received attention under a content-free causal mask
# ---------------------------------------------------------------------------
# Positions 0..n-1. Query i attends over keys 0..i inclusive (causal, with
# self-attention present in the model). Content-free means uniform within the
# row: a[i, j] = 1/(i+1) for j <= i, else 0.
#
#   received(j) = sum_{i=j}^{n-1} 1/(i+1) = sum_{m=j+1}^{n} 1/m = H_n - H_j
#
# H_0 = 0, so received(0) = H_n.
received = sp.harmonic(n) - sp.harmonic(j)

# Verify against the explicit sum at several n, for every j.
ok = True
for nn in (5, 12, 40):
    for jj in range(nn):
        explicit = sum(sp.Rational(1, ii + 1) for ii in range(jj, nn))
        closed = received.subs({n: nn, j: jj})
        if sp.simplify(explicit - closed) != 0:
            ok = False
record("received(j) = H_n - H_j   [closed form vs explicit sum, n in {5,12,40}, all j]", ok)

# Total mass is n (n rows, each summing to 1), so the LAYER MEAN IS EXACTLY 1
# and `received(j)` already IS the "x layer average" quantity the flip reports.
ok = True
for nn in (5, 12, 40, 264):
    total = sum(received.subs({n: nn, j: jj}) for jj in range(nn))
    if sp.simplify(total - nn) != 0:
        ok = False
record("sum_j received(j) = n, so the layer mean is exactly 1 and received(j) IS the ratio",
       ok, "the flip's 'x layer average' and this baseline are directly comparable")

# ---------------------------------------------------------------------------
# 2. The size of the structural tilt, at the project's battery length
# ---------------------------------------------------------------------------
N_BATTERY = 264  # the ~264-token battery core/sink_audit.py sizes its baseline on
H = lambda m: float(sp.harmonic(m))
first = H(N_BATTERY)                      # position 0
last = H(N_BATTERY) - H(N_BATTERY - 1)    # position n-1
median = H(N_BATTERY) - H(N_BATTERY // 2)
record("content-free tilt at n=264 spans three orders of magnitude",
       first > 6.0 and last < 0.005,
       f"pos 0 = {first:.3f}x, median pos = {median:.3f}x, last pos = {last:.5f}x layer average")

# The observed contrast (1.6x vs 0.5x) is far inside that span. Which positions
# does the CONTENT-FREE baseline already assign those values to?
def position_with_ratio(target, nn=N_BATTERY):
    """Smallest j whose baseline received(j) is <= target."""
    Hn = H(nn)
    for jj in range(nn):
        if Hn - H(jj) <= target:
            return jj
    return nn - 1

p16 = position_with_ratio(1.6)
p05 = position_with_ratio(0.5)
record("the observed 1.6x / 0.5x are both attainable with zero content",
       0 < p16 < p05 < N_BATTERY,
       f"baseline hits 1.6x at position ~{p16} and 0.5x at position ~{p05} (n=264): "
       "a population split at those mean positions reproduces the flip with no learned behaviour")

# Consistency: a two-population split of the SAME tokens must average to 1.
# f * 1.6 + (1 - f) * 0.5 = 1  =>  f = 0.5/1.1 ~= 0.4545, i.e. ~45% unclustered.
f = sp.Rational(5, 11)
record("the reported 1.6x / 0.5x pin the unclustered fraction to ~45%",
       sp.simplify(f * sp.Rational(8, 5) + (1 - f) * sp.Rational(1, 2) - 1) == 0,
       f"f = {float(f):.4f}; status-5c reports ~40-50% unclustered -- the numbers are "
       "internally consistent, which is a consistency check, not evidence of content")

# ---------------------------------------------------------------------------
# 3. Row-side baseline: attention entropy
# ---------------------------------------------------------------------------
# core/metrics.py::attention_entropy averages row entropies. Content-free,
# row i is uniform over i+1 keys, so its entropy is log(i+1) exactly.
ok = True
for nn in (5, 12, 40):
    per_row = [sp.log(ii + 1) for ii in range(nn)]
    mean_entropy = sum(per_row) / nn
    # sum_{i=0}^{n-1} log(i+1) = log(n!)
    if sp.simplify(mean_entropy - sp.log(sp.factorial(nn)) / nn) != 0:
        ok = False
record("content-free attention entropy: row i has entropy log(i+1), layer mean log(n!)/n",
       ok, f"at n=264 the mean is {float(sp.log(sp.factorial(264))/264):.3f} nats; "
           "a per-row deviation from log(i+1) is the content, the raw value is mostly position")

# ---------------------------------------------------------------------------
# 4. Z_beta,i, and the direction the mask pushes it
# ---------------------------------------------------------------------------
# Z_beta,i = sum_j exp(beta <x_i, x_j>) is particle i's softmax denominator --
# a ROW quantity, and (math-1.md Sec 1A.6) the per-token weight in the metric
# under which (SA) is a gradient flow: a high-Z token is expensive to move.
#
# In the concentration regime every inner product is a common gamma, so:
beta, gamma = sp.symbols("beta gamma", real=True)
Z_unmasked = n * sp.exp(beta * gamma)          # row i sums over all n
Z_masked = (i + 1) * sp.exp(beta * gamma)      # row i sums over j <= i only

record("unmasked: Z_beta,i is position-independent under concentration",
       sp.simplify(sp.diff(Z_unmasked, i)) == 0,
       "so any spread in Z is content -- e.g. a high-norm sink, whose inner products "
       "with everything are large. This is the regime Sec 1A.6's 'high-Z = sink' reading assumes")

record("masked: Z_beta,i grows linearly in position, and position 0 is the MINIMUM",
       sp.simplify(sp.diff(Z_masked, i) - sp.exp(beta * gamma)) == 0,
       "Z_masked(0) = exp(beta*gamma) is the smallest value any row can take")

# The consequence worth carrying: under the mask the two structural baselines
# point in OPPOSITE directions in position.
# received(j) - received(j+1) = H_{j+1} - H_j = 1/(j+1) > 0, so received
# DECREASES with position. sympy leaves harmonic() unevaluated under simplify,
# so expand_func is what turns the difference into a rational function.
received_step = sp.expand_func(
    (sp.harmonic(n) - sp.harmonic(j)) - (sp.harmonic(n) - sp.harmonic(j + 1))
)
record("the two mask baselines are anti-aligned in position",
       sp.simplify(received_step - 1 / (j + 1)) == 0,
       "received(j) falls as 1/(j+1) per step while Z_masked(i) rises linearly. "
       "So the sink (position 0) is simultaneously the LARGEST received-attention and the "
       "SMALLEST Z -- i.e. under a causal mask the 'high-Z = sink' identification REVERSES. "
       "It is an unmasked-model statement, and Pythia is masked")

# ---------------------------------------------------------------------------
# 5. The baseline under token rule T4 (Phase 10 re-read R3, 2026-10-05)
# ---------------------------------------------------------------------------
# T4 drops the columns of a set D of positions (the sink, massive tokens) and
# renormalises each row over the keys left. Content-free, query i then spreads
# 1/|V_i| over V_i = {j <= i, j not in D}, so for j not in D
#
#   received_T4(j) = sum_{i >= j, |V_i| > 0} 1/|V_i|
#
# (core.parking.t4_received_baseline). With D = {0}: |V_i| = i, so
#   received_T4(j) = sum_{i=j}^{n-1} 1/i = H_{n-1} - H_{j-1}   (j >= 1).
received_t4_0 = sp.harmonic(n - 1) - sp.harmonic(j - 1)
ok = True
for nn in (5, 12, 40):
    for jj in range(1, nn):
        explicit = sum(sp.Rational(1, ii) for ii in range(jj, nn))
        if sp.simplify(explicit - received_t4_0.subs({n: nn, j: jj})) != 0:
            ok = False
record("T4, D = {0}: received(j) = H_{n-1} - H_{j-1}   [vs explicit sum, n in {5,12,40}]", ok)


def t4_explicit(nn, D):
    """received_T4 from the matrix itself: build, drop D's columns, renormalise rows."""
    rows = []
    for ii in range(nn):
        keys = [jj for jj in range(ii + 1) if jj not in D]
        rows.append({jj: sp.Rational(1, len(keys)) for jj in keys} if keys else {})
    return {jj: sum(r.get(jj, 0) for r in rows) for jj in range(nn) if jj not in D}


def t4_closed(nn, D):
    vis = [sum(1 for jj in range(ii + 1) if jj not in D) for ii in range(nn)]
    return {jj: sum(sp.Rational(1, vis[ii]) for ii in range(jj, nn) if vis[ii] > 0)
            for jj in range(nn) if jj not in D}


ok, mass_ok = True, True
for nn, D in ((12, {0, 3}), (20, {0, 1, 7, 19}), (30, {5, 11}), (9, set())):
    e, c = t4_explicit(nn, D), t4_closed(nn, D)
    ok &= all(sp.simplify(e[k] - c[k]) == 0 for k in e)
    # each non-empty row carries mass 1, so the kept positions share exactly that many
    nonempty = sum(1 for ii in range(nn) if any(jj not in D for jj in range(ii + 1)))
    mass_ok &= sp.simplify(sum(c.values()) - nonempty) == 0
record("T4, general D: the closed form is the dropped-and-renormalised matrix's received", ok,
       "exact rationals; D includes 0, interior and last positions, and none")
record("T4: the kept positions' total is the number of rows with a key left", mass_ok,
       "with D = {0} that is n - 1 rows over n - 1 kept positions, so the mean is 1, as before")
# NOT proved here: that dropping D's columns is the right content-free model of a
# network that routes the sink's mass elsewhere; it is the baseline of the rule as
# stated, applied to uniform causal attention, nothing more.

print()
print(f"{sum(results)}/{len(results)} checks passed")
raise SystemExit(0 if all(results) else 1)
