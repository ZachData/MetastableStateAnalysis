"""B7's routing activity: what a partner-persistence rate across depth can and cannot say, and
which nulls see which kind of "frozen" (`docs/ROADMAP.md` §7, B7; `trackB_circuit/design-B7.md`
"The math").

WHY THIS EXISTS
---------------
ROADMAP's B7 row asks how often a token's top attention partner changes from one layer to the
next, against a shuffled-layer null and step 0, so that "condensed and rerouting" and "condensed
and frozen" (2609.28448's distinction between condensation and temporal activity) are told apart.
This check fixes, before any head is read, what the rate is on the cases the design names.

WHAT IT SHOWS
-------------
1. **On diffuse rows the two obvious measures disagree, in opposite directions.** Two independent
   uniform argmaxes over n keys agree with probability 1/n, so the argmax "switch rate" is
   1 − 1/n ≈ 1 ("rerouting"); two uniform rows have soft overlap 1 − TV = 1 ("frozen"). Two
   independent rows of i.i.d. Gaussian logits at s = 1, n = 256, still have soft overlap ≈ 0.5.
   Neither measure is read on diffuse rows; step 0 (near uniform to step 256, B2a) is therefore a
   degenerate null for both, and is read for D_ℓ only.
2. **A condensed row (top share > 1/2) has a unique argmax**, so the partner is well defined
   without a tie rule.
3. **The offset-profile null.** Two independent partners drawn from profiles p and q agree with
   probability c = Σ_k p_k q_k ≤ (Σ p_k² Σ q_k²)^{1/2}; for two one-hot profiles at the same
   offset (two previous-token layers) c = 1, so a positional pair has no persistence beyond it.
4. **The shuffled-layer null cannot see global freezing.** If each row keeps its partner with
   probability r per layer and otherwise redraws from a stationary profile with collision
   probability c, the lag-k agreement is P(k) = r^k + (1 − r^k) c. So: frozen (r = 1) gives
   P(1) = P(k) = 1, an excess of 0 over a shuffled-layer null; reshuffled (r = 0) gives
   P(1) = P(k) = c; only drift (0 < r < 1) gives P(1) > mean_k≥2 P(k). And
   r = (P(1) − c) / (1 − c): the keep rate is identified once c is known. A period-two routing
   has P(1) = c, P(2) = 1.
5. **D_ℓ on unit states.** ‖u − v‖² = 2 − 2 cos θ, so D_ℓ ∈ [0, 2]; a consensus flip has D = 2;
   two independent uniform directions in 1024 dimensions have D ≈ √2.
6. **The nulls at the real design points** (`/challenge-pr` on #271, finding 1). On the 7 v1
   prompts' lengths (rows 16..n − 1), 16 heads per layer, partners reshuffled (uniform on
   1..i − 2): the per-row chance is ≈ 0.0085. A best-of-16 next-layer head chosen on the prompts
   it is read on inflates P to ≈ 0.012; an offset profile pooled over all positions puts c at
   ≈ 0.004 (pooling mixes rows of different reach). Choosing the head leave-one-prompt-out puts P
   back at chance, and the row-valid offset chance within 32-position bins puts N0 at chance or
   up to 15 % above it (conservative: excess leans to 0).

WHAT IT DOES NOT PROVE
----------------------
- That 410m's rows follow the Markov model of check 4 (it is the gate's drift arm, not a claim
  about the model); the real lag profile is read, not fitted.
- That a persistent partner in an untied net is the paper's frozen routing: layer l + 1 has its
  own Q and K, so a kept partner can be two layers' weights agreeing, not one geometry persisting
  (2609.28448 Sec. V B). Only a tied model (S4's tied arm) separates the two.
- That top-1 is the right summary of a condensed row whose mass sits on 2–3 partners; the soft
  overlap on condensed rows is beside for that.
"""
import numpy as np
import sympy as sp

# 1. Diffuse rows: argmax agreement 1/n, soft overlap 1 for identical uniform rows.
n = sp.symbols("n", positive=True, integer=True)
p_agree_uniform = sp.Sum(1 / n * 1 / n, (sp.Symbol("j"), 1, n)).doit()
assert sp.simplify(p_agree_uniform - 1 / n) == 0
tv_uniform = sp.Rational(1, 2) * sp.Sum(sp.Abs(1 / n - 1 / n), (sp.Symbol("j"), 1, n)).doit()
assert 1 - tv_uniform == 1

rng = np.random.default_rng(0)
m, s, draws = 256, 1.0, 400
z1, z2 = rng.normal(0, s, (draws, m)), rng.normal(0, s, (draws, m))
a1 = np.exp(z1) / np.exp(z1).sum(1, keepdims=True)
a2 = np.exp(z2) / np.exp(z2).sum(1, keepdims=True)
soft = 1 - 0.5 * np.abs(a1 - a2).sum(1)
arg = (a1.argmax(1) == a2.argmax(1)).mean()
assert soft.mean() > 0.4 and arg < 0.02, (soft.mean(), arg)

# 2. Top share > 1/2 makes the argmax unique: any other entry b ≤ 1 − a, and a − (1 − a) > 0.
e = sp.symbols("e", positive=True)
a = sp.Rational(1, 2) + e
assert (a - (1 - a)).is_positive                        # = 2e, positive by e's assumption
assert (sp.Rational(1, 2) - (1 - sp.Rational(1, 2))).is_positive is False   # at exactly 1/2 a tie is possible
rows2 = rng.dirichlet(np.ones(30) * 0.05, 20000)
top = np.sort(rows2, 1)
big = top[:, -1] > 0.5
assert big.any() and (top[big, -1] > top[big, -2]).all()

# 3. Offset-profile null: Cauchy–Schwarz bound, and c = 1 for two equal one-hot profiles.
p = rng.dirichlet(np.ones(20), 1000)
q = rng.dirichlet(np.ones(20), 1000)
c = (p * q).sum(1)
assert (c <= np.sqrt((p**2).sum(1) * (q**2).sum(1)) + 1e-12).all()
onehot = np.eye(20)[3]
assert (onehot * onehot).sum() == 1

# 4. Lag-k agreement under keep-or-redraw: P(k) = r^k + (1 − r^k) c, by the two-state recursion.
r, cc, k = sp.symbols("r c k", nonnegative=True)
P = lambda kk: r**kk + (1 - r**kk) * cc
# One more layer: kept (prob r) leaves agreement as it was; a redraw agrees with prob c.
assert sp.simplify(r * P(k) + (1 - r) * cc - P(k + 1)) == 0
assert sp.simplify(P(1).subs(r, 1) - 1) == 0 and sp.simplify(P(5).subs(r, 1) - 1) == 0
assert sp.simplify(P(1).subs(r, 0) - cc) == 0 and sp.simplify(P(5).subs(r, 0) - cc) == 0
assert sp.simplify((P(1) - cc) / (1 - cc) - r) == 0
rv, cv = 0.8, 0.05
assert P(1).subs({r: rv, cc: cv}) > np.mean([float(P(kk).subs({r: rv, cc: cv})) for kk in range(2, 12)])
# Numerically, on simulated chains (the gate's drift arm in miniature).
rows, L, K = 4000, 24, 50
part = np.empty((L, rows), int)
part[0] = rng.integers(0, K, rows)
for l in range(1, L):
    keep = rng.random(rows) < rv
    part[l] = np.where(keep, part[l - 1], rng.integers(0, K, rows))
p1 = (part[1:] == part[:-1]).mean()
assert abs((p1 - 1 / K) / (1 - 1 / K) - rv) < 0.01, p1
# Period two: lag 1 at chance, lag 2 exact.
A, B = rng.integers(0, K, rows), rng.integers(0, K, rows)
alt = np.stack([A if l % 2 == 0 else B for l in range(L)])
assert abs((alt[1:] == alt[:-1]).mean() - 1 / K) < 0.01 and (alt[2:] == alt[:-2]).all()

# 5. D on unit states.
th = sp.symbols("theta", real=True)
u, v = sp.Matrix([1, 0]), sp.Matrix([sp.cos(th), sp.sin(th)])
assert sp.simplify((u - v).dot(u - v) - (2 - 2 * sp.cos(th))) == 0
assert sp.simplify(((u - v).dot(u - v)).subs(th, sp.pi)) == 4
x = rng.normal(size=(2000, 1024)); x /= np.linalg.norm(x, axis=1, keepdims=True)
y = rng.normal(size=(2000, 1024)); y /= np.linalg.norm(y, axis=1, keepdims=True)
assert abs(np.sqrt(((x - y) ** 2).sum(1).mean()) - np.sqrt(2)) < 0.01

# 6. The nulls at the real design points: the pooled-profile N0 and an in-sample best-of-16 are
# biased; leave-one-prompt-out matching and a row-valid within-bin N0 are not (or err high).
NS, H, BIN = (465, 242, 512, 446, 286, 482, 467), 16, 32
rws = [np.arange(16, nn) for nn in NS]
chance = np.mean(np.concatenate([1 / (r_ - 2) for r_ in rws]))
lens = np.array([len(r_) for r_ in rws])


def draw():
    return [np.stack([rng.integers(1, r_ - 1) for _ in range(H)]) for r_ in rws]


def best(Aa, Bb, h, prompts):
    return int(np.argmax([sum((Aa[pp][h] == Bb[pp][g]).sum() for pp in prompts) for g in range(H)]))


def n0_rows(Aa, Bb, h, g, qq):
    ia = np.concatenate(rws)
    oa = np.concatenate([r_ - Aa[pp][h] for pp, r_ in enumerate(rws)])
    ob = np.concatenate([r_ - Bb[pp][g] for pp, r_ in enumerate(rws)])
    K, out = int(ia.max()) + 1, []
    for b0 in range(16, int(rws[qq].max()) + 1, BIN):
        m_ = (ia >= b0) & (ia < b0 + BIN)
        pa, pb = np.bincount(oa[m_], minlength=K) * 1.0, np.bincount(ob[m_], minlength=K) * 1.0
        for i in rws[qq][(rws[qq] >= b0) & (rws[qq] < b0 + BIN)]:
            va, vb = pa[1:i], pb[1:i]
            out.append((va / va.sum() * vb / vb.sum()).sum() if va.sum() and vb.sum() else 0.0)
    return np.mean(out)


p_in, c_pool, p_loo, n0 = [], [], [], []
for _ in range(2):
    Aa, Bb = draw(), draw()
    for h in range(0, H, 2):
        hs = best(Aa, Bb, h, range(7))
        p_in.append(sum((Aa[pp][h] == Bb[pp][hs]).sum() for pp in range(7)) / lens.sum())
        oa = np.concatenate([r_ - Aa[pp][h] for pp, r_ in enumerate(rws)])
        ob = np.concatenate([r_ - Bb[pp][hs] for pp, r_ in enumerate(rws)])
        K = max(oa.max(), ob.max()) + 1
        c_pool.append((np.bincount(oa, minlength=K) / len(oa) * np.bincount(ob, minlength=K) / len(ob)).sum())
        num, n0q = 0, []
        for qq in range(7):
            g = best(Aa, Bb, h, [pp for pp in range(7) if pp != qq])
            num += (Aa[qq][h] == Bb[qq][g]).sum()
            n0q.append(n0_rows(Aa, Bb, h, g, qq))
        p_loo.append(num / lens.sum())
        n0.append(np.average(n0q, weights=lens))
p_in, c_pool, p_loo, n0 = map(np.mean, (p_in, c_pool, p_loo, n0))
assert p_in > 1.25 * chance and c_pool < 0.6 * chance, (p_in, c_pool, chance)
assert abs(p_loo - chance) < 0.1 * chance, (p_loo, chance)
assert chance * 0.97 < n0 < chance * 1.15, (n0, chance)

print("b7_partner_persistence: 6 checks pass (diffuse rows: argmax agreement 1/n, soft overlap 1, "
      f"independent s = 1 rows overlap {soft.mean():.2f}; top share > 1/2 gives a unique argmax; "
      "offset-profile null c = Σ p q, 1 for a shared one-hot offset; P(k) = r^k + (1 − r^k) c, "
      "so a shuffled-layer null sees drift but not freezing, r identified; D ∈ [0, 2], √2 at random; "
      f"at the real n, chance {chance:.4f}: in-sample best-of-16 P {p_in:.4f}, pooled-profile c "
      f"{c_pool:.4f}, leave-one-out P {p_loo:.4f}, row-valid binned N0 {n0:.4f})")
