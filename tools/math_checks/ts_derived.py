"""Training symmetry, first deriving session: the closed forms of `MATH_TRAINING_SYMMETRY.md` §8.

WHY THIS EXISTS
---------------
§8 of `MATH_TRAINING_SYMMETRY.md` answers part of the register's §4. CLAUDE.md asks for a check
of each closed form, saying what it does not prove. `ts_conservation.py` holds F1-F6; this holds
D1-D8. Symbolic where the claim is algebraic, Monte Carlo (seeded) where it is a law of a
random matrix or of Adam on noise.

WHAT IT SHOWS
-------------
D1. **Spread of the random-init symmetric fraction.** For M = W_Q^T W_K with W_Q, W_K k x d iid
    Gaussian and d >> k, s = (|M|_F^2 + tr M^2) / (2 |M|_F^2) has mean 1/2 + 1/(2d) and
    sd 1/(sqrt(2) d), whatever k: tr M^2 = tr N^2 with N = W_K W_Q^T (k x k, entries nearly
    iid N(0, d sigma^4)), Var tr G^2 = 2 k^2 for a k x k standard Gaussian G, |M|_F^2 -> k d^2.
D2. **Real eigenvalues at init.** The nonzero eigenvalues of W_Q^T W_K (or W_O W_V) are those of
    a k x k product that is close to a real Ginibre matrix; the expected number of real ones is
    Edelman-Kostlan-Shub's E_k = 1/2 + sqrt(2/pi) Gamma(k+1/2)/Gamma(k) 2F1(1,-1/2;k;1/2).
D3. **Adam on a coordinate.** With g_t = mu + sigma xi_t (iid) and a slowly varying rate, the
    displacement after T steps has mean close to rho tau_1 and sd close to
    (1-rho^2) sqrt(1 - rho^2/2) sqrt(tau_2), rho = mu / sqrt(mu^2 + sigma^2), tau_1 = sum eta_t,
    tau_2 = sum eta_t^2 (delta method on m / sqrt(v), Gaussian noise: the noise in m and in v
    partly cancel, so the sd is below sqrt((1-rho^2) tau_2)). A zero-mean coordinate walks
    sqrt(tau_2); a consistent sign moves tau_1.
D4. **What Adam conserves.** (a) Scalar sign flow on f(u v): |u| - |v| is conserved (u^2 - v^2 is
    not). (b) A flow preconditioned by a constant diagonal metric conserves the metric-weighted
    imbalance of a rescale symmetry (gain against column). (c) Sign flow on matrix factors does
    not conserve Psi = Q Q^T - K K^T, nor its diagonal (numeric counterexample).
D5. **Imbalance sets the speed.** For s = u v with u^2 - v^2 = c conserved,
    ds/dt = -L'(s) sqrt(c^2 + 4 s^2): linear in time at s -> 0 when c != 0, exponential when c = 0.
D6. **Weight decay with RoPE is a group penalty.** A rotary pair's score is Re(conj(q.x_i) (k.x_j)
    e^{i theta}) for complex rows q, k; the pair's symmetry is q -> c q, k -> k / conj(c), and the
    least |q|^2 + |k|^2 on that orbit is 2 |q| |k| (the non-rotary block's is 2 |M|_*).
D7. **A zero-sum gradient does not make a zero-sum Adam step.** Rows of the unembedding's
    gradient sum to zero over the vocabulary; their signs do not.
D8. **Orbit dimension.** The continuous symmetries of §8.2's table span 0.88 % of Pythia-410m's
    parameters. D8b: the two families added after `/challenge-pr` on #354 (the value-bias shift,
    reader rows along 1/gamma) leave a head's output unchanged.

WHAT IT DOES NOT PROVE
----------------------
- D1, D2: the k x k product is Gaussian only as d -> infinity; the checks are Monte Carlo at
  Pythia's (d, k), not a bound on the finite-d error.
- D3: iid noise, one coordinate, beta_2 = 0.95 (so sqrt(v) fluctuates and the measured mean and
  sd sit a few percent off the beta_2 -> 1 formulas; the tolerance is 15 %). It says nothing
  about correlated noise, a restoring force, or weight decay.
- D4 (b): the metric is held constant. Adam's sqrt(v) moves, so the weighted imbalance is
  conserved only while it is stationary. (c) is one counterexample, not a classification;
  Marcotte et al. 2506.06194 (sign flow, their appendix) and Jacobs et al. 2603.02064 treat that.
- D5: gradient flow, scalar factors. Not Adam, not a matrix.
- D6: the minimum over the orbit, i.e. what decay prefers at a fixed function; not that training
  reaches it. Biases (not decayed in Pythia) are left out.
- D8: counts the symmetries listed in §8.2's table, term by term; a missed symmetry would raise
  it (two were missed in the first pass). The table is not claimed complete.
"""
import math

import numpy as np
import sympy as sp


def d1_sym_fraction_spread(d=1024, k=48, n=600, seed=0):
    rng = np.random.default_rng(seed)
    s = np.empty(n)
    for i in range(n):
        Q, K = rng.standard_normal((k, d)), rng.standard_normal((k, d))
        N = K @ Q.T
        fro = np.trace((K @ K.T) @ (Q @ Q.T))
        s[i] = 0.5 + np.trace(N @ N) / (2 * fro)
    mean, sd = 0.5 + 1 / (2 * d), 1 / (math.sqrt(2) * d)
    assert abs(s.mean() - mean) < 4 * sd / math.sqrt(n), (s.mean(), mean)
    assert abs(s.std() / sd - 1) < 0.1, (s.std(), sd)
    G = rng.standard_normal((4000, 6, 6))
    v = np.einsum("nij,nji->n", G, G).var()
    assert abs(v / (2 * 36) - 1) < 0.1, v
    print(f"D1: sym fraction at d={d}, k={k}: MC mean {s.mean():.6f} (formula {mean:.6f}), "
          f"sd {s.std():.6f} (formula {sd:.6f}); p99 formula {mean + 2.326 * sd:.5f}  OK")


def eks(k):
    return 0.5 + math.sqrt(2 / math.pi) * math.exp(math.lgamma(k + 0.5) - math.lgamma(k)) * float(
        sp.hyper([1, sp.Rational(-1, 2)], [k], sp.Rational(1, 2)).evalf())


def d2_real_eigs(d=1024, n=300, seed=1):
    rng = np.random.default_rng(seed)
    for k in (48, 64):
        cnt = [np.sum(np.abs(np.linalg.eigvals(rng.standard_normal((k, d)) @ rng.standard_normal((k, d)).T).imag) < 1e-9)
               for _ in range(n)]
        e = eks(k)
        assert abs(np.mean(cnt) / e - 1) < 0.08, (k, np.mean(cnt), e)
        print(f"D2: real eigenvalues, k={k}: MC {np.mean(cnt):.2f}, EKS {e:.2f} ({100 * e / k:.1f} % real)  OK")


def d3_adam_walk(seed=2, n=4000, T=3000, b1=0.9, b2=0.95, eps=1e-8):
    rng = np.random.default_rng(seed)
    eta = 1e-3 * np.minimum(1, np.arange(1, T + 1) / 1000)  # a ramp, then flat
    t1, t2 = eta.sum(), (eta ** 2).sum()
    for rho in (0.0, 0.3, 0.9):
        mu, sig = rho, math.sqrt(1 - rho ** 2)
        m = np.zeros(n); v = np.zeros(n); x = np.zeros(n)
        for t in range(1, T + 1):
            g = mu + sig * rng.standard_normal(n)
            m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
            x -= eta[t - 1] * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        mean, sd = -x.mean(), x.std()
        assert abs(mean - rho * t1) < 0.15 * max(rho * t1, 5 * math.sqrt(t2 / n)), (rho, mean, rho * t1)
        pred = (1 - rho ** 2) * math.sqrt((1 - rho ** 2 / 2) * t2)
        assert abs(sd / pred - 1) < 0.15, (rho, sd, pred)
        print(f"D3: Adam, rho={rho}: drift {mean:.4f} (rho tau1 = {rho * t1:.4f}), "
              f"sd {sd:.5f} ((1-rho^2) sqrt((1-rho^2/2) tau2) = {pred:.5f})  OK")


def d4_adam_laws():
    u, v = sp.symbols("u v", positive=True)
    fp = sp.Symbol("fp", real=True, nonzero=True)  # f'(uv)
    du, dv = -sp.sign(v * fp), -sp.sign(u * fp)    # sign flow on f(uv)
    assert sp.simplify(du * sp.sign(u) - dv * sp.sign(v)) == 0 and sp.simplify(2 * u * du - 2 * v * dv) != 0
    # (b) gain g against a column w, loss through g*w, metric diag(a, b_1..b_n) constant
    n = 3
    g, a = sp.symbols("g a", positive=True)
    w = sp.Matrix(n, 1, lambda i, j: sp.Symbol(f"w{i}"))
    b = [sp.Symbol(f"b{i}", positive=True) for i in range(n)]
    G = sp.Matrix(n, 1, lambda i, j: sp.Symbol(f"G{i}"))
    dg = -(G.T * w)[0] / a
    dw = [-g * G[i] / b[i] for i in range(n)]
    lhs = 2 * a * g * dg - sum(2 * b[i] * w[i] * dw[i] for i in range(n))
    assert sp.simplify(lhs) == 0
    # (c) sign flow on matrix factors: Psi and its diagonal move
    rng = np.random.default_rng(3)
    Q, K, Gm = rng.standard_normal((2, 5)), rng.standard_normal((2, 5)), rng.standard_normal((5, 5))
    dQ, dK = -np.sign(K @ Gm.T), -np.sign(Q @ Gm)
    dPsi = dQ @ Q.T + Q @ dQ.T - dK @ K.T - K @ dK.T
    dl1 = (dQ * np.sign(Q)).sum(1) - (dK * np.sign(K)).sum(1)
    assert np.abs(np.diag(dPsi)).max() > 1e-6 and np.abs(dl1).max() > 1e-6
    print("D4: sign flow keeps |u| - |v| (scalar); a constant diagonal metric keeps a g^2 - sum b_i w_i^2; "
          "matrix sign flow keeps neither Psi's diagonal nor the rows' l1 imbalance  OK")


def d5_imbalance_speed():
    u, v, Lp = sp.symbols("u v Lp", positive=True)
    s, c = u * v, u ** 2 - v ** 2
    ds = -Lp * (u ** 2 + v ** 2)  # du = -Lp v, dv = -Lp u
    assert sp.simplify((u ** 2 + v ** 2) ** 2 - (c ** 2 + 4 * s ** 2)) == 0
    assert sp.simplify(2 * u * (-Lp * v) - 2 * v * (-Lp * u)) == 0
    print("D5: ds/dt = -L'(s) sqrt(c^2 + 4 s^2), c = u^2 - v^2 conserved  OK")


def d6_rope_group_penalty():
    q1, q2, k1, k2, th = sp.symbols("q1 q2 k1 k2 th", real=True)
    R = sp.Matrix([[sp.cos(th), -sp.sin(th)], [sp.sin(th), sp.cos(th)]])
    real = (sp.Matrix([[q1, q2]]) * R * sp.Matrix([k1, k2]))[0]
    cplx = sp.re(sp.expand(sp.conjugate(q1 + sp.I * q2) * (k1 + sp.I * k2) * sp.exp(sp.I * th), complex=True))
    assert sp.simplify(real - cplx) == 0
    r, nq, nk = sp.symbols("r nq nk", positive=True)  # |c|^2, |q|, |k|
    f = r * nq ** 2 + nk ** 2 / r
    rs = sp.solve(sp.diff(f, r), r)
    assert len(rs) == 1 and sp.simplify(f.subs(r, rs[0]) - 2 * nq * nk) == 0
    print("D6: a rotary pair is one complex rank-one term; min over its orbit of |q|^2 + |k|^2 is 2 |q| |k|  OK")


def d7_zero_sum(seed=4, V=2000):
    rng = np.random.default_rng(seed)
    y = rng.zipf(1.3, 20000) % V
    freq = np.bincount(y, minlength=V) / len(y)
    g = np.full(V, 1 / V) - freq  # p - y at a uniform prediction, times a common activation
    assert abs(g.sum()) < 1e-12 and abs(np.sign(g).mean()) > 0.3
    print(f"D7: sum of the gradient over the vocabulary {g.sum():.1e}; mean of its sign {np.sign(g).mean():+.2f}  OK")


def d8_orbit(d=1024, L=24, H=16, V=50304):
    dh = d // H; rot = dh // 4; nr = dh - rot
    per_head = (nr * nr + rot      # QK: GL(48) x (C*)^8
                + dh * dh          # OV: GL(64)
                + nr               # key-bias translations (non-rotary)
                + dh)              # value-bias shift against the output bias
    per_layer = (H * per_head
                 + 2 * d           # each norm's gain against its readers' columns
                 + 2 * d           # each norm's bias against its readers' biases
                 + 5 * d + 2       # writers along 1: 1^T W_O (d), 1^T W_out (4d), the two biases
                 + d               # attention-out against MLP-out bias
                 + 7 * d)          # reader rows along 1/gamma: QKV (3d rows), MLP-in (4d rows)
    k = (L * per_layer + d         # unembedding translation
         + V                       # embedding rows along 1
         + 1                       # joint scale of the writers
         + d + V                   # final gain against W_U's columns; W_U's rows along 1/gamma
         + (d - 1) * (d - 2) // 2)  # rotations of the stream fixing 1
    D = L * (4 * d * d + 4 * d + 8 * d * d + 5 * d + 4 * d) + 2 * V * d + 2 * d
    assert k == 3_550_002 and abs(k / D - 0.00876) < 1e-4, (k, k / D)
    print(f"D8: orbit dimension {k:,} of {D:,} parameters ({100 * k / D:.2f} %); (D - k)/2 = {(D - k) / 2:,.0f}  OK")


def d8b_missed_symmetries(seed=5, d=12, k=4, n=7):
    """The two families the first pass missed, on one head reading a LayerNorm output."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)); g, be = rng.uniform(0.5, 2, d), rng.standard_normal(d)
    ln = g * (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True) + be
    Wq, Wk, Wv, Wo = (rng.standard_normal(s) for s in ((k, d), (k, d), (k, d), (d, k)))
    bv, bo = rng.standard_normal(k), rng.standard_normal(d)

    def out(Wq, Wv, bv, bo, ln):
        sc = (ln @ Wq.T) @ (ln @ Wk.T).T
        a = np.exp(sc - sc.max(1, keepdims=True)); a /= a.sum(1, keepdims=True)
        return a @ (ln @ Wv.T + bv) @ Wo.T + bo
    base = out(Wq, Wv, bv, bo, ln)
    c = rng.standard_normal(k)
    assert np.abs(out(Wq, Wv, bv + c, bo - Wo @ c, ln) - base).max() < 1e-10          # value-bias shift
    a = rng.standard_normal(k)
    ln0 = ln - be                                                                    # the gain part alone
    assert np.abs((ln0 @ (Wq + np.outer(a, 1 / g)).T) - ln0 @ Wq.T).max() < 1e-10     # reader rows along 1/gamma
    print("D8b: value-bias shift and reader rows along 1/gamma leave the head unchanged  OK")


if __name__ == "__main__":
    d1_sym_fraction_spread()
    d2_real_eigs()
    d3_adam_walk()
    d4_adam_laws()
    d5_imbalance_speed()
    d6_rope_group_penalty()
    d7_zero_sum()
    d8_orbit()
    d8b_missed_symmetries()
