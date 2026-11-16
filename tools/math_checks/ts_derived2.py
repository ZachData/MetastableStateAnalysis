"""Training symmetry, second deriving session: the closed forms of `MATH_TRAINING_SYMMETRY.md` §9.

WHY THIS EXISTS
---------------
§9 answers three more parts of the register's §4: Q8 (the first update from the init), Q2's open
increment (is F6's list of laws complete with RoPE?), and Q3/Q4's "what AdamW keeps". CLAUDE.md
asks for a check of each closed form, saying what it does not prove. `ts_conservation.py` holds
F1-F6, `ts_derived.py` D1-D8; this holds D9-D11. The readings from Pythia's weights that §9 sets
beside them are `tools/ts_first_step.py`'s, not this file's.

WHAT IT SHOWS
-------------
D9. **AdamW in integral form.** Whatever the gradients, with u_t = m_hat / (sqrt(v_hat) + eps):
    theta_T = A theta_0 - sum_t w_t u_t,  A = prod_t (1 - eta_t lambda),  w_t = eta_t prod_{s>t} (1 - eta_s lambda).
    (a) An identity (checked on a random gradient stream), with -log A = lambda tau_1 + O(lambda^2 tau_2).
    (b) So two checkpoints give the weighted mean of u between them exactly:
        u_bar = -(theta_b - A_ab theta_a) / sum w. A weight that stands still has u_bar = -lambda theta.
    (c) Any linear functional c with c.u_t = 0 at every step decays as A exactly. Under sign steps
        on f(u v) with decay that is |u| - |v| (rate lambda in tau_1; gradient flow's u^2 - v^2
        decays at 2 lambda).
    (d) Below eps Adam is SGD: for |g| << eps the step is eta g / eps (first step) and a zero-sum
        gradient gives a zero-sum step; above eps it does not (D7).
D10. **The first update from the init.**
    (a) Adam's second update for gradients g_1 and g_2 = x g_1:
        u_2(x) = (beta_1 + x) / (1 + beta_1) * sqrt((1 + beta_2) / (beta_2 + x^2)); u_2(1) = 1,
        its maximum over x is at x = beta_2 / beta_1 and is 1.0004; u_2 = 0.85 at x = 0.270 and 5.06.
        70m's stored value (10 units of 2^-24 against a rate of 11.73) pins the jump only to
        x in [3.4, 8.2]; with steady gradients after, checkpoints 4, 8 and 16 narrow it to
        [4.6, 7.9]. A drop to 0.270 is over 10 % low at those checkpoints.
    (b) LayerNorm's Jacobian J = (P - y y^T / d) / s (y the output, P the centring) has J^T 1 = 0 and
        J^T y = 0: the back-signal reaching a writer has no component along 1 or along the
        token's own normalised stream.
    (c) For Gaussian g and W, sign(W^T g) and sign(W^T sign g) agree on
        1 - arccos(sqrt(2/pi)) / pi = 0.7940 of coordinates.
    (d) A toy pre-LN block (embedding, parallel MLP, final LayerNorm, unembedding; Pythia's two
        init rules; Zipf tokens with a planted bigram): the exact gradient at the init has
        sign(-dL/db_out) = sign(centre(W_U^T (u - 1/V))) and sign(-dL/dW_U[v, j]) =
        sign(u_v - 1/V) sign(mean final stream)_j on most entries, and the reader's bias follows (c).
D11. **F6 is complete on a toy.** For one head with RoPE the loss sees (W_Q, W_K) through
    S = Q_n^T K_n (non-rotary) and, per rotary pair, A_p = Q_p^T K_p and B_p = Q_p^T J K_p. These
    are quadratic, so their gradient fields are linear and the Lie algebra they generate is a
    matrix algebra; Marcotte et al.'s count (2307.00144) is: independent laws = D - dim(g theta).
    Found: k_n (k_n + 1)/2 + P, which is F6's list (the non-rotary block of Psi and one trace per
    pair) and nothing else. With biases under a softmax: + k_n (the non-rotary key bias, frozen),
    and the query's factor is the bias-augmented one. With centred inputs (LayerNorm): + 2k (the
    readers' rows along 1). Without RoPE the same code returns Marcotte's k (k + 1)/2 (control).

WHAT IT DOES NOT PROVE
----------------------
- D9 is an identity of the update rule, so it says nothing about what u_t is. (a)-(c) ignore
  nothing; the tau_1 form of A is first order in eta lambda. (d): one coordinate family, eps 1e-8.
- D10 (a): two updates, the first gradient entering at rate 0 (NeoX's ramp starts at 0); it does
  not say why Pythia's x is near 5; the measured values are typed in as constants, and one jump
  is the simplest history that fits, not the only one. (c): Gaussian g; a heavy-tailed back-signal moves the number.
  (d) is a toy: iid tokens, no attention, one block; thresholds are loose bounds on one seed's
  values (printed), not derived rates. Pythia's own agreement is measured by `ts_first_step.py`.
- D11: numerical rank (float64, singular-value gap printed) at two random points of toys with
  d = 3 to 5 and k_n = 0 to 3. The "+ k_n" (biases) and "+ 2k" (centred inputs) terms follow from
  the generators those two cases leave out (what a softmax row and a centred input cannot see);
  the rank only confirms that nothing else is gained or lost. "Complete" is relative to the reduced model (S, A_p, B_p), i.e. it assumes the data
  and loss can produce every combination of those gradients, and distinct frequencies with at
  least 2P + 1 offsets. Not a proof for general (d, k_n, P), and nothing about MLPs or depth.
"""
import math

import numpy as np
import sympy as sp


# ------------------------------------------------------------------ D9

def _adamw(theta0, grads, etas, lam, b1=0.9, b2=0.95, eps=1e-8):
    th = theta0.copy(); m = np.zeros_like(th); v = np.zeros_like(th); us = []
    for t, (g, eta) in enumerate(zip(grads, etas), 1):
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        u = (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        th = th * (1 - eta * lam) - eta * u; us.append(u)
    return th, np.array(us)


def d9_adamw_integral(seed=0, n=200, T=400, lam=0.1, peak=1e-2, warm=100):
    rng = np.random.default_rng(seed)
    etas = np.array([peak * min(1.0, i / warm) for i in range(T)])
    th0 = rng.standard_normal(n); grads = rng.standard_normal((T, n)) + 0.3 * rng.standard_normal(n)
    thT, us = _adamw(th0, grads, etas, lam)
    dec = np.cumprod((1 - lam * etas)[::-1])[::-1]; A = dec[0]; w = etas * np.append(dec[1:], 1.0)
    assert np.abs(thT - (A * th0 - w @ us)).max() < 1e-12                                   # (a)
    t1, t2 = etas.sum(), (etas ** 2).sum()
    assert 0 < -math.log(A) - lam * t1 < lam ** 2 * t2                                      # (a) first order
    a = 150; tha, _ = _adamw(th0, grads[:a], etas[:a], lam)                                  # (b) a pair of checkpoints
    Aab = np.prod(1 - lam * etas[a:]); ubar = -(thT - Aab * tha) / w[a:].sum()
    assert np.abs(ubar - (w[a:] @ us[a:]) / w[a:].sum()).max() < 1e-12
    # (c) sign steps with decay on f(u v) = (u v - 1)^2 / 2: |u| - |v| decays as A exactly
    u_, v_ = 1.7, 0.4; d0 = abs(u_) - abs(v_); Ac = 1.0
    for eta in etas:
        r = u_ * v_ - 1.0
        u_, v_ = u_ * (1 - eta * lam) - eta * np.sign(r * v_), v_ * (1 - eta * lam) - eta * np.sign(r * u_)
        Ac *= 1 - eta * lam; assert u_ > 0 and v_ > 0
    assert abs((abs(u_) - abs(v_)) - Ac * d0) < 1e-12
    # (d) below eps Adam is SGD: a zero-sum gradient gives a zero-sum first step; above eps it does not
    g = np.full(n, 1.0 / n); g[0] -= 1.0                                                     # a softmax row's gradient: p - onehot
    lo = _adamw(np.zeros(n), [1e-12 * g], [1.0], 0.0)[0]; hi = _adamw(np.zeros(n), [g], [1.0], 0.0)[0]
    assert abs(lo.sum()) / np.abs(lo).sum() < 1e-3 and np.abs(lo + 1e-12 * g / 1e-8).max() < 1e-3 * np.abs(lo).max()
    assert abs(hi.sum()) / np.abs(hi).sum() > 1e-2
    print(f"D9: theta_T = A theta_0 - sum w u exact; -log A = lambda tau_1 + {(-math.log(A) - lam * t1) / (lam ** 2 * t2):.2f} lambda^2 tau_2; "
          f"checkpoint pairs give u_bar; |u|-|v| decays as A under sign steps; below eps the step is eta g/eps (zero-sum kept)  OK")


# ------------------------------------------------------------------ D10

def d10a_second_update(b1=0.9, b2=0.95):
    x = sp.symbols("x", positive=True); B1, B2 = sp.Rational(9, 10), sp.Rational(19, 20)
    u2 = (B1 + x) / (1 + B1) * sp.sqrt((1 + B2) / (B2 + x ** 2))
    assert sp.simplify(u2.subs(x, 1) - 1) == 0
    xs = sp.solve(sp.diff(u2, x), x); assert xs == [B2 / B1]
    umax = float(u2.subs(x, B2 / B1)); assert abs(umax - 1.00036) < 1e-5
    f = sp.lambdify(x, u2 - sp.Rational(85, 100)); lo = sp.nsolve(u2 - sp.Rational(85, 100), x, 0.3); hi = sp.nsolve(u2 - sp.Rational(85, 100), x, 5)
    assert abs(f(float(lo))) < 1e-12 and abs(f(float(hi))) < 1e-12 and abs(float(lo) - 0.2696) < 2e-4 and abs(float(hi) - 5.063) < 2e-3
    # one jump in gradient size, then steady, on a ramp whose rate is (t - 1) units at update t:
    # the cumulative displacement at checkpoints 2, 4, 8, 16 (ts_first_step.py measures 0.852, 5.455, 26.59, 117.3 on 70m)
    def cum(xv):
        m = v = c = 0.0; out = {}
        for t in range(1, 17):
            g = 1.0 if t == 1 else xv
            m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
            c += (t - 1) * (m / (1 - b1 ** t)) / math.sqrt(v / (1 - b2 ** t))
            if t in (2, 4, 8, 16):
                out[t] = c
        return out
    meas = ((4, 5.455), (8, 26.59), (16, 117.3)); down = cum(float(lo))
    assert all(down[t] / w - 1 < -0.10 for t, w in meas)
    # what the measurement pins (`/challenge-pr` on #355, finding 4): 70m's bias at checkpoint 2 is 10 units of 2^-24
    # against a rate of 11.73, so u_2 is only known to lie in [9.5, 10.5) / 11.73
    rate = 1e-3 / 1430 / 2.0 ** -24; grid = np.arange(1.1, 40, 0.01); u2f = sp.lambdify(x, u2)
    q = [xv for xv in grid if 9.5 / rate <= u2f(xv) < 10.5 / rate]
    fit = [xv for xv in q if all(abs(cum(xv)[t] / w - 1) < 0.015 for t, w in meas)]
    assert abs(q[0] - 3.36) < 0.02 and abs(q[-1] - 8.22) < 0.02 and abs(fit[0] - 4.61) < 0.02 and abs(fit[-1] - 7.94) < 0.02
    print(f"D10a: u_2(1) = 1, max {umax:.5f} at x = beta_2/beta_1; u_2 = 0.85 at x = {float(lo):.3f} and {float(hi):.2f}; fp16 pins the jump to "
          f"x in [{q[0]:.2f}, {q[-1]:.2f}], and [{fit[0]:.2f}, {fit[-1]:.2f}] if the gradients were steady after (checkpoints 4, 8, 16 within 1.5 %); "
          f"a drop to {float(lo):.2f} gives " + ", ".join(f"{down[t]:.2f}" for t in (4, 8, 16)) + " against 5.455, 26.59, 117.3 (over 10 % low)  OK")


def d10b_layernorm_jacobian(d=4):
    xs = sp.symbols(f"x0:{d}", real=True); X = sp.Matrix(xs); m = sum(xs) / d
    s = sp.sqrt(sum((xi - m) ** 2 for xi in xs) / d); y = (X - sp.ones(d, 1) * m) / s
    J = y.jacobian(X); pt = dict(zip(xs, [sp.Rational(1, 3), -2, sp.Rational(5, 7), 4][:d]))
    Jn = J.subs(pt); yn = y.subs(pt)
    assert sp.simplify(Jn.T * sp.ones(d, 1)) == sp.zeros(d, 1) and sp.simplify(Jn.T * yn) == sp.zeros(d, 1)
    P = sp.eye(d) - sp.ones(d, d) / d
    assert sp.simplify(Jn - (P - yn * yn.T / d) / s.subs(pt)) == sp.zeros(d, d)
    print("D10b: LayerNorm's J = (P - y y^T/d)/s; J^T 1 = 0 and J^T y = 0  OK")


def d10c_sign_through_matrix(seed=1, d=512, n=2048, reps=20):
    rng = np.random.default_rng(seed); want = 1 - math.acos(math.sqrt(2 / math.pi)) / math.pi
    got = np.mean([np.mean(np.sign(W.T @ g) == np.sign(W.T @ np.sign(g)))
                   for W, g in ((rng.standard_normal((d, n)), rng.standard_normal(d)) for _ in range(reps))])
    assert abs(want - 0.7940) < 1e-4 and abs(got - want) < 0.01
    print(f"D10c: sign(W^T g) against sign(W^T sign g): {got:.4f}, formula {want:.4f}  OK")


def d10d_toy_first_gradient(seed=2, V=2000, d=64, N=200_000, L=6):
    import torch
    torch.manual_seed(seed); rng = np.random.default_rng(seed); dt = torch.float64
    Vs = V - 100                                                                              # the last 100 ids never appear (Pythia: 3 % of rows)
    zipf = np.zeros(V); zipf[:Vs] = 1.0 / np.arange(1, Vs + 1); zipf /= zipf.sum()
    pin = np.zeros(V); pin[:Vs] = zipf[:Vs][rng.permutation(Vs)]; succ = rng.permutation(Vs)
    x = rng.choice(V, N, p=pin)
    y = np.where(rng.random(N) < 0.5, succ[x], rng.choice(V, N, p=zipf))                      # a planted bigram on half the tokens
    u = np.bincount(y, minlength=V) / N; ui = np.bincount(x, minlength=V) / N
    s_small, s_wang = math.sqrt(2 / (5 * d)), 2 / (L * math.sqrt(d))
    par = {k: v.requires_grad_() for k, v in {
        "E": s_small * torch.randn(V, d, dtype=dt), "Win": s_small * torch.randn(4 * d, d, dtype=dt), "bin": torch.zeros(4 * d, dtype=dt),
        "Wout": s_wang * torch.randn(d, 4 * d, dtype=dt), "bout": torch.zeros(d, dtype=dt),
        "bf": torch.zeros(d, dtype=dt), "U": s_small * torch.randn(V, d, dtype=dt)}.items()}
    ln = lambda z: (z - z.mean(-1, keepdim=True)) / z.std(-1, keepdim=True, unbiased=False)
    e = par["E"][torch.as_tensor(x)]
    h = e + torch.nn.functional.gelu(ln(e) @ par["Win"].T + par["bin"]) @ par["Wout"].T + par["bout"]
    xf = ln(h) + par["bf"]; loss = torch.nn.functional.cross_entropy(xf @ par["U"].T, torch.as_tensor(y))
    loss.backward(); G = {k: -v.grad.numpy() for k, v in par.items()}                          # the descent direction
    U0, E0 = par["U"].detach().numpy(), par["E"].detach().numpy(); t = U0.T @ (u - 1 / V)
    a_out = np.mean(np.sign(G["bout"]) == np.sign(t - t.mean()))
    a_f = np.mean(np.sign(G["bf"]) == np.sign(t))
    a_in = np.mean(np.sign(G["bin"]) == np.sign(par["Wout"].detach().numpy().T @ np.sign(G["bout"])))
    col = np.sign(xf.detach().numpy().mean(0)); row = np.sign(u - 1 / V)
    a_U = np.mean(np.sign(G["U"]) == np.outer(row, col)); a_U0 = np.mean(np.sign(G["U"][u == 0]) == -col)
    lnE = (E0 - E0.mean(1, keepdims=True)) / E0.std(1, keepdims=True); r0 = np.sign(ui @ lnE)
    a_Win = np.mean(np.sign(G["Win"]) == np.outer(np.sign(G["bin"]), r0))
    want = 1 - math.acos(math.sqrt(2 / math.pi)) / math.pi
    assert a_out > 0.9 and a_f > 0.85 and abs(a_in - want) < 0.05 and a_U > 0.75 and a_U0 > 0.9 and a_Win > 0.6
    print(f"D10d (toy, exact gradient at the init, loss {loss.item():.3f} against log V {math.log(V):.3f}): writer bias against "
          f"sign centre(W_U^T(u - 1/V)) {a_out:.3f}; final-LN bias against the uncentred form {a_f:.3f}; reader bias against "
          f"sign(W_out^T sign b_out) {a_in:.3f} (D10c {want:.3f}); W_U against sign(u_v - 1/V) x sign(mean stream) {a_U:.3f} "
          f"(never-seen rows {a_U0:.3f}); W_in against sign(b_in) x sign(unigram mean of LN(E)) {a_Win:.3f}  OK")


# ------------------------------------------------------------------ D11

def _generators(d, kn, P, bias=False, centred=False):
    """Hessians (D x D, constant) of every scalar the loss can see of theta = (Q, K), Q and K in R^{k x de}."""
    de = d + 1 if bias else d; k = kn + 2 * P; D = 2 * k * de
    qi = lambda a, j: a * de + j; ki = lambda a, j: k * de + a * de + j
    J = np.array([[0.0, -1.0], [1.0, 0.0]])
    B = np.linalg.qr(np.eye(d) - 1 / d)[0][:, :d - 1] if centred else np.eye(d)                # input directions
    if bias:
        Bq = np.zeros((de, B.shape[1] + 1)); Bq[:d, :-1] = B; Bq[d, -1] = 1; B = Bq            # the constant input carries the bias
    nb = B.shape[1]; out = []

    def add(rows, C, softmax_blind):
        for i in range(nb):
            for j in range(nb):
                if softmax_blind and bias and j == nb - 1:
                    continue                                                                   # key-independent: a softmax row cannot see it
                H = np.zeros((D, D))
                for a, ra in enumerate(rows):
                    for b, rb in enumerate(rows):
                        if C[a, b]:
                            M = C[a, b] * np.outer(B[:, i], B[:, j])
                            H[np.ix_([qi(ra, s) for s in range(de)], [ki(rb, t) for t in range(de)])] += M
                H = H + H.T; out.append(H)
    if kn:
        add(list(range(kn)), np.eye(kn), True)
    for p in range(P):
        add([kn + 2 * p, kn + 2 * p + 1], np.eye(2), False); add([kn + 2 * p, kn + 2 * p + 1], J, False)
    return out, D


def _laws(G, D, rng, tol=1e-8):
    """D - dim(g theta) for the matrix Lie algebra g generated by G, at a random theta; and the gap."""
    basis = np.zeros((0, D * D))

    def new(M):
        nonlocal basis
        v = M.ravel() / np.linalg.norm(M)
        for _ in range(2):
            v = v - basis.T @ (basis @ v)
        if np.linalg.norm(v) > tol:
            basis = np.vstack([basis, v / np.linalg.norm(v)]); return True
        return False
    front = [g / np.linalg.norm(g) for g in G if new(g)]
    while front:
        nxt = []
        for f in front:
            for g in G:
                C = f @ g - g @ f
                if np.linalg.norm(C) > 1e-12 and new(C):
                    nxt.append(C / np.linalg.norm(C))
        front = nxt
    out = []
    for _ in range(2):
        s = np.linalg.svd(basis.reshape(-1, D, D) @ rng.standard_normal(D), compute_uv=False)
        r = int((s > 1e-7 * s[0]).sum()); out.append((D - r, s[r - 1] / s[r] if r < len(s) else math.inf))
    assert out[0][0] == out[1][0]
    return out[0][0], min(o[1] for o in out), basis.shape[0]


def d11_rope_laws_complete(seed=3):
    rng = np.random.default_rng(seed)
    cases = [("no RoPE, k = 2 (control)", 4, 2, 0, False, False), ("no RoPE, k = 3 (control)", 4, 3, 0, False, False),
             ("RoPE, 2 non-rotary + 1 pair", 4, 2, 1, False, False), ("RoPE, 2 + 2 pairs", 5, 2, 2, False, False),
             ("RoPE, 1 + 1", 4, 1, 1, False, False), ("RoPE, 3 + 1", 5, 3, 1, False, False),
             ("RoPE, 2 pairs only", 4, 0, 2, False, False), ("RoPE 2 + 1, biases, softmax", 3, 2, 1, True, False),
             ("RoPE 2 + 1, centred inputs", 4, 2, 1, False, True)]
    for name, d, kn, P, bias, cen in cases:
        k = kn + 2 * P; want = kn * (kn + 1) // 2 + P + (kn if bias else 0) + (2 * k if cen else 0)
        G, D = _generators(d, kn, P, bias, cen); got, gap, dim = _laws(G, D, rng)
        assert got == want and gap > 1e8, (name, got, want, gap)
        full = k * (k + 1) // 2
        print(f"D11 {name}: D = {D}, Lie algebra dim {dim}, laws {got} = k_n(k_n+1)/2 + P"
              f"{' + k_n' if bias else ''}{' + 2k' if cen else ''} (F1 without RoPE would give {full}); gap {gap:.0e}  OK")


if __name__ == "__main__":
    d9_adamw_integral()
    d10a_second_update()
    d10b_layernorm_jacobian()
    d10c_sign_through_matrix()
    d10d_toy_first_gradient()
    d11_rope_laws_complete()
