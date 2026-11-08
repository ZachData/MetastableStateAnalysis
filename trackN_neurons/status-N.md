<!-- trackN_neurons/status-N.md -->
# Track N — STATUS

What the MLP's neurons do across training (`docs/ROADMAP.md` §7c, the user's, 2026-11-02).
No card yet: the first result lands at N1 U1.

## N1 U0, the scan and the design (2026-11-08)

Writing only, no forward pass. Branch `claude/n1-design`.

- **Scan:** `docs/lit-N.md`. The companion paper to the essay is not out. **N3's core test has
  been run on `pythia-410m` outside** (neuron-lineage: new jobs sprout next to old ones more
  often than for random directions, 10.4 % vs 6.5 % on its role embedding, most births at
  256–2000; README summary only), and on `pythia-160m` with LLM grading (two agents agree on
  ~40 %). Known-job neurons exist on Pythia in the literature: 410m 23.417 (token frequency),
  70m L3N669 (German, dated across checkpoints). 2509.24198 finds GeLU's negative
  pre-activations functional on a Pythia and forming over training.
- **The unit:** proposed in `trackN_neurons/design-N1.md` "What a pole is under GeLU": both
  poles read on the pre-activation z, the write band at z* = −0.75179 beside. The closed-form
  facts it rests on pass (`tools/math_checks/n1_gelu_pole.py`, 8 of 8; re-run:
  `python3 tools/math_checks/n1_gelu_pole.py`). What the check does not prove is in its docstring.
- **Grades so far** (design-N1.md's scale): Q1 confirmed for the essay (local copy), unknown for
  the companion paper (not out); Q2 leaning (READMEs and a blog, not code); Q3 confirmed for
  L3N669's job (2311.00863's text), leaning for 23.417 (summary of 2406.16254), unknown for
  L3N669's step; Q4 the math confirmed, the training GeLU unknown; Q5 not started; Q6 open.
- **Blocked 144** (the user's): D1 the pole, D2 the budget, D3 N3's direction (design-N1.md
  "Decisions").

## Corrections received

