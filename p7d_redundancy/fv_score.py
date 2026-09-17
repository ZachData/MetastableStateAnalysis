"""Function-vector score on the redundancy set: is §3.12-U's puzzle the
induction -> FV transition?

WHY THIS EXISTS
---------------
`PROJECT.md` §3.16. The verified literature scan's single most actionable
product is `2502.14010` (Yin & Steinhardt, ICML 2025), which reports induction
heads *becoming* **function-vector heads**: a head's induction score declines
over training while its FV score rises, and the head goes on mattering for
in-context learning the whole time.

§3.12-U measured something that looks like that from the causal side and filed
it as a puzzle: **`L5H2`'s induction score falls roughly twentyfold while its
causal effect on the readout goes +0.01 -> +4.97**. If `L5H2` is acquiring a
function-vector role as it sheds its induction role, the puzzle is not a puzzle
-- it is a known transition, seen through a different instrument. That is
cheap to test and this runner tests it.

WHAT AN FV SCORE IS (Todd et al., 2310.15213, as Yin & Steinhardt use it)
------------------------------------------------------------------------
For a task `t` with in-context prompts `p_i`, take head `(l, h)`'s output at
the **last token position**, averaged over the clean prompts: that mean vector
is the head's contribution to the task's function vector. Then build a
**shuffled-label** prompt -- same format, same query, labels permuted so the
mapping is destroyed -- and patch the clean mean into that head's slice at the
last position. The causal indirect effect is

    CIE = P(correct answer | corrupted + patch) - P(correct answer | corrupted)

and the FV score is the average indirect effect over prompts and tasks. A head
with a high FV score carries task-identity information that survives the
destruction of the in-context mapping. **It is a different quantity from the
induction score in kind, not in degree**: induction is measured on repeated
random tokens where there is no task at all.

WHAT THIS RUNNER MEASURES, AND ON ONE INSTRUMENT
------------------------------------------------
Both scores, per head, per checkpoint, in the same process off the same loaded
weights. That matters: the claim under test is about the two trajectories'
**joint** shape, so reading the induction score off a stored series measured
with another batch and another convention would put the comparison's whole
weight on an uncontrolled difference.

THE NULL IS MEASURED, NOT ASSUMED (§3.12-V3, `design-8.md`). An FV score is a
probability difference and its chance level is not zero -- patching *any*
vector into a corrupted prompt perturbs it. `--controls N` draws N heads
uniformly from the non-member population and scores them identically, so the
members are read against this checkpoint's own spread rather than against 0.

REPORTED BOTH WAYS (§3.13). `fv_score` is the mean CIE in probability, which is
Todd's own definition and is declared primary HERE, in advance, so it cannot be
swapped after the fact. `fv_score_logp` (the same patch scored in log
probability) and `fv_score_median` are carried beside it and never swapped in.
The mean/median gap is the quantity §3.13 exists for and 70m's probe work found
a factor of 5.7 in.

PROMPTS ARE BUILT AND THEN GROUPED BY LENGTH, never padded. GPT-NeoX has no
native pad token and left-padding needs a position-id fix that is easy to get
silently wrong; equal-length batches need neither.

COST: one model load per step, then per task one clean pass plus `n_heads + 1`
corrupted passes. Six members and three controls over four tasks is ~40 short
forward passes per checkpoint.

NO p-value; nothing registered. pythia-410m is spent under `check_registry`
rule 3 -- exploratory, and not registrable on this data.
"""
import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", "/run/media/system/WDS_500/Mets"))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))
_want = str(REPO / ".venv")
if not sys.prefix.startswith(_want):
    raise SystemExit(f"wrong interpreter: {sys.prefix!r}, need {_want!r}")

import numpy as np
import torch

from core.lm_loading import load_causal_lm
from tools.run.induction_rank_sweep import (
    EVAL_SEED, N_REP, PROBE_ARMS, arch_dims,
)

from p7d_redundancy.member_formation_curves import batch, catalog_path_for, members

OUT = DATA / "analysis" / "fv_score.json"
DEFAULT_MODEL = "pythia-410m"

#: Word-pair tasks, in Todd et al.'s families. These are CANDIDATES: a pair is
#: kept only if the whole prompt tokenises to the length the builder expects,
#: so the effective list is model-dependent and is recorded in the output.
#: Deliberately mixed in kind -- two lexical-semantic (antonym, country) and
#: two morphological (past tense, plural) -- because a function vector that
#: only ever shows up on one family is a property of that family.
TASKS = {
    "antonym": [
        ("hot", "cold"), ("big", "small"), ("fast", "slow"), ("up", "down"),
        ("good", "bad"), ("light", "dark"), ("high", "low"), ("old", "new"),
        ("long", "short"), ("hard", "soft"), ("rich", "poor"), ("open", "closed"),
        ("true", "false"), ("young", "old"), ("full", "empty"), ("wet", "dry"),
        ("strong", "weak"), ("early", "late"), ("happy", "sad"), ("clean", "dirty"),
    ],
    "country_capital": [
        ("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"),
        ("Russia", "Moscow"), ("China", "Beijing"), ("Spain", "Madrid"),
        ("Germany", "Berlin"), ("Egypt", "Cairo"), ("Greece", "Athens"),
        ("Cuba", "Havana"), ("Austria", "Vienna"), ("Poland", "Warsaw"),
        ("Portugal", "Lisbon"), ("Ireland", "Dublin"), ("Norway", "Oslo"),
        ("Sweden", "Stockholm"), ("Denmark", "Copenhagen"), ("Kenya", "Nairobi"),
        ("Peru", "Lima"), ("Iran", "Tehran"),
    ],
    "present_past": [
        ("go", "went"), ("see", "saw"), ("take", "took"), ("give", "gave"),
        ("run", "ran"), ("come", "came"), ("know", "knew"), ("think", "thought"),
        ("make", "made"), ("find", "found"), ("write", "wrote"), ("speak", "spoke"),
        ("break", "broke"), ("drive", "drove"), ("eat", "ate"), ("fall", "fell"),
        ("hold", "held"), ("keep", "kept"), ("leave", "left"), ("send", "sent"),
    ],
    "singular_plural": [
        ("cat", "cats"), ("dog", "dogs"), ("book", "books"), ("car", "cars"),
        ("tree", "trees"), ("house", "houses"), ("bird", "birds"), ("hand", "hands"),
        ("year", "years"), ("word", "words"), ("game", "games"), ("road", "roads"),
        ("king", "kings"), ("song", "songs"), ("door", "doors"), ("river", "rivers"),
        ("table", "tables"), ("letter", "letters"), ("number", "numbers"),
        ("window", "windows"),
    ],
}


def build_prompt(tok, shots, query_x):
    """Token ids for `x: y\\n` repeated, then `query_x:`.

    Tokenised as one string rather than assembled from pieces, so the model
    sees the segmentation it would see in the wild. The caller groups the
    results by length; nothing is padded.
    """
    text = "".join(f"{x}: {y}\n" for x, y in shots) + f"{query_x}:"
    return tok(text, return_tensors=None)["input_ids"]


def answer_token(tok, y):
    """The single id for ` y`, or None if it is not one token.

    The readout is the probability of ONE token, so a multi-token answer would
    need a length-normalised score with a different chance level. Dropping
    those pairs is the honest option and the drops are recorded.
    """
    ids = tok(f" {y}", return_tensors=None)["input_ids"]
    return ids[0] if len(ids) == 1 else None


def make_prompts(tok, pairs, n_shot, n_prompts, rng):
    """`(clean, corrupted)` prompt pairs sharing a query and an answer.

    Corruption is Todd et al.'s **shuffled labels**: the demonstrations keep
    their inputs and their format and get each other's outputs, under a
    derangement so no demonstration keeps its own. The query and the correct
    answer are untouched, so the only thing destroyed is the mapping the model
    would have to infer -- which is exactly the thing a function vector is
    supposed to supply from elsewhere.
    """
    usable = [(x, y, t) for x, y in pairs
              if (t := answer_token(tok, y)) is not None]
    out = []
    for _ in range(n_prompts):
        pick = rng.choice(len(usable), size=n_shot + 1, replace=False)
        shots = [usable[i] for i in pick[:n_shot]]
        qx, qy, qtok = usable[pick[n_shot]]

        # Derangement of the shot labels: index i never keeps label i.
        for _try in range(100):
            perm = rng.permutation(n_shot)
            if not any(perm == np.arange(n_shot)):
                break
        else:                                   # n_shot == 1 has no derangement
            continue

        clean = build_prompt(tok, [(x, y) for x, y, _ in shots], qx)
        corrupt = build_prompt(
            tok, [(shots[i][0], shots[perm[i]][1]) for i in range(n_shot)], qx)
        out.append({"clean": clean, "corrupt": corrupt,
                    "query": qx, "answer": qy, "answer_id": qtok})
    return out, len(usable)


def group_by_length(prompts, key):
    """`{length: [index, ...]}` so every batch is rectangular without padding."""
    g = {}
    for i, p in enumerate(prompts):
        g.setdefault(len(p[key]), []).append(i)
    return g


@torch.no_grad()
def last_token_head_means(model, prompts, heads, chunk=8):
    """`{(L, H): mean output vector at the LAST position}` over clean prompts.

    The last position is where the model must emit the answer, and it is the
    position Todd et al. read and patch. A mean over all positions -- which is
    what `induction_rank_sweep.head_means` takes, for a different purpose --
    would average the task signal into the demonstrations' own token
    predictions and is not the same quantity.
    """
    _, d_head, _ = arch_dims(model)
    by_layer = {}
    for (L, H) in heads:
        by_layer.setdefault(L, []).append(H)
    acc = {}

    def make_hook(L, idxs):
        def hook(mod, args):
            x = args[0]
            for H in idxs:
                v = x[:, -1, H * d_head:(H + 1) * d_head]      # (B, d_head)
                s, n = acc.get((L, H), (0.0, 0))
                acc[(L, H)] = (s + v.double().sum(0), n + v.shape[0])
        return hook

    handles = [model.gpt_neox.layers[L].attention.dense
               .register_forward_pre_hook(make_hook(L, idxs))
               for L, idxs in by_layer.items()]
    try:
        for ln, idxs in group_by_length(prompts, "clean").items():
            for i in range(0, len(idxs), chunk):
                ids = torch.tensor([prompts[j]["clean"] for j in idxs[i:i + chunk]],
                                   dtype=torch.long)
                model(ids)
    finally:
        for h in handles:
            h.remove()
    return {k: (s / n) for k, (s, n) in acc.items()}


class patched:
    """Write one head's vector into its slice at the LAST position only.

    Everything before the last position is left alone, so the patch cannot
    change how the corrupted demonstrations were read -- only what the model
    has available at the moment it answers. That is what makes the effect
    attributable to the head's task representation rather than to a
    re-processing of the prompt.
    """

    def __init__(self, model, head, vec):
        self.model, self.head, self.vec, self.handles = model, head, vec, []

    def __enter__(self):
        L, H = self.head
        _, d_head, _ = arch_dims(self.model)
        sl = slice(H * d_head, (H + 1) * d_head)
        vec = self.vec

        def hook(mod, args):
            x = args[0].clone()
            x[:, -1, sl] = vec.to(x.dtype)
            return (x,) + args[1:]

        self.handles.append(
            self.model.gpt_neox.layers[L].attention.dense
            .register_forward_pre_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        return False


@torch.no_grad()
def answer_probs(model, prompts, key, chunk=8):
    """`(P(answer), logP(answer))` at the final position, one row per prompt."""
    p = np.zeros(len(prompts))
    lp = np.zeros(len(prompts))
    for ln, idxs in group_by_length(prompts, key).items():
        for i in range(0, len(idxs), chunk):
            sel = idxs[i:i + chunk]
            ids = torch.tensor([prompts[j][key] for j in sel], dtype=torch.long)
            logits = model(ids).logits[:, -1, :].float()
            logprob = torch.log_softmax(logits, dim=-1)
            for r, j in enumerate(sel):
                lp[j] = float(logprob[r, prompts[j]["answer_id"]])
                p[j] = float(np.exp(lp[j]))
    return p, lp


@torch.no_grad()
def induction_scores(model, ids, heads, chunk=2):
    """Mean post-softmax attention from each second-copy query to the
    first-copy occurrence of the same token, per head.

    The same quantity `induction_qk_sweep.induction_attention` reads, batched
    over heads so one pass serves them all.
    """
    want = {}
    for (L, H) in heads:
        want.setdefault(L, []).append(H)
    q = torch.arange(N_REP, 2 * N_REP)
    k = torch.arange(0, N_REP)
    tot = {h: 0.0 for h in heads}
    n = 0
    for i in range(0, len(ids), chunk):
        out = model(ids[i:i + chunk], output_attentions=True)
        b = ids[i:i + chunk].shape[0]
        for L, Hs in want.items():
            att = out.attentions[L].float()                   # (B, n_head, T, T)
            for H in Hs:
                tot[(L, H)] += float(att[:, H, q, k].mean()) * b
        n += b
        del out
    return {h: v / n for h, v in tot.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="registry family prefix (design-8.md rung policy: "
                         "1b/1.4b are reserved -- do not pass those without a "
                         "registered prediction)")
    ap.add_argument("--steps", default="1000,2000,4000,8000,16000,143000",
                    help="§3.12-U's window plus the endpoint -- the interval "
                         "where L5H2's induction score falls while its causal "
                         "effect rises is (1000, 16000]")
    ap.add_argument("--top", type=int, default=6,
                    help="catalogue members to score")
    ap.add_argument("--heads", default="", help="explicit 'L5H2,L7H8' override")
    ap.add_argument("--controls", type=int, default=3,
                    help="non-member heads drawn uniformly, scored identically. "
                         "The FV chance level is NOT zero -- patching any "
                         "vector perturbs a corrupted prompt -- so this is the "
                         "measured null the members are read against")
    ap.add_argument("--tasks", default=",".join(TASKS),
                    help="comma-separated subset of " + ", ".join(TASKS))
    ap.add_argument("--n-shot", type=int, default=10)
    ap.add_argument("--n-prompts", type=int, default=16,
                    help="prompts per task; CIE is averaged over all of them")
    ap.add_argument("--seqs", type=int, default=8,
                    help="repeated-random sequences for the induction score")
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--probe", default="wide", choices=tuple(PROBE_ARMS),
                    help="token range for the INDUCTION half only; the FV half "
                         "uses natural-language tasks and is unaffected")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out = Path(args.out) if args.out else (
        OUT if args.model == DEFAULT_MODEL else
        DATA / "analysis" / f"fv_score_{args.model}.json")

    def parse(spec):
        return [(int(h[1:h.index("H")]), int(h[h.index("H") + 1:]))
                for h in spec.split(",") if h]

    if args.heads:
        heads, source = parse(args.heads), "--heads"
    else:
        heads, source = members(args.top, catalog_path_for(args.model))
    task_names = [t for t in args.tasks.split(",") if t]
    for t in task_names:
        if t not in TASKS:
            raise SystemExit(f"unknown task {t!r}; have {', '.join(TASKS)}")
    steps = [int(x) for x in args.steps.split(",") if x]

    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    res = {"_what_this_is": __doc__, "git_sha": git_sha, "model": args.model,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
           "membership_source": source, "probe": args.probe,
           "primary_statistic": "fv_score (mean CIE in probability)",
           "eval": {"tasks": task_names, "n_shot": args.n_shot,
                    "n_prompts_per_task": args.n_prompts, "seed": EVAL_SEED,
                    "n_seqs_induction": args.seqs, "n_controls": args.controls},
           "members": [f"L{L}H{H}" for L, H in heads],
           "steps": steps, "per_step": {}}

    print(f"model:   {args.model}")
    print(f"members: {', '.join(f'L{L}H{H}' for L, H in heads)}   ({source})")
    print(f"tasks:   {', '.join(task_names)}   "
          f"{args.n_shot}-shot x {args.n_prompts} prompts")
    print(f"FV score = mean over prompts of "
          f"P(answer | corrupted + patch) - P(answer | corrupted)\n")

    for s in steps:
        model, tok = load_causal_lm(f"{args.model}-step{s}")
        model.eval()
        _, _, n_heads = arch_dims(model)
        n_layers = model.config.num_hidden_layers

        rng = np.random.default_rng(EVAL_SEED)
        pool = [(L, H) for L in range(n_layers) for H in range(n_heads)
                if (L, H) not in set(heads)]
        ctrl = [pool[i] for i in
                rng.choice(len(pool), size=args.controls, replace=False)]
        scored = list(heads) + ctrl
        is_member = {h: (h in set(heads)) for h in scored}

        # --- FV half -------------------------------------------------------
        per_task, cie = {}, {h: [] for h in scored}
        cie_lp = {h: [] for h in scored}
        for t in task_names:
            prompts, n_usable = make_prompts(
                tok, TASKS[t], args.n_shot, args.n_prompts,
                np.random.default_rng(EVAL_SEED))
            base_p, base_lp = answer_probs(model, prompts, "corrupt", args.chunk)
            clean_p, _ = answer_probs(model, prompts, "clean", args.chunk)
            mus = last_token_head_means(model, prompts, scored, args.chunk)
            per_task[t] = {
                "n_pairs_usable": n_usable, "n_pairs_total": len(TASKS[t]),
                "p_correct_clean": float(clean_p.mean()),
                "p_correct_corrupt": float(base_p.mean()),
                "icl_gap": float(clean_p.mean() - base_p.mean()), "heads": {}}
            for h in scored:
                with patched(model, h, mus[h]):
                    pp, plp = answer_probs(model, prompts, "corrupt", args.chunk)
                d, dlp = pp - base_p, plp - base_lp
                cie[h].extend(d.tolist())
                cie_lp[h].extend(dlp.tolist())
                per_task[t]["heads"][f"L{h[0]}H{h[1]}"] = {
                    "cie": float(d.mean()), "cie_median": float(np.median(d)),
                    "cie_logp": float(dlp.mean())}
            print(f"step {s} / {t:>16}: clean P {clean_p.mean():.4f}  "
                  f"corrupt P {base_p.mean():.4f}  "
                  f"ICL gap {clean_p.mean() - base_p.mean():+.4f}  "
                  f"({n_usable}/{len(TASKS[t])} pairs usable)", flush=True)

        # --- induction half, same weights, same process --------------------
        ids = batch(np.random.default_rng(EVAL_SEED), args.seqs,
                    *PROBE_ARMS[args.probe])
        ind = induction_scores(model, ids, scored, max(1, args.chunk // 2))

        rows = {}
        print(f"\n{'head':>8} {'kind':>7} {'induction':>10} {'FV score':>10} "
              f"{'FV median':>10} {'FV logP':>9}")
        for h in scored:
            v = np.array(cie[h])
            rows[f"L{h[0]}H{h[1]}"] = {
                "member": is_member[h],
                "induction_score": ind[h],
                "fv_score": float(v.mean()),
                "fv_score_median": float(np.median(v)),
                "fv_score_logp": float(np.mean(cie_lp[h])),
                "fv_score_sd": float(v.std()),
                "n_cie": int(v.size)}
            print(f"{f'L{h[0]}H{h[1]}':>8} "
                  f"{'member' if is_member[h] else 'control':>7} "
                  f"{ind[h]:>10.4f} {v.mean():>+10.4f} "
                  f"{np.median(v):>+10.4f} {np.mean(cie_lp[h]):>+9.3f}")

        mem = [r["fv_score"] for r in rows.values() if r["member"]]
        con = [r["fv_score"] for r in rows.values() if not r["member"]]
        print(f"\n  members mean FV {np.mean(mem):+.4f}   "
              f"controls mean FV {np.mean(con):+.4f}   "
              f"gap {np.mean(mem) - np.mean(con):+.4f}\n")

        res["per_step"][str(s)] = {
            "per_task": per_task, "heads": rows,
            "member_mean_fv": float(np.mean(mem)),
            "control_mean_fv": float(np.mean(con)),
            "controls": [f"L{L}H{H}" for L, H in ctrl]}

        # Written per step: a killed run must not lose the steps it finished.
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(res, fh, indent=2)

        del model, tok
        gc.collect()

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
