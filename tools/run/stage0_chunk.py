"""
Phase 10 Stage 0 in time-boxed chunks: every battery-v2 prompt through Phase 1
at every `pythia-410m` checkpoint of the WDS sweep, a budget of hours at a time.

`p10_cluster_function/handoff-10.md` §0: 20 prompts × 19 checkpoints = 380 run
directories, all under battery v2 (`06790b90dcfe`) from one commit, so the
enlarged battery has one hash, one toolchain and a native partition throughout
(the user's option B, 2026-09-22). At ~3–4 min a run that is ~22 h of CPU, and
the machine is only available in blocks, so the work is cut into chunks.

**Resumable by construction.** Nothing here keeps a ledger that can drift from
the disk: what is done is re-read from `METS_RESULTS_DIR` on every invocation.
A (checkpoint, prompt) pair is done iff some run directory holds a
`manifest.json` for it whose battery hash is v2 **and** whose `git_sha` is the
pinned commit, with a populated `hdbscan_labels.json` and a populated
`pair_agreement` entry. A chunk killed mid-run leaves a directory with no
manifest (it is written last), which therefore does not count and is re-run.

**Pinned commit.** `--pin` must equal `HEAD` of the tree the runs are launched
from, and that tree must be clean — `run_1` writes `git rev-parse HEAD` into
every manifest, so a pull between chunks would otherwise split the battery
across commits without anything failing.

**Budget.** A chunk only starts a prompt whose estimated finish fits inside
`--budget-hours`, and with `--hard-stop` (default) it also kills the running
invocation at the deadline. Estimates start from the 2026-09-22 probe
(`wiki_byzantium` × step143000: 203 s at 404 tokens, `handoff-10.md` §0.2) and
are re-fitted from this sweep's own manifests as soon as it has some.

**Fail fast.** After the first invocation of a chunk, checks 2 and 3 of
`handoff-10.md` §0.3 are re-read off disk; a run that came back with an empty
partition or a zeroed `pair_agreement` stops the chunk rather than being
followed by 379 more (`LESSONS.md`: an instrument that degrades instead of
refusing).

    # what would the next chunk do, and how many chunks remain?
    python -m tools.run.stage0_chunk --pin <sha> plan
    # run one chunk (from the pinned worktree, conda `mets` env)
    python -m tools.run.stage0_chunk --pin <sha> run --budget-hours 10
    # what is done
    python -m tools.run.stage0_chunk --pin <sha> status

The module is import-light on purpose (no torch at import): the planning logic
is tested in the pure tier; only `main()` imports the prompt battery and the
tokenizer.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

MODEL_FAMILY = "pythia-410m"

# The 19 checkpoints of the WDS sweep (`data/phase12/2026-08-31_*`,
# `2026-09-01_*`), in the order they are run: the ends first, so a sweep that
# stops early still spans the training axis. step0 and step1 are the same
# weights upstream (PROJECT.md §3.51 (7)) and both are run anyway, to keep the
# grid identical to the 152 v1 directories.
CHECKPOINTS: Tuple[int, ...] = (
    0, 143000, 1000, 8000, 64, 32000, 256, 2000, 16, 4000,
    1, 2, 4, 8, 32, 128, 512, 16000, 54000,
)

# v2 minus `short_heterogeneous` (115 characters; handoff-10.md §0.1).
EXCLUDED_PROMPTS = frozenset({"short_heterogeneous"})

BATTERY_HASH_V2 = "06790b90dcfe"

# The probe (handoff-10.md §0.2): 203 s in the manifest at 404 tokens.
PROBE_SECONDS = 203.0
PROBE_TOKENS = 404
# Model load + V-spectrum per invocation: 3 min 51 s wall minus 203 s.
LOAD_SECONDS = 30.0
# Attention is n², activations n; between the two the probe cannot tell, so
# the prior leans on the steeper term, and a safety factor covers the rest.
LENGTH_EXPONENT = 1.5
SAFETY = 1.25
# Below this many of the sweep's own runs, keep the probe's scale.
MIN_REFIT = 5


@dataclass(frozen=True)
class Done:
    step: int
    prompt: str
    run_dir: Path
    wall_time_seconds: float
    n_tokens: int


def _populated_partition(run_dir: Path) -> bool:
    try:
        labels = json.loads((run_dir / "hdbscan_labels.json").read_text())
    except (OSError, ValueError):
        return False
    return bool(labels)


def _populated_pair_agreement(exp_dir: Path, prompt: str) -> bool:
    try:
        entry = json.loads((exp_dir / "pair_agreement.json").read_text())[prompt]
    except (OSError, ValueError, KeyError, TypeError):
        return False
    total = sum(entry.get(k, 0) or 0 for k in ("n_semantic", "n_artifact", "n_noise"))
    return total > 0


def _n_tokens(run_dir: Path) -> int:
    try:
        return len((run_dir / "tokens.txt").read_text().split("\n"))
    except OSError:
        return 0


def scan_done(results_dir: Path, pin: str) -> Dict[Tuple[int, str], Done]:
    """Every (step, prompt) with a complete, pinned, populated v2 run on disk."""
    done: Dict[Tuple[int, str], Done] = {}
    for manifest in sorted(results_dir.glob(f"*/{MODEL_FAMILY}-step*_*/manifest.json")):
        run_dir = manifest.parent
        try:
            m = json.loads(manifest.read_text())
        except (OSError, ValueError):
            continue
        if m.get("prompt_battery_hash") != BATTERY_HASH_V2 or m.get("git_sha") != pin:
            continue
        model, step, prompt = m.get("model", ""), m.get("checkpoint_step"), m.get("prompt_key")
        if not model.startswith(f"{MODEL_FAMILY}-step") or step is None or not prompt:
            continue
        if not (_populated_partition(run_dir)
                and _populated_pair_agreement(run_dir.parent, prompt)):
            continue
        done[(int(step), prompt)] = Done(int(step), prompt, run_dir,
                                         float(m.get("wall_time_seconds") or 0.0),
                                         _n_tokens(run_dir))
    return done


def fit_scale(done: Iterable[Done]) -> float:
    """Seconds at PROBE_TOKENS, re-fitted from the sweep's own runs once it has some.

    Median of t_i / (n_i / PROBE_TOKENS)^LENGTH_EXPONENT; the probe until
    MIN_REFIT runs exist.
    """
    ratios = sorted(d.wall_time_seconds / (d.n_tokens / PROBE_TOKENS) ** LENGTH_EXPONENT
                    for d in done if d.n_tokens > 0 and d.wall_time_seconds > 0)
    if len(ratios) < MIN_REFIT:
        return PROBE_SECONDS
    mid = len(ratios) // 2
    return ratios[mid] if len(ratios) % 2 else 0.5 * (ratios[mid - 1] + ratios[mid])


def estimate_seconds(n_tokens: int, scale: float) -> float:
    return SAFETY * scale * (n_tokens / PROBE_TOKENS) ** LENGTH_EXPONENT


def todo(prompts: Sequence[str], done: Dict[Tuple[int, str], Done]
         ) -> List[Tuple[int, List[str]]]:
    """Missing prompts per checkpoint, in CHECKPOINTS order, empty ones dropped."""
    out = []
    for step in CHECKPOINTS:
        missing = [p for p in prompts if (step, p) not in done]
        if missing:
            out.append((step, missing))
    return out


def plan_chunk(remaining: List[Tuple[int, List[str]]], n_tokens: Dict[str, int],
               scale: float, budget_s: float) -> List[Tuple[int, List[str]]]:
    """Greedy: whole checkpoints in order, then the prompts of the next that fit.

    One invocation per checkpoint amortises the model load over its prompts.
    Prompts are never reordered within a checkpoint, so a partly-run
    checkpoint is finished by the next chunk rather than skipped.
    """
    chunk, used = [], 0.0
    for step, prompts in remaining:
        taken, cost = [], LOAD_SECONDS
        for p in prompts:
            c = estimate_seconds(n_tokens[p], scale)
            if used + cost + c > budget_s:
                break
            taken.append(p)
            cost += c
        if taken:
            chunk.append((step, taken))
            used += cost
        if len(taken) < len(prompts):
            break
    return chunk


def n_chunks_left(remaining, n_tokens, scale, budget_s) -> int:
    """How many chunks of this budget the rest of Stage 0 takes, on the estimate."""
    count, rem = 0, [(s, list(p)) for s, p in remaining]
    while rem:
        chunk = plan_chunk(rem, n_tokens, scale, budget_s)
        if not chunk:
            raise ValueError("a single prompt does not fit in the budget")
        count += 1
        taken = {(s, p) for s, ps in chunk for p in ps}
        rem = [(s, [p for p in ps if (s, p) not in taken]) for s, ps in rem]
        rem = [(s, ps) for s, ps in rem if ps]
    return count


def chunk_seconds(chunk, n_tokens, scale) -> float:
    return sum(LOAD_SECONDS + sum(estimate_seconds(n_tokens[p], scale) for p in ps)
               for _, ps in chunk)


# ---------------------------------------------------------------------------
# Runtime (not in the pure tier)
# ---------------------------------------------------------------------------

def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def _require_pinned_tree(pin: str) -> None:
    head = _git("rev-parse", "HEAD")
    if head != pin:
        sys.exit(f"HEAD is {head[:12]}, --pin is {pin[:12]}: run from the pinned tree")
    dirty = [l for l in _git("status", "--porcelain").splitlines()
             if not l[3:].startswith("data/")]
    if dirty:
        sys.exit("tree is dirty; the manifests would record a commit that is not "
                 "what ran:\n  " + "\n  ".join(dirty))


def _battery() -> Tuple[List[str], Dict[str, int]]:
    from core.config import PROMPTS
    from core.prompts import PROMPT_BATTERY_HASH
    if PROMPT_BATTERY_HASH != BATTERY_HASH_V2:
        sys.exit(f"battery is {PROMPT_BATTERY_HASH}, Stage 0 is {BATTERY_HASH_V2}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m", revision="step143000")
    prompts = [k for k in PROMPTS if k not in EXCLUDED_PROMPTS]
    return prompts, {k: len(tok(PROMPTS[k])["input_ids"]) for k in prompts}


def _log(fh, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def _run_chunk(chunk, pin, results_dir, budget_s, hard_stop, log_dir) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    deadline = time.monotonic() + budget_s
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    first = True
    with open(log_dir / f"chunk_{stamp}.log", "a") as fh:
        _log(fh, f"chunk start, pin {pin[:12]}, budget {budget_s / 3600:.1f} h, "
                 f"{sum(len(p) for _, p in chunk)} runs over {len(chunk)} checkpoints")
        for step, prompts in chunk:
            left = deadline - time.monotonic()
            if left <= LOAD_SECONDS:
                _log(fh, "budget exhausted before step%d; stopping" % step)
                break
            cmd = [sys.executable, "-m", "p1_mstate_tracking.run_1",
                   "--models", f"{MODEL_FAMILY}-step{step}", "--prompts", *prompts]
            _log(fh, f"step{step}: {len(prompts)} prompts: {' '.join(prompts)}")
            with open(log_dir / f"chunk_{stamp}_step{step}.out", "w") as out:
                proc = subprocess.Popen(cmd, env=env, stdout=out,
                                        stderr=subprocess.STDOUT,
                                        start_new_session=True)
                try:
                    rc = proc.wait(timeout=left if hard_stop else None)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait()
                    _log(fh, f"step{step}: killed at the deadline; its unfinished "
                             "prompts have no manifest and will re-run next chunk")
                    return 0
            done = scan_done(results_dir, pin)
            got = [p for p in prompts if (step, p) in done]
            _log(fh, f"step{step}: rc {rc}, {len(got)}/{len(prompts)} complete")
            if first and not got:
                _log(fh, "FIRST INVOCATION PRODUCED NO COMPLETE RUN (empty partition, "
                         "zeroed pair_agreement, or a crash) -- stopping the chunk. "
                         "See the .out file.")
                return 2
            first = False
        _log(fh, "chunk end")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pin", required=True, help="full commit sha the runs come from")
    ap.add_argument("--results-dir", default=os.environ.get("METS_RESULTS_DIR"))
    ap.add_argument("--budget-hours", type=float, default=10.0)
    ap.add_argument("--no-hard-stop", dest="hard_stop", action="store_false")
    ap.add_argument("action", choices=["status", "plan", "run"])
    a = ap.parse_args(argv)
    if not a.results_dir:
        sys.exit("set METS_RESULTS_DIR or --results-dir")
    results_dir = Path(a.results_dir)

    prompts, n_tokens = _battery()
    done = scan_done(results_dir, a.pin)
    total = len(CHECKPOINTS) * len(prompts)
    print(f"Stage 0: {len(done)}/{total} done at pin {a.pin[:12]}")
    if a.action == "status":
        return 0

    scale = fit_scale(done.values())
    budget_s = a.budget_hours * 3600
    remaining = todo(prompts, done)
    if not remaining:
        print("nothing left")
        return 0
    chunk = plan_chunk(remaining, n_tokens, scale, budget_s)
    print(f"scale {scale:.0f} s @ {PROBE_TOKENS} tok ({'probe' if len(done) < MIN_REFIT else 'refit'}); "
          f"this chunk: {sum(len(p) for _, p in chunk)} runs, "
          f"~{chunk_seconds(chunk, n_tokens, scale) / 3600:.1f} h estimated; "
          f"chunks left incl. this: {n_chunks_left(remaining, n_tokens, scale, budget_s)}")
    for step, ps in chunk:
        print(f"  step{step}: {len(ps)} prompts")
    if a.action == "plan":
        return 0

    _require_pinned_tree(a.pin)
    return _run_chunk(chunk, a.pin, results_dir, budget_s, a.hard_stop,
                      results_dir / "stage0_logs")


if __name__ == "__main__":
    sys.exit(main())
