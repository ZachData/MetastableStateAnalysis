"""
tools/session_cost.py -- what a Claude Code session cost, read off its transcript.

    python tools/session_cost.py ~/.claude/projects/<project>/<session>.jsonl
    python tools/session_cost.py <transcript> --row "unit name" --pr "#66"

Stdlib only. The transcript is one JSON record per line. What is counted:

- **Model calls**: distinct `message.id` over `assistant` records. One API call
  is written as several records (thinking, text, each tool_use), all carrying
  the same `usage`, so counting records would over-count by ~2-3x.
- **Context per call**: `input_tokens + cache_read_input_tokens +
  cache_creation_input_tokens`, i.e. everything the model re-read that call.
  Total is the sum over calls, peak the max. This, not tool output, is the
  cost driver (`LESSONS.md` lesson 9).
- **Output tokens**: `output_tokens` per call, summed.
- **Tool results**: characters of each `tool_result`, attributed to the tool
  named by the matching `tool_use` id. Characters, not tokens (~4 chars/token).
- **Files read**: `file_path` of every `Read` call, with offset/limit if given.

What it does *not* see: subagent transcripts live in separate files and are
not followed; tool-result size is in characters, not tokens.
"""
import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _result_chars(content):
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        n = 0
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    n += len(block.get("text", ""))
                elif block.get("type") == "image":
                    n += len(json.dumps(block.get("source", {})))
        return n
    return 0


def analyse(lines):
    """Return a dict of cost figures from an iterable of transcript lines."""
    usage_by_id = {}      # message.id -> usage (last record for that id wins)
    tool_names = {}       # tool_use id -> tool name
    reads = []            # (file_path, offset, limit)
    results = []          # (chars, tool name, tool_use id)
    bad_lines = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            bad_lines += 1
            continue
        msg = rec.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if rec.get("type") == "assistant":
            mid = msg.get("id")
            if mid and isinstance(msg.get("usage"), dict):
                usage_by_id[mid] = msg["usage"]
            for block in content if isinstance(content, list) else []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_names[block.get("id")] = block.get("name", "?")
                    if block.get("name") == "Read":
                        inp = block.get("input") or {}
                        reads.append((inp.get("file_path", "?"),
                                      inp.get("offset"), inp.get("limit")))
        elif rec.get("type") == "user" and isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tid = block.get("tool_use_id")
                    results.append((_result_chars(block.get("content")),
                                    tool_names.get(tid, "?"), tid))

    ctx = [u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
           + u.get("cache_creation_input_tokens", 0) for u in usage_by_id.values()]
    per_tool = defaultdict(lambda: [0, 0])
    for chars, name, _ in results:
        per_tool[name][0] += 1
        per_tool[name][1] += chars
    return {
        "calls": len(usage_by_id),
        "context_total": sum(ctx),
        "context_peak": max(ctx, default=0),
        "context_median": int(statistics.median(ctx)) if ctx else 0,
        "output_tokens": sum(u.get("output_tokens", 0) for u in usage_by_id.values()),
        "tool_result_chars": sum(r[0] for r in results),
        "per_tool": {k: {"calls": v[0], "chars": v[1]} for k, v in per_tool.items()},
        "largest_results": sorted(results, key=lambda r: -r[0])[:10],
        "reads": reads,
        "bad_lines": bad_lines,
    }


def _k(n):
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}k" if n >= 1e3 else str(n)


def report(r):
    out = [
        f"model calls        {r['calls']}",
        f"context, total     {r['context_total']:,}  ({_k(r['context_total'])})",
        f"context, peak      {r['context_peak']:,}",
        f"context, median    {r['context_median']:,}",
        f"output tokens      {r['output_tokens']:,}",
        f"tool results       {r['tool_result_chars']:,} chars (~{_k(r['tool_result_chars'] // 4)} tokens)",
        "",
        "per tool           calls      chars",
    ]
    for name, v in sorted(r["per_tool"].items(), key=lambda kv: -kv[1]["chars"]):
        out.append(f"  {name:<16} {v['calls']:>5} {v['chars']:>10,}")
    out += ["", "10 largest tool results (chars, tool, id)"]
    out += [f"  {c:>9,}  {n:<12} {t}" for c, n, t in r["largest_results"]]
    counts = Counter(p for p, _, _ in r["reads"])
    out += ["", f"files Read ({len(r['reads'])} calls)"]
    for path, n in counts.most_common():
        ranged = sum(1 for p, o, l in r["reads"] if p == path and (o or l))
        out.append(f"  {n}x ({ranged} ranged)  {path}")
    if r["bad_lines"]:
        out.append(f"\n{r['bad_lines']} unparseable line(s) skipped")
    return "\n".join(out)


def row(r, unit, pr, date):
    """One `docs/cost_log.md` table row."""
    return (f"| {date} | {unit} | {r['calls']} | {_k(r['context_peak'])} "
            f"| {_k(r['context_total'])} | {_k(r['tool_result_chars'] // 4)} | {pr} |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("transcript", type=Path)
    ap.add_argument("--row", metavar="UNIT", help="print a cost_log.md row for this unit")
    ap.add_argument("--pr", default="—")
    ap.add_argument("--date", default=None, help="default: today")
    a = ap.parse_args(argv)
    if not a.transcript.is_file():
        sys.exit(f"no such transcript: {a.transcript}")
    with a.transcript.open(encoding="utf-8") as f:
        r = analyse(f)
    if r["calls"] == 0:
        sys.exit(f"{a.transcript}: no assistant usage records; not a transcript?")
    if a.row:
        import datetime
        print(row(r, a.row, a.pr, a.date or datetime.date.today().isoformat()))
    else:
        print(report(r))


if __name__ == "__main__":
    main()
