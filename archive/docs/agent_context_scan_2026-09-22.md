# Agent-context scan, 2026-09-22: what the docs and the literature say about CLAUDE.md, STATE.md and token cost

Scope: how this repo feeds Claude context, and what to change so that sessions
spend fewer tokens without following fewer rules. Not a phase literature scan.

**Reading level.** Claude Code docs: read as primary text (fetched 2026-09-22,
`code.claude.com/docs/en/{memory,hooks,skills,context-window}`). Papers: arXiv
is blocked from the machine this was written on, so each paper below is known
**from its abstract or search-result summaries only**. None was read as primary
text. Treat the paper rows as leads, not findings.

## 1. What Claude Code does (docs, primary)

| # | fact | consequence here |
|---|---|---|
| D1 | Root `CLAUDE.md` loads every session. Target **< 200 lines**; "longer files consume more context and reduce adherence." | Ours is 81 lines. Fine. |
| D2 | `@path` imports **load at launch**; they organise, they do not save tokens. | Never "split" `CLAUDE.md` by import to save tokens. |
| D3 | `.claude/rules/*.md` with `paths:` frontmatter load **only when Claude reads a matching file**. Rules without `paths:` load at launch. | Rules that apply only under `claims/` or `data/` can leave `CLAUDE.md` (plan A4). |
| D4 | A subdirectory's `CLAUDE.md` loads on demand, when Claude reads a file in that directory. | A `p10_cluster_function/CLAUDE.md` could carry the thread's rules (A4). |
| D5 | Skills: only name + description sit in context (listing capped at 1,536 chars per skill); the body loads on use and stays. `disable-model-invocation: true` keeps a skill out of context entirely until `/name`. Keep `SKILL.md` < 500 lines. | The Stop protocol is a procedure, and the docs say procedures belong in skills, not `CLAUDE.md` (A4). |
| D6 | "Claude treats [CLAUDE.md] as context, not enforced configuration. To block an action regardless of what Claude decides, use a PreToolUse hook." | "Never read PROJECT.md whole" and "never `git add` under `data/`" should be hooks (A1 does the first). |
| D7 | PreToolUse hook: `tool_input` for Read has `file_path`, `offset`, `limit`; deny via `hookSpecificOutput.permissionDecision: "deny"` + `permissionDecisionReason`, which **Claude sees**. | Implemented in `scripts/hooks/guard_large_read.py`. |
| D8 | SessionStart stdout is added to context. A SessionStart hook matching `compact` re-runs after compaction. Ours has no matcher, so it matches every source. | `STATE.md` is re-injected after `/compact` and on resume. Keep it capped (it is: 150 lines, lint). |
| D9 | After compaction: root `CLAUDE.md` re-read; path rules / nested `CLAUDE.md` reload only when a matching file is read again; up to 5 recent files re-read, any over 5k tokens only as a path. | Rules that must survive compaction stay in root `CLAUDE.md` or `STATE.md`. |
| D10 | "File reads dominate context usage"; delegate research-heavy reading to a subagent, whose reads stay out of the main window. | The index (A2) cuts read size. A cheap-model search subagent (A5) cuts where reads land. |
| D11 | Auto memory `MEMORY.md`: first 200 lines / 25 KB load each session; machine-local, **not shared across machines or cloud environments**. | Don't rely on it here; `STATE.md` in git is the cross-machine memory. |

## 2. What the literature suggests (abstracts only — see reading level)

| paper | claim (from abstract/summary) | bearing |
|---|---|---|
| Gloaguen et al., *Evaluating AGENTS.md*, arXiv 2602.11988 (ETH, Feb 2026) | Context files did not generally raise task success and raised inference cost > 20 %. Instructions in them *are* followed. Repository overviews did not help. Useful for non-standard practices. | Keep `CLAUDE.md` to non-standard rules (ours is). Don't add repo overviews. `INDEX.md`/`README.md` stay on-demand. |
| McMillan, *Instruction Adherence in Coding Agent Configuration Files*, arXiv 2605.10039 (May 2026; 1,650 Claude Code sessions) | File size (25–500 lines), instruction position, and single vs nested architecture: **no detectable effect** after correction. Largest effect is within-session: ~5.6 % lower odds of compliance per additional function generated. | Structure matters less than session length. That argues for short sessions per unit of work and for hooks over prose on rules that must hold late in a session. |
| Shepard & Albrecht, *Probe-and-Refine Tuning of Repository Guidance*, arXiv 2606.20512 (Jun 2026) | Iteratively refined guidance: 33.0 % vs 28.3 % (static) vs 25.5 % (none) on SWE-bench Verified with one open model. The gain is from reaching the right file. Guidance tuned for one model hurt another. | Guidance that says *where to look* is what helps: the index and the `STATE.md` map. Don't over-tune the text to one model. |
| Jaroslawicz et al., *How Many Instructions Can LLMs Follow at Once?* (IFScale), arXiv 2507.11538 (Jul 2025) | Adherence falls as the instruction count rises (10 → 500). Best models 68 % at 500. Bias towards earlier instructions. | Our always-loaded rules number in the dozens, far below where this bites. Put the rules that matter most first. |
| Hong, Troynikov, Huber, *Context Rot*, Chroma technical report (Jul 2025) | Across 18 models, performance becomes less reliable as input length grows, even on simple tasks. | Supports cutting read size (A1, A2) over cutting rule count. |

## 3. What this changes in the plan

1. **Where the tokens go.** The fixed per-session load is small (`CLAUDE.md` + `STATE.md` ≈ 2.5k tokens). The cost is task-time reads of 25–480 KB documents. So A1 (bounded reads) and A2 (section index) come first. Trimming `CLAUDE.md` further is not the lever.
2. **Enforce with hooks, not prose** (D6, lessons 1 and 9): A1 now. A `data/` add-guard is next.
3. **Move procedures out of the always-loaded file** (D3–D5): the Stop protocol becomes a skill, and claims/registry rules become a path rule. That is A4. It is justified by D1/D5 more than by the adherence evidence, which (McMillan) found no size effect.
4. **Short sessions per unit of work** (McMillan's within-session decay) matches the existing Stop protocol. No new rule.
5. **Not supported by anything read:** adding overviews or architecture summaries to `CLAUDE.md` (Gloaguen).

## 4. Open

- Measure, don't estimate: the old box's `~/.claude/projects/*.jsonl` transcripts record per-turn usage and every `Read`. A script over them would show which files actually cost the most.
- Read the three 2026 papers as primary text from a machine that can reach arXiv before citing them anywhere outside this note.
