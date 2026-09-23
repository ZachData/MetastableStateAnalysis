---
name: challenge-pr
description: Adversarial review of a pull request's intent and design choices, in a fresh context with none of the author's conversation. Reconstructs what the PR is for, proposes alternatives before reading the diff, then argues whether the chosen design was the best one, and posts the verdict as a PR comment. Use after opening any PR (CLAUDE.md Stop step 9), or when the user asks to challenge a PR.
argument-hint: "<pr-number>"
arguments: [pr]
context: fork
agent: general-purpose
background: false
effort: high
allowed-tools: Bash(gh *) Bash(git *) Bash(ls *) Bash(grep *) Bash(python *) Bash(python3 *) Read Grep Glob Write
---

# Challenge PR #$pr

You are the **second pair of eyes** on this PR. You did not write it and you
have none of the author's reasoning, only what the PR says about itself and
what the repository shows. That is deliberate: the job is to catch what the
author could not see because they already believed the design was right.

CodeRabbit already reviews line by line. **Do not duplicate it.** Your subject
is **intent and choice**: was this the right thing to build, built the right
way, and does it do what it claims?

## The PR, as it describes itself

!`gh pr view $pr --json number,title,author,headRefName,baseRefName,body,additions,deletions,commits --template '#{{.number}} {{.title}}  ({{.headRefName}} -> {{.baseRefName}}, +{{.additions}}/-{{.deletions}}, {{len .commits}} commits)
{{.body}}'`

## Files touched

!`gh pr diff $pr --name-only`

## Procedure (in this order; the order is the point)

**1. Reconstruct the intent, before reading the diff.** From the description
above, plus `STATE.md` and whatever doc the PR points at (a handoff, a
`status-N.md`, a `PROJECT.md` §3.x via `docs/index/`), write down in two or
three sentences: what problem this PR solves, why now, and what "done" would
look like. If the description does not let you say this, that is finding #1.

**2. Design it yourself, still before the diff.** Name 2–3 genuinely different
ways to solve that problem, including "don't build it" or "a smaller thing".
One line each on what each costs and what it risks. This is what makes the
review adversarial rather than a read-through: once you have seen the author's
design you will anchor on it.

**3. Now read the diff** (`gh pr diff $pr`; for a large one, read file by file)
and the code around it. Answer:
- Does it do what the description claims? List every gap between claim and
  implementation, with `file:line`.
- Which of your alternatives did it pick, or is it a fourth? Steelman it
  first, then attack it: under what input, state or future use does it fail
  or cost more than an alternative would have?
- What does it silently assume? Check each assumption against the tree (`ls`,
  `grep`, manifests, `gh run list`). Docs in this repo have been wrong about
  whether things exist, are populated and are green, so do not trust a doc's
  claim without looking.

**4. Check this project's standing rules** (`CLAUDE.md`, `LESSONS.md`). The
ones most often broken:
- **Refuse rather than degrade.** Does any new path return zeros, `{}`, or a
  default where it should raise?
- **Name the input.** Does a recorded result state its battery hash, key list
  and env?
- **Registry.** Is `claims/registry.json` touched? If so, was it before the
  scoring data was seen?
- **Statistics.** A new null, merger, threshold or e-value: is it valid under
  the dependence these units actually have? Is a threshold labelled
  `placed` / `calibrated`?
- **Write once.** Is a number now copied into two files?
- **Tests.** Do they test the claim, or only that the code runs?

**5. Verdict.** One of: **accept** · **accept with changes** · **rethink**.

## Output

Post exactly one comment on the PR with `gh pr comment $pr --body-file <file>`
(write the file in your scratchpad or `/tmp`), in this shape:

```
## Adversarial review (fresh context, /challenge-pr)

**Verdict:** accept | accept with changes | rethink — one sentence why.

**What this PR is for** (reconstructed before reading the diff): …

**Alternatives considered before reading the diff:**
1. … 2. … 3. …
**Chosen design vs these:** …

**Findings** (most serious first)
1. **[wrong | risk | different-defensible-choice | unclear-intent]** — claim, evidence `file:line`, what would break, suggested change. Confidence: high/medium/low.
…

**What held up:** the parts you tried to break and could not, and how you tried.

**For the author to answer:** questions whose answers would change the verdict.
```

Label each finding honestly. **`wrong`** means you can show it breaks.
**`different-defensible-choice`** means you would have done it otherwise but
theirs is reasonable. Mixing the two up is the failure mode of adversarial
review. Aim for a few findings you can stand behind, not a long list. Explain
the reasoning behind each finding, not only the conclusion: the user reads
these to learn the trade-offs.

**Do not** edit files, push, approve, request changes or merge. Your only
write is the one comment. Return the comment's URL and the verdict line.
