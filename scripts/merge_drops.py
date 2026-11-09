#!/usr/bin/env python3
"""List lines a branch added that a merge into it no longer has (LESSONS.md 8, 2026-11-09).

    python scripts/merge_drops.py [MERGE]      # default HEAD; exit 1 if anything was dropped

MERGE's first parent is the branch, its second the commit merged in (e.g. main). For each
file the branch changed since the merge base, every non-blank line the branch added that is
in neither the base nor MERGE is printed. A line moved or reworded on purpose shows up too:
read the list, do not just count it. Lines are compared as a set, so a duplicate line
dropped while one copy survives is not caught.
"""
import subprocess
import sys


def git(*a, ok=False):
    r = subprocess.run(["git", *a], capture_output=True, text=True)
    if r.returncode and not ok:
        sys.exit(f"git {' '.join(a)}: {r.stderr.strip()}")
    return r.stdout if r.returncode == 0 else None


def lines(rev, path):
    out = git("show", f"{rev}:{path}", ok=True)
    return None if out is None else set(out.split("\n"))


def main(merge="HEAD"):
    parents = git("rev-list", "--parents", "-n1", merge).split()[1:]
    if len(parents) != 2:
        print(f"{merge} is not a two-parent merge; nothing to check")
        return 0
    branch, other = parents
    base = git("merge-base", branch, other).strip()
    lost_any = 0
    for path in git("diff", "--name-only", base, branch).split("\n"):
        if not path:
            continue
        b = lines(branch, path)
        if b is None:
            continue
        m = lines(merge, path)
        before = lines(base, path) or set()
        added = {l for l in b - before if l.strip()}
        lost = sorted(added if m is None else added - m)
        if lost:
            lost_any += len(lost)
            print(f"== {path}: {len(lost)} added line(s) not in {merge}" + (" (file gone)" if m is None else ""))
            for l in lost:
                print("  " + (l if len(l) <= 160 else l[:157] + "..."))
    return 1 if lost_any else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
