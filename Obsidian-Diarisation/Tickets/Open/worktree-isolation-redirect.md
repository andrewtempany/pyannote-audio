---
status: done
created: 2026-09-17
---

> **Superseded.** This ticket is complete and has been rewritten as reference documentation at
> [[Worktree Isolation]]. This file is awaiting deletion — the agent that completed the work
> could not delete it. Read the Docs page instead; it is the current source of truth.

# Worktree isolation redirect error

## Goal

Isolated-worktree agent launches (`Agent` tool with `isolation: "worktree"`) currently fail
in this repo and need to fall back to running agents directly in the main working
directory instead.

## Symptom

Launching an agent with worktree isolation fails immediately with:

```
Refusing to use <repo>\.claude\worktrees\agent-<id> as an isolation worktree: git resolves
its working tree to <repo>/.claude/worktrees/agent-<id> (a core.worktree redirect, or a
checkout discovered above it), so commands run there would write outside the worktree.
Remove the redirect, restore the worktree's own .git, or recreate the worktree, then retry.
```

Reproduced when launching two concurrent agents (for [[Oracle Segmentation Provider]]
and [[Oracle Assignment Strategy]]) — both failed identically.

## What's been checked

- `.git/config` in the main repo has no explicit `core.worktree` override.
- `git worktree list` shows the two worktrees as legitimately registered and `locked`,
  not stale/orphaned entries — `git worktree prune -v` found nothing to remove.
- Workaround used at the time: ran both agents directly in the main working directory
  instead of isolated worktrees, relying on them touching disjoint files.

## Suspected cause (unconfirmed)

Likely a Windows-specific quirk in how `.claude/worktrees/agent-*` checkouts were created
(possibly path casing, or the worktree's `.git` file pointing somewhere unexpected) rather
than a problem with the repo's actual commit history.

## Scope (when picked up)

- Reproduce in isolation (a single worktree launch, not concurrent) to confirm it's not
  a race between the two concurrent launches.
- Inspect the `.git` file inside `.claude/worktrees/agent-*` (should be a `gitdir:` pointer
  file) and compare against what a freshly created worktree looks like.
- Fix or document the correct recreation steps so future concurrent-agent work
  (e.g. the rest of the oracle/ceiling batch) can use real isolation instead of the
  disjoint-files-in-shared-directory workaround.

## Coordinator finding, 2026-09-25 — the suspected cause above is a FALSE PREMISE

Checked before delegating. The "Suspected cause" section (Windows path casing, or the
worktree's `.git` pointing somewhere unexpected) is wrong, and so is the error message's own
first suggestion ("Remove the redirect, restore the worktree's own `.git`").

Evidence, from the two worktrees currently on disk:

- Each worktree's `.git` file is a correct `gitdir:` pointer file
  (`gitdir: C:/Users/Andrew/Code/pyannote-audio/.git/worktrees/agent-<id>`).
- `.git/worktrees/agent-<id>/gitdir` points correctly back at the worktree's own `.git`.
- `git config --get core.worktree` inside a worktree exits 1 — **no redirect exists**, in
  either the worktree config or the main `.git/config`.
- `git rev-parse --show-toplevel` run inside a worktree correctly returns the worktree's own
  path, not the parent repo.
- `git worktree list` shows both as legitimately registered; `git worktree prune -v` finds
  nothing.

Nothing is corrupt. The actual cause is **location**: the worktrees are created at
`<repo>/.claude/worktrees/agent-<id>`, i.e. *inside the main checkout's working tree*. The
tool's guard is the clause "**or a checkout discovered above it**" — git discovery walking up
from that path finds the parent repository, so the guard cannot distinguish this from a real
redirect and refuses. It is a nested-path rejection, not a corruption.

Consequence: "restore the worktree's own `.git`" and "recreate the worktree" cannot fix this,
because recreating it in the same nested location reproduces it exactly. The fix has to move
the worktree root outside the repo, or be documented as a standing limitation with the
shared-directory workaround as the sanctioned approach.

## Acceptance criteria (added by the coordinator — the ticket had none)

Each must be able to fail on the current state.

1. The two stale worktrees under `.claude/worktrees` are removed via `git worktree remove`
   (they are `locked`, so the lock must be released first), and `git worktree list` afterwards
   shows only the main checkout. Fails today: it currently lists three entries.
2. The two now-orphaned branches `worktree-agent-a654406a2579f47ca` and
   `worktree-agent-ae222f9403780bc80` are reported — deleted only if they hold no commits
   unique to them. Check before deleting; report the commit status either way.
3. The false premise above is corrected in the ticket body rather than left to mislead the
   next reader, and the nested-location cause is stated with the evidence.
4. The pre-existing test-failure set is unchanged (this ticket should touch no Python at all;
   if it does, that is a finding).

**Do not** attempt to relocate the worktree root by editing Claude Code's own settings or
tooling — that is outside this repo's scope. Document the limitation instead.

## Non-goals

- Not blocking any ticket — the workaround (shared directory, disjoint files, incremental
  commits) is sufficient for now.

## Implementation Notes

### Independent verification of the coordinator's diagnosis (2026-09-25)

Re-checked every evidence point before touching anything. All confirmed:

- `.claude/worktrees/agent-a654406a2579f47ca/.git` and `.../agent-ae222f9403780bc80/.git` are
  both plain pointer files reading
  `gitdir: C:/Users/Andrew/Code/pyannote-audio/.git/worktrees/agent-<id>`. Correct form.
- `.git/worktrees/agent-<id>/gitdir` points back at
  `C:/Users/Andrew/Code/pyannote-audio/.claude/worktrees/agent-<id>/.git`. Correct, and the
  casing matches on both sides — no Windows casing mismatch.
- `git config --get core.worktree` exits 1 in the main repo and in both worktrees.
  `grep -ri worktree .git/config .git/worktrees/*/config.worktree` returns nothing (no
  `config.worktree` files exist at all). No redirect anywhere.
- `git rev-parse --show-toplevel` inside each worktree returns the worktree's own path.
  `--git-dir` returns its own admin dir, `--git-common-dir` returns the shared `.git`. Exactly
  right for a linked worktree.
- `git worktree prune -v --dry-run` prints nothing.

So: nothing is corrupt, and the nested-location explanation is the only one left standing —
the worktrees live under `<repo>/.claude/worktrees/`, inside the main checkout's own working
tree, so git discovery walking up from there finds the parent repo. That matches the guard's
"or a checkout discovered above it" clause.

One correction to the ticket's own history: the worktrees were **not locked** at the time of
this work. `.git/worktrees/agent-<id>/locked` did not exist for either, and
`git worktree unlock <path>` returned `fatal: '<path>' is not locked`. Either the lock was
released earlier or the original "locked" reading was mistaken. No unlock step was needed.

### Pre-removal safety check

Both were destroyed only after confirming there was nothing in them:

- `git status --porcelain` inside each: zero lines. No modified, staged, or untracked files.
- `git status --porcelain --ignored=matching`: one line each, `!! .claude/settings.local.json`
  — gitignored local settings, not work product.
- `git stash list` inside each: empty.
- Both branches at `4894adf2`, identical to `main`'s tip. `git log --oneline main..<branch>`
  and `oracle-experiment..<branch>` both empty. `git merge-base --is-ancestor 4894adf2 main`
  succeeds.
- `git branch -a --contains <branch>` lists `main`, `oracle-experiment`, `scoring-harness`,
  `origin/main` and friends — the tip is reachable from everywhere.
- Worktree reflogs (`.git/worktrees/agent-<id>/logs/HEAD`) contain only the initial checkout
  plus a no-op `reset: moving to HEAD`. Branch reflogs contain only
  `branch: Created from origin/main`. `refs/` subdirs are empty. Nothing was ever committed in
  either worktree.

Conclusion: the worktrees were freshly created, never used, and held zero unique history.

### Removal

Used git's own commands rather than `rm -rf`, so `.git/worktrees/` admin entries are cleaned
up in the same step:

```
git worktree remove .claude/worktrees/agent-a654406a2579f47ca
git worktree remove .claude/worktrees/agent-ae222f9403780bc80
git branch -d worktree-agent-a654406a2579f47ca
git branch -d worktree-agent-ae222f9403780bc80
```

`git branch -d` (safe delete) rather than `-D` — it refuses if the branch holds unmerged
commits, so it doubles as a final guard. Both deleted without complaint, confirming
merged-ness.

### Scope discipline

No Python touched. This ticket only removed worktrees/branches and edited vault markdown, so
the pre-existing 28-failure baseline cannot have moved. Verified via `git status` showing no
`.py` files changed beyond the pre-existing in-progress edits to `harness/refinement.py` and
`harness/segmentation.py`, which this ticket did not author and left alone.
