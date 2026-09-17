---
status: not-started
created: 2026-09-17
---

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

## Non-goals

- Not blocking any ticket — the workaround (shared directory, disjoint files, incremental
  commits) is sufficient for now.
