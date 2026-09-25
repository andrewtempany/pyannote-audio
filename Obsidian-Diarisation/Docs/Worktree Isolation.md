---
status: done
created: 2026-09-17
aliases: [worktree-isolation, worktree-isolation-redirect]
---

# Worktree Isolation

Reference for why isolated-worktree agent launches do not work in this repo, and what to do
instead.

**Short version:** `Agent` with `isolation: "worktree"` fails here, and it is **not** a
corruption problem. Claude Code creates the worktree at `<repo>/.claude/worktrees/agent-<id>`,
which is *inside* the main checkout's working tree. The isolation guard rejects any worktree
that has another checkout discoverable above it, so a nested path is refused by design.
Recreating the worktree cannot fix it. The sanctioned approach is to run agents in the main
working directory on disjoint files, committing incrementally.

---

## What worktree isolation is meant to do

Passing `isolation: "worktree"` to the `Agent` tool is supposed to give a subagent its own git
worktree — a separate checkout on its own branch, sharing the main repo's object database — so
several agents can edit the same repo concurrently without stepping on each other's files. The
worktree is cleaned up automatically if the agent leaves it unchanged.

## Why it is refused in this repo

Launching an isolated agent fails immediately with:

```
Refusing to use <repo>\.claude\worktrees\agent-<id> as an isolation worktree: git resolves
its working tree to <repo>/.claude/worktrees/agent-<id> (a core.worktree redirect, or a
checkout discovered above it), so commands run there would write outside the worktree.
Remove the redirect, restore the worktree's own .git, or recreate the worktree, then retry.
```

The guard tests two distinct conditions and reports them in one message. Only the second one
applies here: **"or a checkout discovered above it."** The worktree root sits under
`<repo>/.claude/worktrees/`, so git repository discovery walking upward from that path finds
the parent repository at `<repo>`. The guard cannot distinguish that from a genuine
`core.worktree` redirect pointing outside the worktree, so it refuses.

### The error message's own advice does not apply

The message's first two suggestions — "remove the redirect, restore the worktree's own `.git`"
— presuppose corruption. There is none. Verified on the two worktrees that existed here:

| Check | Result |
| --- | --- |
| Worktree's `.git` file | Correct pointer: `gitdir: C:/Users/Andrew/Code/pyannote-audio/.git/worktrees/agent-<id>` |
| `.git/worktrees/agent-<id>/gitdir` | Points correctly back at the worktree's own `.git`, casing matched |
| `git config --get core.worktree` | Exits 1 in the main repo *and* in each worktree — no redirect exists |
| `config.worktree` files | None exist at all |
| `git rev-parse --show-toplevel` (inside worktree) | Returns the worktree's own path, not the parent |
| `git rev-parse --git-common-dir` | Returns the shared `<repo>/.git`, as expected for a linked worktree |
| `git worktree prune -v` | Finds nothing |

The third suggestion, "recreate the worktree", also cannot help: recreating it at the same
nested location reproduces the rejection exactly. A fix would require moving the worktree root
outside the repo, which means changing Claude Code's own tooling or settings — out of scope for
this repo.

This is worth stating plainly because the symptom reads like corruption and invites a long
hunt for a Windows path-casing bug or a broken `.git` pointer. There is no such bug here.

## The sanctioned workaround

Run concurrent agents **directly in the main working directory**, with these conventions:

- **Disjoint files.** Give each agent an explicit, non-overlapping set of files to touch. The
  isolation is by convention, not enforced by git, so the file split has to be stated in the
  agent's instructions.
- **Incremental commits.** Have each agent commit its own work as it goes, so concurrent edits
  do not pile up into one ambiguous working tree.
- **Never `git stash`.** With shared-directory agents, a bare `git stash` will sweep up other
  agents' in-progress edits.

This has been sufficient in practice — it was used for the concurrent
[[Oracle Segmentation Provider]] and [[Oracle Assignment Strategy]] work, which was the
occasion that first surfaced the error.

## Cleaning up leftover worktrees

Stale `.claude/worktrees/agent-*` checkouts accumulate from failed isolation launches. Remove
them with git's own commands, never `rm -rf` — `git worktree remove` also cleans the
corresponding admin entry under `.git/worktrees/`, which a manual delete would orphan.

```bash
# 1. See what exists
git worktree list

# 2. Safety check, per worktree — all three must come back empty
cd .claude/worktrees/agent-<id>
git status --porcelain              # uncommitted work
git status --porcelain --ignored=matching   # untracked/ignored content worth keeping
git stash list

# 3. Confirm the branch holds nothing unique
git log --oneline main..worktree-agent-<id>
git branch -a --contains worktree-agent-<id>

# 4. Remove (add --force only if locked or dirty and you have decided to discard)
git worktree remove .claude/worktrees/agent-<id>

# 5. Delete the orphaned branch with -d, never -D
git branch -d worktree-agent-<id>

# 6. Verify
git worktree list        # only the main checkout
git worktree prune -v    # silent
```

Notes on the steps:

- **Use `git branch -d`, not `-D`.** Safe delete refuses a branch with unmerged commits, so it
  acts as a last-line guard against discarding real work.
- **Check for a lock first.** Worktrees may be registered as `locked`, in which case
  `git worktree unlock <path>` is needed before removal. Check by looking for
  `.git/worktrees/agent-<id>/locked`; `git worktree unlock` on an unlocked worktree exits with
  `fatal: '<path>' is not locked`, which is harmless.
- **A freshly failed worktree holds nothing.** Its `logs/HEAD` will show only the initial
  checkout and a no-op `reset: moving to HEAD`, its branch reflog only
  `branch: Created from origin/main`, and its `refs/` subdir will be empty. Worth confirming
  rather than assuming — removal is hard to reverse.
- Once the last linked worktree is removed, `.git/worktrees/` disappears entirely. That is
  normal, not a sign anything went wrong.

## Current state

As of 2026-09-25 the repo has a single worktree (the main checkout), `.claude/worktrees/` is
empty, `.git/worktrees/` does not exist, and the two branches
`worktree-agent-a654406a2579f47ca` and `worktree-agent-ae222f9403780bc80` have been deleted.
Both had sat at `4894adf2`, identical to `main`'s tip, with no unique commits.
