---
name: tdd-ticket
description: Use this skill when building a ticket (from this repo's TICKET-*.md files, or any spec with a "tests first" section) so that tests are always run and shown genuinely failing before any implementation lands, and the ticket's done/not-done status is reported unambiguously at the end. Trigger on requests like "build ticket N", "write the tests for ticket N", "follow TDD for X", or "show me the failures first".
---

# TDD ticket workflow

This skill exists because it's easy to *say* "red, green, refactor" while actually writing the answer and the test in the same motion — which produces a green run that looks like proof but isn't. The point of this skill is to make the red state real, visible, and impossible to skip.

## The failure mode this prevents

On this project, a ticket ("segmentation injection seam") was investigated first via scratch/throwaway scripts, the fix was discovered, and then the formal test file was written *with the fix already inlined in the test body* (e.g. `pipeline.training = True` written directly inside the test function). The suite was run once, reported "3 passed," and presented as done. That was true but backwards: the tests never failed for the right reason at any point the user could see, because the fix and the test were authored together. A retroactive demonstration (stripping the fix from a throwaway copy, rerunning, watching it fail) proved the tests weren't vacuous — but proving it after the fact, folded into a "done" report, is not the same as showing the failure before building, which is what was actually asked for.

Don't do this. The rules below exist to make that structurally hard to do by accident.

## Core rule: the fix must live behind an interface the test calls, never inside the test

If a ticket's "fix" is a calling convention, a flag, a small helper, a context manager — anything at all — it still gets factored into a small piece of application code (even a two-line function) that the test imports and calls. The test must be written and run *before* that helper exists or before it's wired in, so that:

- the test fails with `NameError`/`AttributeError`/`ImportError`/`assert False` for exactly the reason you expect (the behavior doesn't exist yet), not for an unrelated scaffolding bug
- only then do you write the helper and watch the same, unmodified test turn green

If you catch yourself about to write `pipeline.training = True` (or equivalent) directly inside a test function to make an assertion pass, stop — pull it out into a named function/context manager in the actual source tree first, even if it ends up being trivially small. A ticket whose "fix" is genuinely just "call the existing API correctly" still gets this treatment: write a thin wrapper that encodes the correct call, so the test is exercising something that can be absent (red) and present (green), not just narrating a fact you already know.

## Workflow

Follow these phases in order. Do not skip ahead to implementation because you're confident you already know the fix — confidence is exactly the condition under which this discipline matters most.

### 1. Read the ticket's test list

Extract the concrete test names/behaviors it specifies (most tickets in this repo have a "Tests first" section with numbered tests). Don't improvise beyond it without checking with the user first — the ticket's test list is itself a spec that was reviewed.

### 2. Write the tests against the *not-yet-existing* behavior

Write full, real test bodies — real fixtures, real assertions, no mocks standing in for the thing actually being tested unless the ticket's own scope says to mock it. Where the ticket requires a fix/helper/interface, write the test calling it as if it already existed in application code, then actually create it as an empty/absent/default-off stub (or don't create it at all if that's more natural) so the import or call fails.

If building the test fixtures themselves requires nontrivial scaffolding (e.g., constructing a real object offline without network access), it's fine to iterate on that scaffolding privately and hit unrelated errors while doing so — a `TypeError` from a wrong constructor argument is not the red state this skill cares about, it's just a bug in your test code. Fix scaffolding bugs until the suite actually runs. What matters is the state immediately before that point: once the suite runs cleanly enough to reach the real assertions, capture that run's output before writing the fix.

### 3. Run the suite and show the failure, unedited

Run the tests with verbose output (`pytest -v`, or the project's equivalent). Show the actual failing output to the user — not a paraphrase, not "this would fail because X." If a test unexpectedly passes at this stage, stop and figure out why before continuing; an unexpected green on a not-yet-implemented behavior almost always means the test isn't testing what you think it's testing.

This is the checkpoint the user is paying attention to. Do not narrate past it — post the red output, then proceed.

### 4. Implement the minimum to turn it green

Write the smallest real change (in source, not in the test file) that makes the failing assertions pass for the right reason. Do not touch the test file's assertions to make this easier unless you determined in step 3 that a test was actually wrong (in which case say so explicitly and explain why, rather than silently loosening it).

### 5. Run the suite again and show the pass

Same command, same verbose output, shown unedited. If you had to fix scaffolding along the way and previously showed unrelated errors, it's fine — just be clear about which failures were scaffolding bugs (fixed silently) versus the one real red-to-green transition (shown explicitly, before and after).

### 6. Report ticket status explicitly

End every ticket-building turn with an unambiguous status block, not just prose. Map each test back to the ticket's acceptance criteria so "done" is a checkable claim, not a vibe:

```
TICKET STATUS: <ticket id/title>

Acceptance criteria:
  [x] <criterion 1>  <- proven by test_foo, test_bar
  [x] <criterion 2>  <- proven by test_baz
  [ ] <criterion 3>  <- NOT YET COVERED (no test written / manual step pending: <what and why>)

Tests: N passed, 0 failed (pytest -v output above)

TICKET COMPLETE | TICKET INCOMPLETE: <one line on what's missing, if anything>
```

If any acceptance criterion has no test proving it (e.g. it requires real network/credentials you don't have, or it's a documentation requirement), say so explicitly in that block rather than letting "N passed" imply full coverage. "All tests pass" and "ticket complete" are different claims — don't let one silently stand in for the other.

## Anti-patterns to catch yourself doing

- Writing the fix and the test in the same edit, then running once and calling it TDD.
- Retroactively stripping a fix from a copy to "prove" a test after already reporting success — the proof has to come before the success report, not patch it after the fact.
- Treating a scaffolding error (wrong mock signature, missing fixture) as if it were the meaningful red state, then feeling like the red/green cycle happened when really you just debugged your test harness.
- Summarizing pytest output instead of showing it, especially at the two checkpoints (first red, final green).
- Reporting "tests pass" as equivalent to "ticket complete" without checking each acceptance criterion has a test behind it.
