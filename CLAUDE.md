# Communication rules

These rules govern how you talk to the user while working in this repo. The goal: the user stays in control of the work as it happens, and builds a real understanding of the codebase alongside you — not just a pile of finished diffs.

- **Narrate as you go, not just at the end.** Before starting each meaningful step (editing a file, running a command, making a design choice), say in one short sentence what you're about to do and why. Don't batch it all into a single summary at the end of the turn.
- **Keep it short and human.** Default to plain, simple language a person can read at a glance — no jargon, no code-speak, no internal reasoning dumps. A couple of sentences is normally enough.
- **Technical/dense mode is opt-in only.** Only switch to denser, technical language (e.g. drafting a message for another agent, a machine-readable payload, etc.) when the user explicitly says the message is for that purpose. Otherwise always default to the plain-language style above, even for technical work.
- **Explain new territory, skip repeats.** The first time a session touches a module, pattern, or part of the architecture, give a brief (1-2 sentence) orientation: what it is and how it fits into the rest of the codebase. Don't repeat this explanation on later visits to the same area in the same session — keep those terse.

# Ticket-driven workflow (Obsidian vault)

The `Obsidian-Diarisation/` vault is the source of truth for in-progress work and finished documentation. It's built to be read by agents, so keep it current rather than letting knowledge live only in chat history.

- **No new feature starts without a ticket.** Before writing code for a new feature or non-trivial change, create a ticket note first: `Obsidian-Diarisation/Tickets/Open/<short-title>.md`. All tickets live in the vault, not the repo root — no floating `TICKET-*.md` files.
- **Ticket frontmatter.** Each ticket note starts with frontmatter tracking status:
  ```yaml
  ---
  status: not-started   # not-started | in-progress | blocked | done
  created: <date>
  ---
  ```
  Update `status` as work progresses (e.g. to `in-progress` when you start, `blocked` if stuck).
- **Not-yet-completed tickets live in `Tickets/Open/`.** They stay there, with `status` kept current, for the entire duration of the work.
- **Log implementation notes in the ticket as you build.** Keep an "Implementation Notes" section in the ticket and append to it while the work is happening — not just at the end. Capture things a doc reader would need: decisions made and why, alternatives rejected and why, non-obvious gotchas, key files/functions touched, and how it hooks into existing code. Treat it as informal running commentary, not polished prose — it just needs to be there when it's time to write the real doc.
- **On completion, convert the ticket into documentation — don't just close it.** Rewrite the note from a task-shaped ticket into reference documentation that describes the feature as it now exists (what it does, how it fits into the architecture, how to use it), drawing on the Implementation Notes rather than reconstructing that context from memory. The final doc should read as reference material, not a log of the work done to build it. Set `status: done`, then move the rewritten note to `Obsidian-Diarisation/Docs/<short-title>.md`. If the original ticket also exists as a loose file in the repo (e.g. a root-level `TICKET-*.md` predating the vault), delete that file once the vault doc covers it — the repo should never hold a stale duplicate of something the vault now documents properly.
- Before starting new feature work, check `Obsidian-Diarisation/Docs/` and `Obsidian-Diarisation/Tickets/Open/` for relevant prior context rather than re-deriving it from scratch.
