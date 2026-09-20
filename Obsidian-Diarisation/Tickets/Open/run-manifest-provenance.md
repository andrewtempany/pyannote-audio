---
status: not-started
created: 2026-09-18
---

# Run manifest provenance gaps

## Goal

Make a run manifest sufficient to reconstruct what a run actually measured, without
consulting chat history or guessing from timestamps.

Both gaps below were exposed by the investigation in [[oracle-segmentation-seam-no-op]].
Split out deliberately: they touch the manifest writer rather than the seam, so bundling
them would stop that ticket closing on unrelated work.

## Gap 1: `--oracle-rttm` is not recorded

The reference RTTM defines what an oracle condition *is*. It is currently untracked in the
manifest, so an oracle run's headline number cannot be tied to the ground truth that
produced it.

Worth noting the scoring reference and the oracle reference are currently the same file
(`harness-data/IHM/test.rttm`). That is correct and is what makes the ~4% DER prediction in
[[oracle-segmentation-seam-no-op]] valid — but it is an assumption nobody recorded, and it
would be invisible if it ever stopped holding.

Record the path, and consider a content hash alongside it.

## Gap 2: `git_commit` is unreliable

Runs `1db2ba34` and `c93fb9f9` record commit `3d3d9fa6`, but their notes say "post
cache-key fix" — and `a4a3101a` (the cache-key fix) is *newer* than `3d3d9fa6` in the log.
Most likely the fix was uncommitted in the working tree at run time, so the recorded commit
describes code that was not what ran.

A bare commit hash is therefore not a reliable provenance record. Add a dirty-tree flag
(e.g. `git_dirty: true`) alongside it so the discrepancy is visible rather than inferred
months later. Consider also recording a diff hash when the tree is dirty.

## Acceptance criteria

1. A new run's manifest records the oracle RTTM path (and hash, if adopted).
2. A run made from a dirty working tree is flagged as such in its manifest.
3. Existing manifests remain readable — additive fields only, no breaking schema change.

## See also

- [[Run Manifest]] — current manifest schema and writer.
- [[oracle-segmentation-seam-no-op]] — the investigation that surfaced both gaps.
