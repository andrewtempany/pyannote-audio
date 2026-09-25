---
status: done
created: 2026-09-24
aliases: [unreachable-pair-boundary-distance]
---

# Unreachable Pair Boundary Distance

Reference for what the oracle refinement strategy's unassignable pairs actually are, and what
they do and do not bound.

**Short version:** the `no_reference_overlap` counter conflated two unrelated things. 96.3% of
what it reported was empty tensor slots, not pairs sitting in silence. Genuinely
silence-dwelling pairs number **six**, totalling **0.2024 s** across 16 meetings. The planning
conclusion once drawn from that counter does not hold, and the 1.68 pt gap to the assignment
floor remains unexplained.

---

## The three populations

Every `(chunk, local_speaker)` slot in `segmentations.data` falls into one of three groups.
Measured over all 16 AMI IHM test meetings under oracle segmentation:

| Population | Count | Share |
| --- | --- | --- |
| Has ≥1 active frame and overlaps a reference speaker | 69,682 | 55.9% |
| **Empty support — no active frames at all** | **54,888** | **44.1%** |
| **Active frames but no reference overlap ("genuine silence")** | **6** | **0.005%** |
| Total reconstructed slots | 124,576 | |

### Empty support is a property of tensor layout

`segmentations.data` has shape `(num_chunks, num_frames, local_num_speakers)`. The
`local_num_speakers` axis is a **fixed-width slot count**, so every chunk containing fewer
active speakers than there are slots contributes empty pairs.

This says nothing about segmentation quality. These pairs were never candidates for assignment:
there is no support to relabel, and no speaker to relabel it toward. They are structural
padding.

### Genuine silence is negligible

Six pairs, 0.2024 s in total. Oracle segmentation does **not** produce meaningful tracks over
regions the reference scores as silence. The premise that motivated this analysis was false.

---

## Why the counter conflated them

`_dominant_reference_speaker` ([refinement.py](../../harness/refinement.py)) returned `None` on
two quite different conditions:

- `len(support) == 0` — empty support
- the support overlaps no reference speaker

The strategy loop funnelled both into `no_reference_overlap`. A count of 56,998 therefore read
as "56,998 pairs sit in silence" when it meant "54,888 slots are empty padding and 6 pairs sit
in silence".

**This has since been fixed.** The two dispositions are now separate paths, distinguished by a
module-level `object()` sentinel rather than a string — reference labels come from RTTMs the
harness does not control, so a string sentinel could collide with a real speaker name.

**Expect an asymmetry in the new counters.** `_is_overlap_degraded` also returns `False` on
empty support, and the scope check runs before the dominant-speaker call, so under
`overlap_degraded` scope empty-support pairs are absorbed into `out_of_scope` and never reach
the new branch. `empty_support` therefore reads **0 for `overlap_degraded` and roughly 44% for
`all_pairs`**. That is correct behaviour, pinned deliberately by test, not a bug.

---

## The grid-mismatch inconsistency: real, confirmed, and worth 0.2 s

All six genuine-silence pairs are identical in shape:

| uri | chunk | distance | in frame steps | first active frame | span |
| --- | --- | --- | --- | --- | --- |
| ES2004a | 1 | 0.0069 s | 0.41 | 590/593 | 10.949–10.983 |
| ES2004a | 384 | 0.0069 s | 0.41 | 590/593 | 393.949–393.983 |
| IS1009d | 1070 | 0.0069 s | 0.41 | 590/593 | 1079.949–1079.983 |
| TS3003c | 1840 | 0.0069 s | 0.41 | 590/593 | 1849.949–1849.983 |
| TS3003d | 1599 | 0.0069 s | 0.41 | 590/593 | 1608.949–1608.983 |
| TS3003d | 1952 | 0.0069 s | 0.41 | 590/593 | 1961.949–1961.983 |

**The cause.** `_pair_timeline` reconstructs a pair's support by subdividing the chunk
uniformly: `frame_duration = chunk.duration / num_frames`. `oracle_segmentation()` rasterises
onto the model's `receptive_field`, whose frames overlap and are wider than their step. The two
grids therefore drift apart across a chunk.

**Measured against prediction.** Predicted drift at the chunk end was −0.006875 s = 0.4074
steps; measured 0.0069 s = 0.41 steps, at frame 590 of 593, in all six cases. Prediction and
measurement agree to two decimal places and the frame index is exactly where the mechanism
says drift peaks. The drift is negative, so the reconstructed support sits slightly *earlier*
than the rasterised speech, which is why all six classify as `turn_start`.

**Deliberately not fixed.** Total cost is 0.2024 s. The refinement path is load-bearing for
four recorded conditions and changing it would break comparability for no measurable gain.
Documented here so it is understood rather than rediscovered.

Measured constants: frame step **0.016875 s** (read from
`pipeline._segmentation.model.receptive_field`), 593 frames per chunk, 7,493 reference turns
across the 16 files.

---

## What this does not bound

**The 1.68 pt gap to the assignment floor is unexplained.** Under oracle segmentation, every
pair with active support had a reachable correct answer, and oracle relabelling still left
1.68 points on the table. The unreachable-pairs explanation is dead.

The leading remaining candidate is a **granularity** limit rather than a labelling one: the
oracle assigns one cluster per `(chunk, local_speaker)` pair, but a pair's support can span a
speaker change within the chunk, and whole-pair relabelling cannot split it. **This is
untested.**

**The 141.13 s of false alarm under oracle segmentation remains unapportioned.** The
quantisation prior (≈15.8 s expected from turn-end rounding, upper bound 63.2 s) could not be
tested by this route: that false alarm lands on pairs which *do* overlap reference speech, so
it never reaches the population this analysis examined.

---

## Caveats on the numbers

**Pair total reconciliation.** 124,576 reconstructed slots against 126,925 recorded, a gap of
2,349 (~147 chunks per file). Cause: file duration was derived from the reference extent,
whereas the scored run used audio duration. AMI files run roughly 147 s past the last reference
turn, and those trailing chunks are pure silence, so they would be empty supports too — which
is why 54,888 undershoots 56,998 by a similar order. The gap reinforces the finding rather than
threatening it. `Audio.get_duration` would close it at the cost of touching the audio files.

**Two denominators are in play and both are correct.** 44.1% is empty supports against the
reconstructed 124,576; 96.3% is empty supports against the recorded 56,998
`no_reference_overlap` figure. A bare "44.1%" next to a table reading "44.9%" looks like an
error and is not.

**`hard_clusters` was not recoverable.** The manifests hold only `run_config` and `summary`, and
the strategy skips `cluster < 0` pairs before `_pair_timeline` is reached, so the exact 56,998
population could not be reconstructed. The analysis ran over every slot with active frames — a
superset. The geometry is unaffected, since boundary distance depends only on the segmentation
grid and the reference, neither of which involves clustering. Only the counts inherit the
caveat.

---

## Method, for anyone repeating it

Analysis-only, no GPU, no scored run. `harness/refinement.py` untouched; `_pair_timeline`
imported read-only.

Supports were re-derived rather than loaded, because `segmentations.data` is not persisted in
the run artifacts. Calling `oracle_segmentation()` directly on the reference with the
pipeline's `window` and `frames` — the same three lines as `OracleSegmentation.populate()` —
reproduces the tensor exactly and needs the pipeline object only for its receptive field, not a
forward pass.

Reading the receptive field needs `PYANNOTE_SKIP_DEPENDENCY_CHECK=1` (the installed tree is a
dev build; `harness/config.py` already sets this), and `Pipeline.from_pretrained` takes
`token=`, not `use_auth_token=`, in this version.

---

## Why this is in the record

The analysis set out to explain a number and instead found the number meant something else.
That is the fourth instance in this project of instrumentation measuring other than what it
claimed, and the pattern is consistent: **a counter is only as good as the distinction it
draws.** Keeping four separate exit paths is what made `unmapped_speaker = 0` knowable; failing
to separate a fifth is what made `no_reference_overlap` misleading.

---

## See also

- [[Oracle 2x2 Combined Cells]] — §4, corrected on the strength of this analysis.
- [[Oracle Assignment Strategy]] — the exit paths, now five.
- [[Oracle Segmentation Provider]] — `populate()` and the `oracle_segmentation()` rasterisation.
- [[Oracle Segmentation Findings Report]] — the 3.96% run and the 141.13 s false alarm.
- [[Post-Clustering Refinement Hook]] — `_pair_timeline` and `cluster < 0` handling.
