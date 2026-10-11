---
status: in-progress
created: 2026-09-24
---

# Boundary-distance analysis of unreachable assignment pairs

## Goal

Explain where the `no_reference_overlap` pairs come from in the combined oracle runs, and
report what share of them boundary quantisation accounts for.

## The observation

In the combined oracle runs (oracle segmentation + oracle assignment, AMI IHM test, 16
meetings — see [[Oracle 2x2 Combined Cells]]) the refinement strategy's exit-path
instrumentation recorded:

| Path | Run 1 (`all_pairs`) |
| --- | --- |
| `no_reference_overlap` | 56,998 of 126,925 (44.9%) |
| `unmapped_speaker` | 0 |

A `no_reference_overlap` pair is one whose support intersects no reference speaker at all, so
there is nothing to relabel it toward. No labelling strategy can reach these pairs by
construction — which is why §4 of [[Oracle 2x2 Combined Cells]] treats them as the real
blocker on the 1.68 pt gap to the floor, in place of the `unmapped_speaker` mechanism that
was predicted and fired zero times.

**The puzzle:** these runs used oracle segmentation, derived from the reference RTTM via
`oracle_segmentation()`. If the segmentation came from the reference, how does it produce
tracks over regions the reference scores as silence?

## Hypothesis

Frame quantisation. `oracle_segmentation()` rasterises continuous reference turns onto the
model's receptive-field grid, so turn edges move to frame edges and short inter-turn gaps are
absorbed. Where the rasterised edge lands outside the true turn, the segmentation says speech
and the reference says silence.

Two pieces of corroborating evidence already exist:

- False alarm rose to **141.13 s** under oracle segmentation. Perfect segmentation should
  drive FA toward zero; boundary displacement produces exactly this.
- Segment count fell **1000 → 796** on the single-meeting check, consistent with adjacent
  turns merging across absorbed gaps.

**Expect a partial explanation only.** A displacement of at most one frame per boundary is
small, and 44.9% of pairs is large. Anticipate a residual requiring a second mechanism. That
residual is a finding in its own right, not a failure.

### The rasteriser rounds to nearest — it does not dilate symmetrically

Stated here because it changes what the histogram should look like, and because the intuitive
version of this hypothesis ("a partially covered frame becomes fully active, so every turn
grows at both ends") is **not** what the code does.

`oracle_segmentation()` delegates to `Annotation.discretize(..., resolution=frames)`, which
crops the frame grid with `mode="center"`. That selects the frames whose **centres** are
closest to the turn's start and end — rounding to nearest, not expanding to cover. Measured
directly against the community-1 frame step, sweeping a turn's sub-frame offset:

| offset into frame | start edge | end edge |
| --- | --- | --- |
| 0.0 | −0.00001 s | +0.01693 s |
| 0.2 | −0.00340 s | +0.01354 s |
| 0.4 | −0.00679 s | +0.01015 s |
| 0.6 | −0.01018 s | +0.00676 s |
| 0.8 | −0.01357 s | +0.00337 s |

(negative = eroded into the turn, positive = extended past it)

Two consequences for the analysis:

1. **The start edge erodes; only the end edge extends.** A rasterised turn is displaced later
   at both ends rather than grown at both ends. So false alarm at turn *starts* should be near
   zero, and the FA this mechanism can produce is roughly one frame per turn *end*, not two
   per turn. Output 5's start/end split is therefore a direct test of this, not just a
   breakdown: **a symmetric start/end split falsifies the mechanism** even if the distances
   themselves are small.
2. **The per-boundary budget is bounded above by one frame step**, and averages about half of
   one. Use that when apportioning the 141.13 s — multiply by turn *ends*, not by boundaries.

The table above was produced with an approximate step (0.016949 s) to establish the *sign* and
shape of the effect. The real step is **0.016875 s** (see the prior below); the displacements
scale by 0.9956 and the asymmetry is unchanged.

### Quantitative prior: quantisation can explain at most a small share

**Computed before the measurement, so the result cannot be read as a post-hoc rationalisation.**

Measured values (community-1, read from `pipeline._segmentation.model.receptive_field`):

| Quantity | Value |
| --- | --- |
| receptive field step | **0.016875 s** (16.875 ms) |
| receptive field duration | **0.0619375 s** (61.94 ms — 3.67× the step, i.e. overlapping) |
| chunk window | 10.0 s duration, 1.0 s step |
| frames per chunk | **593** |
| reference turns in `test.rttm`, 16 files | **7,493** |

The end-edge displacement is uniform on (−step/2, +step/2), and only its positive part creates
false alarm, so the expected FA contribution per turn end is `step/8` = **2.11 ms**.

| | Seconds | Share of 141.13 s |
| --- | --- | --- |
| Expected FA from end-edge rounding | **15.8 s** | **11.2%** |
| Absolute upper bound (step/2 every end) | 63.2 s | 44.8% |
| Observed false alarm | 141.13 s | 100% |

**So boundary rounding cannot account for the observed false alarm even in the worst case, and
on expectation explains about an eighth of it.** The prior is therefore: quantisation should
account for roughly 15–16 s of the 141.13 s, and a correspondingly minor share of the 44.9% of
pairs. If the measurement lands near that, a second mechanism dominates and is the real
finding. Treat a result far above ~63 s as evidence the measurement itself is wrong, since
that exceeds what the grid can physically produce.

### A second candidate mechanism — weaker than it first appeared

`_pair_timeline` ([refinement.py:72](../../../harness/refinement.py)) reconstructs a pair's
support by subdividing the chunk uniformly: `frame_duration = chunk_segment.duration /
num_frames`, with frame `f` spanning `[chunk.start + f·d, chunk.start + (f+1)·d)`.
`oracle_segmentation()` rasterises onto the receptive field, whose frames overlap (duration
3.67× step), so the two grids are not the same construction.

**Its predicted magnitude was computed and it is small.** With 593 frames in a 10 s chunk the
uniform pitch is 0.0168634 s against a receptive-field step of 0.016875 s:

- slope = `d − step` = **−0.0116 ms per frame**
- total drift across a whole chunk = **−6.9 ms**, i.e. **0.41 of one frame step**

So the grids nearly coincide, and the divergence never reaches even one frame width before the
chunk ends. This mechanism cannot place a support far from a reference boundary on its own.
It stays in the ticket as a falsifiable check rather than as the leading hypothesis, and the
prediction is specific: **distance should vary linearly with frame index at a slope of
−0.0116 ms/frame, reaching −6.9 ms at frame 593.** A measured ramp of materially different
slope is a different mechanism wearing the same shape, not confirmation of this one.

**Record frame index alongside each distance** so this is testable in the first pass. Note the
sign: the drift is *negative*, so reconstructed supports sit slightly **earlier** than the
rasterised speech, growing toward the chunk end.

### Both mechanisms may be present — report the joint distribution

Outputs 1 and 5 alone cannot separate them. Add a small 2D table of **distance binned against
frame index** (distance bins as in output 1; frame index in, say, 10 buckets across 0–592):

- a spike at low distance that is **flat across frame index** → quantisation;
- a **ramp with frame index** → grid mismatch (check the slope against −0.0116 ms/frame);
- **both visible** → both present, and each needs its share of the duration reported separately.

No extra computation — it reuses the per-pair distances and frame indices already collected.

## What to measure

For each `no_reference_overlap` pair, compute the signed distance from the pair's support to
the nearest reference speech boundary (any speaker, any direction). Report the distribution.

Required outputs:

1. **Histogram of distances**, bin width at or below the frame duration.
2. **The model's actual receptive field / frame duration** for community-1. Read it from
   `pipeline._segmentation.model.receptive_field` — do not assume a value.
3. **Fraction of unreachable pairs within 1, 2, 5 and 10 frame widths** of a boundary.
4. **Total duration covered by unreachable pairs**, and how much of that falls within one
   frame width of a boundary.
5. **The same figures split** by whether the nearest boundary is a turn start, a turn end, or
   an inter-turn gap shorter than one frame.
6. **The residual** — unreachable pairs further than 10 frame widths from any boundary. Count,
   total duration, and a sample of 10 with URI and time span for manual inspection.

## Method constraints

- **No GPU runs.** This is an analysis over data already computed.
- Source the pair supports the same way `make_oracle_strategy` does — `_pair_timeline` over
  active frames of `segmentations.data` — and the reference from the same RTTM the run used:
  `C:/Users/Andrew/Code/AMI-diarization-setup/harness-data/IHM/test.rttm`, read from
  `run_config.oracle_rttm` in `runs/20260920T082056Z-2aada13d.json`. Note this path is
  **outside the repo**, and manifests are flat JSON files in `runs/`, not directories.
- **`segmentations.data` is not persisted in the run artifacts.** The manifests record config
  and scores, not the segmentation tensor, so the supports must be re-derived. Rebuild them by
  calling `oracle_segmentation()` directly on the reference with the pipeline's `window` and
  `frames` — the same three lines as `OracleSegmentation.populate()`
  ([segmentation.py:98-102](../../../harness/segmentation.py)). This needs the pipeline object
  for its `_segmentation.step`/`.duration`/`.model.receptive_field`, but **not** a forward
  pass, so it costs no GPU and reproduces the tensor exactly. **Do not launch a scored run.**
- **Do not modify `harness/refinement.py`'s behaviour.** If instrumentation is needed, add it
  behind a flag or in a separate analysis script. The refinement path is now load-bearing for
  four recorded conditions and must not change.
- **`hard_clusters` is not recoverable, and this bounds the claim.** The manifest holds only
  `run_config` and `summary`; the clustering output was not persisted. The strategy skips
  pairs with `cluster < 0` *before* reaching `_pair_timeline`, so the exact 56,998 population
  cannot be reconstructed from the reference alone.

  Analyse instead **every (chunk, local_speaker) pair with active frames**, which is a
  superset, and state the reconstructed total beside the recorded 126,925. If the two are
  close, the masked pairs are negligible and the distances carry over. **If they diverge
  materially, say so and report the distribution as describing the superset** — do not quietly
  present superset figures as if they were the 56,998. The boundary-distance question is a
  property of the segmentation grid versus the reference, and neither depends on clustering,
  so the geometry is unaffected either way; it is only the pair *counts* in outputs 3 and 6
  that inherit this caveat.
- Work over the **oracle-segmentation condition primarily**. If it is cheap, repeat for the
  baseline-segmentation oracle-assignment runs and report both — a difference between them is
  informative.

## Interpretation, fixed in advance

Recorded before the analysis so a surprise is visible, in the same spirit as §3 of
[[Oracle 2x2 Combined Cells]].

- **Distances cluster sharply within one to two frame widths** → quantisation confirmed as the
  *shape*. Report the fraction of the 141.13 s attributable to it, and reconcile against the
  ~15.8 s prior. This makes the post-clustering ceiling an *architectural* bound rather than a
  methodological one — but note it can only be the whole story if the FA share also comes out
  near 100%, which the prior says is impossible.
- **Distances are scattered through silent regions, far from boundaries** → quantisation
  rejected. The cause lies elsewhere, most likely in how the pipeline forms tracks from the
  segmentation array. `_pair_timeline` grid mismatch is now a weak candidate (bounded at 6.9 ms
  per chunk), so if the residual is large, look past it. Report that plainly and name the next
  place to look.
- **Mixed, with a spike at boundaries and a substantial tail** → report both components
  separately with their durations. **This is the expected outcome, and the prior says the tail
  should carry roughly seven eighths of the false alarm.** The interesting number is what the
  tail is, not that it exists.

## Deliverable

The distribution, the frame duration used, the four fractions from output 3, the distance ×
frame-index table, and a one-paragraph reading naming which interpretation the data supports,
what share of the 44.9% is explained, and how the measured quantisation share compares to the
~15.8 s prior.

**Do not append recommended next steps. The interpretation is the deliverable.**

## Acceptance criteria

1. Frame duration is read from the model, not assumed, and stated in the output.
2. Distance distribution reported with bin width at or below one frame.
3. The residual (output 6) is reported with samples, whether or not the hypothesis holds. A
   confirmed hypothesis does not excuse skipping the unexplained remainder.
4. No scored pipeline run was launched.
5. `harness/refinement.py`'s behaviour is unchanged for all four recorded conditions.
6. The start/end split (output 5) is reported and read against the rounding asymmetry: the
   mechanism predicts turn ends dominate and turn starts contribute almost nothing. A
   roughly symmetric split is a falsification and must be reported as one.
7. The reconstructed pair total is stated beside the recorded 126,925, so any divergence from
   the `hard_clusters` population is visible rather than assumed away.
8. The measured quantisation share of the 141.13 s is reconciled against the ~15.8 s prior
   (upper bound 63.2 s). A measured share above the upper bound means the measurement is
   wrong, not that the bound is.
9. The distance × frame-index joint table is reported, and the grid-mismatch slope is checked
   against the predicted −0.0116 ms/frame rather than merely tested for correlation.

## Implementation Notes

**Run date:** 2026-09-24. Script: `scratchpad/boundary_distance.py` (analysis-only, outside the
repo; `harness/refinement.py` untouched — `_pair_timeline` imported read-only).

### The headline: the question in the ticket was based on a misreading of the counter

`no_reference_overlap` does **not** mean "a pair sitting in a silent region". 96.3% of it is
pairs with **no active frames at all**.

`_dominant_reference_speaker` ([refinement.py:119](../../../harness/refinement.py)) returns
`None` on two quite different conditions — `len(support) == 0` (empty support) and a support
that overlaps no speaker — and the strategy funnels both into the same counter. Verified
directly: an empty `Timeline` returns `None`, as does a support in pure silence.

Reconstructed over all 16 files:

| | Count | Share |
| --- | --- | --- |
| Total (chunk, speaker) slots | 124,576 | — |
| With ≥1 active frame | 69,688 | 55.9% |
| **Empty support (no active frame)** | **54,888** | **44.1%** |
| **Genuinely in silence** | **6** | **0.005%** |

Against the recorded 56,998: empty supports are **96.3%** of it, true silence is **0.01%**.
The `local_num_speakers` dimension of `segmentations.data` is a fixed-width slot count, so
every chunk with fewer active speakers than slots contributes empty pairs. That is a property
of the tensor layout, not of the segmentation's quality.

**The premise "oracle segmentation produces tracks over silent regions" is therefore false.**
It produces almost none — six, totalling 0.2024 s across 16 meetings.

### Reconciliation of the pair totals

Mine 124,576 vs recorded 126,925, a gap of 2,349 (~147 chunks/file). Cause: I derived file
duration from the reference extent, whereas the run used the audio duration. AMI files run
~147 s past the last reference turn; those trailing chunks are pure silence and would be
empty supports too, which is why my 54,888 undershoots 56,998 by a similar order. The gap
reinforces the finding rather than threatening it. Using `Audio.get_duration` would close it
at the cost of touching the audio files.

### Grid mismatch: confirmed exactly, but it explains six pairs

All six genuine-silence pairs are identical in shape:

| uri | chunk | dist | in steps | first active frame | span |
| --- | --- | --- | --- | --- | --- |
| ES2004a | 1 | 0.0069 s | 0.41 | 590/593 | 10.949–10.983 |
| ES2004a | 384 | 0.0069 s | 0.41 | 590/593 | 393.949–393.983 |
| IS1009d | 1070 | 0.0069 s | 0.41 | 590/593 | 1079.949–1079.983 |
| TS3003c | 1840 | 0.0069 s | 0.41 | 590/593 | 1849.949–1849.983 |
| TS3003d | 1599 | 0.0069 s | 0.41 | 590/593 | 1608.949–1608.983 |
| TS3003d | 1952 | 0.0069 s | 0.41 | 590/593 | 1961.949–1961.983 |

Predicted drift was −0.006875 s = **0.4074 steps** at the chunk end; measured **0.0069 s =
0.41 steps**, at frame **590 of 593** in every case. Prediction and measurement agree to two
decimal places, and the frame index is exactly where the mechanism says the drift peaks. This
is the `_pair_timeline` uniform-subdivision vs receptive-field-step divergence, confirmed —
and it is a real (if tiny) implementation inconsistency in the harness, not a pipeline
property. Its total cost is 0.2024 s.

All six are `turn_start` by the nearest-boundary classification: the support drifts *earlier*
(negative drift), landing just before a turn starts.

### Consequences for the ticket as written

- Outputs 1, 3, 4 and 5 are degenerate on a population of six (all in the first bin, all
  `turn_start`, 6/6 within one frame width). Reported, but the distribution is not meaningful.
- Output 6's residual is **empty**: no pair is further than 10 frame widths from a boundary.
- The quantisation prior (~15.8 s of FA) could not be tested this way. It concerned false
  alarm from turn-end rounding; that FA is real but it lands on pairs that *do* overlap
  reference speech, so it never reaches `no_reference_overlap`. The 141.13 s remains
  unapportioned by this analysis.
- Acceptance criterion 6's falsification test does not apply: with six pairs all at turn
  starts, the start/end split is not evidence about the rounding asymmetry.

## On completion

Rewrite this note as reference documentation describing what the unreachable pairs *are* and
what bounds they impose, drawing on the Implementation Notes rather than reconstructing the
context from memory. Set `status: done` and move to
`Obsidian-Diarisation/Docs/Unreachable Pair Boundary Distance.md`.

## See also

- [[Oracle 2x2 Combined Cells]] — §4 records the 56,998 pairs and why they, not
  `unmapped_speaker`, are the blocker.
- [[Oracle Segmentation Findings Report]] — the 3.96% oracle-segmentation run and its error
  breakdown, source of the 141.13 s false alarm this analysis apportions.
- [[Oracle Assignment Strategy]] — the refinement strategy, its scopes, and the four exit
  paths whose counts this ticket explains.
- [[Oracle Segmentation Provider]] — `OracleSegmentation.populate()` and the
  `oracle_segmentation()` rasterisation under investigation.
