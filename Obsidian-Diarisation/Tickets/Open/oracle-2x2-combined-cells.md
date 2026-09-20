---
status: not-started
created: 2026-09-20
---

# Oracle 2x2: combined segmentation + assignment cells

## Goal

Measure the two unmeasured cells of the oracle 2x2 on AMI IHM test (16 meetings,
community-1, collar=0, skip_overlap=False): oracle segmentation combined with oracle
assignment, at both oracle scopes.

Current state of the grid:

|                             | baseline seg | oracle seg   |
| --------------------------- | ------------ | ------------ |
| baseline assignment         | 17.05%       | 3.96%        |
| oracle assignment, degraded | 16.58%       | UNMEASURED   |
| oracle assignment, all      | 16.08%       | UNMEASURED   |

## Why this matters

Under oracle segmentation DER is 3.96%, and 1057.51 s of the 1215 s total error (87%) is
confusion. These two runs measure how much of that confusion is reachable by
post-clustering relabelling once segmentation is perfect.

The existing 0.47 pt and 0.97 pt assignment ceilings were measured under *baseline*
segmentation, where segmentation error dominates and masks whatever headroom relabelling
has. The result of these two runs decides whether the project's remaining effort goes to
a learned assignment classifier or to a clustering hyperparameter sweep.

## Prediction

Recorded before running. Do not tune toward it — the runs are the measurement, the
prediction is only there to make a surprise visible.

Baseline oracle-seg components: MD 16.43 s, FA 141.13 s, Conf 1057.51 s, over ~30714 s
reference speech.

If relabelling were perfect, confusion goes to zero and DER lands at the floor:

    floor = (16.43 + 141.13) / 30714 = 0.51%

It will not reach the floor. The oracle strategy leaves a pair unchanged when its dominant
reference speaker has no mapped cluster, and 9 of 16 meetings over-cluster under oracle
segmentation (exact speaker-count matches fall from 12/16 to 7/16, over-counting in all 9
mismatches).

- **Predicted range: 0.5% to 1.5% DER for `all_pairs`**, implying confusion of roughly
  100–350 s (see criterion 4).
- `overlap_degraded` is *expected* to be higher than `all_pairs`, since it refines a
  narrower set of pairs — but held as an expectation rather than a requirement, because the
  overlap-degraded population is recomputed over different tracks under oracle
  segmentation. See the note under the acceptance criteria.

### Interpretation, fixed in advance

- **Lands near 1%** — the assignment budget under clean segmentation is roughly 3 pt, not
  0.47 pt. The baseline-segmentation ceilings understated the target because segmentation
  error was masking it. A learned assignment classifier is chasing something real.
- **Lands near 3.9%** — relabelling is close to useless even when handed perfect input.
  That is a decisive negative result for the classifier direction, and points at the
  clustering hyperparameter sweep instead.

Anything outside 0.5%–1.5% is a finding to report, not a bug to fix. Stop and report.

## Scope check: is new experimental code needed?

The brief's assumption was that `--run-condition oracle_segmentation` and
`--refinement-strategy oracle` are orthogonal and already compose, so no new experimental
code is required. **That assumption does not hold as written.** No new *scientific* code is
needed — `OracleSegmentation` and the oracle refinement strategy are both implemented and
genuinely independent of each other — but the plumbing that selects them is not orthogonal,
and three separate places conflate "condition" with "segmentation source".

Each is a wiring fix, not new experimental logic. They are listed as work items below.

## Work items

### 1. Schema: new condition value

Add `oracle_segmentation_assignment` to `VALID_CONDITIONS` in
[run_manifest.py](harness/run_manifest.py#L35). Do not overload an existing value — T5's
cross-condition table keys on `condition` alone, so reusing `oracle_segmentation` would
silently merge two different experiments into one row.

Update the matching `--run-condition` choices list in
[run_harness.py](run_harness.py#L176) and the usage line in
[run_experiment.sh](run_experiment.sh#L27).

### 2. Segmentation routing (the real trap)

[run_harness.py:225](run_harness.py#L225) selects the segmentation source with an exact
equality test:

```python
if args.run_condition == "oracle_segmentation":
    ...OracleSegmentation(...)
else:
    segmentation_source = BaselineSegmentation()
```

A new condition value falls through to the `else` branch. The run would complete, write a
valid manifest claiming oracle segmentation, and quietly score **baseline** segmentation.
That failure is silent — the DER would land somewhere in the 16% range and look like a
plausible experimental result rather than a bug.

Widen the test to cover both oracle-segmentation conditions, and widen the companion
`--oracle-rttm` required-argument check on the following line with it. Add a regression
test asserting that `oracle_segmentation_assignment` yields an `OracleSegmentation`
instance.

### 3. Aggregation: scope suppression

[`_effective_scope`](harness/aggregate_runs.py#L52) blanks `oracle_scope` for every
condition except `oracle_assignment`, on the reasoning that the `--oracle-scope` default is
recorded even when nothing chose it. That reasoning is correct but the condition list is
now incomplete: the two new runs differ *only* by scope, so both would render as identical
rows in the cross-condition table. Extend the check to treat the combined condition as one
where scope is meaningful.

### 4. Aggregation: budget deltas — reference is `oracle_segmentation`, not baseline

`_BUDGET_CONDITIONS` has no entry for the combined condition, and the obvious
implementation is wrong. Do not add it to that dict unchanged.

[`_baseline_der`](harness/aggregate_runs.py#L128) finds the row where
`condition == "baseline"` — the 17.05% baseline-segmentation run. So a `baseline - combined`
delta computes `17.05 - ~1.0 = ~16 pt` and reads as a budget for the combined intervention.
That figure **double-counts**: 13.09 pt of it is the segmentation budget already reported on
its own line. The table would show three budget lines summing to far more than 17.05, and
the reader has to reverse-engineer why.

The meaningful delta for these cells is against **oracle segmentation**:

    oracle_seg (3.96) - combined (~1.0) = ~3 pt

That is the assignment budget *given clean segmentation*, which is the entire point of the
runs. It is also directly comparable to the 0.47 / 0.97 pt ceilings measured under baseline
segmentation, which is the comparison the deliverable turns on.

So: `_build_delta_lines` needs a per-condition reference, not one shared baseline. The
combined condition's reference row is `oracle_segmentation`; the existing two keep
`baseline`. Label the line so the reference is visible in the table itself — e.g.
`Assignment budget under oracle segmentation (oracle_segmentation - combined)` — rather than
leaving the reader to infer it. If the `oracle_segmentation` row is absent, emit the same
kind of explicit "cannot be computed" line the baseline path already emits rather than
silently falling back to baseline.

### 5. Instrumentation: how many pairs each scope actually refined

`make_oracle_strategy` ([refinement.py:150](harness/refinement.py#L150)) currently counts
nothing — it returns only the relabelled array. Add counters and log them per run.

This is the number that explains any surprise in criteria 2 or 3, and it costs nothing to
collect. The loop already has four distinct exit paths worth separating, because they mean
different things:

1. **out of scope** — `overlap_degraded` only, pair not overlap-degraded
2. **no reference overlap** — `_dominant_reference_speaker` returned `None`
3. **unmapped speaker** — dominant reference speaker maps to no produced cluster
   (this is the one predicted to keep DER off the 0.51% floor, so it is the most
   diagnostically valuable of the four)
4. **relabelled** — actually changed

Report the totals for both runs, and — since the interesting comparison is against
baseline segmentation — for the existing `oracle_segmentation`-era counts too if they can
be obtained cheaply. A scope population that changes size dramatically between segmentation
conditions is itself the explanation for a criterion-3 surprise.

Keep this out of the manifest schema unless it falls out naturally; a logged number the
report can quote is sufficient.

### 6. Provenance: record `--oracle-rttm`

[[run-manifest-provenance]] is still `status: not-started`, so Gap 1 is still open and this
ticket must close it rather than inherit the fix. Record the oracle RTTM path in
`run_config`. Additive field only — existing manifests must stay readable.

If the provenance ticket lands first, drop this item and note that in the Implementation
Notes rather than doing it twice.

## Cache

`OracleSegmentation.id` is `oracle-v2`. The runner's cache key is
`pipeline_config_id | segmentation_source.id | refinement_id | uri`
([runner.py:37](harness/runner.py#L37)), and `refinement_id` becomes
`oracle:all_pairs` / `oracle:overlap_degraded` for these runs rather than `identity`.
The key therefore already distinguishes them from the existing `oracle_segmentation` run.

**Do not clear `.harness_cache`.** Confirm the distinction holds before launching rather
than assuming it — a collision here would silently serve the previous run's hypotheses.

## Runs

Both on AMI IHM test, full 16 meetings, from a clean committed tree, both with
`--counts-toward-results`:

- **Run 1** — oracle segmentation + oracle assignment, scope `all_pairs`
- **Run 2** — oracle segmentation + oracle assignment, scope `overlap_degraded`

## Acceptance criteria

1. Both runs complete over all 16 meetings and write valid manifests under the new
   condition value.
2. DER for `all_pairs` falls within 0.5%–1.5%. Outside that range: stop and report, do not
   adjust.
3. Missed detection and false alarm are approximately unchanged from the
   `oracle_segmentation` run (16.43 s, 141.13 s). Relabelling must not move them — if it
   does, the refinement is touching something it should not.
4. Confusion for `all_pairs` falls to roughly **100–350 s**, from 1057.51 s.

   Derived from the DER prediction and the floor arithmetic, so it cross-checks criterion 2
   rather than restating it:

       DER 1.0%  ->  total error ~307 s  ->  confusion ~150 s
       DER 1.5%  ->  total error ~461 s  ->  confusion ~300 s

   If DER lands in range but confusion does not, the error has moved somewhere it should
   not have and criterion 3 is the place to look.
5. Pair counts (work item 5) are reported for both runs, broken down by the four exit
   paths.
6. `runs/comparison.csv` contains both runs and the cross-condition table renders without
   error, with the two scopes shown as distinct rows and the combined delta computed
   against `oracle_segmentation`.

### Expectation, not a criterion

`overlap_degraded` DER is **expected** to be higher than `all_pairs` DER. It is not a
pass/fail gate, because it may legitimately fail to hold under oracle segmentation.

`_is_overlap_degraded` ([refinement.py:137](harness/refinement.py#L137)) classifies a pair
by whether its support is predominantly coincident with another speaker — and oracle
segmentation changed the tracks that classification runs over. On the test meeting,
segments fell from 1000 to 796 and `region_t_minus_d` collapsed from 1394 s to 5.8 s. The
set of pairs classified as overlap-degraded is therefore computed over a different
population and may be a different size and shape entirely. If most residual confusion
happens to sit in overlap-degraded pairs, the two scopes could land very close, or in
principle invert.

The 16.58 vs 16.08 ordering was measured under baseline segmentation and does not transfer
automatically.

If the ordering does not hold: report the scope population sizes for both segmentation
conditions and treat it as a finding about where confusion concentrates under clean
segmentation. It is not a bug and nothing should be adjusted to restore the expected
ordering.

## Deliverable

The completed 2x2 table, the confusion delta for each cell, and a one-paragraph reading of
which of the two interpretations above the result supports.

The interpretation *is* the deliverable — do not append recommended next steps.

## Implementation Notes

_Append while the work happens, not at the end._

- **Pre-flight audit (2026-09-20, before any code change).** Verified the brief's
  orthogonality assumption against the source rather than assuming it. Result: the two
  features are independent, but three call sites treat `condition` as a proxy for
  segmentation source. The routing one (item 2) is the dangerous one because it fails
  silently into a plausible-looking wrong number; the two aggregation ones degrade the
  table but would not corrupt a result. Item 3 was not in the original brief and was found
  by reading `_effective_scope`.
- Confirmed `refinement_id` already varies by scope, so the cache key separates these runs
  from the existing `oracle_segmentation` run without touching `OracleSegmentation.id`.
  No cache clear needed.
- Confirmed [[run-manifest-provenance]] is still open, so the `--oracle-rttm` recording
  falls to this ticket.
- **Ticket review (2026-09-20, pre-launch).** Three corrections, all accepted:
  - Item 4 was left open ("decide whether a delta is meaningful"), which would have let the
    obvious-but-wrong implementation through: `_baseline_der` resolves to the 17.05%
    baseline-segmentation run, so a `baseline - combined` delta double-counts the 13.09 pt
    segmentation budget. Now specified explicitly — the combined condition's reference is
    `oracle_segmentation`, giving the ~3 pt assignment budget under clean segmentation,
    which is the quantity the deliverable actually turns on.
  - Scope ordering demoted from acceptance criterion to expectation. The 16.58 vs 16.08
    ordering was measured under baseline segmentation; `_is_overlap_degraded` reclassifies
    over tracks that oracle segmentation substantially changed (1000 -> 796 segments,
    `region_t_minus_d` 1394 s -> 5.8 s), so an inversion is a finding about where confusion
    concentrates, not a failure.
  - Criterion 5 ("falls substantially") was unfalsifiable; replaced with a 100–350 s range
    derived from the DER prediction, which makes it an arithmetic cross-check on
    criterion 2.
  - Added work item 5 (pair-count instrumentation) as a consequence. `make_oracle_strategy`
    counts nothing today. Separated four exit paths rather than one total — "unmapped
    speaker" is the path predicted to hold DER off the 0.51% floor, so collapsing it into a
    single skip count would discard the most diagnostic number.

## See also

- [[Oracle Segmentation Findings Report]] — the 3.96% run and its error breakdown.
- [[Oracle Assignment Strategy]] — the refinement strategy and its scopes.
- [[Oracle Ceiling Metrics]] — the baseline-segmentation ceilings this run reframes.
- [[Cross-Condition Deltas]] — T5 table and budget-delta logic touched by items 3 and 4.
- [[Run Manifest]] — manifest schema and controlled vocabularies.
- [[run-manifest-provenance]] — open ticket owning the provenance gaps generally.
