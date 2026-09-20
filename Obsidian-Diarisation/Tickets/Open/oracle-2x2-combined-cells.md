---
status: in-progress
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

### The completed 2x2 (DER, AMI IHM test, 16 meetings, collar=0, skip_overlap=False)

|                             | baseline seg | oracle seg |
| --------------------------- | ------------ | ---------- |
| baseline assignment         | 17.05%       | 3.96%      |
| oracle assignment, degraded | 16.58%       | **2.75%**  |
| oracle assignment, all      | 16.08%       | **2.19%**  |

### Confusion per cell (seconds)

|                             | baseline seg | oracle seg |
| --------------------------- | ------------ | ---------- |
| baseline assignment         | 1212.2       | 1057.5     |
| oracle assignment, degraded | n/a          | **686.1**  |
| oracle assignment, all      | n/a          | **516.0**  |

The `n/a` cells are real: those two runs predate the DER component breakdown in the summary,
so their confusion was never recorded. Not re-run, since the 2x2 turns on DER and the
confusion deltas that matter are against `oracle_segmentation`.

Confusion delta for each new cell, against the 1057.51 s under oracle segmentation:

- `all_pairs`: 516.0 s, **-541.5 s (51.2% of confusion removed)**
- `overlap_degraded`: 686.1 s, **-371.4 s (35.1% removed)**

Missed detection (16.427 s) and false alarm (141.127 s) are byte-identical to the
`oracle_segmentation` run in both new cells.

### Reading

The assignment budget under clean segmentation is **1.76 pt** (3.96% -> 2.19%), against
**0.97 pt** measured under baseline segmentation — 1.8x larger, in the direction the ticket
anticipated but well short of the ~3 pt the prediction implied, and DER lands at 2.19%
rather than the predicted 0.5–1.5%. This is neither of the two interpretations fixed in
advance. It is not "near 1%", so the baseline-segmentation ceilings did not merely understate
a large recoverable budget; nor is it "near 3.9%", so relabelling is emphatically not useless
on clean input — it removes over half the confusion (541 s of 1057 s) and is the single
largest post-segmentation improvement measured on this project. The instrumentation says why
it stops there, and contradicts the predicted mechanism: `unmapped_speaker` fired **zero**
times in both runs, so over-clustering is not what holds DER off the 0.51% floor. Instead
44.9% of all pairs (56998 of 126925) hit `no_reference_overlap` — their support intersects no
reference speaker at all, leaving the oracle nothing to relabel toward. The residual 1.68 pt
gap to the floor is therefore not an assignment problem in the sense a classifier could
learn: it is frame-level disagreement between the segmentation mask and the reference over
regions the reference scores as silence, which a *labelling* strategy cannot reach by
construction, however good its labels.

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

### Plumbing implemented (2026-09-20)

Built test-first (`tests/test_oracle_combined_cells.py`, 23 tests). Red run showed 17
failures across all six work items before any implementation; the 4 initial greens were
deliberate regression guards on existing routing/delta behaviour.

- **One vacuous pass caught and fixed before implementing.**
  `test_combined_condition_requires_oracle_rttm` passed in the red run for the wrong
  reason: argparse rejected `oracle_segmentation_assignment` as an *invalid choice* and
  exited, so `pytest.raises(SystemExit)` was satisfied without the required-argument check
  existing at all. Tightened to assert on argparse's message (`--oracle-rttm` present,
  `invalid choice` absent) so it can only pass for the real reason. Worth remembering as a
  pattern: any `SystemExit` assertion against an argparse CLI is suspect while the value
  under test isn't yet a valid choice.
- **Item 1 (schema).** Added `oracle_segmentation_assignment` to `VALID_CONDITIONS`, plus a
  new `ORACLE_SEGMENTATION_CONDITIONS` tuple in the same module. The tuple is the fix for
  the *class* of bug behind item 2, not just the instance: routing now keys on membership
  in a named shared fact that lives beside the vocabulary, so the next
  oracle-segmentation condition can't silently miss it. `run_harness.py`'s
  `--run-condition` choices are now derived from `VALID_CONDITIONS` rather than retyped,
  removing the drift risk entirely.
- **Item 2 (routing).** Equality test replaced with membership; the `--oracle-rttm`
  required check widened with it and its error message now names the actual condition.
  Four routing tests pin the matrix: combined → oracle, plain oracle_segmentation →
  oracle, baseline → baseline, oracle_assignment-alone → baseline (that last one matters —
  widening the test must not capture the already-measured baseline-segmentation cell).
- **Item 3 (scope suppression).** `_effective_scope` now checks membership in
  `_SCOPED_CONDITIONS` instead of equality with `oracle_assignment`.
- **Item 4 (budget deltas).** `_BUDGET_CONDITIONS` values became
  `(reference_condition, label)` tuples, and `_build_delta_lines` resolves each row's own
  reference via a new `_condition_der`. `_baseline_der` is retained as a thin wrapper so
  the existing two lines are provably unchanged (a test asserts they still read +0.1309 /
  +0.0097 against baseline). Missing reference row emits an explicit "cannot be computed"
  line rather than falling back to baseline. Header changed from `baseline - oracle` to
  `reference - oracle` since it now covers both.
- **Item 5 (instrumentation).** `make_oracle_strategy`'s closure carries a `counts` dict
  (`_new_counts()`), reset per call, with the four exit paths incremented in the loop. A
  test asserts the four paths *partition* every pair (totals sum to pair count), which is
  what makes the numbers quotable. `run_harness` accumulates them across files and prints
  one line per run.
  - **Caveat worth knowing before reading the numbers:** a cache hit skips the pipeline
    entirely, so the strategy never runs and its counts stay zero. The totals describe
    uncached files only. For these two runs the cache key is fresh (see below) so they'll
    be genuine, but a re-run of the same condition will report zeros.
  - `relabelled` counts every pair assigned a mapped cluster, including where the label
    was already correct — not "changed value". Chosen so the four paths partition cleanly;
    a "changed" counter would leave a fifth unlabelled path.
- **Item 6 (provenance).** `oracle_rttm` recorded in `run_config`, additive and defaulting
  to `None`, deliberately *not* added to `REQUIRED_RUN_CONFIG_FIELDS` — a test asserts old
  manifests without the key still validate. Closes Gap 1 of [[run-manifest-provenance]];
  that ticket was still open at implementation time, so it did not land first.
- `run_experiment.sh` usage line and examples updated with the combined condition.
- **Cache separation confirmed empirically, not assumed** (the ticket asked for this).
  Computed the real keys for `oracle-v2` segmentation across the three refinement ids:
  `identity` → `c8e3977c…`, `oracle:all_pairs` → `eec684fe…`,
  `oracle:overlap_degraded` → `0db89dc3…`. Three distinct keys, so the existing
  `oracle_segmentation` run can't be served to either new run. No cache clear needed.
- Full harness suite re-run after the changes: 118 passed, 1 skipped (pre-existing), no
  regressions.

### Results (2026-09-20)

Both runs completed over 16 meetings from commit `1dab5122`, oracle RTTM
`harness-data/IHM/test.rttm`, collar 0, skip_overlap False.

| Run | scope | manifest | DER | MD | FA | Conf | wall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | all_pairs | `20260920T082056Z-2aada13d` | **2.193%** | 16.427 | 141.127 | 516.00 | 35.0 min |
| 2 | overlap_degraded | `20260920T101248Z-5479daf2` | **2.747%** | 16.427 | 141.127 | 686.09 | fast (warm cache) |

Run 2 completed well under Run 1's 35 min because segmentation/embeddings were already
warm. The strategy still executed — its pair counts are non-zero — so the numbers are
genuine, not a cached replay of Run 1.

**Criterion 2 MISSED, criterion 4 MISSED, both reported rather than adjusted** per the
ticket's standing instruction. Criteria 1, 3, 5, 6 pass.

Criterion 3 passes *exactly*: MD 16.42709375 and FA 141.12684375 are identical to the
oracle-seg run to 8 decimal places in both runs. Relabelling moved confusion and nothing
else, which is the strongest available evidence the refinement is touching only what it
should.

Scope ordering held (`overlap_degraded` 2.75% > `all_pairs` 2.19%), as expected but not
required.

#### The prediction's mechanism was wrong, not just its magnitude

`unmapped_speaker = 0` in **both** runs. The ticket predicted that path — dominant
reference speaker mapping to no produced cluster — would be what held DER off the 0.51%
floor, reasoning from 9/16 meetings over-clustering under oracle segmentation. It never
fired once across 126925 pairs.

Over-clustering does not strand reference speakers. Verified directly against
`optimal_mapping` rather than reasoned about: with 3 reference speakers split across 5
hypothesis clusters, all 3 speakers map (`{A:0, B:2, C:4}`) and the surplus clusters 1 and 3
go unmapped in the *other* direction, which costs nothing because the oracle only ever
reads `speaker_to_cluster`. A reference speaker is stranded only when it has **zero**
hypothesis overlap across the whole file — confirmed as the one case that does produce an
unmapped speaker. Under oracle segmentation every reference speaker has speech by
construction, so `unmapped_speaker = 0` is structural rather than luck, and the ticket's
inference from "9/16 meetings over-cluster" to "pairs will be left unmapped" does not
follow.

The real residual is `no_reference_overlap`: **56998 pairs, 44.9% of all pairs** in Run 1.
These are pairs whose temporal support intersects no reference speaker at all, so the
strategy leaves them untouched by design. Under oracle segmentation the segment boundaries
come from the reference, so this is not boundary noise — it is the frame-level active-speaker
mask disagreeing with the reference annotation over regions the reference scores as silence
for every speaker. Relabelling cannot reach them: there is no dominant reference speaker to
relabel toward.

The four paths partition exactly (69927 + 56998 = 126925 in Run 1; 98541 + 28384 = 126925 in
Run 2), so these counts are a complete account of every pair, not a sample.

#### Scope populations differ radically between segmentation conditions

Run 2's `out_of_scope = 98541` means only 28384 pairs (22.4%) classified as
overlap-degraded under oracle segmentation. Note also `no_reference_overlap = 0` in Run 2:
every pair that passes `_is_overlap_degraded` necessarily has reference overlap, so the
narrower scope structurally cannot encounter that path. This confirms the ticket's
suspicion that `_is_overlap_degraded` reclassifies over a substantially changed track
population — the two scopes are not nested subsets of the same pair set they were under
baseline segmentation.

### Second trap found: stale reference row (2026-09-20, after item 4 landed)

Rendering the table against the **real** `runs/` manifests — not just the synthetic CSVs
the unit tests use — surfaced a second way to get the wrong combined delta. Worth doing
routinely; the synthetic fixtures had one row per condition and so couldn't express this.

`runs/` holds two `oracle_segmentation` runs, **both** marked `counts_toward_results`:

| created_at          | seg id     | DER    |
| ------------------- | ---------- | ------ |
| 2026-09-17T01:36:30 | oracle     | 0.1705 |
| 2026-09-18T12:53:23 | oracle-v2  | 0.0396 |

The first is the pre-seam-fix run ([[oracle-segmentation-seam-no-op]]), where oracle
segmentation was a silent no-op and DER matched baseline. `_condition_der` as first
written returned the *first* matching row, which is the stale one, so the combined delta
came out at `0.1705 - 0.0100 = +0.1605` — precisely the double-counted ~16 pt figure item 4
exists to prevent, reached by a different route and equally plausible-looking.

Fixed by resolving the reference to the most recent matching run by `created_at` (ISO-8601
UTC, so lexical max is chronological). Test `test_reference_der_uses_the_most_recent_
matching_run` encodes both generations and asserts `+0.0296`, not `+0.1605`.

**Known pre-existing wart, deliberately not fixed here:** `_build_delta_lines` emits one
line per *row*, so the stale run still produces its own `Downstream budget: +0.0000` line
next to the real `+0.1309`, and both oracle_segmentation generations appear as table rows.
That predates this ticket and doesn't affect the combined delta now that references resolve
by recency. Fixing it properly means deciding whether superseded runs should be
de-marked, filtered by segmentation id, or de-duplicated by condition — a call for the
results-table owner, not a silent change here. Candidate follow-up ticket.

## See also

- [[Oracle Segmentation Findings Report]] — the 3.96% run and its error breakdown.
- [[Oracle Assignment Strategy]] — the refinement strategy and its scopes.
- [[Oracle Ceiling Metrics]] — the baseline-segmentation ceilings this run reframes.
- [[Cross-Condition Deltas]] — T5 table and budget-delta logic touched by items 3 and 4.
- [[Run Manifest]] — manifest schema and controlled vocabularies.
- [[run-manifest-provenance]] — open ticket owning the provenance gaps generally.
