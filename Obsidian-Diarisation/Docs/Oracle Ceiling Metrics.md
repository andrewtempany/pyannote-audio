---
status: done
created: 2026-09-13
---

# Oracle Ceiling Metrics

> Part of the [[Evaluation Harness]]. Extends [[Scorer]], [[Harness Reporter]], and [[Run Manifest]] with the metrics and manifest fields needed for the oracle/ceiling-analysis experiment batch (T1 in `Obsidian-Diarisation/Tickets/`). **All code here is project-contributed** (`harness/scorer.py`, `harness/reporter.py`, `harness/run_manifest.py`, `run_harness.py`) — no changes to upstream `pyannote-audio`.

## Why this exists

The oracle/ceiling-analysis experiment tests whether diarisation error is fixable downstream of the frozen neural network (in post-clustering speaker assignment), without retraining. That claim rests on being able to distinguish, per file and corpus-wide:

- how much error the whole pipeline produces on overlap regions it detected vs. missed, and
- how much of the error *within detected overlap* is a pure assignment mistake, as opposed to a detection mistake.

The pre-existing harness (see [[Scorer]]) only reported a single blended DER and a single `overlap_der` scored over ground-truth overlap regions — enough to say "the pipeline struggles on overlap" but not enough to say *why*. This extension adds the metrics needed to separate those causes.

## The two overlap regions: T and D

Let **T** (truth) be the reference's ground-truth overlap regions — `reference.get_overlap()`, as before. Let **D** (detected) be the same computation on the **hypothesis**: `hypothesis.get_overlap()`. These are two different, independently-computed regions:

- **T ∩ D** — overlap the pipeline both should have found and did detect as multi-speaker. Error scored here is a pure assignment/detection-quality signal, holding detection constant.
- **T \ D** — overlap the pipeline missed entirely (it only emitted a single speaker there).
- **D \ T** — overlap the pipeline hallucinated (it emitted multiple speakers where the reference has only one).

The gap between `der_overlap_system` (scored over all of T) and `der_overlap_assigned` (scored over T ∩ D only) is, by construction, detection error — obtained without any extra computation.

## New metrics in `score()`

`harness/scorer.py`'s `score()` signature grew a 4th positional metric parameter:

```python
def score(reference, hypothesis, uem, der, overlap_der, der_overlap_assigned, jer) -> dict:
    ...
```

`der_overlap_assigned` must be its own `DiarizationErrorRate(collar=0, skip_overlap=False)` instance, exactly like the pre-existing `der`/`overlap_der` separation described in [[Scorer]] — it is scored over a different region (T ∩ D) than either `der` (whole UEM) or `overlap_der`/`der_overlap_system` (T), so reusing an existing instance would corrupt that instance's corpus-level `abs(metric)` total.

The returned dict now contains:

| Key | Meaning | Scored over |
|---|---|---|
| `der` | Overall DER | whole UEM |
| `der_overlap_system` | Renamed from `overlap_der`. Same semantics: how the whole pipeline handles overlap, including missed/spurious overlap | T |
| `der_overlap_assigned` | **The thesis metric.** How the post-clustering stage handles overlap it was actually handed | T ∩ D |
| `missed_detection`, `false_alarm`, `confusion` | DER's raw component durations (seconds), broken out separately | whole UEM, alongside `der` |
| `jer` | Unchanged | whole UEM |
| `count_ref`, `count_hyp`, `count_error` | Unchanged | — |
| `region_t_and_d`, `region_t_minus_d`, `region_d_minus_t` | Region census: duration (seconds) of T∩D, T\D, D\T | — |

### Getting DER components without a second scoring pass

`missed_detection`/`false_alarm`/`confusion` come from a single call, `der(reference, hypothesis, uem=uem, detailed=True)`, which returns both the DER ratio (keyed by `pyannote.metrics.diarization.DER_NAME`) and the raw component durations (keyed by the `pyannote.metrics.matcher` constants `"missed detection"`, `"false alarm"`, `"confusion"`) in the same dict. This was a deliberate choice to preserve the existing "each metric instance called exactly once per file" contract (see [[Scorer]]) — a second call to `der` with the same arguments would double-count that file into the corpus accumulator.

### Region census via `Timeline` set operations

```python
t = reference.get_overlap().crop(uem, mode="intersection")
d = hypothesis.get_overlap().crop(uem, mode="intersection")
t_and_d = t.crop(d, mode="intersection")
t_minus_d = t.extrude(d)
d_minus_t = d.extrude(t)
```

`t_and_d`, `t_minus_d`, `d_minus_t` are disjoint by construction — no runtime check needed, it follows from how `crop`/`extrude` partition a timeline.

## Fixture gotcha: the Hungarian mapper defeats naive "swap the label" fixtures

`DiarizationErrorRate` re-optimizes the reference/hypothesis label mapping *within whatever region it's scored over*, on every call (`optimal_mapping` via `HungarianMapper`, called fresh inside `compute_components`). A test fixture that just swaps which hypothesis label corresponds to which reference speaker gets silently "fixed" by the mapper and scores as correct — it does not produce real confusion.

To force a genuine, non-recoverable error inside T ∩ D, the hypothesis must actually detect **fewer distinct speakers** than the reference within that region (a real detection/assignment shortfall), not just wrong labels. `tests/test_scorer.py::test_der_overlap_assigned_scored_over_intersection_of_reference_and_hypothesis_overlap` uses a 3-speaker reference overlap where the hypothesis only emits 2 speakers in the intersection window — this is the pattern to reuse for any future fixture involving `der_overlap_assigned`.

## `write_report()` changes

`harness/reporter.py`'s `write_report()` grew the matching 4th positional parameter:

```python
summary = write_report(rows, der, overlap_der, der_overlap_assigned, jer, per_file_csv_path, summary_path)
```

`FIELDNAMES` now includes all the new per-file columns (`der_overlap_system` replacing `overlap_der`, plus `missed_detection`, `false_alarm`, `confusion`, `der_overlap_assigned`, `region_t_and_d`, `region_t_minus_d`, `region_d_minus_t`).

**Region census is the one place the corpus summary deviates from "always read accumulator totals, never recompute from rows"** (see [[Harness Reporter]]'s existing explanation of why that rule exists for DER/JER). Region census durations are raw seconds, not error-rate ratios — there is no `pyannote.metrics` accumulator object for them to live in — so the corpus-level `region_t_and_d`/`region_t_minus_d`/`region_d_minus_t` in `summary.json` are a plain `sum()` over the per-file rows. This is a deliberate, necessary exception, not a violation of the anti-averaging principle (which only applies to ratios, where a sum-of-sums is wrong; a sum of durations is exactly right).

## Manifest fields (`harness/run_manifest.py`)

Four new required fields, plus one new optional field with a default:

- **`condition`** — controlled vocabulary: `baseline`, `oracle_segmentation`, `oracle_assignment`, `nearest_centroid`. Identifies which of the four oracle/ceiling-analysis conditions a run belongs to. `write_manifest()` raises `RunManifestError` for any other value.
- **`refinement_strategy`** — controlled vocabulary: `identity`, `oracle`, `nearest_centroid`. See [[Post-Clustering Refinement Hook]]. `"oracle"` is accepted even though only `identity`/`nearest_centroid` are implemented as of this writing — the vocabulary was set up in advance so a later ticket implementing the oracle strategy doesn't need to touch `run_manifest.py`.
- **`corpus`**, **`mic_condition`** — free strings (e.g. `"AMI"`, `"IHM"`).
- **`git_commit`** — short git commit hash, populated automatically by `run_harness.py` via `git rev-parse --short HEAD`; not user-supplied.
- **`counts_toward_results`** (optional, defaults to `False`) — marks whether a run should be included when computing the batch's cross-condition comparison table. `write_manifest()` merges this into the written `run_config` (`{**run_config, "counts_toward_results": run_config.get("counts_toward_results", False)}`) *before* computing the `run_id` hash, so the hash and the written value are always consistent — a caller that omits the field gets a manifest whose `run_config` is not byte-identical to what it passed in (it has one field added).

### A naming collision, resolved

The pre-existing `--condition` CLI flag on `run_harness.py` means the AMI **mic condition** (`IHM`/`SDM`) — it predates this batch and is unrelated to the new `condition` controlled vocabulary above. Renaming it would have broken any existing script calling `--condition IHM`. Resolution: `--condition` keeps its old meaning and feeds the new `mic_condition` manifest field; a new `--run-condition` flag (default `"baseline"`) feeds the new `condition` field.

```bash
python run_harness.py \
  --data-root /path/to/ami/harness-data \
  --condition IHM \
  --run-condition baseline \
  --refinement-strategy identity \
  --corpus AMI \
  --counts-toward-results \
  --notes "baseline community-1, no modifications"
```

`aggregate_runs.py` needed **no code changes** for any of this — `_flatten_manifest` already dynamically unions every `run_config`/`summary` key into `comparison.csv` columns, so every new field flows through automatically.

## Verification (real GPU run, not just unit tests)

Ran on an RTX 3060 against the real AMI IHM test split with `--refinement-strategy identity --run-condition baseline`:

```
{'der': 0.17049287033463784, 'der_overlap_system': 0.3339467146571433,
 'der_overlap_assigned': 0.21945306864112454, 'jer': 0.22187246998379553,
 'counting_mae': 0.25, 'counting_exact_match_percent': 75.0,
 'region_t_and_d': 2433.117, 'region_t_minus_d': 1393.939, 'region_d_minus_t': 383.126}
```

- `der` is **bit-for-bit identical** to the pre-T1 baseline manifest (`0.17049287033463784`) — confirms none of these changes altered the actual DER computation, only what's reported alongside it.

> **Why this project records two baseline DERs — RESOLVED 2026-09-25.** Two values appear
> across the vault: `0.17049287033463784` (here and in [[Post-Clustering Refinement Hook]]) and
> `0.17048543579940637` (the oracle-era and later runs). The gap was previously attributed to
> "first-computation nondeterminism that the cache then freezes". **That was a guess and it is
> wrong.** This closes a previously open question rather than adding a new caveat.
>
> The cause is RTTM quantisation. `Annotation._iter_rttm` writes every boundary with `:.3f`, so
> a run scored from **cached RTTMs** is millisecond-truncated, while the **first** run of a
> configuration scores full-precision in-memory `Annotation` objects. Every later run of the
> same configuration reads the truncated file.
>
> Evidence: the run above agrees with the known RTTM-cached run `20260925T080105Z-7ddbf6ae` to
> full precision on `der_overlap_system`, `der_overlap_assigned` and all three region terms, and
> its region values are exact multiples of 0.001 where the cold run's are not.
>
> So `0.17049287…` is a **warm (RTTM-cached)** figure and `0.17048543…` is the **cold,
> full-precision** one. Both are correct for what they measure; they are not two attempts at
> one number. **Cite the cold figure against the published benchmark.** See
> [[Pre-Clustering Cache]] for the tier-by-tier statement.
- `der_overlap_assigned` (0.219) sits between `der` (0.170) and `der_overlap_system` (0.334), matching the expected shape: assignment error confined to detected overlap is worse than the whole-file average but better than the full reference-overlap number (which also folds in missed-overlap detection).
- Region census is non-zero and non-trivial on real data: 2433s / 1394s / 383s.

## Non-goals

- No new metrics beyond those listed here (see the parent ticket's explicit out-of-scope note).
- No interpretation of what the numbers mean for the go/no-go decision — that's a separate, human-authored decision record, not part of the harness.
- Does not implement the oracle-segmentation or oracle-assignment conditions themselves — only the metrics/manifest schema they'll report through. See the (still open, as of this doc) oracle-segmentation-provider and oracle-assignment-strategy tickets.

## Key files

- `harness/scorer.py` — `score()`'s new signature and metrics
- `harness/reporter.py` — `write_report()`'s new signature, `FIELDNAMES`
- `harness/run_manifest.py` — `VALID_CONDITIONS`, `VALID_REFINEMENT_STRATEGIES`, `counts_toward_results` defaulting
- `run_harness.py` — 4th accumulator wiring, `--run-condition`/`--refinement-strategy`/`--corpus`/`--counts-toward-results` CLI flags, `_current_git_commit()`
- `tests/test_scorer.py`, `tests/test_reporter.py`, `tests/test_run_manifest.py`, `tests/test_aggregate_runs.py`
