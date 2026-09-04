---
status: done
created: 2026-09-04
---

# Scorer

> Part of the [[Evaluation Harness]]. **`harness/scorer.py` is code contributed by this project — not upstream `pyannote-audio`.** It does no metric math of its own: it is a thin, deliberate wrapper around pre-existing *sister-package* classes, `pyannote.metrics.diarization.DiarizationErrorRate` and `pyannote.metrics.diarization.JaccardErrorRate`, and calls `pyannote.core.Annotation.get_overlap()` / `pyannote.core.Timeline.crop()`. `pyannote.metrics` and `pyannote.core` are dependencies of this repo but live outside it (not under `src/pyannote/audio`), so there is no in-repo file path to cite for them.

## Purpose

Turn one file's `(reference, hypothesis, uem)` into a metrics dict, while accumulating DER/overlap-DER/JER into corpus-level totals across every file in a run.

## Where it fits

Called once per file by the [[Orchestrator]], which owns and constructs the three metric accumulator instances and threads them through every call. The resulting per-file dicts, plus the (now corpus-accumulated) metric instances, are handed to the [[Harness Reporter]].

## The stateless-vs-accumulator design resolution

The original design described this component as "a pure function `(reference, hypothesis, uem) -> dict`," but also required that DER, overlap-DER, and JER be accumulated across the whole corpus by exactly **one** metric instance each, constructed once at the start of a run. A function that is both stateless and accumulates state across calls is a contradiction unless the accumulator objects themselves are parameters, owned by the caller.

Resolved as:

```python
def score(reference, hypothesis, uem, der, overlap_der, jer) -> dict:
    ...
```

`der`, `overlap_der`, `jer` are pre-constructed `DiarizationErrorRate`/`JaccardErrorRate` instances, owned and constructed once by the caller (the [[Orchestrator]]), and passed into every call. This keeps `scorer.py` itself free of model loading and file I/O — the meaning of "pure" that actually matters here — while making the shared-accumulator requirement an explicit, visible parameter instead of an implicit contract a future edit could quietly break. This rationale is repeated as a module docstring in `harness/scorer.py` itself, specifically so a future reader doesn't "simplify" it back into a genuinely stateless function.

## Metrics and exact settings

- **DER (overall)** — `DiarizationErrorRate(collar=0, skip_overlap=False)`.
- **Overlap-region DER** (primary metric) — a **separate** `DiarizationErrorRate(collar=0, skip_overlap=False)` instance, called with `uem=reference.get_overlap().crop(uem, mode="intersection")`. Both `Annotation.get_overlap()` and `Timeline.crop(..., mode="intersection")` were confirmed against the installed `pyannote.core==5.0.0` API before use.
- **JER** — `JaccardErrorRate(collar=0, skip_overlap=False)`. `JaccardErrorRate.__bases__ == (DiarizationErrorRate,)` and already defaults to the same settings, but they're passed explicitly anyway so nobody "aligns" JER to a different collar later without noticing.
- **Speaker counting** — `len(reference.labels())` vs `len(hypothesis.labels())`, plus their absolute difference. Per-file; no accumulator object involved.

## Why three separate metric instances, not two

Overall DER and overlap-DER must be **separate `DiarizationErrorRate` instances**. A `pyannote.metrics` metric object accumulates its internal components on every call — calling the *same* instance twice per file (once with the full UEM, once with the overlap-only UEM) silently sums both regions into one accumulator, producing a meaningless corpus total. `der`, `overlap_der`, and `jer` must each be constructed once at the start of a run and called **exactly once per file**.

## API

```python
from harness.scorer import score

row = score(reference, hypothesis, uem, der, overlap_der, jer)
# {"der": ..., "overlap_der": ..., "jer": ..., "count_ref": ..., "count_hyp": ..., "count_error": ...}
```

DER/overlap-DER/JER accumulate into `der`, `overlap_der`, `jer` as a side effect of the call — read their corpus-level totals later via `abs(metric)` (see [[Harness Reporter]]).

## Gotcha: constructing overlapping tracks in tests

Building a reference `Annotation` with two genuinely overlapping speakers requires **explicit, distinct track keys** — e.g. `annotation[segment, "track_a"] = "A"`. Assigning to the *same* segment with no explicit track (or the same default track) silently **overwrites** the previous label rather than adding a second concurrent track:

```python
annotation[Segment(3, 5)] = "A"
annotation[Segment(3, 5)] = "B"
# annotation now contains only "B" — not both
```

Every overlap fixture in `tests/test_scorer.py` uses explicit track ids for this reason. (Companion to [[Dataset Adapter]]'s `Annotation.__eq__` gotcha.)

## Non-goals

- No corpus-level aggregation beyond calling the passed-in accumulators — that's the [[Orchestrator]]'s/[[Harness Reporter]]'s job.
- No reimplementation of DER/JER math — everything delegates to `pyannote.metrics`.
- No I/O, no model loading anywhere in this module.
