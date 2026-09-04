"""Per-file scoring. See TICKET-05-scorer.md.

Design note: the epic describes this as "a pure function
`(reference, hypothesis, uem) -> dict`", but also requires DER/overlap-DER/JER
to be accumulated across the whole corpus by exactly one metric instance
each, constructed once at the start of a run -- a function that is both
stateless and accumulates state across calls is a contradiction. Resolved by
having the caller (the orchestrator, TICKET-08) own and construct the three
`DiarizationErrorRate`/`JaccardErrorRate` accumulator instances once, and
pass them into `score()` on every call. That keeps this module itself free
of model loading and file I/O -- the actual meaning of "pure" that matters
here -- while making the shared-accumulator requirement an explicit
parameter instead of something a future reader could "fix" back into a
stateless function and silently break corpus-level accumulation.
"""

from __future__ import annotations

from typing import Any, Dict

from pyannote.core import Annotation, Timeline


def score(
    reference: Annotation,
    hypothesis: Annotation,
    uem: Timeline,
    der: Any,
    overlap_der: Any,
    jer: Any,
) -> Dict[str, float]:
    """Score one file and accumulate into the three metric instances.

    `der`, `overlap_der`, and `jer` must be pre-constructed
    `DiarizationErrorRate`/`JaccardErrorRate` instances (collar=0,
    skip_overlap=False), owned by the caller and shared across every file in
    a run -- each is called here exactly once, so its corpus-level total
    (via `abs(metric)`) reflects every file scored through it.
    """
    overlap_uem = reference.get_overlap().crop(uem, mode="intersection")

    der_value = der(reference, hypothesis, uem=uem)
    overlap_der_value = overlap_der(reference, hypothesis, uem=overlap_uem)
    jer_value = jer(reference, hypothesis, uem=uem)

    count_ref = len(reference.labels())
    count_hyp = len(hypothesis.labels())

    return {
        "der": der_value,
        "overlap_der": overlap_der_value,
        "jer": jer_value,
        "count_ref": count_ref,
        "count_hyp": count_hyp,
        "count_error": abs(count_ref - count_hyp),
    }
