"""Per-file scoring. See TICKET-05-scorer.md, extended by T1 (oracle/ceiling
analysis batch: der components, der_overlap_system rename, der_overlap_
assigned, region census).

Design note: the epic describes this as "a pure function
`(reference, hypothesis, uem) -> dict`", but also requires DER/overlap-DER/JER
to be accumulated across the whole corpus by exactly one metric instance
each, constructed once at the start of a run -- a function that is both
stateless and accumulates state across calls is a contradiction. Resolved by
having the caller (the orchestrator, TICKET-08) own and construct the
`DiarizationErrorRate`/`JaccardErrorRate` accumulator instances once, and
pass them into `score()` on every call. That keeps this module itself free
of model loading and file I/O -- the actual meaning of "pure" that matters
here -- while making the shared-accumulator requirement an explicit
parameter instead of something a future reader could "fix" back into a
stateless function and silently break corpus-level accumulation.

T1 note: der_overlap_assigned is scored over a region (T-intersect-D) that is
distinct from both der's (whole uem) and der_overlap_system's (T) -- it needs
its own accumulator instance, so `score()` grows a 4th metric parameter
alongside der/overlap_der/jer, mirroring the existing pattern rather than
refactoring to a dict/object param.
"""

from __future__ import annotations

from typing import Any, Dict

from pyannote.core import Annotation, Timeline
from pyannote.metrics.diarization import DER_NAME

# pyannote.metrics.matcher component keys, as returned by
# metric(..., detailed=True).
_MISSED_DETECTION = "missed detection"
_FALSE_ALARM = "false alarm"
_CONFUSION = "confusion"


def score(
    reference: Annotation,
    hypothesis: Annotation,
    uem: Timeline,
    der: Any,
    overlap_der: Any,
    der_overlap_assigned: Any,
    jer: Any,
) -> Dict[str, float]:
    """Score one file and accumulate into the four metric instances.

    `der`, `overlap_der`, `der_overlap_assigned`, and `jer` must be
    pre-constructed `DiarizationErrorRate`/`JaccardErrorRate` instances
    (collar=0, skip_overlap=False), owned by the caller and shared across
    every file in a run -- each is called here exactly once, so its
    corpus-level total (via `abs(metric)`) reflects every file scored
    through it.
    """
    t = reference.get_overlap().crop(uem, mode="intersection")
    d = hypothesis.get_overlap().crop(uem, mode="intersection")
    t_and_d = t.crop(d, mode="intersection")
    t_minus_d = t.extrude(d)
    d_minus_t = d.extrude(t)

    der_components = der(reference, hypothesis, uem=uem, detailed=True)
    overlap_der_value = overlap_der(reference, hypothesis, uem=t)
    der_overlap_assigned_value = der_overlap_assigned(reference, hypothesis, uem=t_and_d)
    jer_value = jer(reference, hypothesis, uem=uem)

    count_ref = len(reference.labels())
    count_hyp = len(hypothesis.labels())

    return {
        "der": der_components[DER_NAME],
        "der_overlap_system": overlap_der_value,
        "der_overlap_assigned": der_overlap_assigned_value,
        "jer": jer_value,
        "count_ref": count_ref,
        "count_hyp": count_hyp,
        "count_error": abs(count_ref - count_hyp),
        "missed_detection": der_components[_MISSED_DETECTION],
        "false_alarm": der_components[_FALSE_ALARM],
        "confusion": der_components[_CONFUSION],
        "region_t_and_d": t_and_d.duration(),
        "region_t_minus_d": t_minus_d.duration(),
        "region_d_minus_t": d_minus_t.duration(),
    }
