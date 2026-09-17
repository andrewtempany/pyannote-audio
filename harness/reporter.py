"""Reporter: rows -> per-file CSV + corpus summary. See TICKET-07-reporter.md,
extended by T1 (oracle/ceiling analysis batch: der components,
der_overlap_system rename, der_overlap_assigned, region census).

Corpus-level DER/der_overlap_system/der_overlap_assigned/JER are read from
the caller-owned accumulator instances (via `abs(metric)`) that the scorer's
`score()` was also called with -- never recomputed by averaging the per-file
rows, which the epic explicitly forbids (a corpus DER is a ratio over the
whole corpus, not a mean of per-file ratios). Region census durations have no
such accumulator (they're raw durations, not error rates), so those three
fields are summed across the per-file rows directly.

Corpus-level DER components (missed detection / false alarm / confusion, added
for T5) follow the accumulator rule too, read from `der.accumulated_` rather
than summed from the per-file rows -- the rows carry each file's own
`detailed=True` breakdown from the same accumulator, so re-summing them would
duplicate work the accumulator already did.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Union

FIELDNAMES = [
    "uri",
    "der",
    "der_overlap_system",
    "jer",
    "count_ref",
    "count_hyp",
    "count_error",
    "missed_detection",
    "false_alarm",
    "confusion",
    "der_overlap_assigned",
    "region_t_and_d",
    "region_t_minus_d",
    "region_d_minus_t",
]

_REGION_CENSUS_FIELDS = ("region_t_and_d", "region_t_minus_d", "region_d_minus_t")

# summary field name -> pyannote.metrics component key on the DER accumulator
# (DiarizationErrorRate.metric_components() -> [..., 'false alarm',
# 'missed detection', 'confusion']).
_DER_COMPONENT_KEYS = {
    "missed_detection": "missed detection",
    "false_alarm": "false alarm",
    "confusion": "confusion",
}


class ReporterError(ValueError):
    """Raised when the reporter is given nothing to report."""


def write_report(
    rows: List[Dict[str, Any]],
    der: Any,
    overlap_der: Any,
    der_overlap_assigned: Any,
    jer: Any,
    per_file_csv_path: Union[str, Path],
    summary_path: Union[str, Path],
) -> Dict[str, float]:
    if not rows:
        raise ReporterError("no rows to report: at least one scored file is required")

    per_file_csv_path = Path(per_file_csv_path)
    summary_path = Path(summary_path)
    per_file_csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(per_file_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in FIELDNAMES})

    count_errors = [row["count_error"] for row in rows]
    counting_mae = sum(count_errors) / len(count_errors)
    counting_exact_match_percent = (
        100.0 * sum(1 for e in count_errors if e == 0) / len(count_errors)
    )

    summary = {
        "der": abs(der),
        "der_overlap_system": abs(overlap_der),
        "der_overlap_assigned": abs(der_overlap_assigned),
        "jer": abs(jer),
        "counting_mae": counting_mae,
        "counting_exact_match_percent": counting_exact_match_percent,
        **{
            field: sum(row[field] for row in rows)
            for field in _REGION_CENSUS_FIELDS
        },
        **{
            field: der.accumulated_.get(key, 0.0)
            for field, key in _DER_COMPONENT_KEYS.items()
        },
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary
