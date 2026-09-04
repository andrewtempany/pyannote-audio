"""Reporter: rows -> per-file CSV + corpus summary. See TICKET-07-reporter.md.

Corpus-level DER/overlap-DER/JER are read from the caller-owned accumulator
instances (via `abs(metric)`) that TICKET-05's `score()` was also called
with -- never recomputed by averaging the per-file rows, which the epic
explicitly forbids (a corpus DER is a ratio over the whole corpus, not a
mean of per-file ratios).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Union

FIELDNAMES = ["uri", "der", "overlap_der", "jer", "count_ref", "count_hyp", "count_error"]


class ReporterError(ValueError):
    """Raised when the reporter is given nothing to report."""


def write_report(
    rows: List[Dict[str, Any]],
    der: Any,
    overlap_der: Any,
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
        "overlap_der": abs(overlap_der),
        "jer": abs(jer),
        "counting_mae": counting_mae,
        "counting_exact_match_percent": counting_exact_match_percent,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary
