"""Aggregate a directory of run manifests (harness/run_manifest.py) into a
flat comparison table, one row per run. Generated on demand, not
hand-maintained. See the run-manifest ticket in
Obsidian-Diarisation/Tickets/Open/run-manifest.md.

Also builds T5's cross-condition table: the same data grouped by experimental
condition, with the two ceiling-analysis budget deltas computed rather than
transcribed by hand.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def _flatten_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    row = {"run_id": manifest["run_id"], "created_at": manifest["created_at"]}
    row.update(manifest["run_config"])
    row.setdefault("notes", "")
    row.update(manifest["summary"])
    row["duration_seconds"] = manifest.get("duration_seconds") or ""
    return row


_TABLE_COLUMNS = (
    ("condition", "Condition"),
    ("oracle_scope", "Oracle scope"),
    ("der", "DER"),
    ("der_overlap_system", "DER overlap (system)"),
    ("der_overlap_assigned", "DER overlap (assigned)"),
    ("missed_detection", "Missed detection"),
    ("false_alarm", "False alarm"),
    ("confusion", "Confusion"),
)

# Conditions whose distance from a reference condition is a "budget": how much
# DER the corresponding oracle intervention could recover if it were perfect.
#
# Each entry carries its OWN reference condition, because they aren't all
# measured against baseline. The combined cell (oracle segmentation + oracle
# assignment) is measured against `oracle_segmentation`: a baseline-referenced
# delta would double-count the segmentation budget already reported on its own
# line, and the quantity of interest is the assignment budget *given clean
# segmentation*, directly comparable to the assignment ceilings measured under
# baseline segmentation.
_BUDGET_CONDITIONS = {
    "oracle_segmentation": (
        "baseline", "Downstream budget (baseline - oracle segmentation)"
    ),
    "oracle_assignment": (
        "baseline", "Assignment budget (baseline - oracle assignment)"
    ),
    "oracle_segmentation_assignment": (
        "oracle_segmentation",
        "Assignment budget under oracle segmentation "
        "(oracle_segmentation - oracle_segmentation_assignment)",
    ),
}

# Conditions for which oracle_scope reflects a decision somebody actually
# made, rather than --oracle-scope's recorded default.
_SCOPED_CONDITIONS = ("oracle_assignment", "oracle_segmentation_assignment")


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _effective_scope(row: Dict[str, Any]) -> str:
    """oracle_scope only means something for runs that actually refine, i.e.
    the conditions in _SCOPED_CONDITIONS.

    run_harness.py's --oracle-scope defaults to "all_pairs" and is recorded in
    every manifest regardless, so for any other condition the stored value is
    a default that was never actually chosen -- showing it would imply a
    decision nobody made.

    The combined condition belongs here too: its two runs differ *only* by
    scope, so blanking it would render them as identical rows.
    """
    if row.get("condition") not in _SCOPED_CONDITIONS:
        return ""
    return str(row.get("oracle_scope", "")).strip()


def _fmt(value: Any) -> str:
    """Render a CSV cell for the table: numbers to 4dp, blanks as an em dash."""
    text = str(value).strip()
    if not text:
        return "--"
    try:
        return f"{float(text):.4f}"
    except ValueError:
        return text


def _render_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def build_cross_condition_table(comparison_csv_path: Union[str, Path]) -> str:
    """T5: group an aggregated comparison.csv by experimental condition and
    render a Markdown table plus the ceiling-analysis budget deltas.

    Only rows with `counts_toward_results` true are included -- exploratory
    and debugging runs are excluded by design, so the table always reflects
    the deliberately-tracked batch rather than everything ever run.

    Rows are grouped by (condition, oracle_scope): the two oracle_assignment
    scopes (`all_pairs`, `overlap_degraded`) measure different ceilings and
    are always reported separately, never averaged into one number.

    Deltas are `baseline - oracle`, so a POSITIVE value means the oracle
    condition lowered DER (i.e. that much error is recoverable). A negative
    value honestly means the condition was worse than baseline.
    """
    comparison_csv_path = Path(comparison_csv_path)
    if not comparison_csv_path.exists() or not comparison_csv_path.read_text().strip():
        return "No runs recorded yet."

    with open(comparison_csv_path, newline="") as f:
        all_rows = list(csv.DictReader(f))

    rows = [r for r in all_rows if _is_true(r.get("counts_toward_results"))]
    if not rows:
        return "No runs with counts_toward_results=True yet."

    headers = [label for _, label in _TABLE_COLUMNS]
    table_rows = [
        [
            _fmt(_effective_scope(row) if key == "oracle_scope" else row.get(key, ""))
            for key, _ in _TABLE_COLUMNS
        ]
        for row in rows
    ]
    table = _render_markdown_table(headers, table_rows)

    deltas = _build_delta_lines(rows)
    if deltas:
        table += "\n\n" + "\n".join(deltas)
    return table


def _condition_der(rows: List[Dict[str, Any]], condition: str) -> Optional[float]:
    """DER of the first row for `condition`, or None if absent/unparseable.

    Used to resolve each budget delta's own reference row: not every delta is
    measured against baseline (see _BUDGET_CONDITIONS).
    """
    for row in rows:
        if row.get("condition") == condition:
            try:
                return float(row["der"])
            except (KeyError, ValueError):
                return None
    return None


def _baseline_der(rows: List[Dict[str, Any]]) -> Optional[float]:
    return _condition_der(rows, "baseline")


def _build_delta_lines(rows: List[Dict[str, Any]]) -> List[str]:
    if _baseline_der(rows) is None and not any(
        row.get("condition") in _BUDGET_CONDITIONS for row in rows
    ):
        return ["_No baseline run recorded -- budget deltas cannot be computed._"]

    lines = [
        "**Budget deltas** (`reference - oracle`; positive = oracle lowered DER)",
        "",
    ]
    for row in rows:
        entry = _BUDGET_CONDITIONS.get(row.get("condition", ""))
        if not entry:
            continue
        reference_condition, label = entry
        try:
            der = float(row["der"])
        except (KeyError, ValueError):
            continue

        scope = _effective_scope(row)
        suffix = f" [{scope}]" if scope else ""

        reference = _condition_der(rows, reference_condition)
        if reference is None:
            # Explicit rather than falling back to baseline: for the combined
            # condition a baseline-referenced delta would double-count the
            # segmentation budget and read as an assignment budget.
            lines.append(
                f"- {row['condition']}{suffix}: no {reference_condition} run "
                f"recorded -- this delta cannot be computed."
            )
            continue

        lines.append(f"- {label}{suffix}: {reference - der:+.4f}")

    return lines if len(lines) > 2 else []


def aggregate_runs(runs_dir: Union[str, Path], output_csv_path: Union[str, Path]) -> Path:
    runs_dir = Path(runs_dir)
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_paths = sorted(runs_dir.glob("*.json"))
    rows = [_flatten_manifest(json.loads(p.read_text())) for p in manifest_paths]

    if not rows:
        output_csv_path.write_text("")
        return output_csv_path

    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return output_csv_path


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output-csv", default=None,
                        help="Defaults to <runs-dir>/comparison.csv.")
    parser.add_argument(
        "--cross-condition-table", action="store_true",
        help="After refreshing the CSV, print T5's cross-condition table "
             "(grouped by condition, with the ceiling-analysis budget deltas).",
    )
    args = parser.parse_args(argv)

    output_csv = Path(args.output_csv or Path(args.runs_dir) / "comparison.csv")
    aggregate_runs(runs_dir=args.runs_dir, output_csv_path=output_csv)

    if args.cross_condition_table:
        print(build_cross_condition_table(output_csv))
    else:
        print(output_csv)


if __name__ == "__main__":
    main()
