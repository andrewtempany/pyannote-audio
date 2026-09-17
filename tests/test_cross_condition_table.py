"""Tests for T5: the cross-condition deltas table.

build_cross_condition_table() reads an already-aggregated comparison.csv and
returns a Markdown table: one row per (condition, oracle_scope) group, plus
the two budget deltas (baseline minus oracle segmentation, baseline minus
oracle assignment).

All inputs are fabricated CSV files -- no model, no audio, no manifests.
"""

import csv

import pytest

from harness.aggregate_runs import build_cross_condition_table

_COLUMNS = [
    "run_id",
    "condition",
    "oracle_scope",
    "counts_toward_results",
    "notes",
    "der",
    "der_overlap_system",
    "der_overlap_assigned",
    "missed_detection",
    "false_alarm",
    "confusion",
]


def _row(
    run_id,
    condition,
    der,
    oracle_scope="",
    counts_toward_results="True",
    der_overlap_system=0.0,
    der_overlap_assigned=0.0,
    missed_detection=0.0,
    false_alarm=0.0,
    confusion=0.0,
    notes="",
):
    return {
        "run_id": run_id,
        "condition": condition,
        "oracle_scope": oracle_scope,
        "counts_toward_results": counts_toward_results,
        "notes": notes,
        "der": der,
        "der_overlap_system": der_overlap_system,
        "der_overlap_assigned": der_overlap_assigned,
        "missed_detection": missed_detection,
        "false_alarm": false_alarm,
        "confusion": confusion,
    }


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _full_batch():
    """The four tracked conditions, shaped like the real comparison.csv."""
    return [
        _row("r1", "baseline", 0.1705, der_overlap_system=0.3339, der_overlap_assigned=0.2195),
        _row("r2", "nearest_centroid", 0.1705, der_overlap_system=0.3339, der_overlap_assigned=0.2195),
        _row("r3", "oracle_segmentation", 0.1704, der_overlap_system=0.3339, der_overlap_assigned=0.2195),
        _row("r4", "oracle_assignment", 0.1608, oracle_scope="all_pairs",
             der_overlap_system=0.3137, der_overlap_assigned=0.1920),
        _row("r5", "oracle_assignment", 0.1658, oracle_scope="overlap_degraded",
             der_overlap_system=0.3189, der_overlap_assigned=0.1993),
    ]


def test_table_includes_every_tracked_condition(tmp_path):
    csv_path = _write_csv(tmp_path / "comparison.csv", _full_batch())

    table = build_cross_condition_table(csv_path)

    for condition in (
        "baseline", "nearest_centroid", "oracle_segmentation", "oracle_assignment"
    ):
        assert condition in table


def test_excludes_rows_not_counting_toward_results(tmp_path):
    rows = _full_batch() + [
        _row("exploratory", "baseline", 0.9999, counts_toward_results="False",
             notes="scratch debugging run"),
    ]
    csv_path = _write_csv(tmp_path / "comparison.csv", rows)

    table = build_cross_condition_table(csv_path)

    assert "0.9999" not in table
    assert "scratch debugging run" not in table


def test_both_oracle_assignment_scopes_appear_separately(tmp_path):
    csv_path = _write_csv(tmp_path / "comparison.csv", _full_batch())

    table = build_cross_condition_table(csv_path)

    assert "all_pairs" in table
    assert "overlap_degraded" in table
    # both scopes' DERs present -- neither dropped nor averaged into one row
    assert "0.1608" in table
    assert "0.1658" in table
    # the average of the two would be 0.1633 -- must NOT appear
    assert "0.1633" not in table


def test_reports_downstream_budget_delta_baseline_minus_oracle_segmentation(tmp_path):
    rows = [
        _row("r1", "baseline", 0.2000),
        _row("r2", "oracle_segmentation", 0.1500),
    ]
    csv_path = _write_csv(tmp_path / "comparison.csv", rows)

    table = build_cross_condition_table(csv_path)

    # 0.2000 - 0.1500 = +0.0500, positive because oracle lowered DER
    assert "0.0500" in table
    assert "-0.0500" not in table


def test_reports_assignment_budget_delta_for_each_oracle_scope(tmp_path):
    rows = [
        _row("r1", "baseline", 0.2000),
        _row("r2", "oracle_assignment", 0.1500, oracle_scope="all_pairs"),
        _row("r3", "oracle_assignment", 0.1800, oracle_scope="overlap_degraded"),
    ]
    csv_path = _write_csv(tmp_path / "comparison.csv", rows)

    table = build_cross_condition_table(csv_path)

    assert "0.0500" in table  # baseline - all_pairs
    assert "0.0200" in table  # baseline - overlap_degraded


def test_delta_is_negative_when_oracle_is_worse_than_baseline(tmp_path):
    # Sign convention must be honest: a condition that made DER *worse* has
    # to surface as a negative delta, not be silently abs()'d into looking
    # like an improvement.
    rows = [
        _row("r1", "baseline", 0.1500),
        _row("r2", "oracle_segmentation", 0.2000),
    ]
    csv_path = _write_csv(tmp_path / "comparison.csv", rows)

    table = build_cross_condition_table(csv_path)

    assert "-0.0500" in table


def test_table_includes_der_component_breakdown(tmp_path):
    rows = [
        _row("r1", "baseline", 0.1705,
             missed_detection=111.5, false_alarm=222.5, confusion=333.5),
    ]
    csv_path = _write_csv(tmp_path / "comparison.csv", rows)

    table = build_cross_condition_table(csv_path)

    assert "111.5" in table
    assert "222.5" in table
    assert "333.5" in table


def test_output_is_a_markdown_table(tmp_path):
    csv_path = _write_csv(tmp_path / "comparison.csv", _full_batch())

    table = build_cross_condition_table(csv_path)

    lines = [line for line in table.splitlines() if line.strip()]
    assert any(line.lstrip().startswith("|") for line in lines)
    # a markdown table needs a header separator row of dashes
    assert any(set(line.replace("|", "").replace(" ", "")) <= {"-", ":"}
               and "-" in line for line in lines)


def test_oracle_scope_ignored_for_conditions_that_do_not_use_it(tmp_path):
    # run_harness.py's --oracle-scope has a default of "all_pairs" that gets
    # recorded in every manifest, even for runs where it's meaningless (it's
    # only read when --refinement-strategy is oracle). Showing it against
    # oracle_segmentation would imply a scope choice that was never made.
    rows = [
        _row("r1", "baseline", 0.2000),
        _row("r2", "oracle_segmentation", 0.1500, oracle_scope="all_pairs"),
    ]
    csv_path = _write_csv(tmp_path / "comparison.csv", rows)

    table = build_cross_condition_table(csv_path)

    seg_line = next(
        line for line in table.splitlines() if "oracle_segmentation" in line and "|" in line
    )
    assert "all_pairs" not in seg_line
    # ...and the delta line for it must not carry a scope label either
    delta_line = next(
        line for line in table.splitlines()
        if "Downstream budget" in line
    )
    assert "all_pairs" not in delta_line


def test_missing_baseline_does_not_crash(tmp_path):
    # A partial batch (no baseline scored yet) still renders the condition
    # rows it does have -- deltas just can't be computed without a baseline.
    rows = [
        _row("r1", "oracle_assignment", 0.1608, oracle_scope="all_pairs"),
    ]
    csv_path = _write_csv(tmp_path / "comparison.csv", rows)

    table = build_cross_condition_table(csv_path)

    assert "oracle_assignment" in table


def test_empty_csv_does_not_crash(tmp_path):
    csv_path = _write_csv(tmp_path / "comparison.csv", [])

    table = build_cross_condition_table(csv_path)

    assert isinstance(table, str)
