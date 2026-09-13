"""Tests for the run-manifest ticket: harness/aggregate_runs.py.

harness/aggregate_runs.py doesn't exist yet at the time these tests are
written -- the first run of this suite is expected to fail on import.
"""

import csv
import json

import pytest

from harness.aggregate_runs import aggregate_runs


def _write_manifest(runs_dir, run_id, run_config, summary):
    runs_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "created_at": "2026-09-07T12:00:00Z",
        "run_config": run_config,
        "summary": summary,
    }
    (runs_dir / f"{run_id}.json").write_text(json.dumps(manifest))


def test_aggregate_builds_one_row_per_manifest(tmp_path):
    runs_dir = tmp_path / "runs"
    _write_manifest(
        runs_dir, "run-1",
        {"pipeline_config_id": "p1", "segmentation_source_id": "baseline", "clustering_model": "cm1"},
        {"der": 0.1, "counting_mae": 0.5},
    )
    _write_manifest(
        runs_dir, "run-2",
        {"pipeline_config_id": "p2", "segmentation_source_id": "oracle", "clustering_model": "cm2"},
        {"der": 0.2, "counting_mae": 0.7},
    )

    output_csv = tmp_path / "comparison.csv"
    aggregate_runs(runs_dir=runs_dir, output_csv_path=output_csv)

    with open(output_csv, newline="") as f:
        rows = {row["run_id"]: row for row in csv.DictReader(f)}

    assert len(rows) == 2
    assert rows["run-1"]["clustering_model"] == "cm1"
    assert rows["run-1"]["segmentation_source_id"] == "baseline"
    assert float(rows["run-1"]["der"]) == 0.1
    assert rows["run-2"]["clustering_model"] == "cm2"
    assert float(rows["run-2"]["counting_mae"]) == 0.7


def test_aggregate_includes_notes_column_when_present(tmp_path):
    runs_dir = tmp_path / "runs"
    _write_manifest(
        runs_dir, "run-1",
        {
            "pipeline_config_id": "p1", "segmentation_source_id": "baseline",
            "clustering_model": "dbscan", "notes": "dbscan clustering experiment",
        },
        {"der": 0.1},
    )

    output_csv = tmp_path / "comparison.csv"
    aggregate_runs(runs_dir=runs_dir, output_csv_path=output_csv)

    with open(output_csv, newline="") as f:
        rows = {row["run_id"]: row for row in csv.DictReader(f)}

    assert rows["run-1"]["notes"] == "dbscan clustering experiment"


def test_aggregate_fills_blank_notes_for_manifest_missing_the_field(tmp_path):
    runs_dir = tmp_path / "runs"
    _write_manifest(
        runs_dir, "run-1",
        {"pipeline_config_id": "p1", "segmentation_source_id": "baseline", "clustering_model": "cm1"},
        {"der": 0.1},
    )

    output_csv = tmp_path / "comparison.csv"
    aggregate_runs(runs_dir=runs_dir, output_csv_path=output_csv)

    with open(output_csv, newline="") as f:
        rows = {row["run_id"]: row for row in csv.DictReader(f)}

    assert "notes" in rows["run-1"]
    assert rows["run-1"]["notes"] == ""


def test_aggregate_includes_t1_condition_and_metric_fields(tmp_path):
    # T1: condition, refinement_strategy, counts_toward_results, and the new
    # scorer metrics (der_overlap_system, der_overlap_assigned, component
    # breakdown) must flow through to comparison.csv, not just per_file.csv
    # / summary.json. _flatten_manifest already unions run_config and summary
    # keys dynamically, so this is a regression guard rather than new
    # aggregation logic -- but T1 explicitly requires every new field to
    # reach comparison.csv, so it needs its own assertion.
    runs_dir = tmp_path / "runs"
    _write_manifest(
        runs_dir, "run-1",
        {
            "pipeline_config_id": "p1",
            "segmentation_source_id": "baseline",
            "clustering_model": "cm1",
            "condition": "oracle_assignment",
            "refinement_strategy": "oracle",
            "corpus": "AMI",
            "mic_condition": "IHM",
            "git_commit": "abc1234",
            "counts_toward_results": True,
        },
        {
            "der": 0.1,
            "der_overlap_system": 0.2,
            "der_overlap_assigned": 0.05,
            "missed_detection": 0.01,
            "false_alarm": 0.02,
            "confusion": 0.03,
            "region_t_and_d": 4.0,
            "region_t_minus_d": 1.0,
            "region_d_minus_t": 0.5,
        },
    )

    output_csv = tmp_path / "comparison.csv"
    aggregate_runs(runs_dir=runs_dir, output_csv_path=output_csv)

    with open(output_csv, newline="") as f:
        rows = {row["run_id"]: row for row in csv.DictReader(f)}

    row = rows["run-1"]
    assert row["condition"] == "oracle_assignment"
    assert row["refinement_strategy"] == "oracle"
    assert row["corpus"] == "AMI"
    assert row["mic_condition"] == "IHM"
    assert row["git_commit"] == "abc1234"
    assert row["counts_toward_results"] == "True"
    assert float(row["der_overlap_system"]) == pytest.approx(0.2)
    assert float(row["der_overlap_assigned"]) == pytest.approx(0.05)
    assert float(row["missed_detection"]) == pytest.approx(0.01)
    assert float(row["false_alarm"]) == pytest.approx(0.02)
    assert float(row["confusion"]) == pytest.approx(0.03)
    assert float(row["region_t_and_d"]) == pytest.approx(4.0)
    assert float(row["region_t_minus_d"]) == pytest.approx(1.0)
    assert float(row["region_d_minus_t"]) == pytest.approx(0.5)


def test_aggregate_handles_empty_runs_dir(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    output_csv = tmp_path / "comparison.csv"
    aggregate_runs(runs_dir=runs_dir, output_csv_path=output_csv)

    with open(output_csv, newline="") as f:
        rows = list(csv.reader(f))

    assert rows == [] or rows == [["run_id"]]
