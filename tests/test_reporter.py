"""Tests for TICKET-07 (the original reporter) and T1 (oracle/ceiling
analysis batch: der components, der_overlap_system rename,
der_overlap_assigned, region census).

write_report()'s signature has grown a 4th metric parameter,
der_overlap_assigned, alongside der/overlap_der/jer -- mirrors the same
change made to harness/scorer.py's score() for T1, so corpus-level
der_overlap_assigned (scored over a region distinct from both der and
der_overlap_system) gets its own accumulator total via abs(metric).

All inputs here are fabricated dicts/fake metric objects matching the
scorer's score() output shape and the "accumulator instances read via
abs(metric)" contract -- no model, no audio, no real dataset.
"""

import csv
import json

import pytest

from harness.reporter import ReporterError, write_report


class _FakeMetric:
    """Stands in for a pyannote.metrics accumulator: only `abs(metric)` is
    read by the reporter (it must not recompute the total from per-file
    rows), so that's the only thing this fake needs to support."""

    def __init__(self, total: float):
        self._total = total

    def __abs__(self) -> float:
        return self._total


def _row(
    uri,
    der,
    der_overlap_system,
    jer,
    count_ref,
    count_hyp,
    missed_detection=0.0,
    false_alarm=0.0,
    confusion=0.0,
    der_overlap_assigned=0.0,
    region_t_and_d=0.0,
    region_t_minus_d=0.0,
    region_d_minus_t=0.0,
):
    return {
        "uri": uri,
        "der": der,
        "der_overlap_system": der_overlap_system,
        "jer": jer,
        "count_ref": count_ref,
        "count_hyp": count_hyp,
        "count_error": abs(count_ref - count_hyp),
        "missed_detection": missed_detection,
        "false_alarm": false_alarm,
        "confusion": confusion,
        "der_overlap_assigned": der_overlap_assigned,
        "region_t_and_d": region_t_and_d,
        "region_t_minus_d": region_t_minus_d,
        "region_d_minus_t": region_d_minus_t,
    }


def _fresh_fake_metrics():
    return _FakeMetric(0.0), _FakeMetric(0.0), _FakeMetric(0.0), _FakeMetric(0.0)


def test_per_file_csv_has_expected_columns(tmp_path):
    rows = [_row("ES2002a", 0.1, 0.2, 0.3, 2, 2)]
    der, overlap_der, der_overlap_assigned, jer = (
        _FakeMetric(0.1), _FakeMetric(0.2), _FakeMetric(0.0), _FakeMetric(0.3),
    )

    write_report(
        rows, der, overlap_der, der_overlap_assigned, jer,
        per_file_csv_path=tmp_path / "per_file.csv",
        summary_path=tmp_path / "summary.json",
    )

    with open(tmp_path / "per_file.csv", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

    assert header == [
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


def test_per_file_csv_values_match_input_rows(tmp_path):
    rows = [
        _row("ES2002a", 0.123456, 0.2, 0.3, 3, 5),
        _row("ES2002b", 0.4, 0.5, 0.6, 2, 2),
    ]
    der, overlap_der, der_overlap_assigned, jer = (
        _FakeMetric(0.25), _FakeMetric(0.35), _FakeMetric(0.0), _FakeMetric(0.45),
    )

    write_report(
        rows, der, overlap_der, der_overlap_assigned, jer,
        per_file_csv_path=tmp_path / "per_file.csv",
        summary_path=tmp_path / "summary.json",
    )

    with open(tmp_path / "per_file.csv", newline="") as f:
        reader = csv.DictReader(f)
        read_rows = list(reader)

    assert len(read_rows) == 2
    assert read_rows[0]["uri"] == "ES2002a"
    assert float(read_rows[0]["der"]) == pytest.approx(0.123456)
    assert int(read_rows[0]["count_ref"]) == 3
    assert int(read_rows[0]["count_hyp"]) == 5
    assert int(read_rows[0]["count_error"]) == 2
    assert read_rows[1]["uri"] == "ES2002b"
    assert float(read_rows[1]["jer"]) == pytest.approx(0.6)


def test_per_file_csv_includes_component_and_region_census_values(tmp_path):
    # T1: missed_detection/false_alarm/confusion and the region census
    # durations must round-trip through the CSV, not just exist as keys.
    rows = [
        _row(
            "ES2002a", 0.1, 0.2, 0.3, 2, 2,
            missed_detection=1.5, false_alarm=0.5, confusion=2.0,
            der_overlap_assigned=0.25,
            region_t_and_d=4.0, region_t_minus_d=1.0, region_d_minus_t=0.5,
        )
    ]
    der, overlap_der, der_overlap_assigned, jer = _fresh_fake_metrics()

    write_report(
        rows, der, overlap_der, der_overlap_assigned, jer,
        per_file_csv_path=tmp_path / "per_file.csv",
        summary_path=tmp_path / "summary.json",
    )

    with open(tmp_path / "per_file.csv", newline="") as f:
        reader = csv.DictReader(f)
        read_row = next(reader)

    assert float(read_row["missed_detection"]) == pytest.approx(1.5)
    assert float(read_row["false_alarm"]) == pytest.approx(0.5)
    assert float(read_row["confusion"]) == pytest.approx(2.0)
    assert float(read_row["der_overlap_assigned"]) == pytest.approx(0.25)
    assert float(read_row["region_t_and_d"]) == pytest.approx(4.0)
    assert float(read_row["region_t_minus_d"]) == pytest.approx(1.0)
    assert float(read_row["region_d_minus_t"]) == pytest.approx(0.5)


def test_corpus_summary_includes_accumulated_der_overlap_der_jer(tmp_path):
    # deliberately inconsistent with what averaging the rows would produce,
    # so this test fails if the reporter recomputes instead of reading the
    # accumulator totals.
    rows = [
        _row("ES2002a", 0.1, 0.1, 0.1, 2, 2),
        _row("ES2002b", 0.9, 0.9, 0.9, 2, 2),
    ]
    der, overlap_der, der_overlap_assigned, jer = (
        _FakeMetric(0.42), _FakeMetric(0.17), _FakeMetric(0.0), _FakeMetric(0.88),
    )

    write_report(
        rows, der, overlap_der, der_overlap_assigned, jer,
        per_file_csv_path=tmp_path / "per_file.csv",
        summary_path=tmp_path / "summary.json",
    )

    summary = json.loads((tmp_path / "summary.json").read_text())

    assert summary["der"] == pytest.approx(0.42)
    assert summary["der_overlap_system"] == pytest.approx(0.17)
    assert summary["jer"] == pytest.approx(0.88)
    # sanity: these are NOT the per-file averages (0.5, 0.5, 0.5)
    assert summary["der"] != pytest.approx(0.5)


def test_corpus_summary_includes_accumulated_der_overlap_assigned(tmp_path):
    # der_overlap_assigned needs its own accumulator total in the summary,
    # distinct from der and der_overlap_system -- deliberately set to a value
    # that doesn't match either of the others, so a bug that wired the wrong
    # accumulator into the summary would be caught.
    rows = [_row("ES2002a", 0.1, 0.2, 0.3, 2, 2, der_overlap_assigned=0.3)]
    der, overlap_der, der_overlap_assigned, jer = (
        _FakeMetric(0.1), _FakeMetric(0.2), _FakeMetric(0.55), _FakeMetric(0.3),
    )

    write_report(
        rows, der, overlap_der, der_overlap_assigned, jer,
        per_file_csv_path=tmp_path / "per_file.csv",
        summary_path=tmp_path / "summary.json",
    )

    summary = json.loads((tmp_path / "summary.json").read_text())

    assert summary["der_overlap_assigned"] == pytest.approx(0.55)


def test_corpus_summary_includes_accumulated_region_census(tmp_path):
    # Region census is corpus-wide too: durations summed across every file,
    # not read from a single accumulator (there's no pyannote.metrics
    # accumulator for a raw duration total, so the reporter must sum the
    # per-file rows itself for these three fields specifically).
    rows = [
        _row("a", 0, 0, 0, 1, 1, region_t_and_d=2.0, region_t_minus_d=1.0, region_d_minus_t=0.0),
        _row("b", 0, 0, 0, 1, 1, region_t_and_d=3.0, region_t_minus_d=0.5, region_d_minus_t=1.0),
    ]
    der, overlap_der, der_overlap_assigned, jer = _fresh_fake_metrics()

    write_report(
        rows, der, overlap_der, der_overlap_assigned, jer,
        per_file_csv_path=tmp_path / "per_file.csv",
        summary_path=tmp_path / "summary.json",
    )

    summary = json.loads((tmp_path / "summary.json").read_text())

    assert summary["region_t_and_d"] == pytest.approx(5.0)
    assert summary["region_t_minus_d"] == pytest.approx(1.5)
    assert summary["region_d_minus_t"] == pytest.approx(1.0)


def test_counting_mae_computed_correctly(tmp_path):
    rows = [
        _row("a", 0, 0, 0, 2, 2),  # count_error = 0
        _row("b", 0, 0, 0, 2, 5),  # count_error = 3
        _row("c", 0, 0, 0, 4, 5),  # count_error = 1
    ]
    der, overlap_der, der_overlap_assigned, jer = _fresh_fake_metrics()

    write_report(
        rows, der, overlap_der, der_overlap_assigned, jer,
        per_file_csv_path=tmp_path / "per_file.csv",
        summary_path=tmp_path / "summary.json",
    )

    summary = json.loads((tmp_path / "summary.json").read_text())

    assert summary["counting_mae"] == pytest.approx((0 + 3 + 1) / 3)


def test_counting_exact_match_percentage_computed_correctly(tmp_path):
    rows = [
        _row("a", 0, 0, 0, 2, 2),  # exact match
        _row("b", 0, 0, 0, 2, 5),  # not
        _row("c", 0, 0, 0, 4, 4),  # exact match
        _row("d", 0, 0, 0, 1, 3),  # not
    ]
    der, overlap_der, der_overlap_assigned, jer = _fresh_fake_metrics()

    write_report(
        rows, der, overlap_der, der_overlap_assigned, jer,
        per_file_csv_path=tmp_path / "per_file.csv",
        summary_path=tmp_path / "summary.json",
    )

    summary = json.loads((tmp_path / "summary.json").read_text())

    assert summary["counting_exact_match_percent"] == pytest.approx(50.0)


def test_empty_rows_raises_clear_error(tmp_path):
    der, overlap_der, der_overlap_assigned, jer = _fresh_fake_metrics()

    with pytest.raises(ReporterError):
        write_report(
            [], der, overlap_der, der_overlap_assigned, jer,
            per_file_csv_path=tmp_path / "per_file.csv",
            summary_path=tmp_path / "summary.json",
        )

    assert not (tmp_path / "per_file.csv").exists()
    assert not (tmp_path / "summary.json").exists()


def test_output_files_written_to_configured_paths(tmp_path):
    rows = [_row("a", 0.1, 0.1, 0.1, 1, 1)]
    der, overlap_der, der_overlap_assigned, jer = (
        _FakeMetric(0.1), _FakeMetric(0.1), _FakeMetric(0.1), _FakeMetric(0.1),
    )

    custom_csv = tmp_path / "nested" / "custom_per_file.csv"
    custom_summary = tmp_path / "nested" / "custom_summary.json"

    write_report(
        rows, der, overlap_der, der_overlap_assigned, jer,
        per_file_csv_path=custom_csv,
        summary_path=custom_summary,
    )

    assert custom_csv.exists()
    assert custom_summary.exists()
