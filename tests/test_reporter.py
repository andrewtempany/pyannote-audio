"""Tests for TICKET-07: the reporter.

harness/reporter.py doesn't exist yet at the time these tests are written --
the first run of this suite is expected to fail on import.

All inputs here are fabricated dicts/fake metric objects matching TICKET-05's
score() output shape and the "der/overlap_der/jer accumulator instances read
via abs(metric)" contract -- no model, no audio, no real dataset.
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


def _row(uri, der, overlap_der, jer, count_ref, count_hyp):
    return {
        "uri": uri,
        "der": der,
        "overlap_der": overlap_der,
        "jer": jer,
        "count_ref": count_ref,
        "count_hyp": count_hyp,
        "count_error": abs(count_ref - count_hyp),
    }


def test_per_file_csv_has_expected_columns(tmp_path):
    rows = [_row("ES2002a", 0.1, 0.2, 0.3, 2, 2)]
    der, overlap_der, jer = _FakeMetric(0.1), _FakeMetric(0.2), _FakeMetric(0.3)

    write_report(
        rows, der, overlap_der, jer,
        per_file_csv_path=tmp_path / "per_file.csv",
        summary_path=tmp_path / "summary.json",
    )

    with open(tmp_path / "per_file.csv", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

    assert header == ["uri", "der", "overlap_der", "jer", "count_ref", "count_hyp", "count_error"]


def test_per_file_csv_values_match_input_rows(tmp_path):
    rows = [
        _row("ES2002a", 0.123456, 0.2, 0.3, 3, 5),
        _row("ES2002b", 0.4, 0.5, 0.6, 2, 2),
    ]
    der, overlap_der, jer = _FakeMetric(0.25), _FakeMetric(0.35), _FakeMetric(0.45)

    write_report(
        rows, der, overlap_der, jer,
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


def test_corpus_summary_includes_accumulated_der_overlap_der_jer(tmp_path):
    # deliberately inconsistent with what averaging the rows would produce,
    # so this test fails if the reporter recomputes instead of reading the
    # accumulator totals.
    rows = [
        _row("ES2002a", 0.1, 0.1, 0.1, 2, 2),
        _row("ES2002b", 0.9, 0.9, 0.9, 2, 2),
    ]
    der, overlap_der, jer = _FakeMetric(0.42), _FakeMetric(0.17), _FakeMetric(0.88)

    write_report(
        rows, der, overlap_der, jer,
        per_file_csv_path=tmp_path / "per_file.csv",
        summary_path=tmp_path / "summary.json",
    )

    summary = json.loads((tmp_path / "summary.json").read_text())

    assert summary["der"] == pytest.approx(0.42)
    assert summary["overlap_der"] == pytest.approx(0.17)
    assert summary["jer"] == pytest.approx(0.88)
    # sanity: these are NOT the per-file averages (0.5, 0.5, 0.5)
    assert summary["der"] != pytest.approx(0.5)


def test_counting_mae_computed_correctly(tmp_path):
    rows = [
        _row("a", 0, 0, 0, 2, 2),  # count_error = 0
        _row("b", 0, 0, 0, 2, 5),  # count_error = 3
        _row("c", 0, 0, 0, 4, 5),  # count_error = 1
    ]
    der, overlap_der, jer = _FakeMetric(0.0), _FakeMetric(0.0), _FakeMetric(0.0)

    write_report(
        rows, der, overlap_der, jer,
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
    der, overlap_der, jer = _FakeMetric(0.0), _FakeMetric(0.0), _FakeMetric(0.0)

    write_report(
        rows, der, overlap_der, jer,
        per_file_csv_path=tmp_path / "per_file.csv",
        summary_path=tmp_path / "summary.json",
    )

    summary = json.loads((tmp_path / "summary.json").read_text())

    assert summary["counting_exact_match_percent"] == pytest.approx(50.0)


def test_empty_rows_raises_clear_error(tmp_path):
    der, overlap_der, jer = _FakeMetric(0.0), _FakeMetric(0.0), _FakeMetric(0.0)

    with pytest.raises(ReporterError):
        write_report(
            [], der, overlap_der, jer,
            per_file_csv_path=tmp_path / "per_file.csv",
            summary_path=tmp_path / "summary.json",
        )

    assert not (tmp_path / "per_file.csv").exists()
    assert not (tmp_path / "summary.json").exists()


def test_output_files_written_to_configured_paths(tmp_path):
    rows = [_row("a", 0.1, 0.1, 0.1, 1, 1)]
    der, overlap_der, jer = _FakeMetric(0.1), _FakeMetric(0.1), _FakeMetric(0.1)

    custom_csv = tmp_path / "nested" / "custom_per_file.csv"
    custom_summary = tmp_path / "nested" / "custom_summary.json"

    write_report(
        rows, der, overlap_der, jer,
        per_file_csv_path=custom_csv,
        summary_path=custom_summary,
    )

    assert custom_csv.exists()
    assert custom_summary.exists()
