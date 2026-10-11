"""Tests for the oracle 2x2 combined-cells ticket
(Obsidian-Diarisation/Tickets/Open/oracle-2x2-combined-cells.md): the plumbing
needed to run oracle segmentation and oracle assignment *together*.

No new scientific code is under test here. `OracleSegmentation` and
`make_oracle_strategy` are both already implemented and independent of each
other; what's missing is that three separate call sites treat the manifest's
`condition` field as a proxy for "which segmentation source", so a combined
run can't be expressed. These tests pin each of those, plus the two
instrumentation/provenance additions.

The dangerous one is `test_combined_condition_uses_oracle_segmentation`. Today
the condition falls through run_harness.py's `else` branch to
`BaselineSegmentation()`, so a combined run would complete, write a manifest
claiming oracle segmentation, and silently score baseline -- landing somewhere
around 16% DER, which looks like a plausible experimental result rather than a
bug. That is the failure this file exists to make impossible.

At the time of writing, none of the behavior below exists: the new condition
value isn't in the vocabulary, the routing/aggregation call sites test for
exact equality against `oracle_segmentation`, `make_oracle_strategy` counts
nothing, and `--oracle-rttm` is never recorded. The first run of this suite is
expected to fail throughout.
"""

import csv
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pyannote.core import (
    Annotation,
    Segment,
    SlidingWindow,
    SlidingWindowFeature,
    Timeline,
)

from harness.aggregate_runs import build_cross_condition_table
from harness.config import HarnessConfig
from harness.refinement import make_oracle_strategy
from harness.run_manifest import VALID_CONDITIONS, write_manifest
from harness.segmentation import BaselineSegmentation, OracleSegmentation
import run_harness
from run_harness import run_harness as run_harness_fn

_COMBINED = "oracle_segmentation_assignment"


# --------------------------------------------------------------------------
# Work item 1: schema -- the new condition value
# --------------------------------------------------------------------------


def test_combined_condition_is_a_valid_condition():
    # A distinct value, not an overload of oracle_segmentation: T5's
    # cross-condition table keys on `condition` alone, so reusing the existing
    # value would silently merge two different experiments into one row.
    assert _COMBINED in VALID_CONDITIONS
    assert "oracle_segmentation" in VALID_CONDITIONS  # unchanged


def _run_config(**overrides):
    config = {
        "pipeline_config_id": "pyannote/speaker-diarization-community-1",
        "segmentation_source_id": "oracle-v2",
        "clustering_model": "pyannote-default",
        "extra_pipeline_steps": [],
        "split": "test",
        "condition": _COMBINED,
        "der_collar": 0.0,
        "der_skip_overlap": False,
        "refinement_strategy": "oracle",
        "oracle_scope": "all_pairs",
        "corpus": "AMI",
        "mic_condition": "IHM",
        "git_commit": "abc1234",
    }
    config.update(overrides)
    return config


def test_manifest_accepts_the_combined_condition(tmp_path):
    path = write_manifest(_run_config(), {"der": 0.01}, tmp_path)

    assert path.exists()


def test_cli_offers_the_combined_condition_as_a_run_condition_choice():
    # The vocabulary and the CLI's --run-condition choices must stay in step,
    # or the value is valid in the manifest but unreachable from the runner.
    action = next(
        a for a in _cli_actions() if "--run-condition" in getattr(a, "option_strings", [])
    )
    assert _COMBINED in action.choices


def _cli_actions():
    import argparse

    # Rebuild the parser the same way main() does, without running anything:
    # parse a --help-free argv and capture the parser via a patch.
    captured = {}
    real_parse_args = argparse.ArgumentParser.parse_args

    def _capture(self, *args, **kwargs):
        captured["parser"] = self
        raise _StopParsing()

    with patch.object(argparse.ArgumentParser, "parse_args", _capture):
        try:
            run_harness.main([])
        except _StopParsing:
            pass
        finally:
            argparse.ArgumentParser.parse_args = real_parse_args

    return captured["parser"]._actions


class _StopParsing(Exception):
    """Raised to abort main() once its parser has been built."""


# --------------------------------------------------------------------------
# Work item 2: segmentation routing -- the silent-failure trap
# --------------------------------------------------------------------------


def _oracle_rttm(tmp_path):
    rttm = tmp_path / "oracle.rttm"
    rttm.write_text(
        "SPEAKER fake-uri 1 0.000 1.000 <NA> <NA> A <NA> <NA>\n"
        "SPEAKER fake-uri 1 1.000 1.000 <NA> <NA> B <NA> <NA>\n"
    )
    return rttm


def _main_argv(tmp_path, condition, extra=()):
    data_root = tmp_path / "data"
    (data_root / "IHM").mkdir(parents=True, exist_ok=True)
    return [
        "--data-root", str(data_root),
        "--condition", "IHM",
        "--cache-dir", str(tmp_path / "cache"),
        "--dotenv", str(tmp_path / ".env"),
        "--per-file-csv", str(tmp_path / "per_file.csv"),
        "--summary", str(tmp_path / "summary.json"),
        "--runs-dir", str(tmp_path / "runs"),
        "--run-condition", condition,
        *extra,
    ]


def _segmentation_source_from_main(tmp_path, condition, extra=()):
    """Drive run_harness.main() far enough to see which segmentation source it
    chose, with the pipeline load and the run itself stubbed out."""
    (tmp_path / ".env").write_text("HF_TOKEN=fake\n")

    with patch("run_harness.Pipeline") as mock_pipeline_cls, \
         patch("run_harness.run_harness") as mock_run:
        mock_pipeline_cls.from_pretrained.return_value = MagicMock()
        mock_run.return_value = {"der": 0.01}
        run_harness.main(_main_argv(tmp_path, condition, extra))

    return mock_run.call_args.kwargs["segmentation_source"]


def test_combined_condition_uses_oracle_segmentation(tmp_path):
    # THE trap this ticket exists to close. An exact-equality test against
    # "oracle_segmentation" lets the combined condition fall through to
    # BaselineSegmentation(), which fails silently: the run completes, the
    # manifest claims oracle segmentation, and baseline gets scored.
    source = _segmentation_source_from_main(
        tmp_path, _COMBINED,
        extra=["--refinement-strategy", "oracle",
               "--oracle-rttm", str(_oracle_rttm(tmp_path))],
    )

    assert isinstance(source, OracleSegmentation)


def test_combined_condition_requires_oracle_rttm(tmp_path, capsys):
    # The companion required-argument check has to widen with the routing
    # test, or the combined condition reaches OracleSegmentation with no
    # reference to build ground truth from.
    #
    # The assertion checks argparse's *message*, not just SystemExit: before
    # the condition joins --run-condition's choices, argparse exits anyway
    # with "invalid choice", which would make a bare pytest.raises(SystemExit)
    # pass vacuously without the required-argument check existing at all.
    (tmp_path / ".env").write_text("HF_TOKEN=fake\n")

    with patch("run_harness.Pipeline") as mock_pipeline_cls, \
         patch("run_harness.run_harness"):
        mock_pipeline_cls.from_pretrained.return_value = MagicMock()
        with pytest.raises(SystemExit):
            run_harness.main(
                _main_argv(tmp_path, _COMBINED,
                           extra=["--refinement-strategy", "oracle"])
            )

    message = capsys.readouterr().err
    assert "--oracle-rttm" in message
    assert "invalid choice" not in message


def test_plain_oracle_segmentation_still_routes_to_oracle(tmp_path):
    source = _segmentation_source_from_main(
        tmp_path, "oracle_segmentation",
        extra=["--oracle-rttm", str(_oracle_rttm(tmp_path))],
    )

    assert isinstance(source, OracleSegmentation)


def test_baseline_condition_still_routes_to_baseline(tmp_path):
    source = _segmentation_source_from_main(tmp_path, "baseline")

    assert isinstance(source, BaselineSegmentation)


def test_oracle_assignment_alone_still_routes_to_baseline(tmp_path):
    # oracle *assignment* under baseline segmentation is the existing
    # measured cell -- widening the routing test must not capture it.
    source = _segmentation_source_from_main(
        tmp_path, "oracle_assignment",
        extra=["--refinement-strategy", "oracle"],
    )

    assert isinstance(source, BaselineSegmentation)


# --------------------------------------------------------------------------
# Work items 3 and 4: aggregation -- scope suppression and budget deltas
# --------------------------------------------------------------------------

_COLUMNS = [
    "run_id", "created_at", "condition", "oracle_scope", "counts_toward_results",
    "notes", "der", "der_overlap_system", "der_overlap_assigned",
    "missed_detection", "false_alarm", "confusion",
]


def _row(run_id, condition, der, oracle_scope="", counts_toward_results="True",
         created_at="2026-09-20T00:00:00", **extra):
    row = {
        "run_id": run_id,
        "created_at": created_at,
        "condition": condition,
        "oracle_scope": oracle_scope,
        "counts_toward_results": counts_toward_results,
        "notes": "",
        "der": der,
        "der_overlap_system": 0.0,
        "der_overlap_assigned": 0.0,
        "missed_detection": 0.0,
        "false_alarm": 0.0,
        "confusion": 0.0,
    }
    row.update(extra)
    return row


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _combined_batch():
    """The 2x2 as it looks once both new cells are measured."""
    return [
        _row("r1", "baseline", 0.1705),
        _row("r2", "oracle_segmentation", 0.0396),
        _row("r3", "oracle_assignment", 0.1608, oracle_scope="all_pairs"),
        _row("r4", _COMBINED, 0.0100, oracle_scope="all_pairs"),
        _row("r5", _COMBINED, 0.0150, oracle_scope="overlap_degraded"),
    ]


def test_combined_condition_scopes_render_as_distinct_rows(tmp_path):
    # Work item 3. The two new runs differ ONLY by scope, so if
    # _effective_scope blanks it for this condition they become
    # indistinguishable rows.
    csv_path = _write_csv(tmp_path / "comparison.csv", _combined_batch())

    table = build_cross_condition_table(csv_path)

    combined_lines = [
        line for line in table.splitlines() if _COMBINED in line and "|" in line
    ]
    assert len(combined_lines) == 2
    assert any("all_pairs" in line for line in combined_lines)
    assert any("overlap_degraded" in line for line in combined_lines)


def test_combined_budget_delta_is_measured_against_oracle_segmentation(tmp_path):
    # Work item 4. `baseline - combined` (0.1705 - 0.0100 = 0.1605) would
    # double-count the 13.09 pt segmentation budget already reported on its
    # own line. The meaningful figure is the assignment budget *given clean
    # segmentation*: oracle_segmentation - combined = 0.0396 - 0.0100.
    csv_path = _write_csv(tmp_path / "comparison.csv", _combined_batch())

    table = build_cross_condition_table(csv_path)

    assert "+0.0296" in table
    assert "0.1605" not in table


def test_combined_delta_line_names_its_reference_condition(tmp_path):
    # The reader must not have to reverse-engineer which row the delta is
    # against, given the other two deltas use a different reference.
    csv_path = _write_csv(tmp_path / "comparison.csv", _combined_batch())

    table = build_cross_condition_table(csv_path)

    combined_delta = next(
        line for line in table.splitlines()
        if line.startswith("- ") and "0.0296" in line
    )
    assert "oracle_segmentation" in combined_delta


def test_existing_budget_deltas_keep_baseline_as_reference(tmp_path):
    # Per-condition references must not disturb the two existing lines.
    csv_path = _write_csv(tmp_path / "comparison.csv", _combined_batch())

    table = build_cross_condition_table(csv_path)

    seg_delta = next(
        line for line in table.splitlines() if "Downstream budget" in line
    )
    assert "+0.1309" in seg_delta
    assign_delta = next(
        line for line in table.splitlines()
        if "Assignment budget" in line and "oracle segmentation" not in line
    )
    assert "+0.0097" in assign_delta


def test_reference_der_uses_the_most_recent_matching_run(tmp_path):
    # runs/ holds two generations of oracle_segmentation runs, both marked
    # counts_toward_results: a stale pre-seam-fix one (segmentation id
    # "oracle", DER 0.1705) and the current one ("oracle-v2", 0.0396). The
    # stale row sorts first, so taking the FIRST match would measure the
    # combined cell against 0.1705 and yield a meaningless ~0.16 delta while
    # looking perfectly plausible.
    rows = [
        _row("r1", "baseline", 0.1705, created_at="2026-09-16T21:39:18"),
        _row("stale", "oracle_segmentation", 0.1705, created_at="2026-09-17T01:36:30"),
        _row("current", "oracle_segmentation", 0.0396, created_at="2026-09-18T12:53:23"),
        _row("r4", _COMBINED, 0.0100, oracle_scope="all_pairs",
             created_at="2026-09-20T12:00:00"),
    ]
    csv_path = _write_csv(tmp_path / "comparison.csv", rows)

    table = build_cross_condition_table(csv_path)

    # 0.0396 - 0.0100 = +0.0296, against the CURRENT oracle_segmentation run
    assert "+0.0296" in table
    # 0.1705 - 0.0100 = +0.1605 would mean it used the stale row
    assert "+0.1605" not in table


def test_missing_oracle_segmentation_row_says_so_explicitly(tmp_path):
    # Not a silent fallback to baseline, which would emit the double-counted
    # ~16 pt figure as though it were the assignment budget.
    rows = [
        _row("r1", "baseline", 0.1705),
        _row("r2", _COMBINED, 0.0100, oracle_scope="all_pairs"),
    ]
    csv_path = _write_csv(tmp_path / "comparison.csv", rows)

    table = build_cross_condition_table(csv_path)

    assert "0.1605" not in table
    combined_line = next(
        line for line in table.splitlines()
        if line.startswith("- ") and _COMBINED in line
    )
    assert "cannot be computed" in combined_line.lower()


# --------------------------------------------------------------------------
# Work item 5: pair-count instrumentation
# --------------------------------------------------------------------------


def _segmentations(binary_masks, chunk_duration=1.0):
    data = np.stack(binary_masks, axis=0).astype(float)
    sliding_window = SlidingWindow(start=0.0, duration=chunk_duration, step=chunk_duration)
    return SlidingWindowFeature(data, sliding_window)


def _call(strategy, hard_clusters, segmentations):
    num_chunks, local_num_speakers = hard_clusters.shape
    return strategy(
        np.zeros((num_chunks, local_num_speakers, 4)),
        hard_clusters,
        None,
        None,
        segmentations,
    )


def test_oracle_strategy_reports_five_exit_path_counts():
    # Two chunks, both fully active, reference speaker A then B, clusters
    # swapped -- both pairs are relabelled, nothing skipped.
    reference = Annotation()
    reference[Segment(0.0, 1.0)] = "A"
    reference[Segment(1.0, 2.0)] = "B"
    segmentations = _segmentations([np.ones((4, 1)), np.ones((4, 1))])

    strategy = make_oracle_strategy(reference, oracle_scope="all_pairs")
    _call(strategy, np.array([[1], [0]]), segmentations)

    # `empty_support` is a fifth path, split out of `no_reference_overlap`
    # (which used to absorb both no-active-frames and no-reference-speaker).
    # Both pairs here are fully active, so it must read 0.
    assert strategy.counts == {
        "relabelled": 2,
        "out_of_scope": 0,
        "empty_support": 0,
        "no_reference_overlap": 0,
        "unmapped_speaker": 0,
    }


def test_oracle_strategy_counts_pairs_with_no_reference_overlap():
    # Chunk 1's support falls in a reference gap, so
    # _dominant_reference_speaker returns None for it.
    reference = Annotation()
    reference[Segment(0.0, 1.0)] = "A"
    segmentations = _segmentations([np.ones((4, 1)), np.ones((4, 1))])

    strategy = make_oracle_strategy(reference, oracle_scope="all_pairs")
    _call(strategy, np.array([[0], [1]]), segmentations)

    assert strategy.counts["no_reference_overlap"] == 1


def test_oracle_strategy_counts_out_of_scope_pairs_under_overlap_degraded():
    # Neither pair is predominantly overlapping speech, so under the narrower
    # scope both are skipped as out of scope rather than relabelled.
    reference = Annotation()
    reference[Segment(0.0, 1.0)] = "A"
    reference[Segment(1.0, 2.0)] = "B"
    segmentations = _segmentations([np.ones((4, 1)), np.ones((4, 1))])

    strategy = make_oracle_strategy(reference, oracle_scope="overlap_degraded")
    _call(strategy, np.array([[1], [0]]), segmentations)

    assert strategy.counts["out_of_scope"] == 2
    assert strategy.counts["relabelled"] == 0


def test_oracle_strategy_counts_unmapped_dominant_speakers():
    # Three reference speakers but only two produced clusters, so at least one
    # dominant reference speaker maps to no cluster. That path is the one
    # predicted to hold DER off the 0.51% floor, so it must be counted
    # separately rather than collapsed into a single skip total.
    reference = Annotation()
    reference[Segment(0.0, 1.0)] = "A"
    reference[Segment(1.0, 2.0)] = "B"
    reference[Segment(2.0, 3.0)] = "C"
    segmentations = _segmentations(
        [np.ones((4, 1)), np.ones((4, 1)), np.ones((4, 1))]
    )

    strategy = make_oracle_strategy(reference, oracle_scope="all_pairs")
    _call(strategy, np.array([[0], [1], [0]]), segmentations)

    assert strategy.counts["unmapped_speaker"] >= 1


def test_oracle_strategy_counts_every_pair_exactly_once():
    reference = Annotation()
    reference[Segment(0.0, 1.0)] = "A"
    reference[Segment(1.0, 2.0)] = "B"
    segmentations = _segmentations([np.ones((4, 2)), np.ones((4, 2))])

    strategy = make_oracle_strategy(reference, oracle_scope="all_pairs")
    _call(strategy, np.array([[0, 1], [1, 0]]), segmentations)

    assert sum(strategy.counts.values()) == 4


def test_counts_reset_between_calls():
    # The harness rebuilds the closure per file, but a strategy invoked twice
    # must not accumulate across calls or the reported totals are meaningless.
    reference = Annotation()
    reference[Segment(0.0, 1.0)] = "A"
    reference[Segment(1.0, 2.0)] = "B"
    segmentations = _segmentations([np.ones((4, 1)), np.ones((4, 1))])

    strategy = make_oracle_strategy(reference, oracle_scope="all_pairs")
    _call(strategy, np.array([[1], [0]]), segmentations)
    _call(strategy, np.array([[1], [0]]), segmentations)

    assert sum(strategy.counts.values()) == 2


def test_harness_reports_pair_counts_summed_over_files(tmp_path, capsys):
    # Criterion 5 wants the totals for a whole run, so the per-file closures'
    # counts have to be accumulated -- each file gets a fresh closure.
    config = _config_for_run(tmp_path)
    fake_pipeline = MagicMock()

    rows = [_fake_adapter_row(), ("fake-uri-2",) + _fake_adapter_row()[1:]]

    with patch("run_harness.AMIDatasetAdapter") as mock_adapter_cls, \
         patch("run_harness.Runner") as mock_runner_cls, \
         patch("run_harness.make_oracle_strategy") as mock_make_oracle:
        mock_adapter_cls.return_value = rows
        mock_runner = MagicMock()
        mock_runner.run.return_value = Annotation()
        mock_runner_cls.return_value = mock_runner

        def _fake_strategy(reference, oracle_scope="all_pairs"):
            strategy = MagicMock()
            strategy.counts = {
                "relabelled": 3, "out_of_scope": 1,
                "no_reference_overlap": 0, "unmapped_speaker": 2,
            }
            return strategy

        mock_make_oracle.side_effect = _fake_strategy

        run_harness_fn(
            config, fake_pipeline, "test-pipeline", BaselineSegmentation(),
            tmp_path / "per_file.csv", tmp_path / "summary.json",
            refinement_strategy="oracle",
            oracle_scope="all_pairs",
        )

    output = capsys.readouterr().out
    assert "relabelled=6" in output       # 3 per file, two files
    assert "unmapped_speaker=4" in output  # 2 per file
    assert "out_of_scope=2" in output


def test_non_oracle_run_reports_no_pair_counts(tmp_path, capsys):
    config = _config_for_run(tmp_path)
    fake_pipeline = MagicMock()
    uri, reference, uem = _fake_adapter_row()

    with patch("run_harness.AMIDatasetAdapter") as mock_adapter_cls, \
         patch("run_harness.Runner") as mock_runner_cls:
        mock_adapter_cls.return_value = [(uri, reference, uem)]
        mock_runner = MagicMock()
        mock_runner.run.return_value = Annotation()
        mock_runner_cls.return_value = mock_runner

        run_harness_fn(
            config, fake_pipeline, "test-pipeline", BaselineSegmentation(),
            tmp_path / "per_file.csv", tmp_path / "summary.json",
            refinement_strategy="identity",
        )

    assert "oracle pair disposition" not in capsys.readouterr().out


# --------------------------------------------------------------------------
# Work item 6: provenance -- record --oracle-rttm
# --------------------------------------------------------------------------


def test_manifest_records_the_oracle_rttm_path(tmp_path):
    import json

    config = _config_for_run(tmp_path)
    uri, reference, uem = _fake_adapter_row()
    fake_pipeline = MagicMock()
    runs_dir = tmp_path / "runs"

    with patch("run_harness.AMIDatasetAdapter") as mock_adapter_cls, \
         patch("run_harness.Runner") as mock_runner_cls:
        mock_adapter_cls.return_value = [(uri, reference, uem)]
        mock_runner = MagicMock()
        mock_runner.run.return_value = Annotation()
        mock_runner_cls.return_value = mock_runner

        run_harness_fn(
            config, fake_pipeline, "test-pipeline", BaselineSegmentation(),
            tmp_path / "per_file.csv", tmp_path / "summary.json",
            runs_dir=runs_dir,
            run_condition=_COMBINED,
            oracle_rttm="path/to/only_words.rttm",
        )

    manifest = json.loads(next(runs_dir.glob("*.json")).read_text())
    assert manifest["run_config"]["oracle_rttm"] == "path/to/only_words.rttm"


def test_oracle_rttm_field_is_additive_and_defaults_to_none(tmp_path):
    # Existing manifests have no oracle_rttm key at all; adding it must not
    # become a required field, or every previously-written manifest and every
    # baseline run breaks.
    import json

    config = _config_for_run(tmp_path)
    uri, reference, uem = _fake_adapter_row()
    fake_pipeline = MagicMock()
    runs_dir = tmp_path / "runs"

    with patch("run_harness.AMIDatasetAdapter") as mock_adapter_cls, \
         patch("run_harness.Runner") as mock_runner_cls:
        mock_adapter_cls.return_value = [(uri, reference, uem)]
        mock_runner = MagicMock()
        mock_runner.run.return_value = Annotation()
        mock_runner_cls.return_value = mock_runner

        run_harness_fn(
            config, fake_pipeline, "test-pipeline", BaselineSegmentation(),
            tmp_path / "per_file.csv", tmp_path / "summary.json",
            runs_dir=runs_dir,
        )

    manifest = json.loads(next(runs_dir.glob("*.json")).read_text())
    assert manifest["run_config"]["oracle_rttm"] is None
    # and a manifest written without the field at all still validates
    write_manifest(_run_config(condition="baseline"), {"der": 0.17}, runs_dir)


def _config_for_run(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir(exist_ok=True)
    return HarnessConfig.load(
        data_root=data_root, condition="IHM", cache_dir=tmp_path / "cache"
    )


def _fake_adapter_row():
    reference = Annotation()
    reference[Segment(0.0, 1.0)] = "A"
    reference[Segment(1.0, 2.0)] = "B"
    return "fake-uri", reference, Timeline([Segment(0.0, 2.0)])
