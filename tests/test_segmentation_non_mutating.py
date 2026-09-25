"""Tests that OracleSegmentation.populate() never writes `annotation` onto the
caller's file dict.

Background (see Obsidian-Diarisation/Tickets/Open/oracle-experiment-closeout.md,
item 2): `populate()` used to set `file["annotation"] = reference` as
scaffolding for `oracle_segmentation()` and restore it in a `finally`. The
restore worked, and `runner.py` calls the pipeline only after `populate()` has
returned, so no leak ever reached `SpeakerDiarization.apply()`. The
mutate-and-restore dance was nonetheless fragile: it depended on the `finally`
and on nothing observing `file` in between.

Asserting the post-state of `file` cannot detect the difference -- the restore
already makes that assertion pass. So these tests record every `__setitem__`
instead, which is what makes them fail on the pre-change code.
"""

from pathlib import Path
from types import SimpleNamespace

from pyannote.core import SlidingWindow
from pyannote.database.util import load_rttm

from harness.segmentation import OracleSegmentation

_ORACLE_FIXTURE_RTTM = Path(__file__).parent / "data" / "oracle_only_words.rttm"


class _FakeSegmentationInference:
    def __init__(self):
        self.step = 1.0
        self.duration = 2.0
        self.model = SimpleNamespace(
            receptive_field=SlidingWindow(start=0.0, step=0.5, duration=0.5)
        )


class _FakePipelineForOracle:
    def __init__(self):
        self.training = False
        self._segmentation = _FakeSegmentationInference()

    @property
    def CACHED_SEGMENTATION(self):
        return "training_cache/segmentation"


class _RecordingDict(dict):
    """A dict that remembers every key ever assigned, even if later removed.

    The point of the exercise: a plain `dict` cannot distinguish "never set"
    from "set then restored", and the restore makes those two look identical
    from the outside once `populate()` returns.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.assigned_keys = []

    def __setitem__(self, key, value):
        self.assigned_keys.append(key)
        super().__setitem__(key, value)


def test_populate_never_assigns_annotation_onto_the_caller_file():
    """Fails on the pre-change code: it recorded an `annotation` assignment
    even though the `finally` removed the key again before returning."""
    pipeline = _FakePipelineForOracle()
    reference_lookup = load_rttm(str(_ORACLE_FIXTURE_RTTM))
    file = _RecordingDict({"uri": "oracle_fixture", "duration": 6.0})

    OracleSegmentation(reference_lookup=reference_lookup).populate(pipeline, file)

    assert "annotation" not in file.assigned_keys
    # the one write it SHOULD make:
    assert pipeline.CACHED_SEGMENTATION in file.assigned_keys


def test_populate_result_is_unchanged_by_the_shallow_copy():
    """The scaffolding change must be invisible in the output. Same assertion
    as test_segmentation.py's fixture test, restated here so a regression in
    the copy shows up beside the mutation test rather than only elsewhere."""
    from pyannote.audio.pipelines.utils.oracle import oracle_segmentation

    pipeline = _FakePipelineForOracle()
    reference_lookup = load_rttm(str(_ORACLE_FIXTURE_RTTM))
    uri = "oracle_fixture"
    file = {"uri": uri, "duration": 6.0}

    OracleSegmentation(reference_lookup=reference_lookup).populate(pipeline, file)
    result = file[pipeline.CACHED_SEGMENTATION]

    expected = oracle_segmentation(
        {"uri": uri, "duration": 6.0, "annotation": reference_lookup[uri]},
        SlidingWindow(
            step=pipeline._segmentation.step, duration=pipeline._segmentation.duration
        ),
        pipeline._segmentation.model.receptive_field,
    )

    assert result.data.shape == expected.data.shape
    assert (result.data == expected.data).all()


def test_populate_preserves_a_pre_existing_annotation_key():
    """If a caller legitimately supplied its own `annotation`, populate() must
    leave that value alone rather than overwrite-then-restore it."""
    pipeline = _FakePipelineForOracle()
    reference_lookup = load_rttm(str(_ORACLE_FIXTURE_RTTM))
    caller_annotation = object()
    file = {
        "uri": "oracle_fixture",
        "duration": 6.0,
        "annotation": caller_annotation,
    }

    OracleSegmentation(reference_lookup=reference_lookup).populate(pipeline, file)

    assert file["annotation"] is caller_annotation


def test_guard_is_retained_and_docstring_describes_refusal_not_leak_prevention():
    """The `_expects_num_speakers` guard stays. It does not prevent a leak --
    the sequencing in runner.py already does that -- but it refuses a
    combination the harness cannot currently run, with a clearer error than
    the library's own `ValueError`. The docstring must not claim the leak
    story, which is what sent a previous reader down the wrong path.
    """
    import pytest

    pipeline = _FakePipelineForOracle()
    pipeline._expects_num_speakers = True
    reference_lookup = load_rttm(str(_ORACLE_FIXTURE_RTTM))
    file = {"uri": "oracle_fixture", "duration": 6.0}

    with pytest.raises(RuntimeError):
        OracleSegmentation(reference_lookup=reference_lookup).populate(pipeline, file)

    # The docstring retains the old phrase only inside its own retraction, so
    # assert on the framing instead of the words: the guard must be described
    # as refusing an unsupported combination, and the sequencing claim must be
    # explicitly marked wrong wherever it still appears.
    docstring = OracleSegmentation.__doc__ or ""
    assert "not leak prevention" in docstring
    if "before it's restored" in docstring:
        assert "was\n    wrong" in docstring or "was wrong" in docstring
