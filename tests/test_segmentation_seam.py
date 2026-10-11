"""Tests for TICKET-01: the CACHED_SEGMENTATION injection seam.

SpeakerDiarization.get_segmentations() checks `file[self.CACHED_SEGMENTATION]`
and skips running the segmentation model if that key is present -- but only
inside `if self.training:` (src/pyannote/audio/pipelines/speaker_diarization.py,
around get_segmentations()). During normal inference (self.training is False
by default) the check is never even reached. This is the seam the harness
epic assumed it could use to inject oracle segmentation later; these tests
first prove the bug, then prove the fix (temporarily setting
`pipeline.training = True`), then prove the fix has no other side effect.

The `pipeline`/`full_pipeline`/`one_file` fixtures used here live in
tests/conftest.py (shared with TICKET-04 and TICKET-06, which need the same
real, fully offline SpeakerDiarization pipeline).
"""

from unittest.mock import MagicMock


def test_reproduces_bug_cached_segmentation_ignored_outside_training(pipeline, one_file):
    """Documents the bug found during harness scoping: outside self.training,
    get_segmentations() ignores a pre-populated CACHED_SEGMENTATION and always
    re-runs the model. If this test starts failing, the seam may have been
    fixed upstream -- re-check TICKET-01 before deleting it."""
    dummy_segmentation = pipeline._segmentation(one_file)

    spy = MagicMock(wraps=pipeline._segmentation)
    pipeline._segmentation = spy
    one_file[pipeline.CACHED_SEGMENTATION] = dummy_segmentation

    assert pipeline.training is False

    pipeline.get_segmentations(one_file)

    assert spy.call_count == 1, (
        "expected the model to be re-run despite a cached segmentation being "
        "present (that's the bug) -- got a different call count, meaning the "
        "seam's behavior has changed since this test was written"
    )


def test_cached_segmentation_is_used_via_training_flag(pipeline, one_file):
    """The fix: temporarily setting pipeline.training = True makes
    get_segmentations() honor a pre-populated CACHED_SEGMENTATION and skip the
    model entirely. This isn't a new hack -- get_embeddings() already gates its
    own caching behind the identical `if self.training:` check, so this is an
    existing, intended seam that just wasn't documented for inference use."""
    dummy_segmentation = pipeline._segmentation(one_file)

    spy = MagicMock(wraps=pipeline._segmentation)
    pipeline._segmentation = spy
    one_file[pipeline.CACHED_SEGMENTATION] = dummy_segmentation

    pipeline.training = True
    try:
        result = pipeline.get_segmentations(one_file)
    finally:
        pipeline.training = False

    assert spy.call_count == 0
    assert result is dummy_segmentation


def test_fix_has_no_side_effect_on_normal_output(full_pipeline, one_file):
    """Flipping pipeline.training = True must not change pipeline output when
    no CACHED_SEGMENTATION is pre-populated. `training` is a plain bool flag on
    the base pyannote.pipeline.Pipeline class (unrelated to nn.Module's
    train()/eval() and its dropout/batchnorm effects), so this should hold --
    this test is what actually proves it for this fork rather than assuming it."""
    full_pipeline.training = False
    baseline_output = full_pipeline(one_file)

    full_pipeline.training = True
    try:
        fixed_output = full_pipeline(one_file)
    finally:
        full_pipeline.training = False

    assert baseline_output.speaker_diarization == fixed_output.speaker_diarization
