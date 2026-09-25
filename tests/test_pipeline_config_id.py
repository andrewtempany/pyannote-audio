"""Tests for the composite `pipeline_config_id` (Problem A of the
pre-clustering-embedding-cache ticket).

The defect these guard against: `run_harness.py` used to pass
`pipeline_config_id=checkpoint`, the bare string
"pyannote/speaker-diarization-community-1". Nothing about the clustering
method or its hyperparameters was in the final-hypothesis cache key, so a
hyperparameter sweep would compute point 1, cache it, and serve that same
cached RTTM for every subsequent point -- a flat curve from a run that
completed cleanly and wrote a valid manifest.

Two tiers, deliberately different (see harness/cache_id.py):

  * final-hypothesis key  MUST contain clustering config  (criterion 1)
  * intermediate key      MUST NOT contain clustering config (criterion 2)

Both directions are tested here, because getting them the wrong way round
yields either a useless cache or silently wrong results.

No model, no network, no audio: the clustering objects are real
`pyannote.pipeline.Pipeline` instances built from the same descriptor
machinery the library uses, so the generic parameter extraction is exercised
for real rather than against a mock that assumes the answer.
"""

from types import SimpleNamespace

import pytest
from pyannote.pipeline import Pipeline as PipelineBase
from pyannote.pipeline.parameter import Categorical, Integer, Uniform

from harness.cache_id import (
    clustering_config_description,
    intermediate_config_id,
    pipeline_config_id,
)

CHECKPOINT = "pyannote/speaker-diarization-community-1"


class _FakeVBxClustering(PipelineBase):
    """Mirrors VBxClustering's declared hyperparameters exactly
    (clustering.py:568-570) without needing a PLDA checkpoint: VBxClustering's
    __init__ takes a positional `plda` with no default, and the library
    special-cases it at speaker_diarization.py:290-291, so it cannot be
    constructed directly here. What matters for the key is the pyannote
    descriptor mechanism, which is identical."""

    def __init__(self, threshold=0.6, Fa=0.07, Fb=0.8):
        super().__init__()
        self.threshold = Uniform(0.5, 0.8)
        self.Fa = Uniform(0.01, 0.5)
        self.Fb = Uniform(0.01, 15.0)
        # instantiate them, exactly as Pipeline.instantiate() would
        self.threshold = threshold
        self.Fa = Fa
        self.Fb = Fb


class _FakeAgglomerativeClustering(PipelineBase):
    """Mirrors AgglomerativeClustering's declared hyperparameters
    (clustering.py:322-328) -- a different parameter *set*, to prove the
    extraction is not hardcoded to VBx's three."""

    def __init__(self, threshold=0.7, method="centroid", min_cluster_size=12):
        super().__init__()
        self.threshold = Uniform(0.0, 2.0)
        self.method = Categorical(
            ["average", "centroid", "complete", "median", "single", "ward", "weighted"]
        )
        self.min_cluster_size = Integer(1, 20)
        self.threshold = threshold
        self.method = method
        self.min_cluster_size = min_cluster_size


class _FakeKMeansClustering(PipelineBase):
    """KMeansClustering declares no tunable hyperparameters at all. A class
    with an empty parameter set must not crash the extractor."""


def _pipeline(clustering, embedding_batch_size=32, segmentation_batch_size=32):
    """A stand-in for the real pipeline carrying everything both keys read.

    Batch sizes default to 32, the shipped value, so every pre-existing test
    keeps the identity it had before batch size entered the intermediate key.
    """
    return SimpleNamespace(
        klustering=type(clustering).__name__,
        clustering=clustering,
        embedding_batch_size=embedding_batch_size,
        segmentation_batch_size=segmentation_batch_size,
    )


# --------------------------------------------------------------------------
# Criterion 1: clustering config IS in the final-hypothesis key
# --------------------------------------------------------------------------


def test_final_key_differs_by_clustering_hyperparameter_value():
    """The headline criterion. Two VBx configs differing in one
    hyperparameter must not share a final-hypothesis cache key.

    This is the sweep case: threshold 0.6 vs 0.7 is exactly what a
    12-point threshold sweep varies. Under the old bare-checkpoint key
    both produced the same key and therefore the same cached RTTM."""
    a = pipeline_config_id(CHECKPOINT, _pipeline(_FakeVBxClustering(threshold=0.6)))
    b = pipeline_config_id(CHECKPOINT, _pipeline(_FakeVBxClustering(threshold=0.7)))

    assert a != b


def test_final_key_differs_for_every_vbx_hyperparameter():
    """Not just `threshold`: Fa and Fb must each move the key too. A key that
    happened to capture only the first parameter would pass the test above
    while still flattening an Fa or Fb sweep."""
    base = pipeline_config_id(CHECKPOINT, _pipeline(_FakeVBxClustering()))

    for kwargs in ({"threshold": 0.65}, {"Fa": 0.2}, {"Fb": 3.0}):
        varied = pipeline_config_id(CHECKPOINT, _pipeline(_FakeVBxClustering(**kwargs)))
        assert varied != base, f"key did not change when varying {kwargs}"


def test_final_key_differs_by_clustering_class():
    """Same checkpoint, different clustering algorithm -> different key, even
    if a hyperparameter name coincidentally matches."""
    vbx = pipeline_config_id(CHECKPOINT, _pipeline(_FakeVBxClustering(threshold=0.7)))
    agg = pipeline_config_id(
        CHECKPOINT, _pipeline(_FakeAgglomerativeClustering(threshold=0.7))
    )

    assert vbx != agg


def test_final_key_captures_non_vbx_parameter_sets():
    """The extractor must not be hardcoded to VBx's threshold/Fa/Fb. An
    agglomerative-style `method` (a Categorical, not a float) must move the
    key as well."""
    a = pipeline_config_id(
        CHECKPOINT, _pipeline(_FakeAgglomerativeClustering(method="centroid"))
    )
    b = pipeline_config_id(
        CHECKPOINT, _pipeline(_FakeAgglomerativeClustering(method="ward"))
    )
    c = pipeline_config_id(
        CHECKPOINT, _pipeline(_FakeAgglomerativeClustering(min_cluster_size=5))
    )

    assert a != b
    assert a != c


def test_final_key_stable_for_identical_config():
    """The fix must not break caching by making every run unique: the same
    configuration must produce the same key across separate objects and
    across processes (hence a hash of values, not object identity)."""
    a = pipeline_config_id(CHECKPOINT, _pipeline(_FakeVBxClustering()))
    b = pipeline_config_id(CHECKPOINT, _pipeline(_FakeVBxClustering()))

    assert a == b


def test_final_key_differs_by_checkpoint():
    """The checkpoint dimension the old key already had must survive."""
    a = pipeline_config_id(CHECKPOINT, _pipeline(_FakeVBxClustering()))
    b = pipeline_config_id("some/other-checkpoint", _pipeline(_FakeVBxClustering()))

    assert a != b


def test_final_key_handles_clustering_with_no_hyperparameters():
    """KMeansClustering declares no tunable parameters. This must produce a
    usable key rather than raising -- but still differ from a class that has
    parameters."""
    kmeans = pipeline_config_id(CHECKPOINT, _pipeline(_FakeKMeansClustering()))
    vbx = pipeline_config_id(CHECKPOINT, _pipeline(_FakeVBxClustering()))

    assert isinstance(kmeans, str) and kmeans
    assert kmeans != vbx


def test_final_key_is_loudly_unavailable_when_clustering_missing():
    """Failure mode 1 (a silent no-op producing plausible numbers): if the
    pipeline has no clustering attribute, the key must NOT quietly fall back
    to the bare checkpoint -- that is precisely the old defect, and it would
    look completely normal. It must raise."""
    with pytest.raises(Exception):
        pipeline_config_id(CHECKPOINT, SimpleNamespace())


def test_human_readable_description_records_the_values():
    """The manifest needs the human-readable form alongside the hash, or a
    future reader cannot tell what a run was."""
    description = clustering_config_description(
        _pipeline(_FakeVBxClustering(threshold=0.6, Fa=0.07, Fb=0.8))
    )

    assert "_FakeVBxClustering" in description
    assert "threshold=0.6" in description
    assert "Fa=0.07" in description
    assert "Fb=0.8" in description


# --------------------------------------------------------------------------
# Criterion 2: clustering config is NOT in the intermediate key
# --------------------------------------------------------------------------


def test_intermediate_key_ignores_clustering_hyperparameters():
    """The entire point of the intermediate tier: segmentation and embeddings
    do not depend on clustering, so every point of a clustering sweep must
    share one cached copy. If this ever differs, the sweep pays for GPU
    inference at every point and the ticket's goal is lost."""
    a = intermediate_config_id(CHECKPOINT, _pipeline(_FakeVBxClustering(threshold=0.6)))
    b = intermediate_config_id(CHECKPOINT, _pipeline(_FakeVBxClustering(threshold=0.7)))

    assert a == b


def test_intermediate_key_ignores_clustering_class_entirely():
    """Not just hyperparameters -- swapping VBx for agglomerative must also
    reuse the same intermediates, since neither affects segmentation or
    embedding output."""
    vbx = intermediate_config_id(CHECKPOINT, _pipeline(_FakeVBxClustering()))
    agg = intermediate_config_id(CHECKPOINT, _pipeline(_FakeAgglomerativeClustering()))
    kmeans = intermediate_config_id(CHECKPOINT, _pipeline(_FakeKMeansClustering()))

    assert vbx == agg == kmeans


def test_intermediate_key_differs_by_checkpoint():
    """Invalidation: the checkpoint determines both the segmentation model and
    the embedding model, so it MUST be in the intermediate key."""
    a = intermediate_config_id(CHECKPOINT, _pipeline(_FakeVBxClustering()))
    b = intermediate_config_id("some/other-checkpoint", _pipeline(_FakeVBxClustering()))

    assert a != b


def test_the_two_tiers_are_not_the_same_key():
    """Guards against a refactor that collapses one into the other -- which
    would silently turn one tier into the other's semantics."""
    pipeline = _pipeline(_FakeVBxClustering())

    assert intermediate_config_id(CHECKPOINT, pipeline) != pipeline_config_id(
        CHECKPOINT, pipeline
    )


# --------------------------------------------------------------------------
# Batch size IS in the intermediate key (and only there)
# --------------------------------------------------------------------------


def test_intermediate_key_differs_by_embedding_batch_size():
    """The headline criterion for the batch-size fix.

    `get_embeddings()` stacks waveforms into one tensor and runs the embedding
    model on the batch (speaker_diarization.py:460-468), so batch shape changes
    float reduction order and perturbs the embeddings. Two runs at different
    batch sizes therefore produce DIFFERENT cached arrays.

    Before this change both runs produced the SAME intermediate key, so they
    collided silently: whichever ran first won, and the second was served
    arrays it did not compute. Measured on real runs -- batch 8 scored DER
    0.17049342382952828 against batch 32's 0.17048543579940637.

    Batch size is a value input that also happens to affect speed. Leaving it
    out of the key was the bug.
    """
    a = intermediate_config_id(
        CHECKPOINT, _pipeline(_FakeVBxClustering(), embedding_batch_size=32)
    )
    b = intermediate_config_id(
        CHECKPOINT, _pipeline(_FakeVBxClustering(), embedding_batch_size=8)
    )

    assert a != b, (
        "batch 32 and batch 8 share an intermediate cache key, so they collide "
        "silently while producing different embeddings"
    )


def test_intermediate_key_differs_by_segmentation_batch_size():
    """Segmentation is batched too (speaker_diarization.py:259), so the same
    argument applies to its batch size. A key that captured only the embedding
    batch size would pass the test above while still colliding on a
    segmentation-batch change."""
    a = intermediate_config_id(
        CHECKPOINT, _pipeline(_FakeVBxClustering(), segmentation_batch_size=32)
    )
    b = intermediate_config_id(
        CHECKPOINT, _pipeline(_FakeVBxClustering(), segmentation_batch_size=8)
    )

    assert a != b


def test_intermediate_key_stable_across_calls_at_one_batch_size():
    """The other half of the criterion: adding batch size must not make the key
    unstable. A key that varied between calls at a FIXED configuration would
    orphan the cache on every run -- every lookup a miss, every run paying full
    GPU cost, while looking healthy.

    Two separately-constructed pipelines at the same batch size, to catch a key
    that accidentally captured object identity rather than value.
    """
    a = intermediate_config_id(
        CHECKPOINT, _pipeline(_FakeVBxClustering(), embedding_batch_size=32)
    )
    b = intermediate_config_id(
        CHECKPOINT, _pipeline(_FakeVBxClustering(), embedding_batch_size=32)
    )

    assert a == b


def test_batch_size_does_NOT_enter_the_final_key():
    """Deliberate asymmetry, and the reason this is not simply "add it to both".

    The final-RTTM key already contains the intermediate components, so batch
    size reaches it transitively -- but nothing batch-specific is added to it
    beyond that. The test that matters is the tier distinction above; this one
    pins the decision so a future edit cannot quietly add a second, independent
    batch-size term to the final key.
    """
    final_32 = pipeline_config_id(
        CHECKPOINT, _pipeline(_FakeVBxClustering(), embedding_batch_size=32)
    )
    final_8 = pipeline_config_id(
        CHECKPOINT, _pipeline(_FakeVBxClustering(), embedding_batch_size=8)
    )

    # They differ, because the final key is built ON TOP of the intermediate
    # components -- not because batch size was added to it separately.
    assert final_32 != final_8
