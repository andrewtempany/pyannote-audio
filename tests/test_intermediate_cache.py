"""Tests for the pre-clustering intermediate cache (harness/intermediate_cache.py).

Criteria 3 and 4 of the pre-clustering-embedding-cache ticket are exercised
here at the largest scale achievable without real audio or a GPU. The fake
pipeline below is not a mock that assumes the answer: it reimplements the
training-gated cache reads and writes from
`src/pyannote/audio/pipelines/speaker_diarization.py` at the verified line
numbers, so a regression in the seam (wrong key name, wrong gate, populated
value not reaching the read) makes these tests fail rather than pass.

What it reproduces, and from where:

  * `CACHED_SEGMENTATION` == "training_cache/segmentation"   (:317-319)
  * read/write of it, gated on `self.training`               (:337-344)
  * read of "training_cache/embeddings", gated on training   (:377-386)
  * write of it, powerset branch: {"embeddings": ...} only   (:483-492)
  * clustering consulted only AFTER both, so clustering
    configuration cannot affect either intermediate          (:609/:647/:656)

community-1 is powerset (`powerset_max_classes=2` in the checkpoint
metadata), so the fake declares `powerset = True` and stores the powerset
dict shape.

The 16-meeting version of criterion 3 is deferred to the coordinator's cold
baseline run; these tests cover the mechanism, not the corpus.
"""

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pyannote.core import Annotation, Segment, SlidingWindow, SlidingWindowFeature

from harness.cache_id import intermediate_config_id, pipeline_config_id
from harness.intermediate_cache import (
    CacheLookup,
    IntermediateCache,
    NonPowersetCheckpointUnsupported,
)
from harness.runner import Runner
from harness.segmentation import BaselineSegmentation

CHECKPOINT = "pyannote/speaker-diarization-community-1"

# Cost per simulated inference step, in seconds. Stands in for GPU work so
# criterion 4's wall-clock isolation is measurable without a GPU.
_SEGMENTATION_COST = 0.05
_EMBEDDING_COST = 0.05


class _FakeSpeakerDiarization:
    """Faithful stand-in for the real pipeline's caching behavior.

    Deliberately mirrors the real control flow rather than short-cutting it:
    segmentation first, then embeddings, then clustering -- because the whole
    design rests on clustering coming last.
    """

    CACHED_SEGMENTATION = "training_cache/segmentation"

    def __init__(self, clustering_threshold=0.6, num_chunks=40, num_speakers=3,
                 dimension=192, batch_size=32):
        self.training = False
        # Real SpeakerDiarization always sets both (speaker_diarization.py:236
        # and :259), and both are part of the intermediate cache key because
        # batch shape changes the float reduction order and so the arrays'
        # values. The fake carries them for the same reason it mirrors
        # _segmentation: so the key logic is exercised for real.
        self.embedding_batch_size = batch_size
        self.segmentation_batch_size = batch_size
        self.klustering = "VBxClustering"
        self.clustering = SimpleNamespace(threshold=clustering_threshold, Fa=0.07, Fb=0.8)
        self._num_chunks = num_chunks
        self._num_speakers = num_speakers
        self._dimension = dimension
        # Mirrors pipeline._segmentation.model.specifications.powerset, which
        # IntermediateCache reads to decide the stored dict shape.
        self._segmentation = SimpleNamespace(
            model=SimpleNamespace(specifications=SimpleNamespace(powerset=True))
        )
        self.segmentation_inferences = 0
        self.embedding_extractions = 0

    # -- the two expensive steps -----------------------------------------

    def _run_segmentation_model(self, uri):
        self.segmentation_inferences += 1
        time.sleep(_SEGMENTATION_COST)
        rng = np.random.default_rng(abs(hash(uri)) % (2**32))
        data = rng.random((self._num_chunks, 589, self._num_speakers)).astype(np.float32)
        return SlidingWindowFeature(data, SlidingWindow(duration=10.0, step=1.0, start=0.0))

    def _run_embedding_model(self, uri):
        self.embedding_extractions += 1
        time.sleep(_EMBEDDING_COST)
        rng = np.random.default_rng((abs(hash(uri)) + 1) % (2**32))
        return rng.random(
            (self._num_chunks, self._num_speakers, self._dimension)
        ).astype(np.float32)

    # -- speaker_diarization.py:337-344 ----------------------------------

    def get_segmentations(self, file):
        if self.training:
            if self.CACHED_SEGMENTATION in file:
                return file[self.CACHED_SEGMENTATION]
            segmentations = self._run_segmentation_model(file["uri"])
            file[self.CACHED_SEGMENTATION] = segmentations
            return segmentations
        return self._run_segmentation_model(file["uri"])

    # -- speaker_diarization.py:377-386 and :483-492 ---------------------

    def get_embeddings(self, file):
        if self.training:
            cache = file.get("training_cache/embeddings", dict())
            if ("embeddings" in cache) and (
                self._segmentation.model.specifications.powerset
                or (cache["segmentation.threshold"] == self.segmentation.threshold)
            ):
                return cache["embeddings"]

        embeddings = self._run_embedding_model(file["uri"])

        if self.training:
            if self._segmentation.model.specifications.powerset:
                file["training_cache/embeddings"] = {"embeddings": embeddings}
            else:
                file["training_cache/embeddings"] = {
                    "segmentation.threshold": self.segmentation.threshold,
                    "embeddings": embeddings,
                }
        return embeddings

    # -- apply() ----------------------------------------------------------

    def __call__(self, file):
        segmentations = self.get_segmentations(file)
        embeddings = self.get_embeddings(file)

        # Clustering, strictly after both intermediates (:656). The hypothesis
        # depends on BOTH the embeddings and the clustering threshold, so a
        # wrongly-shared intermediate OR a wrongly-shared RTTM shows up as a
        # different annotation rather than passing unnoticed.
        hypothesis = Annotation(uri=file["uri"])
        scores = embeddings.mean(axis=(1, 2))
        for index, score in enumerate(scores):
            label = "A" if score > self.clustering.threshold * 0.5 else "B"
            hypothesis[Segment(index, index + 1)] = label
        # Make the output depend on the segmentation too.
        hypothesis[Segment(1000, 1000 + float(segmentations.data.shape[0]))] = "SEG"

        return SimpleNamespace(
            speaker_diarization=hypothesis,
            exclusive_speaker_diarization=Annotation(uri=file["uri"]),
        )


def _cache(tmp_path, pipeline, enabled=True, config_id=None):
    return IntermediateCache(
        tmp_path,
        config_id=config_id or intermediate_config_id(CHECKPOINT, pipeline),
        segmentation_source_id=BaselineSegmentation().id,
        enabled=enabled,
    )


def _runner(cache_dir, pipeline, intermediate_cache, refinement_id="identity"):
    return Runner(
        pipeline,
        pipeline_config_id=pipeline_config_id(CHECKPOINT, pipeline),
        segmentation_source=BaselineSegmentation(),
        cache_dir=cache_dir,
        refinement_id=refinement_id,
        intermediate_cache=intermediate_cache,
    )


def _file(uri):
    return {"uri": uri, "audio": f"/nonexistent/{uri}.wav"}


# ==========================================================================
# The seam: a populated value must actually reach the pipeline's reads
# ==========================================================================


def test_warm_run_skips_segmentation_inference_and_embedding_extraction():
    """The core mechanism. Not "it got faster" -- the actual inference call
    counters must stay at 1 across a second run."""
    pipeline = _FakeSpeakerDiarization()
    file_cold = _file("ES2002a")

    pipeline.training = True
    pipeline(file_cold)
    assert pipeline.segmentation_inferences == 1
    assert pipeline.embedding_extractions == 1

    # Same `file` mapping reused: the in-process cache alone must already work,
    # otherwise the disk tier is built on a broken premise.
    pipeline(file_cold)
    assert pipeline.segmentation_inferences == 1, "segmentation model re-ran"
    assert pipeline.embedding_extractions == 1, "embedding model re-ran"


def test_cache_miss_is_observable_not_silent(tmp_path):
    """Failure mode 1: a cache that silently misses and recomputes yields
    correct numbers and no speedup. The miss must be visible."""
    pipeline = _FakeSpeakerDiarization()
    cache = _cache(tmp_path, pipeline)

    lookup = cache.populate(pipeline, _file("ES2002a"))

    assert lookup.any_hit is False
    assert str(lookup) == "miss"
    assert cache.stats.misses == 1
    assert cache.stats.full_hits == 0


def test_disabled_cache_reports_zero_activity(tmp_path):
    """A disabled cache must look obviously inert -- no writes, no hits --
    rather than resembling a working one."""
    pipeline = _FakeSpeakerDiarization()
    cache = _cache(tmp_path, pipeline, enabled=False)
    runner = _runner(tmp_path / "cold", pipeline, cache)

    runner.run(_file("ES2002a"))

    assert cache.stats.writes == 0
    assert cache.stats.full_hits == 0
    assert cache.entry_count() == 0


def test_partial_and_full_hits_are_distinguishable(tmp_path):
    """Failure mode 3: one value must not stand for two conditions. "no entry
    on disk" and "entry with segmentation but no embeddings" have different
    performance consequences and must be told apart."""
    assert str(CacheLookup()) == "miss"
    assert str(CacheLookup(segmentation_hit=True)) == "partial(segmentation only)"
    assert str(CacheLookup(segmentation_hit=True, embeddings_hit=True)) == (
        "hit(segmentation+embeddings)"
    )
    assert CacheLookup(segmentation_hit=True).full_hit is False
    assert CacheLookup(segmentation_hit=True).any_hit is True


def test_save_records_failure_when_pipeline_left_nothing(tmp_path):
    """If the training gate were broken, the pipeline would leave no
    intermediates on `file`. That must be recorded, not shrugged off: it is
    the signature of a broken seam."""
    pipeline = _FakeSpeakerDiarization()
    cache = _cache(tmp_path, pipeline)

    wrote = cache.save(pipeline, _file("ES2002a"))

    assert wrote is False
    assert "ES2002a" in cache.stats.write_failures


def test_non_powerset_checkpoint_refuses_rather_than_guessing(tmp_path):
    """On a non-powerset model the stored dict shape AND the intermediate key
    would both have to change. Refuse loudly instead of caching embeddings
    that would be reused at the wrong segmentation threshold."""
    pipeline = _FakeSpeakerDiarization()
    pipeline._segmentation.model.specifications.powerset = False
    pipeline.segmentation = SimpleNamespace(threshold=0.5)
    cache = _cache(tmp_path, pipeline)

    file = _file("ES2002a")
    pipeline.training = True
    pipeline(file)

    with pytest.raises(NonPowersetCheckpointUnsupported):
        cache.save(pipeline, file)


def test_stored_embedding_dict_matches_the_powerset_write_shape(tmp_path):
    """The reconstructed cache dict must match
    speaker_diarization.py:484-487 exactly -- `{"embeddings": ...}` with NO
    "segmentation.threshold". Synthesising a threshold would populate a state
    the cold path never produces."""
    pipeline = _FakeSpeakerDiarization()
    cache = _cache(tmp_path, pipeline)

    cold = _file("ES2002a")
    pipeline.training = True
    pipeline(cold)
    cache.save(pipeline, cold)

    warm = _file("ES2002a")
    cache.populate(pipeline, warm)

    assert set(warm["training_cache/embeddings"].keys()) == {"embeddings"}


def test_oracle_segmentation_is_not_overwritten_by_the_disk_cache(tmp_path):
    """An injected ground-truth segmentation must win over a cached one."""
    pipeline = _FakeSpeakerDiarization()
    cache = _cache(tmp_path, pipeline)

    cold = _file("ES2002a")
    pipeline.training = True
    pipeline(cold)
    cache.save(pipeline, cold)

    sentinel = SlidingWindowFeature(
        np.ones((2, 3, 4), dtype=np.float32), SlidingWindow(duration=10.0, step=1.0)
    )
    warm = _file("ES2002a")
    warm[pipeline.CACHED_SEGMENTATION] = sentinel

    cache.populate(pipeline, warm)

    assert warm[pipeline.CACHED_SEGMENTATION] is sentinel


# ==========================================================================
# Criterion 3: warm and cold runs produce BYTE-IDENTICAL RTTMs
# ==========================================================================


def _rttm_bytes(cache_dir: Path, runner: Runner, uri: str) -> bytes:
    return runner.cache_path(uri).read_bytes()


URIS = [
    "ES2002a", "ES2002b", "ES2003a", "ES2003b", "ES2004a", "ES2004b",
    "ES2005a", "ES2005b", "IS1000a", "IS1001a", "IS1002b", "IS1003b",
    "TS3003a", "TS3004a", "TS3006b", "TS3007a",
]


def test_warm_and_cold_rttms_are_byte_identical(tmp_path):
    """Criterion 3, at 16 simulated meetings.

    CRITICAL: the warm run writes its RTTMs into a SEPARATE cache_dir. If both
    runs shared one, `Runner.run()` would return the cached RTTM at
    runner.py:51-52 before the pipeline was touched at all, and the comparison
    would read back the very file it was comparing against -- passing
    unconditionally, measuring nothing.

    Byte comparison, not DER: DER is label-invariant under optimal mapping and
    precision-tolerant, and has already hidden a real difference on this
    project.

    The 16-file corpus here is simulated; the real 16-meeting run is the
    coordinator's cold baseline.
    """
    # -- fully cold: no intermediates on disk, own RTTM cache dir ---------
    cold_dir = tmp_path / "cold"
    cold_pipeline = _FakeSpeakerDiarization()
    cold_cache = _cache(tmp_path / "shared_intermediates", cold_pipeline)
    cold_runner = _runner(cold_dir, cold_pipeline, cold_cache)

    cold_bytes = {}
    for uri in URIS:
        cold_runner.run(_file(uri))
        cold_bytes[uri] = _rttm_bytes(cold_dir, cold_runner, uri)

    assert cold_pipeline.segmentation_inferences == len(URIS)
    assert cold_pipeline.embedding_extractions == len(URIS)
    assert cold_cache.stats.writes == len(URIS)

    # -- warm: same intermediates, FRESH RTTM cache dir -------------------
    warm_dir = tmp_path / "warm"
    warm_pipeline = _FakeSpeakerDiarization()
    warm_cache = _cache(tmp_path / "shared_intermediates", warm_pipeline)
    warm_runner = _runner(warm_dir, warm_pipeline, warm_cache)

    warm_bytes = {}
    for uri in URIS:
        warm_runner.run(_file(uri))
        warm_bytes[uri] = _rttm_bytes(warm_dir, warm_runner, uri)

    # The warm run really did use the cache rather than recomputing...
    assert warm_cache.stats.full_hits == len(URIS)
    assert warm_pipeline.segmentation_inferences == 0
    assert warm_pipeline.embedding_extractions == 0
    # ...and really did write its own separate files.
    assert warm_runner.cache_path(URIS[0]) != cold_runner.cache_path(URIS[0])

    mismatched = [uri for uri in URIS if cold_bytes[uri] != warm_bytes[uri]]
    assert not mismatched, f"warm RTTMs differ from cold for: {mismatched}"


def test_byte_comparison_can_actually_detect_a_difference(tmp_path):
    """Criterion 4's lesson applied to criterion 3: a comparison that cannot
    fail is not a check. Feed the warm run a DIFFERENT cached embedding and
    confirm the byte comparison notices -- proving the test above would have
    caught a real corruption."""
    cold_dir = tmp_path / "cold"
    intermediates = tmp_path / "intermediates"
    pipeline = _FakeSpeakerDiarization()
    cache = _cache(intermediates, pipeline)
    runner = _runner(cold_dir, pipeline, cache)
    runner.run(_file("ES2002a"))
    cold = _rttm_bytes(cold_dir, runner, "ES2002a")

    # Corrupt the stored embeddings in place. Zeroed rather than set to 1.0:
    # the fake's label rule is `mean > threshold * 0.5` (i.e. > 0.3), and
    # uniform-random embeddings average ~0.5, so setting them to 1.0 lands on
    # the SAME side of the boundary and produces a byte-identical RTTM. The
    # corruption has to actually cross the decision boundary for this
    # sensitivity check to mean anything -- which is itself an instance of
    # "an assertion that cannot fail is not an assertion".
    path = cache.path("ES2002a")
    with np.load(path, allow_pickle=False) as stored:
        payload = {key: stored[key] for key in stored.files}
    payload["embeddings"] = np.zeros_like(payload["embeddings"])
    np.savez_compressed(path, **payload)

    warm_dir = tmp_path / "warm"
    warm_pipeline = _FakeSpeakerDiarization()
    warm_cache = _cache(intermediates, warm_pipeline)
    warm_runner = _runner(warm_dir, warm_pipeline, warm_cache)
    warm_runner.run(_file("ES2002a"))
    warm = _rttm_bytes(warm_dir, warm_runner, "ES2002a")

    assert cold != warm, (
        "byte comparison failed to notice deliberately corrupted cached "
        "embeddings -- criterion 3's check would be vacuous"
    )


def test_segmentation_array_round_trips_exactly(tmp_path):
    """Byte-identical RTTMs require bit-identical arrays, so check the
    serialisation directly rather than only through the RTTM."""
    pipeline = _FakeSpeakerDiarization()
    cache = _cache(tmp_path, pipeline)

    cold = _file("ES2002a")
    pipeline.training = True
    pipeline(cold)
    original_segmentation = cold[pipeline.CACHED_SEGMENTATION]
    original_embeddings = cold["training_cache/embeddings"]["embeddings"]
    cache.save(pipeline, cold)

    warm = _file("ES2002a")
    cache.populate(pipeline, warm)
    restored_segmentation = warm[pipeline.CACHED_SEGMENTATION]

    np.testing.assert_array_equal(
        original_segmentation.data, restored_segmentation.data
    )
    np.testing.assert_array_equal(
        original_embeddings, warm["training_cache/embeddings"]["embeddings"]
    )
    assert (
        restored_segmentation.sliding_window.duration
        == original_segmentation.sliding_window.duration
    )
    assert (
        restored_segmentation.sliding_window.step
        == original_segmentation.sliding_window.step
    )
    assert (
        restored_segmentation.sliding_window.start
        == original_segmentation.sliding_window.start
    )


# ==========================================================================
# The two tiers interacting: a clustering sweep
# ==========================================================================


def test_clustering_sweep_shares_intermediates_but_not_rttms(tmp_path):
    """The end-to-end point of the whole ticket, in miniature.

    Two sweep points differing only in clustering threshold must:
      * share the cached intermediates (no second GPU inference), and
      * produce SEPARATE RTTM cache entries with genuinely different content.

    Under the old bare-checkpoint key the second point returned the first
    point's cached RTTM and the sweep curve was flat.
    """
    cache_dir = tmp_path / "cache"
    intermediates = tmp_path / "intermediates"

    point_1 = _FakeSpeakerDiarization(clustering_threshold=0.6)
    cache_1 = _cache(intermediates, point_1)
    runner_1 = _runner(cache_dir, point_1, cache_1)
    hypothesis_1 = runner_1.run(_file("ES2002a"))

    point_2 = _FakeSpeakerDiarization(clustering_threshold=1.6)
    cache_2 = _cache(intermediates, point_2)
    runner_2 = _runner(cache_dir, point_2, cache_2)
    hypothesis_2 = runner_2.run(_file("ES2002a"))

    # Intermediates shared: the second point ran no inference at all.
    assert cache_2.stats.full_hits == 1
    assert point_2.segmentation_inferences == 0
    assert point_2.embedding_extractions == 0
    assert cache_1.path("ES2002a") == cache_2.path("ES2002a")

    # RTTMs NOT shared.
    assert runner_1.cache_path("ES2002a") != runner_2.cache_path("ES2002a")

    labels_1 = sorted({label for _, _, label in hypothesis_1.itertracks(yield_label=True)})
    labels_2 = sorted({label for _, _, label in hypothesis_2.itertracks(yield_label=True)})
    assert labels_1 != labels_2, (
        "the two sweep points produced identical output -- the second served "
        "the first's cached RTTM"
    )


def test_second_sweep_point_is_not_served_a_stale_rttm(tmp_path):
    """Reproduces the defect directly: pre-populate the RTTM cache under point
    1's key, then run point 2 in the SAME cache_dir. Point 2's pipeline must
    actually be invoked."""
    cache_dir = tmp_path / "cache"
    intermediates = tmp_path / "intermediates"

    point_1 = _FakeSpeakerDiarization(clustering_threshold=0.6)
    runner_1 = _runner(cache_dir, point_1, _cache(intermediates, point_1))
    runner_1.run(_file("ES2002a"))

    point_2 = _FakeSpeakerDiarization(clustering_threshold=1.6)
    runner_2 = _runner(cache_dir, point_2, _cache(intermediates, point_2))
    runner_2.run(_file("ES2002a"))

    # Clustering re-ran (the pipeline was called), even though segmentation
    # and embeddings were served from cache.
    assert point_2.segmentation_inferences == 0
    assert point_2.embedding_extractions == 0
    assert runner_2.cache_path("ES2002a").exists()
    assert (
        runner_1.cache_path("ES2002a").read_bytes()
        != runner_2.cache_path("ES2002a").read_bytes()
    )


# ==========================================================================
# Criterion 4: wall clock attributable to the INTERMEDIATE tier specifically
# ==========================================================================


def test_intermediate_tier_reduces_wall_clock_with_final_cache_bypassed(tmp_path):
    """Criterion 4, isolated as the rewritten criterion requires.

    The comparison is NOT warm-vs-cold overall: a warm run hits the RTTM cache
    at runner.py:51-52 and returns before the pipeline is constructed, so it
    would be ~100% faster whether or not one line of intermediate caching
    works. That is why the original criterion could not fail.

    Here the final-hypothesis cache is bypassed in BOTH arms (each run gets a
    fresh RTTM cache_dir, so `run()` can never short-circuit), and the only
    difference is whether the intermediate cache is warm. That is the only
    comparison in which the intermediate tier is on the critical path.

    The simulated inference cost is a fixed sleep, so this measures the
    mechanism, not real GPU time; the real figure comes from the coordinator's
    run.
    """
    intermediates = tmp_path / "intermediates"
    uris = URIS[:8]

    # -- arm A: fully cold ------------------------------------------------
    cold_pipeline = _FakeSpeakerDiarization()
    cold_cache = _cache(intermediates, cold_pipeline)
    cold_runner = _runner(tmp_path / "rttm_cold", cold_pipeline, cold_cache)

    started = time.monotonic()
    for uri in uris:
        cold_runner.run(_file(uri))
    cold_seconds = time.monotonic() - started

    # -- arm B: intermediate cache warm, final cache still bypassed -------
    warm_pipeline = _FakeSpeakerDiarization()
    warm_cache = _cache(intermediates, warm_pipeline)
    warm_runner = _runner(tmp_path / "rttm_warm", warm_pipeline, warm_cache)

    started = time.monotonic()
    for uri in uris:
        warm_runner.run(_file(uri))
    warm_seconds = time.monotonic() - started

    # The final cache genuinely could not short-circuit: the pipeline ran in
    # both arms, producing a fresh RTTM each time.
    assert warm_runner.cache_path(uris[0]).exists()
    assert warm_cache.stats.full_hits == len(uris)
    # ...and the expensive steps were skipped only in arm B.
    assert cold_pipeline.segmentation_inferences == len(uris)
    assert warm_pipeline.segmentation_inferences == 0
    assert warm_pipeline.embedding_extractions == 0

    print(
        f"\ncriterion 4 (intermediate tier isolated, {len(uris)} files):\n"
        f"  fully cold                      : {cold_seconds:.3f}s\n"
        f"  intermediates warm, RTTM bypassed: {warm_seconds:.3f}s\n"
        f"  delta                           : {cold_seconds - warm_seconds:.3f}s "
        f"({100 * (cold_seconds - warm_seconds) / cold_seconds:.1f}% faster)"
    )

    assert warm_seconds < cold_seconds


def test_on_disk_size_is_reported(tmp_path):
    """Criterion 5's mechanism: the per-file and total on-disk size must be
    measurable, since the working directory has a history of filling up."""
    pipeline = _FakeSpeakerDiarization()
    cache = _cache(tmp_path, pipeline)
    runner = _runner(tmp_path / "rttm", pipeline, cache)

    for uri in URIS:
        runner.run(_file(uri))

    total = cache.disk_usage_bytes()
    assert cache.entry_count() == len(URIS)
    assert total > 0

    print(
        f"\ncriterion 5 (simulated arrays, {len(URIS)} files): "
        f"{total / 1e6:.2f} MB total, "
        f"{total / len(URIS) / 1e6:.3f} MB per file"
    )
