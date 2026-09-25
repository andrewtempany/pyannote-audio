"""Segmentation source interface. See TICKET-04-segmentation-source-interface.md.

Each source has a stable `id` (feeds the runner's cache key, TICKET-06) and a
`populate(pipeline, file)` hook that may inject a cached segmentation onto
`file[pipeline.CACHED_SEGMENTATION]` before the pipeline is applied. That
injection only takes effect if `pipeline.training` is True when the pipeline
consults CACHED_SEGMENTATION (SpeakerDiarization.get_segmentations() gates the
read on `self.training`, per the seam established in
TICKET-01-segmentation-injection-seam.md) -- but `populate()` itself does not
manage that flag. `Runner.run()` (harness/runner.py) sets
`pipeline.training = True` around *both* the `populate()` call and the
pipeline call, uniformly for every segmentation source, so the flag is a
run-scoped concern rather than something each source manages independently.
See Obsidian-Diarisation/Docs/Oracle Segmentation Seam.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, MutableMapping

from pyannote.core import Annotation, SlidingWindow
from pyannote.audio.pipelines.utils.oracle import oracle_segmentation


class SegmentationSource(ABC):
    """Interface for a segmentation source: something that can either leave
    the pipeline's own segmentation model alone (BaselineSegmentation) or
    inject a precomputed segmentation for the pipeline to use instead
    (OracleSegmentation, still a stub in this ticket)."""

    id: str

    @abstractmethod
    def populate(self, pipeline: Any, file: MutableMapping) -> None:
        """Called before the pipeline is applied to `file`. May populate
        `file[pipeline.CACHED_SEGMENTATION]`; must not otherwise mutate
        `pipeline` or `file`. Does NOT manage `pipeline.training` -- the
        caller (Runner.run()) is responsible for setting that flag True
        around both this call and the pipeline call, uniformly across all
        segmentation sources, so that CACHED_SEGMENTATION population is the
        only difference between conditions."""
        raise NotImplementedError


class BaselineSegmentation(SegmentationSource):
    """Uses the pipeline's own segmentation -- normal, unmodified behavior.
    Populates nothing on `file` and never touches `pipeline.training`."""

    id = "baseline"

    def populate(self, pipeline: Any, file: MutableMapping) -> None:
        return


class OracleSegmentation(SegmentationSource):
    """Builds ground-truth segmentation from a reference-Annotation lookup
    (e.g. parsed from an only_words RTTM via `pyannote.database.util.load_rttm`)
    using `pyannote.audio.pipelines.utils.oracle.oracle_segmentation(file,
    window, frames, num_speakers=None)`, which discretises a reference
    `Annotation` into the `(num_chunks, num_frames, num_speakers)`
    `SlidingWindowFeature` that `CACHED_SEGMENTATION` expects. Already used
    for oracle clustering (src/pyannote/audio/pipelines/clustering.py:712-715),
    so the shape contract is known-good.

    `window`/`frames` are derived from the pipeline's own segmentation
    Inference (`pipeline._segmentation.step`/`.duration`/
    `.model.receptive_field`), mirroring the exact convention
    `SpeakerDiarization.apply()` itself uses when calling `oracle_segmentation`
    for oracle clustering (speaker_diarization.py:663, clustering.py:710-712).

    Setting `file[pipeline.CACHED_SEGMENTATION]` only takes effect if
    `pipeline.training` is True when the pipeline consults it (see
    Obsidian-Diarisation/Docs/Segmentation Injection Seam.md) -- this method
    does NOT flip that flag itself. `Runner.run()` sets `pipeline.training =
    True` around both the `populate()` call and the subsequent pipeline call
    (uniformly for baseline and oracle alike), and restores it afterwards
    (even on failure), so a failed/aborted run never leaves the pipeline
    permanently in training mode.

    The reference is handed to `oracle_segmentation()` on a shallow copy of
    `file`, so `file` itself never carries an `annotation` key at any point.

    Also raises `RuntimeError` if `pipeline._expects_num_speakers` is True
    (e.g. KMeansClustering or OracleClustering). This guard is a **refusal of
    an unsupported combination, not leak prevention.** Speaker-count-driven
    clustering needs a speaker count, and the harness supplies none:
    `run_harness.py` builds `file` as `{"uri", "audio"}` and `Runner.run()`
    calls the pipeline with no `num_speakers`, so
    `SpeakerDiarization.apply()` would fall through to its own
    `ValueError: num_speakers must be provided ...` raised from inside the
    library (speaker_diarization.py:600-607). Failing here instead gives a
    harness-level error that names the actual problem.

    Where `k` should come from (oracle count, fixed, or estimated) is a design
    decision with a different experiment behind each option, and is a
    prerequisite for the clustering sweep rather than something this class
    should decide implicitly.

    Note on a claim this docstring previously made: it stated that
    `apply()` would read `file["annotation"]` "before it's restored". That was
    wrong -- `populate()` returns, restoring included, before `Runner.run()`
    calls the pipeline at all, so no leak was ever possible by that route.
    The description was taken at face value and cost a round of misdirected
    planning; see Obsidian-Diarisation/Docs/Oracle Segmentation Provider.md."""

    id = "oracle-v2"

    def __init__(self, reference_lookup: Mapping[str, Annotation] = None):
        self._reference_lookup = reference_lookup if reference_lookup is not None else {}

    def populate(self, pipeline: Any, file: MutableMapping) -> None:
        uri = file["uri"]
        reference = self._reference_lookup[uri]  # KeyError -> fail fast

        window = SlidingWindow(
            step=pipeline._segmentation.step,
            duration=pipeline._segmentation.duration,
        )
        frames = pipeline._segmentation.model.receptive_field

        if getattr(pipeline, "_expects_num_speakers", False):
            raise RuntimeError(
                "pipeline._expects_num_speakers is True (e.g. KMeansClustering "
                "or OracleClustering), but the harness supplies no "
                "num_speakers: run_harness.py builds file as {'uri', 'audio'} "
                "and Runner.run() calls the pipeline without it, so "
                "SpeakerDiarization.apply() would raise 'num_speakers must be "
                "provided' from inside the library "
                "(speaker_diarization.py:600-607). Deciding where k comes from "
                "-- oracle count, fixed, or estimated -- is a prerequisite for "
                "the clustering sweep; note that using the oracle count turns "
                "this into an oracle-segmentation-plus-oracle-count condition "
                "and must be recorded as such. See Obsidian-Diarisation/Docs/"
                "Oracle Segmentation Seam.md (Confound A)."
            )

        # `oracle_segmentation()` reads only "duration" (falling back to
        # decoding the audio) and "annotation", and never writes to the mapping
        # it is given -- so a shallow copy carries everything it needs. Passing
        # the copy keeps `annotation` off the caller's dict entirely, which is
        # strictly safer than setting it and restoring it in a `finally`: there
        # is no window in which the key exists, and no restore to depend on.
        scaffold = dict(file)
        scaffold["annotation"] = reference
        file[pipeline.CACHED_SEGMENTATION] = oracle_segmentation(
            scaffold, window, frames
        )
