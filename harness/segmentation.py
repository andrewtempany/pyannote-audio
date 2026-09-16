"""Segmentation source interface. See TICKET-04-segmentation-source-interface.md.

Each source has a stable `id` (feeds the runner's cache key, TICKET-06) and a
`populate(pipeline, file)` hook that may inject a cached segmentation onto
`file[pipeline.CACHED_SEGMENTATION]` before the pipeline is applied. Doing so
requires the `pipeline.training = True` seam established in
TICKET-01-segmentation-injection-seam.md -- SpeakerDiarization.get_segmentations()
only consults CACHED_SEGMENTATION while `pipeline.training` is True.
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
        `pipeline` or `file`."""
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
    flips it for the duration of `populate()` and restores it afterwards
    (even on failure), so a failed/aborted run never leaves the pipeline
    permanently in training mode."""

    id = "oracle"

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

        file["annotation"] = reference

        original_training = pipeline.training
        pipeline.training = True
        try:
            file[pipeline.CACHED_SEGMENTATION] = oracle_segmentation(
                file, window, frames
            )
        finally:
            pipeline.training = original_training
