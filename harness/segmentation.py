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
from typing import Any, MutableMapping


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
    """Stub. Will build segmentation from the reference annotation using
    `pyannote.audio.pipelines.utils.oracle.oracle_segmentation(file, window,
    frames, num_speakers=None)`, which discretises a reference `Annotation`
    into the `(num_chunks, num_frames, num_speakers)` `SlidingWindowFeature`
    that `CACHED_SEGMENTATION` expects. Already used for oracle clustering
    (src/pyannote/audio/pipelines/clustering.py:712-715), so the shape
    contract is known-good. Wiring it up -- computing the value, setting
    `file[pipeline.CACHED_SEGMENTATION]`, and engaging the `pipeline.training
    = True` seam -- is a future ticket; this stub only reserves the shape of
    the interface."""

    id = "oracle"

    def populate(self, pipeline: Any, file: MutableMapping) -> None:
        """Not yet implemented. See TICKET-04-segmentation-source-interface.md
        and pyannote.audio.pipelines.utils.oracle.oracle_segmentation."""
        raise NotImplementedError(
            "OracleSegmentation is a stub (TICKET-04); the real implementation "
            "will build on pyannote.audio.pipelines.utils.oracle.oracle_segmentation"
        )
