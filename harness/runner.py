"""Runner: pipeline + segmentation source + audio -> hypothesis, with disk
caching. See TICKET-06-runner.md.

`pipeline_config_id` is a caller-supplied stable string identifying the
pipeline's configuration (e.g. an HF checkpoint id/revision, or a hash of
its instantiated hyperparameters) -- the runner does not attempt to derive
one by introspecting an arbitrary `Pipeline` object, which may hold
non-serializable state (loaded models, a PLDA instance, ...). Owning that
identifier is the caller's (the orchestrator's, TICKET-08) responsibility.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Union

from pyannote.core import Annotation
from pyannote.database.util import load_rttm


class Runner:
    def __init__(
        self,
        pipeline: Any,
        pipeline_config_id: str,
        segmentation_source: Any,
        cache_dir: Union[str, Path],
        refinement_id: str = "identity",
        intermediate_cache: Any = None,
    ):
        self._pipeline = pipeline
        self._pipeline_config_id = pipeline_config_id
        self._segmentation_source = segmentation_source
        self._cache_dir = Path(cache_dir)
        self._refinement_id = refinement_id
        # Optional second cache tier (harness/intermediate_cache.py). None
        # keeps the historical single-tier behavior, which is what the
        # existing runner tests exercise.
        self._intermediate_cache = intermediate_cache

    def cache_key(self, uri: str) -> str:
        """Key for the FINAL HYPOTHESIS (the RTTM).

        Includes `pipeline_config_id`, which as of the pre-clustering-cache
        work is a composite of the checkpoint, the clustering class name and
        every instantiated clustering hyperparameter (see
        harness/cache_id.pipeline_config_id). Clustering MUST be in this key:
        it determines the speaker labels, so two clustering configurations that
        shared a key would mean the second silently served the first's cached
        RTTM -- a hyperparameter sweep in which every point reports point 1's
        DER, on a run that completes cleanly and writes a valid manifest.

        Contrast `IntermediateCache.key()`, which deliberately EXCLUDES
        clustering so that segmentation and embeddings are shared across
        clustering variants. Getting the two the wrong way round yields either
        a useless cache or silently wrong results.
        """
        raw = (
            f"{self._pipeline_config_id}|{self._segmentation_source.id}|"
            f"{self._refinement_id}|{uri}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def cache_path(self, uri: str) -> Path:
        return self._cache_dir / f"{self.cache_key(uri)}.rttm"

    def run(self, file: Mapping) -> Annotation:
        uri = file["uri"]
        cache_path = self.cache_path(uri)

        if cache_path.exists():
            return load_rttm(str(cache_path))[uri]

        # pipeline.training gates SpeakerDiarization.get_segmentations()'s
        # only read of CACHED_SEGMENTATION (see harness/segmentation.py's
        # module docstring) -- set it True around BOTH populate() and the
        # pipeline call, uniformly for every segmentation source, so the
        # only difference between conditions is whether CACHED_SEGMENTATION
        # was pre-populated. This also makes embedding caching
        # (speaker_diarization.py training-gated read/write) behave
        # identically across conditions instead of only for oracle runs.
        #
        # getattr/hasattr guard: every real pyannote Pipeline sets
        # self.training = False in __init__ (pyannote/pipeline/pipeline.py),
        # so this is always present in production. It's guarded here only so
        # lightweight fakes in tests that don't care about the flag (and
        # never defined it, since it was previously untouched by the runner)
        # aren't forced to grow one just to be called.
        has_training_attr = hasattr(self._pipeline, "training")
        original_training = getattr(self._pipeline, "training", None)
        if has_training_attr:
            self._pipeline.training = True
        try:
            # populate() runs -- and may raise (e.g. OracleSegmentation's
            # fail-fast KeyError) -- before the pipeline is touched at all,
            # so a failure here never leaves a partial/bad cache entry
            # behind.
            self._segmentation_source.populate(self._pipeline, file)

            # Second tier: put cached segmentation/embeddings onto `file` so
            # the pipeline's own training-gated reads
            # (speaker_diarization.py:337-344 and :377-386) find them instead
            # of running the segmentation model and embedding extractor.
            #
            # Deliberately AFTER segmentation_source.populate(): an oracle
            # source's ground-truth segmentation must win over anything on
            # disk, and populate() will not overwrite a key that is already
            # present.
            if self._intermediate_cache is not None:
                self._intermediate_cache.populate(self._pipeline, file)

            output = self._pipeline(file)

            # Harvest whatever the pipeline left behind. Inside the try so it
            # still sees `file` while training mode is on, but after the
            # pipeline call so the arrays exist. A warm run re-saving is a
            # no-op (save() returns early if the path exists).
            if self._intermediate_cache is not None:
                self._intermediate_cache.save(self._pipeline, file)
        finally:
            if has_training_attr:
                self._pipeline.training = original_training

        hypothesis = output.speaker_diarization

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            hypothesis.write_rttm(f)

        return hypothesis
