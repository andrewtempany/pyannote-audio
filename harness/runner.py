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
    ):
        self._pipeline = pipeline
        self._pipeline_config_id = pipeline_config_id
        self._segmentation_source = segmentation_source
        self._cache_dir = Path(cache_dir)

    def cache_key(self, uri: str) -> str:
        raw = f"{self._pipeline_config_id}|{self._segmentation_source.id}|{uri}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def cache_path(self, uri: str) -> Path:
        return self._cache_dir / f"{self.cache_key(uri)}.rttm"

    def run(self, file: Mapping) -> Annotation:
        uri = file["uri"]
        cache_path = self.cache_path(uri)

        if cache_path.exists():
            return load_rttm(str(cache_path))[uri]

        # populate() runs -- and may raise (e.g. OracleSegmentation's stub)
        # -- before the pipeline is touched at all, so a failure here never
        # leaves a partial/bad cache entry behind.
        self._segmentation_source.populate(self._pipeline, file)

        output = self._pipeline(file)
        hypothesis = output.speaker_diarization

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            hypothesis.write_rttm(f)

        return hypothesis
