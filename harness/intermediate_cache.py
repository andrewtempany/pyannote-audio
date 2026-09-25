"""Disk persistence for the pipeline's pre-clustering intermediates.

WHAT THIS IS FOR: segmentation inference and embedding extraction are the
dominant cost of a scored run (47-71 minutes for 16 meetings), and neither
depends on the clustering stage. Cached once per (checkpoint, segmentation
source, file) and reused, a clustering sweep point becomes a clustering call
over cached arrays instead of a full GPU run.

HOW IT HOOKS IN -- no new injection point was needed
----------------------------------------------------
`SpeakerDiarization` already caches both intermediates on the `file` mapping,
gated on `self.training`. Verified by reading the source, not the docstrings
(this project has twice been bitten by the reverse):

* segmentation -- `CACHED_SEGMENTATION` is the literal
  "training_cache/segmentation" (speaker_diarization.py:317-319), read AND
  written in `get_segmentations()` at :337-344 under `if self.training:`.
* embeddings -- the literal "training_cache/embeddings", read in
  `get_embeddings()` at :377-386 under `if self.training:`, written at
  :483-492 under the same gate.

Both are reached on the normal `apply()` path (segmentation at :609,
embeddings at :647-652), so a value pre-populated on `file` does reach the
read. `Runner.run()` (harness/runner.py:69-83) already sets
`pipeline.training = True` around both `populate()` and the pipeline call, for
every condition, and restores it in a `finally`.

So this module only has to do what `OracleSegmentation.populate()` already
does for segmentation: write the arrays to disk after a cold run, put them
back on `file` before a warm one.

THE POWERSET COUPLING -- the one real trap here
-----------------------------------------------
The embedding cache value is a dict, and its shape depends on whether the
segmentation model is powerset (speaker_diarization.py:483-492):

    powerset      -> {"embeddings": ...}                          (no threshold)
    non-powerset  -> {"segmentation.threshold": ..., "embeddings": ...}

while the read at :382-385 is

    if ("embeddings" in cache) and (
        self._segmentation.model.specifications.powerset
        or (cache["segmentation.threshold"] == self.segmentation.threshold)
    ):

Note the asymmetry: a powerset *write* followed by a non-powerset *read* would
raise `KeyError` on `cache["segmentation.threshold"]`. community-1 IS powerset
(verified from the checkpoint metadata: `powerset_max_classes=2`; and
independently, `pipeline.segmentation` has no `threshold` key at all, which
only happens on the powerset branch of `__init__` at :262-271), so `or`
short-circuits on `specifications.powerset` and the subscript is never
reached.

This module therefore reconstructs the powerset shape -- `{"embeddings": ...}`
only -- and deliberately does NOT synthesise a `segmentation.threshold` entry:
on a powerset pipeline that value does not exist, and inventing one would mean
populating a cache state the cold path never produces.

On a NON-powerset checkpoint two things would both have to change, and the
code refuses rather than guessing: the stored dict would need the threshold,
AND `segmentation.threshold` would have to join the intermediate cache key
(it changes the binarization the embeddings are extracted from, so embeddings
cached at one threshold are simply wrong at another). `save()` raises in that
case.

FAILURE MODE THIS IS DESIGNED AGAINST
-------------------------------------
"A cache that silently misses and recomputes" is the same shape of bug as the
segmentation seam no-op: everything completes, the numbers are plausible, and
the only symptom is that nothing got faster. So a miss is *observable*: every
lookup returns a `CacheLookup` recording the outcome, `Runner` counts hits and
misses, and `run_harness.py` prints the tally. A cache that silently did
nothing would show 0 hits and 0 writes rather than looking like a success.

Storage is `.npz` (numpy-native, compressed) plus the `SlidingWindow`
parameters needed to rebuild the `SlidingWindowFeature`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional, Union

import numpy as np
from pyannote.core import SlidingWindow, SlidingWindowFeature

# The literal keys SpeakerDiarization uses. Mirrored here rather than imported
# because `CACHED_SEGMENTATION` is an instance property; the segmentation one
# is read off the live pipeline when available and only falls back to this.
_SEGMENTATION_CACHE_KEY = "training_cache/segmentation"
_EMBEDDING_CACHE_KEY = "training_cache/embeddings"

# Format version. Bump to orphan every existing entry if the stored layout
# changes (e.g. gaining a `segmentation.threshold` field for a non-powerset
# checkpoint). Orphaning is the correct response to a layout change -- old
# entries become unreachable, not silently misread.
_FORMAT_VERSION = 1


class NonPowersetCheckpointUnsupported(RuntimeError):
    """Raised when asked to persist intermediates for a non-powerset model.

    Two coupled changes would be required (see the module docstring): the
    stored embedding dict would need a `segmentation.threshold` field to match
    the write at speaker_diarization.py:489-492, and `segmentation.threshold`
    would have to become part of the intermediate cache key, because it changes
    the binarization the embeddings are extracted from. Guessing at either
    would produce embeddings that are wrong but entirely plausible, so this
    refuses instead.
    """


@dataclass
class CacheLookup:
    """Outcome of one intermediate-cache lookup.

    Deliberately not a bool and not an Optional. Failure mode 3 on this project
    was a function returning `None` for two unrelated conditions whose caller
    conflated them, which turned 96.3% of a headline figure into empty slots.
    Here "no entry on disk" and "entry present but only segmentation, not
    embeddings" are genuinely different states with different performance
    consequences, so they are distinguishable.
    """

    segmentation_hit: bool = False
    embeddings_hit: bool = False

    @property
    def any_hit(self) -> bool:
        return self.segmentation_hit or self.embeddings_hit

    @property
    def full_hit(self) -> bool:
        return self.segmentation_hit and self.embeddings_hit

    def __str__(self) -> str:
        if self.full_hit:
            return "hit(segmentation+embeddings)"
        if self.segmentation_hit:
            return "partial(segmentation only)"
        if self.embeddings_hit:
            return "partial(embeddings only)"
        return "miss"


@dataclass
class CacheStats:
    """Observable tally, so a cache that quietly does nothing looks different
    from one that works. Printed per run by run_harness.py."""

    full_hits: int = 0
    partial_hits: int = 0
    misses: int = 0
    writes: int = 0
    write_failures: Dict[str, str] = field(default_factory=dict)

    def record_lookup(self, lookup: CacheLookup) -> None:
        if lookup.full_hit:
            self.full_hits += 1
        elif lookup.any_hit:
            self.partial_hits += 1
        else:
            self.misses += 1

    def __str__(self) -> str:
        text = (
            f"intermediate cache: {self.full_hits} full hit(s), "
            f"{self.partial_hits} partial, {self.misses} miss(es), "
            f"{self.writes} write(s)"
        )
        if self.write_failures:
            text += "; write failures: " + ", ".join(
                f"{uri}: {reason}" for uri, reason in sorted(self.write_failures.items())
            )
        return text


def _is_powerset(pipeline: Any) -> Optional[bool]:
    """True/False for a real pipeline, None when it cannot be determined
    (e.g. a lightweight test fake with no segmentation model)."""
    try:
        return bool(pipeline._segmentation.model.specifications.powerset)
    except AttributeError:
        return None


def _segmentation_key(pipeline: Any) -> str:
    return getattr(pipeline, "CACHED_SEGMENTATION", None) or _SEGMENTATION_CACHE_KEY


def _encode_window(window: SlidingWindow) -> Dict[str, np.ndarray]:
    """`SlidingWindow` is rebuilt from duration/step/start/end.

    `end` is `inf` for an unbounded window, which survives a float64 npz
    round-trip intact, but is stored via an explicit flag as well so the
    reconstruction does not depend on that.
    """
    end = window.end
    unbounded = end is None or (isinstance(end, float) and math.isinf(end))
    return {
        "window_duration": np.asarray(window.duration, dtype=np.float64),
        "window_step": np.asarray(window.step, dtype=np.float64),
        "window_start": np.asarray(window.start, dtype=np.float64),
        "window_end": np.asarray(0.0 if unbounded else end, dtype=np.float64),
        "window_unbounded": np.asarray(unbounded, dtype=bool),
    }


def _decode_window(stored: Any) -> SlidingWindow:
    unbounded = bool(stored["window_unbounded"])
    return SlidingWindow(
        duration=float(stored["window_duration"]),
        step=float(stored["window_step"]),
        start=float(stored["window_start"]),
        end=None if unbounded else float(stored["window_end"]),
    )


class IntermediateCache:
    """Reads/writes segmentation and embedding arrays for one cache identity.

    `config_id` must be the INTERMEDIATE key's pipeline-configuration portion
    (`harness.cache_id.intermediate_config_id`), which excludes clustering
    configuration. Passing the final-hypothesis id here would key the
    intermediates on clustering and defeat the entire purpose -- every sweep
    point would miss. The distinction is the whole design; see
    `harness/cache_id.py`.
    """

    def __init__(
        self,
        cache_dir: Union[str, Path],
        config_id: str,
        segmentation_source_id: str,
        enabled: bool = True,
    ):
        self._dir = Path(cache_dir) / "intermediates"
        self._config_id = config_id
        self._segmentation_source_id = segmentation_source_id
        self._enabled = enabled
        self.stats = CacheStats()

    # -- identity ---------------------------------------------------------

    def key(self, uri: str) -> str:
        """`config_id | segmentation_source.id | uri` -- deliberately WITHOUT
        refinement_id (refinement runs after clustering, at
        speaker_diarization.py:672, so it cannot affect either intermediate)
        and deliberately WITHOUT clustering config (see class docstring)."""
        import hashlib

        raw = f"{self._config_id}|{self._segmentation_source_id}|{uri}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def path(self, uri: str) -> Path:
        return self._dir / f"{self.key(uri)}.npz"

    # -- warm side --------------------------------------------------------

    def populate(self, pipeline: Any, file: MutableMapping) -> CacheLookup:
        """Put cached intermediates onto `file` so the pipeline's own
        training-gated reads find them. Mirrors
        `OracleSegmentation.populate()`.

        Never overwrites a value already on `file`: `OracleSegmentation`
        populates ground-truth segmentation, and that must win over anything
        on disk. (In practice the key includes `segmentation_source_id`, so an
        oracle entry and a baseline entry never share a path -- this is a
        second, structural guard rather than the primary one.)
        """
        lookup = CacheLookup()
        if not self._enabled:
            return lookup

        path = self.path(file["uri"])
        if not path.exists():
            self.stats.record_lookup(lookup)
            return lookup

        with np.load(path, allow_pickle=False) as stored:
            keys = set(stored.files)

            segmentation_key = _segmentation_key(pipeline)
            if "segmentation" in keys and segmentation_key not in file:
                file[segmentation_key] = SlidingWindowFeature(
                    stored["segmentation"], _decode_window(stored)
                )
                lookup.segmentation_hit = True

            if "embeddings" in keys and _EMBEDDING_CACHE_KEY not in file:
                # Powerset shape only -- matches the write at
                # speaker_diarization.py:484-487 exactly. See module docstring
                # for why no `segmentation.threshold` is synthesised.
                file[_EMBEDDING_CACHE_KEY] = {"embeddings": stored["embeddings"]}
                lookup.embeddings_hit = True

        self.stats.record_lookup(lookup)
        return lookup

    # -- cold side --------------------------------------------------------

    def save(self, pipeline: Any, file: MutableMapping) -> bool:
        """Harvest whatever the pipeline left on `file` and write it to disk.

        Called after the pipeline runs. The pipeline itself populates both keys
        on a cold run (segmentation at speaker_diarization.py:342, embeddings
        at :485), which is why nothing has to be intercepted mid-`apply()`.

        Returns True if a file was written.
        """
        if not self._enabled:
            return False

        uri = file["uri"]
        path = self.path(uri)
        if path.exists():
            return False

        segmentation = file.get(_segmentation_key(pipeline))
        embedding_cache = file.get(_EMBEDDING_CACHE_KEY) or {}
        embeddings = embedding_cache.get("embeddings")

        if segmentation is None and embeddings is None:
            # Nothing to store. Recorded rather than ignored: if this happened
            # for every file it would mean the training-gated caches never
            # populated, i.e. the seam is broken -- exactly the silent no-op
            # this project has already paid for once.
            self.stats.write_failures[uri] = "pipeline left no intermediates on file"
            return False

        if embeddings is not None and _is_powerset(pipeline) is False:
            raise NonPowersetCheckpointUnsupported(
                "the segmentation model is not powerset, so "
                "speaker_diarization.py:489-492 stores a "
                "'segmentation.threshold' alongside the embeddings and the "
                "intermediate cache key would have to include that threshold "
                "too (it changes the binarization embeddings are extracted "
                "from). Both changes are required together; refusing to store "
                "an embedding cache that would be reused at the wrong "
                "threshold. See harness/intermediate_cache.py."
            )

        payload: Dict[str, Any] = {
            "format_version": np.asarray(_FORMAT_VERSION, dtype=np.int64),
        }
        if segmentation is not None:
            payload["segmentation"] = np.asarray(segmentation.data)
            payload.update(_encode_window(segmentation.sliding_window))
        if embeddings is not None:
            payload["embeddings"] = np.asarray(embeddings)

        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary file and replace, so an interrupted run cannot
        # leave a truncated .npz that a later warm run would read as valid.
        #
        # Gotcha: np.savez_compressed APPENDS ".npz" unless the filename
        # already ends in it, so the temp name must keep that suffix last
        # (".partial.npz", not ".npz.partial") or the file lands somewhere
        # other than where we then try to rename from.
        tmp = path.with_name(path.stem + ".partial.npz")
        np.savez_compressed(tmp, **payload)
        tmp.replace(path)
        self.stats.writes += 1
        return True

    # -- reporting --------------------------------------------------------

    def disk_usage_bytes(self) -> int:
        if not self._dir.exists():
            return 0
        return sum(p.stat().st_size for p in self._dir.glob("*.npz"))

    def entry_count(self) -> int:
        if not self._dir.exists():
            return 0
        return len(list(self._dir.glob("*.npz")))
