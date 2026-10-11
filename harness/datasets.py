"""AMI dataset adapter: (condition, split, data_root) -> (uri, reference, uem).

See TICKET-03-dataset-adapter.md.

Directory convention this adapter expects under `config.data_root`:

    {data_root}/{condition}/{split}.rttm
    {data_root}/{condition}/{split}.uem

Both files use the standard multi-uri RTTM/UEM format (one file covering
every meeting in that split), parsed via pyannote.database.util's
load_rttm/load_uem rather than reimplementing RTTM/UEM parsing here.

Design decision (multi-channel IHM): real AMI meetings under the IHM
condition have multiple raw per-participant headset-channel audio files,
but the ground-truth RTTM/UEM are inherently meeting-level -- one
Annotation covering every speaker in the meeting, independent of which
physical mic channel is later decoded for audio. This adapter's uri
space is therefore exactly the set of uris present in the (condition,
split)'s RTTM/UEM files: one `(uri, reference, uem)` per meeting, never
one per channel. Resolving a uri to a specific audio path (e.g. a
pre-mixed "Mix-Headset" file) is out of scope here -- this adapter never
touches audio, only the ground-truth annotation and evaluation map.
"""

from __future__ import annotations

from typing import Iterator, Tuple

from pyannote.core import Annotation, Timeline
from pyannote.database.util import load_rttm, load_uem

from harness.config import HarnessConfig


class DatasetError(RuntimeError):
    """Raised when AMI RTTM/UEM data cannot be located or parsed as expected."""


class AMIDatasetAdapter:
    """Iterates a (condition, split) as (uri, reference, uem) tuples."""

    def __init__(self, config: HarnessConfig):
        self._config = config

    def __iter__(self) -> Iterator[Tuple[str, Annotation, Timeline]]:
        rttm_path = self._config.data_root / self._config.condition / f"{self._config.split}.rttm"
        uem_path = self._config.data_root / self._config.condition / f"{self._config.split}.uem"

        if not rttm_path.exists():
            raise DatasetError(
                f"No RTTM file found for condition {self._config.condition!r}, "
                f"split {self._config.split!r}: expected {rttm_path}"
            )
        if not uem_path.exists():
            raise DatasetError(
                f"No UEM file found for condition {self._config.condition!r}, "
                f"split {self._config.split!r}: expected {uem_path}"
            )

        references = load_rttm(str(rttm_path))
        uems = load_uem(str(uem_path))

        for uri, uem in uems.items():
            if uri not in references:
                raise DatasetError(
                    f"UEM file {uem_path} references uri {uri!r} but no matching "
                    f"entry was found in RTTM file {rttm_path}"
                )
            yield uri, references[uri], uem
