"""Correct-cluster labels for target pairs (assignment-classifier ticket).

The ONLY module in this ticket's code that reads reference RTTMs. The label
for a pair is what the oracle strategy would give it under `all_pairs`
(`harness/refinement.make_oracle_strategy`, read from its per-pair `labels`
record): the cluster mapped to the dominant reference speaker of the pair's
support. -1 means unlabelled (empty support, no reference overlap, or a
reference speaker no cluster maps to).

Test discipline (ticket hard-fail 4): test-split references may be read only
for pair-accuracy evaluation. `load_reference` takes a `purpose` and refuses a
test request for anything else, so feature building and training cannot read
them even by accident. `tests/test_assignment_labels.py` patches this loader
to prove the dataset builder never asks for test URIs.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from pyannote.core import Annotation

from harness.assignment_common import DATA_ROOT, ClusteredMeeting
from harness.refinement import make_oracle_strategy

PURPOSES = ("training", "pair-eval")


class TestReferenceAccess(RuntimeError):
    """A non-evaluation code path asked for a test-split reference."""


@lru_cache(maxsize=4)
def _split_references(split: str) -> dict:
    from pyannote.database.util import load_rttm

    return load_rttm(str(DATA_ROOT / "IHM" / f"{split}.rttm"))


def load_reference(split: str, uri: str, purpose: str) -> Annotation:
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}, got {purpose!r}")
    if split == "test" and purpose != "pair-eval":
        raise TestReferenceAccess(
            f"test reference for {uri} requested for {purpose!r}; only pair-eval may read it"
        )
    return _split_references(split)[uri]


def pair_labels(meeting: ClusteredMeeting, reference: Annotation) -> np.ndarray:
    """(chunks, local_speakers) array of correct clusters; -1 = unlabelled.

    The mapping is computed from the clustering's own `hard_clusters`, exactly
    as the oracle refinement run does.
    """
    oracle = make_oracle_strategy(reference, "all_pairs")
    oracle(meeting.embeddings, meeting.hard_clusters, meeting.soft_clusters,
           meeting.centroids, meeting.segmentations)
    return oracle.labels.copy()
