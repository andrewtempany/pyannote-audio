"""Post-clustering refinement strategies. See
Obsidian-Diarisation/Tickets/Open/T2-post-clustering-refinement-hook.md.

The extension point sits between the clustering call and inactive-speaker
masking in `SpeakerDiarization.apply`. Every strategy here operates on
(chunk, local_speaker) pairs, not frames -- `embeddings` and `hard_clusters`
both have shape (num_chunks, local_num_speakers, ...) per `clustering.py`.

Matches the plain-function house style of `harness/scorer.py` /
`harness/reporter.py` -- no classes.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np


def identity(
    embeddings: Optional[np.ndarray],
    hard_clusters: np.ndarray,
    soft_clusters: Optional[np.ndarray],
    centroids: Optional[np.ndarray],
    segmentations,
) -> np.ndarray:
    """True no-op: return hard_clusters unchanged, without mutating the input."""
    return hard_clusters.copy()


def nearest_centroid(
    embeddings: np.ndarray,
    hard_clusters: np.ndarray,
    soft_clusters: Optional[np.ndarray],
    centroids: Optional[np.ndarray],
    segmentations,
) -> np.ndarray:
    """Reassign each (chunk, local_speaker) pair to whichever centroid is
    closest by cosine similarity.

    When `centroids` is None (the OracleClustering case, which returns no
    centroids when no embeddings are supplied), there is nothing to compare
    against, so this no-ops back to identity behavior rather than raising --
    a documented choice (see T2 Implementation Notes), not inferred.
    """
    if centroids is None:
        return identity(embeddings, hard_clusters, soft_clusters, centroids, segmentations)

    num_chunks, local_num_speakers = hard_clusters.shape

    # cosine similarity between every (chunk, local_speaker) embedding and
    # every centroid: normalize both sides, then dot product.
    flat_embeddings = embeddings.reshape(-1, embeddings.shape[-1])
    embeddings_norm = flat_embeddings / np.clip(
        np.linalg.norm(flat_embeddings, axis=-1, keepdims=True), a_min=1e-8, a_max=None
    )
    centroids_norm = centroids / np.clip(
        np.linalg.norm(centroids, axis=-1, keepdims=True), a_min=1e-8, a_max=None
    )

    similarity = embeddings_norm @ centroids_norm.T
    nearest = np.argmax(similarity, axis=-1)

    return nearest.reshape(num_chunks, local_num_speakers)


REFINEMENT_STRATEGIES: Dict[str, Callable] = {
    "identity": identity,
    "nearest_centroid": nearest_centroid,
}


def get_refinement_strategy(name: str) -> Callable:
    """Look up a refinement strategy by name. Raises KeyError on an unknown
    name -- config-driven selection needs a clear failure for typos."""
    return REFINEMENT_STRATEGIES[name]
