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
from pyannote.core import Annotation, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate


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


_ORACLE_SCOPES = ("all_pairs", "overlap_degraded")


def _pair_timeline(chunk_index: int, active_frames: np.ndarray, sliding_window) -> Timeline:
    """Build a Timeline covering the active frames of one (chunk, local_speaker)
    pair. `active_frames` is a 1-D boolean/0-1 array over frames within the
    chunk; frame `f`'s time span is a linear subdivision of the chunk's own
    segment (sliding_window[chunk_index]).
    """
    chunk_segment = sliding_window[chunk_index]
    num_frames = active_frames.shape[0]
    frame_duration = chunk_segment.duration / num_frames

    timeline = Timeline()
    for f in range(num_frames):
        if active_frames[f]:
            start = chunk_segment.start + f * frame_duration
            end = start + frame_duration
            timeline.add(_make_segment(start, end))
    return timeline.support()


def _make_segment(start: float, end: float):
    from pyannote.core import Segment

    return Segment(start, end)


def _reconstruct_hypothesis(hard_clusters: np.ndarray, segmentations) -> Annotation:
    """Build a hypothesis Annotation from hard_clusters + segmentations, one
    label per cluster id, so it can be handed to
    DiarizationErrorRate.optimal_mapping alongside the reference. Only
    clusters actually present in hard_clusters ever become labels here.
    """
    sliding_window = segmentations.sliding_window
    num_chunks, _num_frames, local_num_speakers = segmentations.data.shape

    hypothesis = Annotation()
    for chunk_index in range(num_chunks):
        for local_speaker in range(local_num_speakers):
            cluster = hard_clusters[chunk_index, local_speaker]
            if cluster < 0:
                continue  # inactive/unassigned pair, e.g. masked speakers
            active_frames = segmentations.data[chunk_index, :, local_speaker] > 0
            support = _pair_timeline(chunk_index, active_frames, sliding_window)
            for segment in support:
                hypothesis[segment, f"pair-{chunk_index}-{local_speaker}"] = str(cluster)
    return hypothesis


def _dominant_reference_speaker(support: Timeline, reference: Annotation) -> Optional[str]:
    """The reference speaker with greatest total overlap with `support`, or
    None if `support` is empty or overlaps no reference speaker at all.
    """
    if len(support) == 0:
        return None

    cropped = reference.crop(support, mode="intersection")
    best_speaker = None
    best_duration = 0.0
    for speaker in cropped.labels():
        duration = cropped.label_timeline(speaker).duration()
        if duration > best_duration:
            best_duration = duration
            best_speaker = speaker
    return best_speaker


def _is_overlap_degraded(support: Timeline, reference: Annotation) -> bool:
    """True when more than half of `support`'s duration is coincident with a
    *second* simultaneously-active reference speaker (i.e. the pair's audio
    is predominantly overlapping speech in the ground truth).
    """
    total = support.duration()
    if total == 0.0:
        return False

    overlapping_regions = reference.get_overlap().crop(support, mode="intersection")
    return overlapping_regions.duration() > 0.5 * total


def _new_counts() -> Dict[str, int]:
    """A fresh tally of the four ways the oracle loop can dispose of a pair.

    Kept as four separate paths rather than one skip total because they mean
    different things diagnostically. `unmapped_speaker` in particular is the
    path that holds DER off its theoretical floor -- the dominant reference
    speaker maps to no cluster the pipeline produced, so the pair can't be
    relabelled without inventing a cluster -- and collapsing it into a
    generic skip count would discard the most informative number.
    """
    return {
        "relabelled": 0,
        "out_of_scope": 0,
        "no_reference_overlap": 0,
        "unmapped_speaker": 0,
    }


def make_oracle_strategy(reference: Annotation, oracle_scope: str = "all_pairs") -> Callable:
    """Build a refinement strategy that assigns each pair the hypothesis
    cluster mapped (by whole-file temporal overlap) to that pair's dominant
    reference speaker -- but only ever among clusters the pipeline actually
    produced. See Obsidian-Diarisation/Tickets/Open/T4-oracle-assignment-
    strategy.md for the full method.

    Unlike `identity`/`nearest_centroid`, this returns a closure rather than
    being a plain module-level function directly in REFINEMENT_STRATEGIES:
    it needs the current file's reference Annotation, which the fixed 5-arg
    refinement interface (embeddings, hard_clusters, soft_clusters,
    centroids, segmentations) has no slot for. The harness is responsible
    for calling this once per file (see run_harness.py) and setting
    `pipeline.refinement` to the result before running that file.

    The returned closure carries a `counts` dict recording how each pair was
    disposed of -- see `_new_counts` for the four paths and why they're kept
    apart. It's reset on every call, so it always describes the most recent
    invocation rather than accumulating across files.
    """
    if oracle_scope not in _ORACLE_SCOPES:
        raise ValueError(
            f"Unsupported oracle_scope {oracle_scope!r}; expected one of {_ORACLE_SCOPES}"
        )

    def strategy(
        embeddings: Optional[np.ndarray],
        hard_clusters: np.ndarray,
        soft_clusters: Optional[np.ndarray],
        centroids: Optional[np.ndarray],
        segmentations,
    ) -> np.ndarray:
        result = hard_clusters.copy()
        counts = _new_counts()
        strategy.counts = counts

        hypothesis = _reconstruct_hypothesis(hard_clusters, segmentations)
        if len(hypothesis) == 0:
            return result

        # hyp_cluster_label (str) -> ref_speaker, by total temporal overlap
        # across the whole file -- method step 1.
        cluster_to_speaker = DiarizationErrorRate().optimal_mapping(reference, hypothesis)
        speaker_to_cluster = {
            speaker: cluster for cluster, speaker in cluster_to_speaker.items()
        }

        sliding_window = segmentations.sliding_window
        num_chunks, _num_frames, local_num_speakers = segmentations.data.shape

        for chunk_index in range(num_chunks):
            for local_speaker in range(local_num_speakers):
                cluster = hard_clusters[chunk_index, local_speaker]
                if cluster < 0:
                    continue

                active_frames = segmentations.data[chunk_index, :, local_speaker] > 0
                support = _pair_timeline(chunk_index, active_frames, sliding_window)

                if oracle_scope == "overlap_degraded" and not _is_overlap_degraded(
                    support, reference
                ):
                    counts["out_of_scope"] += 1
                    continue  # not in scope for this oracle_scope -- leave unchanged

                dominant_speaker = _dominant_reference_speaker(support, reference)
                if dominant_speaker is None:
                    counts["no_reference_overlap"] += 1
                    continue  # no reference overlap at all -- leave unchanged

                mapped_cluster = speaker_to_cluster.get(dominant_speaker)
                if mapped_cluster is None:
                    # method step 5: dominant reference speaker maps to no
                    # cluster the pipeline produced -- never invent one.
                    counts["unmapped_speaker"] += 1
                    continue

                # cluster labels in `hypothesis` are strings (str(cluster));
                # convert back to the original numpy dtype for assignment.
                result[chunk_index, local_speaker] = int(str(mapped_cluster))
                counts["relabelled"] += 1

        return result

    strategy.counts = _new_counts()
    return strategy


REFINEMENT_STRATEGIES: Dict[str, Callable] = {
    "identity": identity,
    "nearest_centroid": nearest_centroid,
}


def get_refinement_strategy(name: str) -> Callable:
    """Look up a refinement strategy by name. Raises KeyError on an unknown
    name -- config-driven selection needs a clear failure for typos."""
    return REFINEMENT_STRATEGIES[name]
