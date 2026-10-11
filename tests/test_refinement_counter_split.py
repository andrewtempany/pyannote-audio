"""Tests for the counter split in the oracle assignment strategy.

Background (see Obsidian-Diarisation/Tickets/Open/oracle-experiment-closeout.md,
item 1): `_dominant_reference_speaker` returned `None` for two unrelated
conditions -- an empty support (the pair has no active frames at all) and a
non-empty support that intersects no reference speaker. The strategy loop
funnelled both into a single `no_reference_overlap` counter, and a planning
conclusion was then drawn from that conflated number: 96.3% of the 56,998
pairs it reported turned out to be empty tensor slots from the fixed-width
`local_num_speakers` axis, not tracks over silence.

These tests pin the two dispositions apart and pin the partition invariant
that makes the counts trustworthy as a complete account rather than a sample.

Fixtures follow tests/test_oracle_refinement.py: real pyannote.core objects,
non-overlapping chunks (step == duration), 1s chunks of 4 frames so the
overlap arithmetic is checkable by hand.
"""

import numpy as np
from pyannote.core import Annotation, Segment, SlidingWindow, SlidingWindowFeature

from harness.refinement import _new_counts, make_oracle_strategy


def _segmentations(binary_masks, chunk_duration=1.0):
    data = np.stack(binary_masks, axis=0).astype(float)
    sliding_window = SlidingWindow(start=0.0, duration=chunk_duration, step=chunk_duration)
    return SlidingWindowFeature(data, sliding_window)


def _run(reference, segmentations, hard_clusters, scope="all_pairs"):
    """Invoke a freshly built strategy and hand back (result, counts)."""
    num_chunks, _num_frames, local_num_speakers = segmentations.data.shape
    embeddings = np.zeros((num_chunks, local_num_speakers, 4))
    num_clusters = int(hard_clusters.max()) + 1 if hard_clusters.size else 1
    soft_clusters = np.zeros((num_chunks, local_num_speakers, num_clusters))
    centroids = np.zeros((num_clusters, 4))

    strategy = make_oracle_strategy(reference, oracle_scope=scope)
    result = strategy(embeddings, hard_clusters, soft_clusters, centroids, segmentations)
    return result, strategy.counts


def test_empty_support_counted_apart_from_no_reference_overlap():
    # Two pairs that both used to land in `no_reference_overlap`, for
    # different reasons:
    #   chunk 0, slot 1 -- all frames zero. An empty tensor slot: the
    #     fixed-width local_num_speakers axis always allocates 2 slots per
    #     chunk whether or not a second speaker is active. Never a candidate
    #     for assignment at all.
    #   chunk 1, slot 0 -- frames ARE active, over [1, 2), but the reference
    #     has no speaker there. A genuine silence-dwelling track.
    # The whole point of the split is that these mean different things.
    segmentations = _segmentations(
        [
            np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),  # slot 1 empty
            np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),  # slot 1 empty
        ]
    )
    hard_clusters = np.array([[0, 0], [0, 0]])

    # reference covers [0, 1) only -- so chunk 1's active slot 0 sits in
    # reference silence, while both slot-1 pairs have no frames at all.
    reference = Annotation()
    reference[Segment(0.0, 1.0)] = "A"

    _result, counts = _run(reference, segmentations, hard_clusters)

    # chunk0/slot0 has active frames over [0,1) and overlaps A -> relabelled.
    assert counts["relabelled"] == 1
    # chunk1/slot0 has active frames but no reference speaker -> the genuine
    # silence-dwelling disposition, and the only pair on that path.
    assert counts["no_reference_overlap"] == 1
    # both slot-1 pairs are empty tensor slots -> the new path.
    assert counts["empty_support"] == 2


def test_five_paths_partition_every_pair_all_pairs_scope():
    # The partition invariant is what makes these counts a complete account
    # of every pair rather than a sample -- it is what made `unmapped_speaker
    # = 0` trustworthy rather than merely unobserved, and it must survive the
    # split. Fixture deliberately exercises four of the five paths at once.
    segmentations = _segmentations(
        [
            np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
            np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
            np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        ]
    )
    hard_clusters = np.array([[0, 0], [0, 0], [0, 0]])

    reference = Annotation()
    reference[Segment(0.0, 1.0)] = "A"
    # chunk 1's support [1, 2) -> speaker C, whom no cluster maps to, so that
    # pair takes the unmapped_speaker path. chunk 2's support [2, 3) has no
    # reference at all -> no_reference_overlap.
    reference[Segment(1.0, 2.0)] = "C"

    _result, counts = _run(reference, segmentations, hard_clusters)

    num_chunks, _num_frames, local_num_speakers = segmentations.data.shape
    total_pairs = num_chunks * local_num_speakers
    assert sum(counts.values()) == total_pairs == 6

    # And specifically: the three empty slot-1 pairs are on the new path, not
    # inflating no_reference_overlap the way they used to.
    assert counts["empty_support"] == 3
    assert counts["no_reference_overlap"] == 1


def test_five_paths_partition_every_pair_overlap_degraded_scope():
    segmentations = _segmentations(
        [
            np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
            np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        ]
    )
    hard_clusters = np.array([[0, 0], [0, 0]])

    reference = Annotation()
    reference[Segment(0.0, 1.0), "a"] = "A"
    reference[Segment(1.0, 2.0), "a2"] = "A"
    reference[Segment(1.0, 2.0), "b"] = "B"

    _result, counts = _run(reference, segmentations, hard_clusters, scope="overlap_degraded")

    assert sum(counts.values()) == 4


def test_empty_support_reads_zero_under_overlap_degraded_scope():
    # A deliberately-asserted artifact, not a bug, and the reason the counter
    # docstring calls it out: `_is_overlap_degraded` also returns False on an
    # empty support, and the scope check runs BEFORE the dominant-speaker
    # call. So under overlap_degraded, empty-support pairs are absorbed into
    # `out_of_scope` and never reach the empty_support branch at all.
    #
    # Asserting it here means the asymmetry (0 under overlap_degraded, ~44%
    # under all_pairs on the real corpus) is a recorded expectation rather
    # than something the next reader discovers and chases.
    segmentations = _segmentations(
        [
            np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
            np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        ]
    )
    hard_clusters = np.array([[0, 0], [0, 0]])

    reference = Annotation()
    reference[Segment(0.0, 1.0)] = "A"

    _result, counts = _run(reference, segmentations, hard_clusters, scope="overlap_degraded")

    assert counts["empty_support"] == 0
    # the two empty slots went here instead, along with the clean pairs.
    assert counts["out_of_scope"] == 4

    # Same fixture under all_pairs DOES route them to empty_support -- the
    # asymmetry is scope-driven, not fixture-driven.
    _result, all_pairs_counts = _run(
        reference, segmentations, hard_clusters, scope="all_pairs"
    )
    assert all_pairs_counts["empty_support"] == 2


def test_new_counts_exposes_five_paths():
    counts = _new_counts()
    assert set(counts) == {
        "relabelled",
        "out_of_scope",
        "empty_support",
        "no_reference_overlap",
        "unmapped_speaker",
    }
    assert all(value == 0 for value in counts.values())


def test_new_counts_docstring_drops_the_falsified_unmapped_speaker_claim():
    # `unmapped_speaker` was documented as "the path that holds DER off its
    # theoretical floor". It fired zero times across 126,925 pairs in both
    # recorded runs, so the claim is falsified; under oracle segmentation
    # every reference speaker has speech by construction, which makes the
    # zero structural. Pin it so the claim cannot quietly return.
    docstring = _new_counts.__doc__ or ""
    assert "holds DER off its theoretical floor" not in docstring


class TestBehaviourUnchanged:
    """Criterion 3: the split changes tallying only.

    Which pairs get relabelled, and to what, must be identical -- the
    refinement path is load-bearing for four recorded conditions
    (`identity`, `nearest_centroid`, `oracle:all_pairs`,
    `oracle:overlap_degraded`), and a changed hypothesis would invalidate
    recorded results rather than merely re-describing them.

    Verified over the returned `hard_clusters` array, not by re-running the
    pipeline: the expected arrays below are the values the pre-split code
    produced for these fixtures.
    """

    def test_all_pairs_relabelling_is_unchanged(self):
        segmentations = _segmentations(
            [
                np.ones((4, 1)),
                np.ones((4, 1)),
                np.ones((4, 1)),
                np.ones((4, 1)),
            ]
        )
        hard_clusters = np.array([[0], [1], [1], [1]])

        reference = Annotation()
        reference[Segment(0.0, 1.0)] = "A"
        reference[Segment(1.0, 2.0)] = "A"
        reference[Segment(2.0, 3.0)] = "B"
        reference[Segment(3.0, 4.0)] = "B"

        result, _counts = _run(reference, segmentations, hard_clusters)

        # chunk 1 reassigned 1 -> 0 (dominant speaker A maps to cluster 0);
        # every other pair already agreed with the mapping.
        assert np.array_equal(result, np.array([[0], [0], [1], [1]]))

    def test_overlap_degraded_relabelling_is_unchanged(self):
        segmentations = _segmentations([np.ones((4, 1)), np.ones((4, 1))])
        hard_clusters = np.array([[0], [1]])

        reference = Annotation()
        reference[Segment(0.0, 1.0), "a_only"] = "A"
        reference[Segment(1.0, 2.0), "a_track"] = "A"
        reference[Segment(1.0, 2.0), "b_track"] = "B"

        result, _counts = _run(reference, segmentations, hard_clusters, scope="overlap_degraded")

        assert np.array_equal(result, np.array([[0], [0]]))

    def test_never_invents_a_cluster_absent_from_the_input(self):
        # The core invariant of the oracle strategy, re-pinned here because
        # the split touches the branch that protects it.
        segmentations = _segmentations([np.ones((4, 1)), np.ones((4, 1))])
        hard_clusters = np.array([[0], [0]])

        reference = Annotation()
        reference[Segment(0.0, 1.0)] = "A"
        reference[Segment(1.0, 2.0)] = "C"

        result, counts = _run(reference, segmentations, hard_clusters)

        assert set(np.unique(result)).issubset(set(np.unique(hard_clusters)))
        assert counts["unmapped_speaker"] == 1

    def test_empty_support_pairs_are_left_unchanged(self):
        # Reclassifying a pair must not start relabelling it.
        segmentations = _segmentations(
            [np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])]
        )
        hard_clusters = np.array([[0, 7]])

        reference = Annotation()
        reference[Segment(0.0, 1.0)] = "A"

        result, counts = _run(reference, segmentations, hard_clusters)

        assert counts["empty_support"] == 1
        assert result[0, 1] == 7  # untouched
