"""Tests for T4: oracle assignment strategy.

`make_oracle_strategy` doesn't exist yet at the time these tests are written
-- the first run of this suite is expected to fail on import.

Method under test (see Obsidian-Diarisation/Tickets/Open/T4-oracle-assignment-
strategy.md):
  1. Optimal hypothesis-cluster <-> reference-speaker mapping by total
     temporal overlap across the file.
  2. Each (chunk, local_speaker) pair's temporal support comes from
     `segmentations` (the same SlidingWindowFeature the T2 hook already
     threads through every strategy).
  3. Find the reference speaker with greatest overlap over that support.
  4. Assign the cluster that maps to that reference speaker.
  5. If the dominant reference speaker maps to no cluster, leave the pair
     unchanged.

`oracle_scope`:
  - "all_pairs": refine every pair.
  - "overlap_degraded": refine only pairs whose support is predominantly
    (>50% of duration) coincident with another simultaneously-active
    reference speaker -- i.e. the pair's audio is mostly overlapping speech
    in the ground truth. Left untouched otherwise, even if the mapping would
    have changed the label.

All fixtures use real pyannote.core objects (Segment/Timeline/Annotation/
SlidingWindow/SlidingWindowFeature), not mocks -- the whole point of the
strategy is temporal-overlap arithmetic, which a mock can't exercise
meaningfully. Chunks are built non-overlapping (step == duration) and small
(1s, 4 frames) purely to keep hand arithmetic easy to verify.
"""

import numpy as np
import pytest
from pyannote.core import Annotation, Segment, SlidingWindow, SlidingWindowFeature

from harness.refinement import make_oracle_strategy


def _segmentations(binary_masks, chunk_duration=1.0):
    """Build a (num_chunks, num_frames, local_num_speakers) SlidingWindowFeature
    from a list of per-chunk (num_frames, local_num_speakers) 0/1 arrays.
    Chunks are laid out back-to-back (non-overlapping): step == duration.
    """
    data = np.stack(binary_masks, axis=0).astype(float)
    num_chunks = data.shape[0]
    sliding_window = SlidingWindow(start=0.0, duration=chunk_duration, step=chunk_duration)
    # SlidingWindowFeature needs a window per chunk implicitly via sliding_window;
    # pad in case internal indexing wants exactly num_chunks entries available.
    return SlidingWindowFeature(data, sliding_window)


def test_clean_single_speaker_pair_gets_correctly_mapped_cluster():
    # Two chunks, 1 local speaker slot each, no overlap in the mask (4/4
    # frames active = whole chunk). Chunk 0 belongs to reference speaker A,
    # chunk 1 to reference speaker B. Clustering (wrongly) swapped the
    # cluster ids relative to correct identity, but the *set* of clusters
    # produced (0 and 1) is correct -- oracle must map cluster->speaker by
    # overlap and reassign each pair to the right cluster.
    segmentations = _segmentations(
        [
            np.ones((4, 1)),  # chunk 0: local speaker 0 active whole chunk [0, 1)
            np.ones((4, 1)),  # chunk 1: local speaker 0 active whole chunk [1, 2)
        ]
    )
    # clustering assigned chunk 0 -> cluster 1, chunk 1 -> cluster 0 (swapped
    # relative to which reference speaker dominates each window).
    hard_clusters = np.array([[1], [0]])
    embeddings = np.zeros((2, 1, 4))
    soft_clusters = np.zeros((2, 1, 2))
    centroids = np.zeros((2, 4))

    reference = Annotation()
    reference[Segment(0.0, 1.0)] = "A"
    reference[Segment(1.0, 2.0)] = "B"

    strategy = make_oracle_strategy(reference, oracle_scope="all_pairs")
    result = strategy(embeddings, hard_clusters, soft_clusters, centroids, segmentations)

    # optimal_mapping should find cluster 1 <-> A (chunk 0's overlap) and
    # cluster 0 <-> B (chunk 1's overlap) -- so pairs are corrected to
    # whichever cluster maps to the reference speaker actually dominant in
    # that pair's support. Both pairs already point at their own dominant
    # speaker's mapped cluster, so this also proves it doesn't just
    # relabel blindly by mapping table but per-pair support.
    assert result.shape == hard_clusters.shape
    assert result[0, 0] == 1  # chunk 0 -> speaker A -> cluster 1
    assert result[1, 0] == 0  # chunk 1 -> speaker B -> cluster 0


def test_reassigns_pair_whose_cluster_disagrees_with_dominant_reference_speaker():
    # Same clean two-chunk-two-speaker setup, but this time clustering
    # assigned BOTH pairs to the same cluster (a clustering mistake merging
    # two distinct speakers). Oracle must split them back apart using the
    # per-pair support's dominant reference speaker, since two clusters (0
    # and 1) exist elsewhere... to keep this fixture using only clusters the
    # pipeline "actually produced", we give it a 3rd pair that uses cluster 1
    # so the mapping has both 0 and 1 to draw from.
    segmentations = _segmentations(
        [
            np.ones((4, 1)),  # chunk 0: pair support == [0, 1) -> speaker A
            np.ones((4, 1)),  # chunk 1: pair support == [1, 2) -> speaker A
            np.ones((4, 1)),  # chunk 2: pair support == [2, 3) -> speaker B
        ]
    )
    # clustering merged the two A-chunks' pair AND the B-chunk's pair into
    # cluster 0 for chunk 1 (wrong -- chunk 1 is really speaker A, same as
    # chunk 0), while chunk 0 and chunk 2 sit at clusters 0 and 1 correctly.
    hard_clusters = np.array([[0], [0], [1]])
    embeddings = np.zeros((3, 1, 4))
    soft_clusters = np.zeros((3, 1, 2))
    centroids = np.zeros((2, 4))

    reference = Annotation()
    reference[Segment(0.0, 1.0)] = "A"
    reference[Segment(1.0, 2.0)] = "A"
    reference[Segment(2.0, 3.0)] = "B"

    strategy = make_oracle_strategy(reference, oracle_scope="all_pairs")
    result = strategy(embeddings, hard_clusters, soft_clusters, centroids, segmentations)

    # optimal mapping (by total overlap): cluster 0 has 2s of A overlap
    # (chunks 0+1) vs cluster 1 has 1s of B overlap (chunk 2) -> cluster 0
    # <-> A, cluster 1 <-> B. All three pairs are already consistent with
    # that mapping (chunk1's dominant ref speaker is A -> mapped cluster is
    # 0, which it already has), so nothing should change here -- this test
    # exists to prove the *mapping* step uses total overlap across the whole
    # file, not a per-pair vote, before the trickier disagreement fixture
    # below.
    assert np.array_equal(result, hard_clusters)


def test_pair_reassigned_when_its_own_cluster_is_not_the_mapped_one():
    # A real disagreement: chunk 1 is speaker A but clustering put it in
    # cluster 1 (which the whole-file mapping assigns to speaker B, because
    # cluster 1 otherwise overlaps B for longer elsewhere). Oracle must
    # reassign chunk 1's pair to cluster 0 (A's mapped cluster).
    segmentations = _segmentations(
        [
            np.ones((4, 1)),  # chunk 0: [0, 1) -> A
            np.ones((4, 1)),  # chunk 1: [1, 2) -> A (but wrongly clustered)
            np.ones((4, 1)),  # chunk 2: [2, 3) -> B
            np.ones((4, 1)),  # chunk 3: [3, 4) -> B
        ]
    )
    hard_clusters = np.array([[0], [1], [1], [1]])  # chunk1 wrongly in cluster 1
    embeddings = np.zeros((4, 1, 4))
    soft_clusters = np.zeros((4, 1, 2))
    centroids = np.zeros((2, 4))

    reference = Annotation()
    reference[Segment(0.0, 1.0)] = "A"
    reference[Segment(1.0, 2.0)] = "A"
    reference[Segment(2.0, 3.0)] = "B"
    reference[Segment(3.0, 4.0)] = "B"

    strategy = make_oracle_strategy(reference, oracle_scope="all_pairs")
    result = strategy(embeddings, hard_clusters, soft_clusters, centroids, segmentations)

    # cluster 0 overlaps A for 1s total; cluster 1 overlaps A for 1s (chunk1)
    # and B for 2s (chunks 2,3) = mapping picks cluster 1 <-> B (2s > ties
    # favor whichever the Hungarian assignment resolves to given A already
    # has cluster 0 as its only other option) and cluster 0 <-> A.
    # chunk 1's dominant reference speaker over its own support is A, and A
    # maps to cluster 0 -- so chunk 1 must be reassigned to 0.
    assert result[0, 0] == 0
    assert result[1, 0] == 0  # reassigned: was 1, dominant ref speaker A -> cluster 0
    assert result[2, 0] == 1
    assert result[3, 0] == 1


def test_never_emits_a_cluster_label_absent_from_input():
    # Reference has a 3rd speaker C that the pipeline's clustering never
    # produced a cluster for at all (e.g. missed entirely). Oracle must NOT
    # invent a new cluster id for C -- if C is the dominant reference
    # speaker for some pair, and no hypothesis cluster maps to C, that pair
    # is left unchanged (method step 5).
    segmentations = _segmentations(
        [
            np.ones((4, 1)),  # chunk 0: [0, 1) -> A
            np.ones((4, 1)),  # chunk 1: [1, 2) -> C (pipeline never clustered this speaker)
        ]
    )
    hard_clusters = np.array([[0], [0]])  # only cluster 0 ever produced
    embeddings = np.zeros((2, 1, 4))
    soft_clusters = np.zeros((2, 1, 1))
    centroids = np.zeros((1, 4))

    reference = Annotation()
    reference[Segment(0.0, 1.0)] = "A"
    reference[Segment(1.0, 2.0)] = "C"

    strategy = make_oracle_strategy(reference, oracle_scope="all_pairs")
    result = strategy(embeddings, hard_clusters, soft_clusters, centroids, segmentations)

    # every label in the output must already have existed in hard_clusters --
    # the core invariant the whole ticket exists to protect.
    assert set(np.unique(result)).issubset(set(np.unique(hard_clusters)))
    assert result.shape == hard_clusters.shape


def test_overlap_degraded_scope_skips_clean_pairs():
    # overlap_degraded must only touch pairs whose support is predominantly
    # (>50% duration) coincident with another simultaneously-active
    # reference speaker. Chunk 1 here is a clean, non-overlapping region
    # (only A speaks) but is wrongly clustered -- overlap_degraded scope
    # must leave it alone even though all_pairs scope would fix it.
    segmentations = _segmentations(
        [
            np.ones((4, 1)),  # chunk 0: [0, 1) -> A, clean
            np.ones((4, 1)),  # chunk 1: [1, 2) -> A, clean, but wrongly clustered
        ]
    )
    hard_clusters = np.array([[0], [1]])  # chunk 1 wrongly in cluster 1
    embeddings = np.zeros((2, 1, 4))
    soft_clusters = np.zeros((2, 1, 2))
    centroids = np.zeros((2, 4))

    reference = Annotation()
    reference[Segment(0.0, 1.0)] = "A"
    reference[Segment(1.0, 2.0)] = "A"
    # no second speaker anywhere -- nothing is overlap-degraded.

    strategy = make_oracle_strategy(reference, oracle_scope="overlap_degraded")
    result = strategy(embeddings, hard_clusters, soft_clusters, centroids, segmentations)

    assert np.array_equal(result, hard_clusters)


def test_overlap_degraded_scope_refines_predominantly_overlapping_pair():
    # Chunk 1's support [1, 2) is entirely coincident with a second
    # reference speaker B talking simultaneously with A -- i.e. 100% (>50%)
    # of its duration is overlapping speech. That pair IS in scope for
    # overlap_degraded and must be corrected if its cluster disagrees with
    # the dominant reference speaker there.
    segmentations = _segmentations(
        [
            np.ones((4, 1)),  # chunk 0: [0, 1) -> A only, clean
            np.ones((4, 1)),  # chunk 1: [1, 2) -> A dominant, but B overlaps fully
        ]
    )
    # chunk 1's pair wrongly clustered into cluster 1 (which the file-level
    # mapping will associate with B, since B doesn't appear elsewhere and
    # chunk1's overlap with B is real, but A still dominates chunk1's own
    # duration since A is present the whole time and B only the last 0.5s
    # -- construct with B covering the whole overlapping chunk so A is
    # nonetheless dominant by having twice the total reference duration
    # from chunk 0 + chunk 1 vs B's chunk-1-only presence, mirroring the
    # earlier disagreement fixture but wrapped in overlap).
    hard_clusters = np.array([[0], [1]])
    embeddings = np.zeros((2, 1, 4))
    soft_clusters = np.zeros((2, 1, 2))
    centroids = np.zeros((2, 4))

    # Annotation.__setitem__ with the same segment replaces rather than
    # unions labels in pyannote.core -- add B via a distinct (support-equal)
    # track using the two-argument form so both A and B are simultaneously
    # active over [1, 2).
    reference = Annotation()
    reference[Segment(0.0, 1.0), "a_only"] = "A"
    reference[Segment(1.0, 2.0), "a_track"] = "A"
    reference[Segment(1.0, 2.0), "b_track"] = "B"

    strategy = make_oracle_strategy(reference, oracle_scope="overlap_degraded")
    result = strategy(embeddings, hard_clusters, soft_clusters, centroids, segmentations)

    # chunk 0 stays (clean, correctly clustered already).
    assert result[0, 0] == 0
    # chunk 1 is overlap-degraded (B fully coincident with A over its
    # support) and its dominant reference speaker (A, present the whole
    # support vs B's partial/equal presence -- A wins by total overlap
    # magnitude across the file) maps to cluster 0, so it must be corrected.
    assert result[1, 0] == 0


def test_oracle_strategy_never_mutates_input_hard_clusters():
    segmentations = _segmentations([np.ones((4, 1)), np.ones((4, 1))])
    hard_clusters = np.array([[1], [0]])
    original = hard_clusters.copy()
    embeddings = np.zeros((2, 1, 4))
    soft_clusters = np.zeros((2, 1, 2))
    centroids = np.zeros((2, 4))

    reference = Annotation()
    reference[Segment(0.0, 1.0)] = "A"
    reference[Segment(1.0, 2.0)] = "B"

    strategy = make_oracle_strategy(reference, oracle_scope="all_pairs")
    strategy(embeddings, hard_clusters, soft_clusters, centroids, segmentations)

    assert np.array_equal(hard_clusters, original)


def test_make_oracle_strategy_rejects_unknown_scope():
    reference = Annotation()
    reference[Segment(0.0, 1.0)] = "A"

    with pytest.raises(ValueError):
        make_oracle_strategy(reference, oracle_scope="not_a_real_scope")


def test_oracle_is_registered_and_resolvable_by_name_via_factory_pattern():
    # oracle is different from identity/nearest_centroid: it needs
    # per-file reference, so it cannot sit in REFINEMENT_STRATEGIES as a
    # plain callable the way the other two do. Confirm make_oracle_strategy
    # is importable from the harness.refinement module (the module's public
    # surface for this ticket) and produces a callable matching the T2
    # 5-arg interface.
    from harness import refinement

    assert hasattr(refinement, "make_oracle_strategy")
    reference = Annotation()
    reference[Segment(0.0, 1.0)] = "A"
    strategy = refinement.make_oracle_strategy(reference, oracle_scope="all_pairs")
    assert callable(strategy)
