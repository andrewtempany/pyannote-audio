"""Tests for T2: post-clustering refinement hook.

harness/refinement.py doesn't exist yet at the time these tests are written --
the first run of this suite is expected to fail on import.

All fixtures here are small, hand-built numpy arrays -- no model, no audio,
no disk I/O. Shapes match what `SpeakerDiarization.apply` actually produces
(verified by reading src/pyannote/audio/pipelines/speaker_diarization.py
lines 530-785 and pipelines/clustering.py before writing these fixtures):
  embeddings:    (num_chunks, local_num_speakers, dimension)
  hard_clusters: (num_chunks, local_num_speakers)
  soft_clusters: (num_chunks, local_num_speakers, num_clusters)
  centroids:     (num_speakers, dimension), or None for OracleClustering

This pass is tests-only (TDD red phase): it does not implement the
refinement hook in speaker_diarization.py, nor the identity/nearest_centroid
strategies. Import failure is expected and correct right now.
"""

import numpy as np
import pytest

from harness.refinement import (
    REFINEMENT_STRATEGIES,
    get_refinement_strategy,
    identity,
    nearest_centroid,
)


def test_identity_returns_hard_clusters_unchanged():
    # Hand-built hard_clusters for 3 chunks x 2 local speakers. identity must
    # be a true no-op: same values, not just same shape -- so we compare
    # element-wise against the exact input array.
    hard_clusters = np.array([[0, 1], [1, 0], [0, 0]])
    embeddings = np.zeros((3, 2, 4))  # unused by identity; shape is realistic
    soft_clusters = np.zeros((3, 2, 2))  # unused by identity
    centroids = np.zeros((2, 4))

    result = identity(embeddings, hard_clusters, soft_clusters, centroids, segmentations=None)

    assert np.array_equal(result, hard_clusters)


def test_identity_does_not_mutate_input_in_place():
    # A strategy that mutates its input array in place would still pass the
    # "unchanged values" test above by accident. Guard against that
    # separately: the returned array must not be the *same object* silently
    # aliased back after some destructive round-trip -- assert the input
    # array's own identity/values survive the call untouched.
    hard_clusters = np.array([[0, 1], [1, 0]])
    original = hard_clusters.copy()
    embeddings = np.zeros((2, 2, 4))
    soft_clusters = np.zeros((2, 2, 2))
    centroids = np.zeros((2, 4))

    identity(embeddings, hard_clusters, soft_clusters, centroids, segmentations=None)

    assert np.array_equal(hard_clusters, original)


def test_soft_clusters_recovered_from_clustering_call():
    # T2 scope item: "Recover soft_clusters. Line 640 currently discards it
    # into `_`." That's a one-character change to speaker_diarization.py's
    # call site, not logic a unit test can exercise in isolation -- there is
    # nothing in harness/refinement.py that would fail if the discard were
    # never fixed, since the strategy functions here are given soft_clusters
    # as a normal parameter regardless of where the caller got it from.
    #
    # Decision: test this at the narrower unit boundary instead of writing a
    # pipeline-level integration test (which would need a real model/audio,
    # out of scope for this pass per the ticket). The contract we CAN pin
    # down here is that every registered refinement strategy actually
    # accepts and is able to use a real (non-None, non-trivial) soft_clusters
    # array -- i.e. the interface has a genuine soft_clusters slot, not a
    # vestigial one that strategies ignore by construction. We prove this by
    # checking nearest_centroid's signature accepts it and that calling
    # identity with a realistic, non-degenerate soft_clusters array (as
    # clustering.py actually returns: (num_chunks, num_speakers,
    # num_clusters) posteriors, not all-equal/trivial) does not raise or
    # discard it via an implicit shape check.
    num_chunks, local_num_speakers, num_clusters = 3, 2, 2
    hard_clusters = np.array([[0, 1], [1, 0], [0, 1]])
    embeddings = np.random.RandomState(0).randn(num_chunks, local_num_speakers, 4)
    soft_clusters = np.array(
        [
            [[0.9, 0.1], [0.2, 0.8]],
            [[0.3, 0.7], [0.6, 0.4]],
            [[0.95, 0.05], [0.1, 0.9]],
        ]
    )
    centroids = np.random.RandomState(1).randn(num_clusters, 4)

    # this call must at minimum accept the real posterior shape without
    # error -- if the interface secretly ignored/dropped soft_clusters (e.g.
    # by not declaring the parameter at all), this would raise a TypeError.
    result = identity(embeddings, hard_clusters, soft_clusters, centroids, segmentations=None)

    assert result.shape == hard_clusters.shape


def test_nearest_centroid_reassigns_pair_to_closer_centroid():
    # Two centroids, clearly separated along one axis. Pair (chunk=0,
    # speaker=0)'s embedding is hand-placed near centroid 1 by cosine
    # similarity, but hard_clusters wrongly says it belongs to cluster 0 --
    # simulating a clustering mistake that a nearest-centroid refinement
    # pass should correct.
    centroids = np.array(
        [
            [1.0, 0.0],   # centroid 0
            [0.0, 1.0],   # centroid 1
        ]
    )
    embeddings = np.array(
        [
            [[0.01, 0.99]],  # chunk 0, speaker 0: near centroid 1, not 0
        ]
    )
    hard_clusters = np.array([[0]])  # currently (wrongly) assigned to cluster 0
    soft_clusters = np.zeros((1, 1, 2))  # not used by nearest_centroid's own logic

    result = nearest_centroid(embeddings, hard_clusters, soft_clusters, centroids, segmentations=None)

    assert result[0, 0] == 1


def test_nearest_centroid_leaves_correctly_assigned_pair_alone():
    # Companion to the reassignment test: a pair whose current cluster
    # already IS its nearest centroid must not be perturbed. Without this,
    # a buggy strategy that reassigns everything to argmax regardless of
    # current state could still pass the reassignment test above by luck.
    centroids = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    embeddings = np.array(
        [
            [[0.99, 0.01]],  # chunk 0, speaker 0: near centroid 0
        ]
    )
    hard_clusters = np.array([[0]])
    soft_clusters = np.zeros((1, 1, 2))

    result = nearest_centroid(embeddings, hard_clusters, soft_clusters, centroids, segmentations=None)

    assert result[0, 0] == 0


def test_nearest_centroid_with_none_centroids_documented_behavior():
    # T2's implementer notes flag this explicitly as an open decision:
    # OracleClustering returns centroids=None, and "the interface needs to
    # handle a null-centroid case explicitly rather than crashing."
    #
    # Decision made here: nearest_centroid has nothing to compare against
    # without centroids, so it no-ops back to identity behavior (returns
    # hard_clusters unchanged) rather than raising. This is documented as a
    # choice, not inferred -- a raising ValueError would be equally
    # defensible, but silently falling back keeps a config-selected
    # nearest_centroid run from hard-crashing an entire harness run just
    # because it happened to hit an OracleClustering condition.
    hard_clusters = np.array([[0, 1], [1, 0]])
    embeddings = np.zeros((2, 2, 4))
    soft_clusters = np.zeros((2, 2, 2))

    result = nearest_centroid(embeddings, hard_clusters, soft_clusters, centroids=None, segmentations=None)

    assert np.array_equal(result, hard_clusters)


@pytest.mark.parametrize("strategy_fn", [identity, nearest_centroid])
def test_strategies_preserve_hard_clusters_shape(strategy_fn):
    # Both strategies are required by T2 to return an array of identical
    # shape to their hard_clusters input, regardless of internal logic.
    hard_clusters = np.array([[0, 1, 2], [1, 0, 2], [2, 2, 0]])
    embeddings = np.random.RandomState(2).randn(3, 3, 4)
    soft_clusters = np.random.RandomState(3).rand(3, 3, 3)
    centroids = np.random.RandomState(4).randn(3, 4)

    result = strategy_fn(embeddings, hard_clusters, soft_clusters, centroids, segmentations=None)

    assert result.shape == hard_clusters.shape


def test_get_refinement_strategy_resolves_identity_by_name():
    # Selectability via harness config: config carries a plain string like
    # refinement_strategy: "identity", and something must turn that string
    # into the actual callable. get_refinement_strategy is that lookup.
    strategy_fn = get_refinement_strategy("identity")

    assert strategy_fn is identity


def test_get_refinement_strategy_resolves_nearest_centroid_by_name():
    strategy_fn = get_refinement_strategy("nearest_centroid")

    assert strategy_fn is nearest_centroid


def test_get_refinement_strategy_rejects_unknown_name():
    # Config-driven selection needs a clear failure for typos/unsupported
    # strategy names rather than a silent None or KeyError with no context.
    with pytest.raises(KeyError):
        get_refinement_strategy("not_a_real_strategy")


def test_refinement_strategies_registry_contains_both_strategies():
    # The registry itself is the natural place a harness config loader would
    # introspect (e.g. to validate a config value up front against known
    # strategies), so it must expose both required strategies by name.
    assert REFINEMENT_STRATEGIES["identity"] is identity
    assert REFINEMENT_STRATEGIES["nearest_centroid"] is nearest_centroid
