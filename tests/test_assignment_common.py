"""Assignment-classifier WS0 checks on a real cached meeting (A3, A4, A5).

Skipped when the pre-clustering cache for the test split is absent. Loads the
real community-1 pipeline on CPU (no inference: segmentation and embeddings
come from `.harness_cache/intermediates/`).
"""

import numpy as np
import pytest

from harness import assignment_common as ac

URI = "IS1009a"  # smallest test meeting


@pytest.fixture(scope="module", params=["vbx", "hdbscan"])
def meeting(request):
    try:
        pipeline = ac.load_pipeline(request.param)
    except Exception as exc:  # no token / offline
        pytest.skip(f"pipeline unavailable: {exc}")
    cache = ac.intermediate_cache(pipeline)
    if not cache.path(URI).exists():
        pytest.skip("test-split intermediates not cached")
    segmentations, embeddings = ac.load_intermediates(cache, URI)
    return request.param, pipeline, ac.cluster_meeting(pipeline, segmentations, embeddings, URI)


def test_held_back_mask_matches_filter_embeddings(meeting):
    _, pipeline, m = meeting
    _, chunk_idx, speaker_idx = pipeline.clustering.filter_embeddings(
        m.embeddings, segmentations=m.segmentations
    )
    kept = np.zeros_like(m.active)
    kept[chunk_idx, speaker_idx] = True
    assert np.array_equal(m.held_back, m.active & ~kept)
    assert m.held_back.sum() > 0


def test_a4_centroids_and_soft_clusters_aligned_with_hard_clusters(meeting):
    _, _, m = meeting
    num_clusters = m.centroids.shape[0]
    assert m.centroids is not None and m.soft_clusters is not None
    assert m.soft_clusters.shape == m.hard_clusters.shape + (num_clusters,)
    labels = m.hard_clusters[m.active]
    assert labels.min() >= 0 and labels.max() < num_clusters
    # every centroid is used by some active pair
    assert set(np.unique(labels)) == set(range(num_clusters))


def test_assignment_rule_per_method(meeting):
    method, _, m = meeting
    if method == "hdbscan":
        # plain nearest centroid for every pair: nc_target == identity holds
        assert np.array_equal(m.hard_clusters, np.argmax(m.soft_clusters, axis=2))
    else:
        # VBx: per-chunk Hungarian, so active pairs in one chunk never share a cluster
        for c in range(m.hard_clusters.shape[0]):
            labels = m.hard_clusters[c][m.active[c]]
            assert len(set(labels.tolist())) == len(labels)


def test_a5_held_back_embeddings_present(meeting):
    _, _, m = meeting
    assert not np.isnan(m.embeddings[m.held_back]).any()


def test_outliers_only_under_hdbscan_and_never_held_back(meeting):
    method, _, m = meeting
    if method == "vbx":
        assert not m.outlier.any()
    else:
        assert m.outlier.any()
        assert not (m.outlier & m.held_back).any()
