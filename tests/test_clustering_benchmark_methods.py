"""Tests for the clustering classes added by the clustering-benchmark ticket.

Routing tests assert on the LIVE pipeline object, like
test_clustering_model_selection.py: the failure being guarded against is a
selection that falls through to another class while the manifest claims
otherwise. Fixtures here are synthetic unit-norm "speaker" blobs, so the
expected structure is known exactly.
"""

import numpy as np
import pytest
from pyannote.core import SlidingWindow, SlidingWindowFeature

from harness.cache_id import pipeline_config_id
from test_clustering_model_selection import (
    CHECKPOINT,
    _patched_loaders,
    build_pipeline,
)

from pyannote.audio.pipelines import clustering as clustering_module


def _blobs(num_blobs=3, per_blob=30, dimension=16, spread=0.05, seed=0):
    """`num_blobs` tight clusters around mutually orthogonal unit vectors."""
    rng = np.random.default_rng(seed)
    centres = np.eye(dimension)[:num_blobs]
    points = np.concatenate(
        [c + spread * rng.standard_normal((per_blob, dimension)) for c in centres]
    )
    labels = np.repeat(np.arange(num_blobs), per_blob)
    return points / np.linalg.norm(points, axis=1, keepdims=True), labels


def _hdbscan(**params):
    instance = clustering_module.HDBSCANClustering(metric="cosine")
    instance.instantiate(
        {"min_cluster_size": 5, "min_samples": 0, "cluster_selection_method": "eom", **params}
    )
    return instance


# --------------------------------------------------------------------------
# HDBSCAN
# --------------------------------------------------------------------------


def test_hdbscan_routing_selects_the_hdbscan_class():
    with _patched_loaders():
        pipeline = build_pipeline(CHECKPOINT, token=None, clustering_model="hdbscan")

    assert isinstance(pipeline.clustering, clustering_module.HDBSCANClustering)
    assert pipeline._expects_num_speakers is False


def test_hdbscan_min_cluster_size_moves_the_final_cache_key():
    params = {"min_samples": 0, "cluster_selection_method": "eom"}
    with _patched_loaders():
        a = build_pipeline(CHECKPOINT, token=None, clustering_model="hdbscan",
                           clustering_params={"min_cluster_size": 5, **params})
        b = build_pipeline(CHECKPOINT, token=None, clustering_model="hdbscan",
                           clustering_params={"min_cluster_size": 10, **params})

    assert pipeline_config_id(CHECKPOINT, a) != pipeline_config_id(CHECKPOINT, b)


def test_hdbscan_never_returns_a_negative_label_with_an_obvious_outlier():
    points, _ = _blobs()
    outlier = np.zeros(points.shape[1])
    outlier[-1] = 1.0  # orthogonal to every blob
    embeddings = np.vstack([points, outlier])

    clusters = _hdbscan().cluster(embeddings, min_clusters=1, max_clusters=len(embeddings))

    assert clusters.min() >= 0
    assert set(np.unique(clusters)) == {0, 1, 2}
    # HDBSCAN itself did call it noise -- otherwise this test proves nothing.
    assert _hdbscan_diagnostics_outliers(embeddings) >= 1


def _hdbscan_diagnostics_outliers(embeddings):
    instance = _hdbscan()
    instance.cluster(embeddings, min_clusters=1, max_clusters=len(embeddings))
    return instance.last_diagnostics["num_outliers"]


def test_hdbscan_outlier_is_folded_into_the_nearest_centroid():
    points, labels = _blobs()
    # Nearer blob 1 (cosine distance 0.9) than any other blob (1.0), and far
    # enough off it that HDBSCAN calls it noise (asserted below; at 0.2 and
    # above HDBSCAN absorbs it into blob 1 instead, which tests nothing).
    near_one = np.zeros(points.shape[1])
    near_one[1], near_one[-1] = 0.1, np.sqrt(1 - 0.1**2)
    embeddings = np.vstack([points, near_one])

    instance = _hdbscan()
    clusters = instance.cluster(embeddings, min_clusters=1, max_clusters=len(embeddings))

    assert instance.last_diagnostics["num_outliers"] >= 1
    blob_one_label = np.bincount(clusters[:-1][labels == 1]).argmax()
    assert clusters[-1] == blob_one_label


def test_hdbscan_all_noise_returns_all_zeros():
    rng = np.random.default_rng(0)
    scattered = rng.standard_normal((6, 16))
    scattered /= np.linalg.norm(scattered, axis=1, keepdims=True)

    instance = _hdbscan(min_cluster_size=7)
    clusters = instance.cluster(scattered, min_clusters=1, max_clusters=6)

    np.testing.assert_array_equal(clusters, np.zeros(6))


def _segmentations(num_chunks, num_frames=50, num_speakers=3):
    """Binary segmentation. In chunks 1+, local speaker 0 is active on frames
    0-29 and speaker 1 on 10-39: each has 10 solo frames (exactly the 20%
    `filter_embeddings` needs) and 20 of its 30 active frames overlapped.
    In chunk 0 both are active on every frame, so neither has a solo frame
    and both are dropped by the filter. Speaker 2 is silent throughout."""
    data = np.zeros((num_chunks, num_frames, num_speakers))
    data[1:, 0:30, 0] = 1
    data[1:, 10:40, 1] = 1
    data[0, :, 0] = 1
    data[0, :, 1] = 1
    return SlidingWindowFeature(data, SlidingWindow(start=0, duration=1, step=1))


def test_hdbscan_full_call_reports_outliers_and_overlap_per_embedding():
    num_chunks = 40
    points, _ = _blobs(num_blobs=2, per_blob=num_chunks, dimension=8)
    embeddings = np.stack(
        [points[:num_chunks], points[num_chunks:], np.full((num_chunks, 8), np.nan)], axis=1
    )

    instance = _hdbscan()
    hard_clusters, _, _ = instance(embeddings, segmentations=_segmentations(num_chunks))

    assert hard_clusters.min() >= 0
    diagnostics = instance.last_diagnostics
    assert diagnostics["num_dropped_by_filter"] == 2  # chunk 0, speakers 0 and 1
    assert diagnostics["num_train"] == 2 * (num_chunks - 1)
    assert diagnostics["mean_overlap_fraction_dropped"] == pytest.approx(1.0)
    rows = diagnostics["rows"]
    assert len(rows) == diagnostics["num_train"]
    assert {"chunk_idx", "speaker_idx", "is_outlier", "assigned_cluster",
            "overlap_fraction"} <= set(rows[0])
    assert all(r["overlap_fraction"] == pytest.approx(2 / 3) for r in rows)
    assert diagnostics["num_outliers"] == sum(r["is_outlier"] for r in rows)


# --------------------------------------------------------------------------
# Spectral with eigengap
# --------------------------------------------------------------------------


def _spectral(**params):
    instance = clustering_module.SpectralEigengapClustering(metric="cosine")
    instance.instantiate({"pruning": 0.3, "max_speakers": 10, **params})
    return instance


def test_spectral_routing_selects_the_spectral_class():
    with _patched_loaders():
        pipeline = build_pipeline(CHECKPOINT, token=None, clustering_model="spectral")

    assert isinstance(pipeline.clustering, clustering_module.SpectralEigengapClustering)
    assert pipeline._expects_num_speakers is False


def test_spectral_pruning_moves_the_final_cache_key():
    with _patched_loaders():
        a = build_pipeline(CHECKPOINT, token=None, clustering_model="spectral",
                           clustering_params={"pruning": 0.1, "max_speakers": 10})
        b = build_pipeline(CHECKPOINT, token=None, clustering_model="spectral",
                           clustering_params={"pruning": 0.2, "max_speakers": 10})

    assert pipeline_config_id(CHECKPOINT, a) != pipeline_config_id(CHECKPOINT, b)


@pytest.mark.parametrize("num_blobs", [2, 3, 5])
def test_eigengap_counts_well_separated_blobs(num_blobs):
    points, labels = _blobs(num_blobs=num_blobs)

    instance = _spectral()
    clusters = instance.cluster(points, min_clusters=1, max_clusters=len(points))

    assert instance.last_diagnostics["k"] == num_blobs
    assert set(np.unique(clusters)) == set(range(num_blobs))
    # each blob lands in exactly one cluster
    for blob in range(num_blobs):
        assert len(np.unique(clusters[labels == blob])) == 1


def test_eigengap_estimate_is_capped_at_max_speakers():
    points, _ = _blobs(num_blobs=5)

    instance = _spectral(max_speakers=3)
    instance.cluster(points, min_clusters=1, max_clusters=len(points))

    assert 1 <= instance.last_diagnostics["k"] <= 3
