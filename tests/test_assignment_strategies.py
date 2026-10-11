"""Assignment-classifier WS3: target-pair strategies."""

import numpy as np
import pytest
from pyannote.core import SlidingWindow, SlidingWindowFeature

from harness import assignment_common as ac
from harness import assignment_strategies as st
from harness.refinement import get_refinement_strategy, is_valid_refinement_strategy, refinement_id_for
from harness.run_manifest import VALID_REFINEMENT_STRATEGIES


def _meeting(hard, emb, centroids, target, active=None):
    chunks, speakers = hard.shape
    active = np.ones_like(target) if active is None else active
    seg = SlidingWindowFeature(np.ones((chunks, 4, speakers), dtype=np.float32),
                               SlidingWindow(start=0, duration=10, step=1))
    return ac.ClusteredMeeting(uri="x", embeddings=emb, segmentations=seg, hard_clusters=hard.copy(),
                               soft_clusters=None, centroids=centroids, active=active,
                               held_back=target, outlier=np.zeros_like(target))


E = np.eye(3)  # three orthogonal centroids


def test_nc_target_changes_targets_only():
    hard = np.array([[0, 1, 2]])
    emb = np.array([[E[2], E[2], E[2]]])  # every pair is nearest to centroid 2
    target = np.array([[False, True, False]])
    out = st.decide_nc(_meeting(hard, emb, E, target))
    assert out.tolist() == [[0, 2, 2]]


def test_constrained_target_avoids_clusters_held_by_non_targets():
    hard = np.array([[0, 1, 1]])
    emb = np.array([[E[0], 0.9 * E[0] + 0.1 * E[2], E[1]]])
    target = np.array([[False, True, False]])  # cluster 0 and 1 held by non-targets
    out = st.decide_constrained(_meeting(hard, emb, E, target))
    assert out.tolist() == [[0, 2, 1]]


def test_knn_target_uses_non_target_members():
    hard = np.array([[0, 1, 1], [0, 1, 0]])
    emb = np.array([[E[0], E[1], E[2]], [E[0], E[1], E[1] * 0.99 + E[0] * 0.01]])
    target = np.array([[False, False, False], [False, False, True]])
    out = st.decide_knn(_meeting(hard, emb, E, target), k=1)
    assert out[1, 2] == 1


def test_non_target_change_is_a_hard_fail():
    hard = np.array([[0, 1, 2]])
    m = _meeting(hard, np.array([[E[0], E[1], E[2]]]), E, np.array([[False, True, False]]))
    with pytest.raises(st.NonTargetLabelChanged):
        st.checked_decision(m, lambda mm: np.array([[1, 1, 2]]))


def test_names_registered_everywhere_and_ids_distinct():
    for name in st.STATIC_NAMES:
        assert name in VALID_REFINEMENT_STRATEGIES
        assert is_valid_refinement_strategy(name)
    ids = {refinement_id_for(n) for n in ("identity", "nearest_centroid", *st.STATIC_NAMES)}
    assert len(ids) == 2 + len(st.STATIC_NAMES)
    assert not is_valid_refinement_strategy("nc_targett")


@pytest.fixture(scope="module")
def hdbscan():
    try:
        pipeline = ac.load_pipeline("hdbscan")
    except Exception as exc:
        pytest.skip(f"pipeline unavailable: {exc}")
    cache = ac.intermediate_cache(pipeline)
    if not cache.path("IS1009a").exists():
        pytest.skip("intermediates not cached")
    seg, emb = ac.load_intermediates(cache, "IS1009a")
    return pipeline, seg, emb


def test_hdbscan_nc_target_reproduces_identity(hdbscan):
    pipeline, seg, emb = hdbscan
    m = ac.cluster_meeting(pipeline, seg, emb, "IS1009a")
    assert np.array_equal(st.decide_nc(m), m.hard_clusters)


@pytest.mark.parametrize("name", ["nc_target", "constrained_target", "knn_target_k10"])
def test_hook_path_matches_offline_path(hdbscan, name):
    pipeline, seg, emb = hdbscan
    offline = st.checked_decision(ac.cluster_meeting(pipeline, seg, emb, "IS1009a"), st.decider(name))
    # in-pipeline order: clustering (sets last_diagnostics), then the hook
    hard, soft, centroids = pipeline.clustering(embeddings=emb.copy(), segmentations=seg,
                                                num_clusters=None, min_clusters=1, max_clusters=np.inf)
    hooked = get_refinement_strategy(name, pipeline)(emb, hard, soft, centroids, seg)
    assert np.array_equal(hooked, offline)
    assert hooked.dtype == hard.dtype
