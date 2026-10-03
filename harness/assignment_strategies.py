"""Assignment strategies for target pairs (assignment-classifier ticket, WS3/WS4).

Every strategy here changes TARGET pairs only (held-back by
`filter_embeddings`, plus HDBSCAN outliers) and leaves every other pair's
label exactly as clustering produced it. The core of each is a
`decide(meeting) -> hard_clusters` function over an
`assignment_common.ClusteredMeeting`, used both:

  * offline, for pair accuracy (`harness/assignment_eval.py`), and
  * inside the pipeline, via `build(name, pipeline)`, which wraps `decide` in
    the 5-arg refinement interface.

Inside the pipeline, outliers are read from `pipeline.clustering.last_diagnostics`.
The refinement hook runs straight after clustering in the same `apply()`
(speaker_diarization.py:656-674), so those diagnostics always describe the
current file. VBx has no diagnostics and no outliers.

The wrapper enforces ticket hard-fail 2: if a non-target pair's label would
change, it raises instead of returning.

Names:
  nc_target               plain nearest centroid (argmax cosine) on target pairs
  constrained_target      nearest-centroid scores, then a per-chunk Hungarian over
                          target pairs and the clusters not already held by
                          non-target active pairs in that chunk
  knn_target_k{5,10,25}   argmax of mean cosine to the k most similar non-target
                          members of each cluster
  classifier:<model_id>, classifier_hungarian:<model_id>   (WS4)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from harness import assignment_common as ac

STATIC_NAMES = ("nc_target", "constrained_target", "knn_target_k5", "knn_target_k10", "knn_target_k25")
MODEL_PREFIXES = ("classifier:", "classifier_hungarian:")


class NonTargetLabelChanged(RuntimeError):
    """Hard-fail 2: a strategy changed a pair it must leave alone."""


def is_assignment_strategy(name: str) -> bool:
    return name in STATIC_NAMES or name.startswith(MODEL_PREFIXES)


def refinement_id(name: str) -> str:
    """Final-cache-key id. Model strategies carry the model file's sha256 (A8)."""
    if name.startswith(MODEL_PREFIXES):
        from harness.assignment_models import model_sha256

        prefix, model_id = name.split(":", 1)
        return f"{prefix}:{model_id}:{model_sha256(model_id)}"
    return name


def _cosine(m: ac.ClusteredMeeting) -> np.ndarray:
    unit = m.embeddings / np.clip(np.linalg.norm(m.embeddings, axis=-1, keepdims=True), 1e-8, None)
    cent = m.centroids / np.clip(np.linalg.norm(m.centroids, axis=-1, keepdims=True), 1e-8, None)
    return unit @ cent.T  # (chunks, speakers, K)


def hungarian_targets(m: ac.ClusteredMeeting, scores: np.ndarray) -> np.ndarray:
    """Per chunk: non-target active pairs keep their clusters; target pairs get a
    one-to-one assignment to the remaining clusters maximising `scores`.
    A target left over (more targets than free clusters) takes its own argmax.
    """
    out = m.hard_clusters.copy()
    target = m.target
    fixed = m.active & ~target
    for c in np.where(target.any(axis=1))[0]:
        speakers = np.where(target[c])[0]
        taken = set(out[c, fixed[c]].tolist())
        free = np.array([k for k in range(scores.shape[2]) if k not in taken])
        if len(free):
            rows, cols = linear_sum_assignment(scores[c][np.ix_(speakers, free)], maximize=True)
            assigned = set()
            for r, col in zip(rows, cols):
                out[c, speakers[r]] = free[col]
                assigned.add(speakers[r])
        else:
            assigned = set()
        for s in speakers:
            if s not in assigned:
                out[c, s] = int(np.argmax(scores[c, s]))
    return out


def _apply_on_targets(m: ac.ClusteredMeeting, decided: np.ndarray) -> np.ndarray:
    out = m.hard_clusters.copy()
    out[m.target] = decided[m.target]
    return out


def decide_nc(m: ac.ClusteredMeeting) -> np.ndarray:
    return _apply_on_targets(m, np.argmax(_cosine(m), axis=2))


def decide_constrained(m: ac.ClusteredMeeting) -> np.ndarray:
    return hungarian_targets(m, _cosine(m))


def knn_scores(m: ac.ClusteredMeeting, k: int) -> np.ndarray:
    """(chunks, speakers, K) mean cosine to the k nearest non-target members; -inf if a
    cluster has no non-target member. Same computation as the dataset's knn_sim."""
    from harness.assignment_dataset import _knn_sims, _unit

    unit = _unit(m.embeddings)
    members = m.active & ~m.target
    tc, ts = np.where(m.target)
    scores = np.full(m.hard_clusters.shape + (m.centroids.shape[0],), -np.inf)
    for cluster in range(m.centroids.shape[0]):
        mc, ms = np.where(members & (m.hard_clusters == cluster))
        sims = _knn_sims(unit[tc, ts], unit[mc, ms], ks=(k,))[k]
        scores[tc, ts, cluster] = np.where(np.isnan(sims), -np.inf, sims)
    return scores


def decide_knn(m: ac.ClusteredMeeting, k: int) -> np.ndarray:
    return _apply_on_targets(m, np.argmax(knn_scores(m, k), axis=2))


def decider(name: str) -> Callable[[ac.ClusteredMeeting], np.ndarray]:
    if name == "nc_target":
        return decide_nc
    if name == "constrained_target":
        return decide_constrained
    if name.startswith("knn_target_k"):
        k = int(name[len("knn_target_k"):])
        return lambda m: decide_knn(m, k)
    if name.startswith(MODEL_PREFIXES):
        from harness.assignment_models import model_decider

        prefix, model_id = name.split(":", 1)
        return model_decider(model_id, hungarian=(prefix == "classifier_hungarian"))
    raise KeyError(name)


def meeting_from_hook(pipeline, embeddings, hard_clusters, soft_clusters, centroids,
                      segmentations) -> ac.ClusteredMeeting:
    active = np.sum(segmentations.data, axis=1) > 0
    outlier = np.zeros_like(active)
    diagnostics = getattr(pipeline.clustering, "last_diagnostics", None) if pipeline is not None else None
    if diagnostics and "rows" in diagnostics:
        for row in diagnostics["rows"]:
            if row["is_outlier"]:
                outlier[row["chunk_idx"], row["speaker_idx"]] = True
    return ac.ClusteredMeeting(
        uri="", embeddings=embeddings, segmentations=segmentations,
        hard_clusters=np.asarray(hard_clusters).copy(),
        soft_clusters=None if soft_clusters is None else np.asarray(soft_clusters),
        centroids=np.asarray(centroids), active=active,
        held_back=ac.held_back_mask(embeddings, segmentations), outlier=outlier,
    )


def checked_decision(m: ac.ClusteredMeeting, decide: Callable) -> np.ndarray:
    out = decide(m)
    keep = ~m.target
    if not np.array_equal(out[keep], m.hard_clusters[keep]):
        raise NonTargetLabelChanged(f"{int(np.sum(out[keep] != m.hard_clusters[keep]))} non-target pairs changed")
    return out.astype(m.hard_clusters.dtype)


def build(name: str, pipeline) -> Callable:
    """5-arg refinement strategy for `name`, bound to `pipeline` for HDBSCAN diagnostics."""
    decide = decider(name)

    def strategy(embeddings, hard_clusters, soft_clusters, centroids, segmentations):
        m = meeting_from_hook(pipeline, embeddings, hard_clusters, soft_clusters, centroids, segmentations)
        return checked_decision(m, decide)

    strategy.assignment_name = name
    return strategy
