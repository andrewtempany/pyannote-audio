"""Shared plumbing for the assignment-classifier ticket.

Rebuilds one meeting's clustering on CPU from the pre-clustering cache, so the
dataset builder, the pair evaluator and the tests all see exactly what
`SpeakerDiarization.apply()` hands the refinement hook
(speaker_diarization.py:656-674):

    hard_clusters, soft_clusters, centroids = clustering(embeddings, binarized_segmentations, ...)
    hard_clusters = refinement(embeddings, hard_clusters, soft_clusters, centroids, segmentations)

community-1 is powerset, so the binarized segmentation IS the segmentation
(:614-615). Speaker-count bounds are what `set_num_speakers(None, None, None)`
gives inside `apply()`: num_clusters None, min 1, max inf.

Nothing here reads a reference RTTM. Labels come from
`harness/assignment_labels.py`, which is the only module allowed to.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from pyannote.core import SlidingWindowFeature

from harness.intermediate_cache import IntermediateCache, _decode_window

CHECKPOINT = "pyannote/speaker-diarization-community-1"
DATA_ROOT = Path.home() / "Code" / "AMI-diarization-setup" / "harness-data"
REPO = Path(__file__).resolve().parent.parent
SWEEP_DIR = REPO / "runs" / "sweeps" / "assignment-classifier"

# The two clustering methods this ticket refines. Values are build_pipeline
# arguments; HDBSCAN is the clustering benchmark's best point HDB-2b-04.
METHODS: Dict[str, dict] = {
    "vbx": {"clustering_model": "pyannote-default", "clustering_params": {}},
    "hdbscan": {
        "clustering_model": "hdbscan",
        "clustering_params": {
            "min_cluster_size": 40,
            "min_samples": 0,
            "cluster_selection_method": "leaf",
        },
    },
}


@dataclass
class ClusteredMeeting:
    """Everything the refinement hook sees for one meeting, plus the target mask."""

    uri: str
    embeddings: np.ndarray  # (chunks, local_speakers, dim), as extracted
    segmentations: SlidingWindowFeature  # binary (chunks, frames, local_speakers)
    hard_clusters: np.ndarray  # (chunks, local_speakers), clustering output
    soft_clusters: np.ndarray  # (chunks, local_speakers, clusters)
    centroids: np.ndarray  # (clusters, dim)
    active: np.ndarray  # (chunks, local_speakers) bool: any active frame
    held_back: np.ndarray  # active and excluded by filter_embeddings
    outlier: np.ndarray  # HDBSCAN noise before fold-in (all False for VBx)

    @property
    def target(self) -> np.ndarray:
        """Pairs a strategy may decide. VBx: held-back. HDBSCAN: held-back | outliers."""
        return self.held_back | self.outlier


def load_pipeline(method: str, token: Optional[str] = None, dotenv: str = ".env"):
    """Build the community-1 pipeline with `method`'s clustering, on CPU."""
    from run_harness import build_pipeline
    from harness.config import HarnessConfig

    if token is None:
        token = HarnessConfig.load(DATA_ROOT, dotenv_path=dotenv).hf_token
    spec = METHODS[method]
    return build_pipeline(
        CHECKPOINT,
        token=token,
        clustering_model=spec["clustering_model"],
        clustering_params=dict(spec["clustering_params"]),
    )


def intermediate_cache(pipeline, cache_dir=REPO / ".harness_cache") -> IntermediateCache:
    from harness.cache_id import intermediate_config_id
    from harness.segmentation import BaselineSegmentation

    return IntermediateCache(
        cache_dir,
        intermediate_config_id(CHECKPOINT, pipeline),
        BaselineSegmentation().id,
    )


def load_intermediates(cache: IntermediateCache, uri: str):
    path = cache.path(uri)
    if not path.exists():
        raise FileNotFoundError(f"no cached intermediates for {uri} at {path}")
    with np.load(path, allow_pickle=False) as stored:
        segmentations = SlidingWindowFeature(stored["segmentation"], _decode_window(stored))
        embeddings = stored["embeddings"]
    return segmentations, embeddings


def held_back_mask(embeddings: np.ndarray, segmentations: SlidingWindowFeature,
                   min_active_ratio: float = 0.2) -> np.ndarray:
    """Active pairs that `BaseClustering.filter_embeddings` leaves out of clustering.

    Recomputed with the same rule (clustering.py:106-123) rather than by calling
    filter_embeddings, so it can be checked against it in a test.
    """
    data = segmentations.data
    num_frames = data.shape[1]
    single = np.sum(data, axis=2, keepdims=True) == 1
    clean = np.sum(data * single, axis=1)
    kept = (clean >= min_active_ratio * num_frames) & ~np.any(np.isnan(embeddings), axis=2)
    active = np.sum(data, axis=1) > 0
    return active & ~kept


def cluster_meeting(pipeline, segmentations: SlidingWindowFeature,
                    embeddings: np.ndarray, uri: str = "") -> ClusteredMeeting:
    """Run the pipeline's clustering exactly as apply() does, on copies."""
    clustering = pipeline.clustering
    if hasattr(clustering, "last_diagnostics"):
        clustering.last_diagnostics = None
    hard, soft, centroids = clustering(
        embeddings=embeddings.copy(),
        segmentations=segmentations,
        num_clusters=None,
        min_clusters=1,
        max_clusters=np.inf,
    )
    active = np.sum(segmentations.data, axis=1) > 0
    outlier = np.zeros_like(active)
    diagnostics = getattr(clustering, "last_diagnostics", None)
    if diagnostics and "rows" in diagnostics:
        for row in diagnostics["rows"]:
            if row["is_outlier"]:
                outlier[row["chunk_idx"], row["speaker_idx"]] = True
    return ClusteredMeeting(
        uri=uri,
        embeddings=embeddings,
        segmentations=segmentations,
        hard_clusters=np.asarray(hard).copy(),
        soft_clusters=np.asarray(soft).copy(),
        centroids=np.asarray(centroids).copy(),
        active=active,
        held_back=held_back_mask(embeddings, segmentations),
        outlier=outlier,
    )


def overlap_fraction(segmentations: SlidingWindowFeature) -> np.ndarray:
    """Share of each pair's active frames where 2+ local speakers are active.

    Same definition as HDBSCANClustering.filter_embeddings (clustering.py:817-822);
    NaN for inactive pairs.
    """
    data = segmentations.data
    active_frames = np.sum(data, axis=1)
    overlap_frames = np.sum(data * (np.sum(data, axis=2, keepdims=True) >= 2), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return overlap_frames / active_frames


OVERLAP_BINS = ("0", "(0,0.25]", "(0.25,0.5]", "(0.5,1]")


def overlap_bin(fraction: np.ndarray) -> np.ndarray:
    out = np.full(fraction.shape, "", dtype=object)
    out[fraction == 0] = OVERLAP_BINS[0]
    out[(fraction > 0) & (fraction <= 0.25)] = OVERLAP_BINS[1]
    out[(fraction > 0.25) & (fraction <= 0.5)] = OVERLAP_BINS[2]
    out[fraction > 0.5] = OVERLAP_BINS[3]
    return out


def meeting_uris(split: str) -> list:
    path = Path.home() / "Code" / "AMI-diarization-setup" / "lists" / f"{split}.meetings.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]
