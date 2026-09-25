"""Cache identity for the harness's two-tier cache.

THE ONE THING TO UNDERSTAND HERE: there are two keys, and they deliberately
differ in exactly one respect -- whether clustering configuration is part of
them.

    tier            key components                               clustering?
    -----------     -----------------------------------------    -----------
    intermediate    intermediate_config_id | seg_source.id | uri     NO
    (segmentation,  = checkpoint only
     embeddings)

    final           pipeline_config_id | seg_source.id |             YES
    (RTTM)            refinement_id | uri
                    = checkpoint + clustering class + every
                      instantiated clustering hyperparameter

WHY THAT WAY ROUND, and why getting it backwards is dangerous:

* Segmentation inference and embedding extraction happen *before* clustering
  in `SpeakerDiarization.apply()` (segmentation at speaker_diarization.py:609,
  embeddings at :647, clustering only at :656). Their output does not depend
  on the clustering configuration in any way. That is the entire premise of
  the intermediate tier: one cached copy per (checkpoint, segmentation source,
  file) serves *every* point of a clustering sweep. If clustering config
  leaked into the intermediate key, every sweep point would miss and pay for
  GPU inference again -- the cache would be useless, though still correct.

* The final RTTM *does* depend on clustering, because clustering is what
  produces the speaker labels. If clustering config is missing from the final
  key -- which was the pre-existing defect, `run_harness.py` passed the bare
  checkpoint string -- then a sweep computes point 1, caches it, and serves
  that same RTTM for every subsequent point. Every point reports an identical
  DER, the run completes, the manifest validates, and the curve is flat. That
  is silently wrong rather than merely slow, and it is much harder to notice.

So: a useless cache if the intermediate key is over-specified, silently wrong
results if the final key is under-specified. The asymmetry in consequence is
why the final key errs toward including things and the intermediate key errs
toward excluding them.

HOW TO ADD A NEW DIMENSION TO EITHER KEY
----------------------------------------
Ask one question: *does this thing change segmentation or embedding output?*

* YES  -> it belongs in `intermediate_config_id` (and therefore, since the
  final key is derived from the same base, in both). Add it to the
  `_intermediate_components` list. Examples: the checkpoint, the segmentation
  model revision, the inference batch sizes, `segmentation_step`,
  `embedding_exclude_overlap`,
  `segmentation.threshold` on a non-powerset checkpoint (see the powerset note
  in `harness/intermediate_cache.py`). Adding a dimension here orphans the
  existing intermediate entries, which is correct -- they become unreachable,
  not wrong.

  Ask the question about VALUES, not about intent. Batch size looks like a
  performance setting and was left out of this key for exactly that reason;
  it changes the floats the model emits, so it belonged here all along.

* NO, but it changes the final RTTM -> it belongs in `pipeline_config_id`
  only. Add it to the `_final_only_components` list. Examples: anything about
  clustering, or a post-clustering step not already covered by
  `refinement_id`.

Clustering hyperparameters need no code change at all: they are enumerated
generically from pyannote's own descriptor machinery (see
`_instantiated_clustering_parameters`), so a clustering class with an entirely
different parameter set is picked up automatically.

Note that `segmentation_source.id` and `refinement_id` are NOT handled here --
`Runner` concatenates those itself (harness/runner.py). This module owns only
the pipeline-configuration portion of each key.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Tuple

# The prefix makes a key self-describing in a traceback or a manifest, and
# guarantees the two tiers can never collide even if their component lists
# ever coincided.
_INTERMEDIATE_PREFIX = "intermediate-v1"
_FINAL_PREFIX = "final-v1"


class ClusteringConfigUnavailable(RuntimeError):
    """Raised when the clustering configuration cannot be read off a pipeline.

    This is deliberately loud. The failure mode this module exists to prevent
    is a *silent* one: if the clustering configuration were quietly omitted
    from the final-hypothesis key -- by falling back to the bare checkpoint
    when introspection failed -- the result would be a sweep in which every
    point returns point 1's cached RTTM. The run would complete, the manifests
    would validate and the curve would be flat. There would be nothing to see.

    So there is no fallback path. If we cannot tell what the clustering
    configuration is, we refuse to build a key that claims to identify it.
    """


def _instantiated_clustering_parameters(clustering: Any) -> Dict[str, Any]:
    """Every instantiated hyperparameter of a clustering object, by name.

    Read generically from pyannote's own parameter machinery rather than from
    a hardcoded per-class list. `pyannote.pipeline.Pipeline.__setattr__`
    (site-packages pyannote/pipeline/pipeline.py:102-149) routes a `Parameter`
    assignment into `self._parameters` and a later concrete assignment into
    `self._instantiated`; `_flattened_parameters(instantiated=True)` (:165-210)
    returns the concrete values, recursing into sub-pipelines with a
    `parent>child` naming convention.

    Being generic is not stylistic here. This ticket's own text, and several
    docs, wrongly asserted that community-1 clusters with
    `AgglomerativeClustering`; it actually uses `VBxClustering`
    (threshold/Fa/Fb, verified from the shipped config.yaml and the live
    pipeline). A hardcoded list built on that stale claim would have omitted
    every real hyperparameter and produced a constant key across a VBx sweep,
    looking perfectly healthy while flattening the curve.

    An empty dict is a legitimate result -- `KMeansClustering` declares no
    tunable parameters -- and is distinguished from "could not introspect" by
    the caller, which requires the attribute to exist at all.
    """
    flattened = getattr(clustering, "_flattened_parameters", None)
    if callable(flattened):
        try:
            return dict(flattened(instantiated=True))
        except Exception as exc:  # pragma: no cover - defensive
            raise ClusteringConfigUnavailable(
                f"could not read instantiated parameters from "
                f"{type(clustering).__name__}: {exc!r}. Refusing to build a "
                "cache key that silently omits clustering configuration."
            ) from exc

    # Not a pyannote Pipeline. Fall back to the object's own public attributes,
    # so a plain/duck-typed clustering object still contributes its values
    # instead of contributing nothing.
    public = {
        name: value
        for name, value in vars(clustering).items()
        if not name.startswith("_") and isinstance(value, (int, float, str, bool, type(None)))
    }
    if not public:
        raise ClusteringConfigUnavailable(
            f"{type(clustering).__name__} exposes no readable hyperparameters "
            "(no pyannote _flattened_parameters, no public scalar attributes). "
            "Refusing to build a cache key that silently omits clustering "
            "configuration."
        )
    return public


def _clustering_of(pipeline: Any) -> Any:
    clustering = getattr(pipeline, "clustering", None)
    if clustering is None:
        raise ClusteringConfigUnavailable(
            "pipeline has no `clustering` attribute, so its clustering "
            "configuration cannot be included in the final-hypothesis cache "
            "key. Refusing to fall back to a clustering-free key: that is the "
            "exact defect this key restructure fixes, and it fails silently "
            "(every sweep point would reuse point 1's cached RTTM)."
        )
    return clustering


def _clustering_class_name(pipeline: Any, clustering: Any) -> str:
    """`klustering` is the pipeline's own record of the selected algorithm
    (speaker_diarization.py:242); fall back to the object's class name."""
    return str(getattr(pipeline, "klustering", None) or type(clustering).__name__)


def _canonical(value: Any) -> str:
    """Stable textual form of a hyperparameter value.

    `repr` on a float is round-trip exact in Python 3, so 0.6 and
    0.6000000000001 do not collide. Values are rendered rather than pickled so
    that a key stays stable across processes and library versions -- a key that
    changed between runs would orphan the cache on every invocation.
    """
    return repr(value)


def _clustering_components(pipeline: Any) -> List[str]:
    clustering = _clustering_of(pipeline)
    params = _instantiated_clustering_parameters(clustering)
    rendered = [f"{name}={_canonical(params[name])}" for name in sorted(params)]
    return [f"clustering={_clustering_class_name(pipeline, clustering)}"] + rendered


class BatchSizeUnavailable(RuntimeError):
    """Raised when the effective inference batch sizes cannot be read.

    Loud for the same reason as `ClusteringConfigUnavailable`. Batch size
    changes the cached arrays' VALUES (see `_batch_size_components`), so a key
    built without it would let two runs at different batch sizes collide on one
    cache entry: the second is served arrays it did not compute, and nothing
    about the run looks wrong. There is no degraded path.
    """


def _batch_size_components(pipeline: Any) -> List[str]:
    """The effective inference batch sizes, which are VALUE inputs.

    This looks like a performance knob and is not one. `get_embeddings()`
    stacks waveforms into a single tensor and runs the embedding model on the
    batch (speaker_diarization.py:460-468), and segmentation is batched the
    same way (:259). Batch shape changes float reduction order inside the
    model, so the arrays that come out genuinely differ.

    Measured on this corpus: a batch-8 run scored DER 0.17049342382952828
    against the batch-32 run's 0.17048543579940637. Before batch size entered
    this key the two shared an entry and collided silently.

    Read through the PROPERTY `segmentation_batch_size`
    (speaker_diarization.py:298-299), which delegates to
    `self._segmentation.batch_size`, so the value recorded is the one inference
    will actually use even if the inference object was reconfigured directly.
    `embedding_batch_size` is a plain attribute (:236).
    """
    components = []
    for name in ("segmentation_batch_size", "embedding_batch_size"):
        value = getattr(pipeline, name, None)
        if value is None:
            raise BatchSizeUnavailable(
                f"pipeline exposes no `{name}`, so the effective inference "
                "batch size cannot be included in the intermediate cache key. "
                "Refusing to build a key that would let two runs at different "
                "batch sizes collide on one cache entry while producing "
                "different embeddings."
            )
        components.append(f"{name}={int(value)}")
    return components


def _intermediate_components(checkpoint: str, pipeline: Any) -> List[str]:
    """Everything that changes segmentation or embedding OUTPUT.

    Invalidation reasoning, written so it can be checked rather than trusted:

    * `checkpoint` selects BOTH the segmentation model and the embedding model
      (speaker_diarization.py:204-217 -- `segmentation`, `embedding` and `plda`
      all default to subfolders of the same checkpoint). Change it and both
      cached arrays are wrong. It is in the key.

    * The segmentation SOURCE (baseline vs oracle) also changes the
      segmentation tensor, and changes the embeddings downstream of it, since
      embeddings are extracted from the binarized segmentation
      (speaker_diarization.py:647-652). It is in the key -- but it is
      contributed by `Runner` as `segmentation_source.id`, not here.

    * `uri` identifies the audio. Contributed by `Runner`.

    * Clustering configuration is deliberately ABSENT. Clustering runs at
      speaker_diarization.py:656, strictly after both intermediates are
      produced, and cannot affect them.

    * The effective inference BATCH SIZES are in the key, via
      `_batch_size_components`. They change the arrays' values, not merely the
      speed of producing them -- see that function. This was originally
      omitted on the reasoning that batching "affects speed, not values";
      that reasoning was wrong and the omission let batch-8 and batch-32 runs
      collide on one entry.

    What is NOT currently in the key, and is safe only because the harness
    holds it fixed: `segmentation_step` (0.1), `embedding_exclude_overlap`
    (True), and -- on a non-powerset checkpoint -- `segmentation.threshold`,
    which would change the binarization the embeddings are extracted from.
    community-1 is powerset so no threshold exists (verified:
    `hasattr(pipeline.segmentation, "threshold")` is False). If the harness
    ever varies any of these, add them here. A reader checking this list
    should confirm those values are still fixed in `run_harness.py`.
    """
    return [f"checkpoint={checkpoint}"] + _batch_size_components(pipeline)


def intermediate_config_id(checkpoint: str, pipeline: Any) -> str:
    """Pipeline-configuration portion of the INTERMEDIATE cache key.

    Contains no clustering configuration, by design -- see the module
    docstring. It DOES contain the effective inference batch sizes, which are
    read off `pipeline`: they change the cached arrays' values, so two runs at
    different batch sizes must not share an entry.

    `pipeline` was previously optional and ignored. It is now required, because
    a default would silently produce a key that omits batch size -- exactly the
    collision this fixes.
    """
    raw = "|".join(
        [_INTERMEDIATE_PREFIX] + _intermediate_components(checkpoint, pipeline)
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def pipeline_config_id(checkpoint: str, pipeline: Any) -> str:
    """Pipeline-configuration portion of the FINAL-HYPOTHESIS cache key.

    The checkpoint plus the clustering class name plus every instantiated
    clustering hyperparameter value. Raises `ClusteringConfigUnavailable`
    rather than degrading to a clustering-free key: see the module docstring
    and that exception's own.
    """
    components = (
        [_FINAL_PREFIX]
        + _intermediate_components(checkpoint, pipeline)
        + _clustering_components(pipeline)
    )
    raw = "|".join(components)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def clustering_config_description(pipeline: Any) -> str:
    """Human-readable form of the clustering configuration, for the manifest.

    The hash alone is write-only: it tells a future reader that two runs
    differed but never what they were. Recorded alongside it so a sweep's
    manifests are self-describing, e.g.
    `VBxClustering(Fa=0.07, Fb=0.8, threshold=0.6)`.
    """
    clustering = _clustering_of(pipeline)
    params = _instantiated_clustering_parameters(clustering)
    name = _clustering_class_name(pipeline, clustering)
    inner = ", ".join(f"{key}={params[key]!r}" for key in sorted(params))
    return f"{name}({inner})"


def describe(checkpoint: str, pipeline: Any) -> Tuple[str, str, str]:
    """Both keys plus the human-readable clustering description, for callers
    that record all three in a manifest."""
    return (
        pipeline_config_id(checkpoint, pipeline),
        intermediate_config_id(checkpoint, pipeline),
        clustering_config_description(pipeline),
    )
