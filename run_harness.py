"""Orchestrator: adapter -> runner -> scorer -> reporter. See
TICKET-08-orchestrator-integration.md, extended by T1/T2 (oracle/ceiling
analysis batch).

Constructs the four corpus-level metric accumulators (der, overlap_der,
der_overlap_assigned, jer) exactly once per run and threads them through
every score() call, per the scorer's resolved contract. Writes both output
files only after every file has been scored -- a failure partway through
(e.g. OracleSegmentation's still-stubbed NotImplementedError) never leaves a
partial/misleading CSV or summary behind.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import torch
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

from harness.cache_id import clustering_config_description, intermediate_config_id
from harness.cache_id import pipeline_config_id as build_pipeline_config_id
from harness.config import HarnessConfig
from harness.datasets import AMIDatasetAdapter
from harness.intermediate_cache import IntermediateCache
from harness.refinement import get_refinement_strategy, make_oracle_strategy
from harness.reporter import write_report
from harness.run_manifest import (
    CLUSTERING_MODEL_CLASSES,
    ORACLE_SEGMENTATION_CONDITIONS,
    VALID_CLUSTERING_MODELS,
    VALID_CONDITIONS,
    write_manifest,
)
from harness.runner import Runner
from harness.scorer import score


def _current_git_commit() -> str:
    """Short git commit hash of the working tree HEAD, recorded on every
    manifest so a run's exact code state is reconstructable later."""
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent
    ).decode().strip()


def _resolve_device(requested: Optional[str]) -> torch.device:
    """--device flag -> torch.device. `requested=None` auto-detects: cuda if
    available, else cpu. An explicit value always wins over auto-detection."""
    if requested is not None:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _audio_path(config: HarnessConfig, uri: str) -> Path:
    """uri -> audio file path convention for this harness (a decision this
    ticket makes, deferred by TICKET-03's dataset adapter, which never
    touches audio at all): {data_root}/{condition}/audio/{uri}.wav."""
    return config.data_root / config.condition / "audio" / f"{uri}.wav"


def run_harness(
    config: HarnessConfig,
    pipeline: Any,
    pipeline_config_id: str,
    segmentation_source: Any,
    per_file_csv_path: Union[str, Path],
    summary_path: Union[str, Path],
    clustering_model: Optional[str] = None,
    runs_dir: Optional[Union[str, Path]] = None,
    notes: str = "",
    run_condition: str = "baseline",
    refinement_strategy: str = "identity",
    oracle_scope: Optional[str] = None,
    corpus: str = "AMI",
    counts_toward_results: bool = False,
    oracle_rttm: Optional[str] = None,
    intermediate_config_id_value: Optional[str] = None,
    clustering_config: Optional[str] = None,
    use_intermediate_cache: bool = True,
) -> Dict[str, float]:
    der = DiarizationErrorRate(collar=config.der_collar, skip_overlap=config.der_skip_overlap)
    overlap_der = DiarizationErrorRate(
        collar=config.der_collar, skip_overlap=config.der_skip_overlap
    )
    der_overlap_assigned = DiarizationErrorRate(
        collar=config.der_collar, skip_overlap=config.der_skip_overlap
    )
    jer = JaccardErrorRate(collar=config.der_collar, skip_overlap=config.der_skip_overlap)

    # "oracle" needs the current file's own reference Annotation, which the
    # fixed 5-arg refinement interface (embeddings, hard_clusters,
    # soft_clusters, centroids, segmentations) has no slot for -- so unlike
    # identity/nearest_centroid (set once, shared across every file),
    # oracle's pipeline.refinement is rebuilt per file inside the loop
    # below, right before that file is run. See T4's Implementation Notes.
    if refinement_strategy != "oracle":
        pipeline.refinement = get_refinement_strategy(refinement_strategy)

    refinement_id = (
        f"{refinement_strategy}:{oracle_scope or 'all_pairs'}"
        if refinement_strategy == "oracle"
        else refinement_strategy
    )
    # Second cache tier. Keyed on `intermediate_config_id_value`, which
    # EXCLUDES clustering configuration so that segmentation and embeddings are
    # shared across every point of a clustering sweep -- the entire reason the
    # tier exists. `pipeline_config_id` (which includes clustering) must never
    # be passed here; see harness/cache_id.py for why the two differ.
    # Read off the live pipeline, never from a flag: the property
    # `segmentation_batch_size` (speaker_diarization.py:298-299) delegates to
    # `self._segmentation.batch_size`, so this is the value inference will
    # actually use.
    def _batch_size(name):
        """Coerce to int, or None if the pipeline has no usable value.

        The manifest is JSON, so a non-numeric value (a test double's
        auto-attribute, say) must become None rather than being written
        out -- an unserialisable object here would fail the manifest
        write at the very end of a completed run, losing its result.
        """
        try:
            return int(getattr(pipeline, name))
        except (AttributeError, TypeError, ValueError):
            return None

    inference_batch_sizes = {
        "segmentation": _batch_size("segmentation_batch_size"),
        "embedding": _batch_size("embedding_batch_size"),
    }

    intermediate_cache = None
    if use_intermediate_cache:
        intermediate_cache = IntermediateCache(
            config.cache_dir,
            config_id=(
                intermediate_config_id_value
                if intermediate_config_id_value is not None
                else intermediate_config_id(pipeline_config_id, pipeline)
            ),
            segmentation_source_id=segmentation_source.id,
        )

    runner = Runner(
        pipeline,
        pipeline_config_id,
        segmentation_source,
        config.cache_dir,
        refinement_id=refinement_id,
        intermediate_cache=intermediate_cache,
    )
    adapter = AMIDatasetAdapter(config)

    run_started_at = time.monotonic()
    rows = []
    # Corpus-level tally of how the oracle strategy disposed of each pair,
    # summed over files. Logged rather than written to the manifest: it
    # explains a surprising DER (particularly via "unmapped_speaker") without
    # needing a schema change to quote in a report.
    oracle_pair_counts: Dict[str, int] = {}
    for uri, reference, uem in adapter:
        oracle_strategy = None
        if refinement_strategy == "oracle":
            oracle_strategy = make_oracle_strategy(
                reference, oracle_scope=oracle_scope or "all_pairs"
            )
            pipeline.refinement = oracle_strategy
        file = {"uri": uri, "audio": str(_audio_path(config, uri))}
        hypothesis = runner.run(file)
        row = score(reference, hypothesis, uem, der, overlap_der, der_overlap_assigned, jer)
        row["uri"] = uri
        rows.append(row)

        if oracle_strategy is not None:
            # A cache hit skips the pipeline entirely, so the strategy is
            # never invoked and its counts stay at zero -- that's a real
            # caveat on these totals, not a bug, and is why they're reported
            # per run alongside the cache state rather than stored.
            for path, count in getattr(oracle_strategy, "counts", {}).items():
                oracle_pair_counts[path] = oracle_pair_counts.get(path, 0) + count

    duration_seconds = time.monotonic() - run_started_at

    if intermediate_cache is not None:
        # Printed unconditionally, including when every lookup missed. A cache
        # that silently misses and recomputes produces correct numbers and no
        # speedup -- the same shape of failure as the segmentation seam no-op.
        # "0 full hits, 0 writes" is the signal that the seam is broken, and it
        # is only visible if the tally is reported even when it is all zeros.
        print(
            f"{intermediate_cache.stats} "
            f"({intermediate_cache.entry_count()} entries, "
            f"{intermediate_cache.disk_usage_bytes() / 1e6:.1f} MB on disk)"
        )

    if oracle_pair_counts:
        total = sum(oracle_pair_counts.values())
        print(
            f"oracle pair disposition (scope={oracle_scope or 'all_pairs'}, "
            f"{total} pairs over {len(rows)} files): "
            + ", ".join(f"{path}={count}" for path, count in sorted(oracle_pair_counts.items()))
        )

    summary = write_report(
        rows, der, overlap_der, der_overlap_assigned, jer, per_file_csv_path, summary_path
    )

    if runs_dir is not None:
        run_config = {
            "pipeline_config_id": pipeline_config_id,
            "segmentation_source_id": segmentation_source.id,
            "clustering_model": clustering_model,
            "extra_pipeline_steps": [],
            "split": config.split,
            "condition": run_condition,
            "der_collar": config.der_collar,
            "der_skip_overlap": config.der_skip_overlap,
            "notes": notes,
            "refinement_strategy": refinement_strategy,
            "oracle_scope": oracle_scope,
            "corpus": corpus,
            "mic_condition": config.condition,
            "git_commit": _current_git_commit(),
            "counts_toward_results": counts_toward_results,
            # Additive provenance field (Gap 1 of the run-manifest-provenance
            # ticket): which reference RTTM built the ground-truth
            # segmentation. None for runs that used the pipeline's own.
            # Deliberately not added to REQUIRED_RUN_CONFIG_FIELDS -- every
            # manifest written before this change lacks the key entirely and
            # must stay readable.
            "oracle_rttm": oracle_rttm,
            # `pipeline_config_id` above is now a hash of the checkpoint plus
            # the clustering class and every instantiated clustering
            # hyperparameter, so on its own it is write-only: it distinguishes
            # two runs but never says what they were. These two fields make a
            # sweep's manifests self-describing -- the human-readable
            # clustering configuration, and the intermediate-tier key (which
            # excludes clustering, so sweep points share it and that is
            # visible rather than inferred).
            #
            # Additive, like `oracle_rttm`: deliberately NOT added to
            # REQUIRED_RUN_CONFIG_FIELDS, because every manifest written
            # before this change lacks both keys and must stay readable.
            "clustering_config": clustering_config,
            "intermediate_config_id": intermediate_config_id_value,
            # Batch size is part of the intermediate key because it changes
            # the cached arrays' VALUES, not just how fast they are produced
            # (speaker_diarization.py:460-468). Recorded in readable form
            # alongside the hash so a future reader can tell why two runs
            # that look identically configured landed on different entries.
            "inference_batch_sizes": inference_batch_sizes,
        }
        write_manifest(run_config, summary, runs_dir, duration_seconds=duration_seconds)

    return summary


from pyannote.audio import Pipeline

# Vocabulary value used when `--clustering-model` is omitted. "pyannote-default"
# means whatever the checkpoint ships, which for community-1 is VBxClustering
# with threshold 0.6 / Fa 0.07 / Fb 0.8 -- the configuration behind the recorded
# baseline DER 0.17048543579940637.
DEFAULT_CLUSTERING_MODEL = "pyannote-default"


def _coerce_param_value(raw: str) -> Any:
    """`name=value` string -> a typed Python value.

    Types matter twice over. The clustering classes declare parameters as
    typed descriptors -- `min_cluster_size` is an `Integer`, `method` a
    `Categorical` of strings, `threshold`/`Fa`/`Fb` are `Uniform` floats -- so
    handing a string where a number belongs would either raise deep inside
    clustering or silently compare wrongly. And the final cache key hashes
    `repr(value)`, so `0.7` and `"0.7"` would produce different keys for the
    same configuration and split a sweep's cache in half.

    Ordering is int-before-float on purpose: `int("12")` succeeds where an
    int-typed parameter needs an int, while `float("12")` would give `12.0`
    and a differently-hashing key for the same setting.
    """
    lowered = raw.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _parse_clustering_params(raw: Optional[Sequence[str]]) -> Dict[str, Any]:
    """Repeatable `--clustering-param name=value` -> a dict.

    Malformed input raises rather than being skipped. A silently dropped
    hyperparameter is the worst outcome available here: the sweep runs every
    point at the default setting, reports a flat curve, and the flat curve
    reads as a real negative result about the hyperparameter.

    Note this only validates the SHAPE (`name=value`). Whether `name` is a
    real parameter of the selected class is checked by
    `pyannote.pipeline.Pipeline.instantiate`, which raises
    `ValueError: parameter '<name>' does not exist`. Deliberately not
    duplicated here -- a second, hand-maintained list of valid names would
    drift from the classes and could accept something the library rejects.
    """
    params: Dict[str, Any] = {}
    for item in raw or []:
        if "=" not in item:
            raise ValueError(
                f"--clustering-param expects name=value, got {item!r}"
            )
        name, _, value = item.partition("=")
        name = name.strip()
        if not name:
            raise ValueError(
                f"--clustering-param expects name=value with a non-empty name, "
                f"got {item!r}"
            )
        params[name] = _coerce_param_value(value)
    return params


def _from_pretrained_with_clustering(checkpoint: str, class_name: str, token=None):
    """Load `checkpoint` but with a different clustering class.

    WHY THIS EXISTS AT ALL. The obvious approach -- pass `clustering=<name>`
    through `Pipeline.from_pretrained` -- does not work:
    `from_pretrained`'s signature is fixed (`checkpoint, revision,
    hparams_file, subfolder, token, cache_dir`, core/pipeline.py:153-161) with
    no `**kwargs`, so it raises `TypeError: unexpected keyword argument
    'clustering'`. There is no seam for overriding a construction parameter, and
    adding one would mean editing `src/pyannote/`, which is out of scope.

    So this reproduces `from_pretrained`'s own sequence (core/pipeline.py:246-291)
    over the shipped config, with `pipeline.params.clustering` swapped:

      1. read the shipped `config.yaml`;
      2. replace `config["pipeline"]["params"]["clustering"]`;
      3. `expand_subfolders` against the REAL checkpoint id, so `$model/segmentation`,
         `$model/embedding` and `$model/plda` still resolve to the same assets --
         passing a config dict to `from_pretrained` instead would set
         `model_id = Path.cwd()` (:191) and break that resolution;
      4. construct `SpeakerDiarization(**params)`, so `__init__`
         (speaker_diarization.py:283-295) performs the enum lookup, the
         `VBxClustering`-only PLDA special case and `_expects_num_speakers`;
      5. apply the shipped `config["params"]`, minus any clustering entry the new
         class does not declare.

    Step 5's filtering is the one place this deviates from the library, and it is
    necessary rather than cosmetic: the shipped config instantiates VBx's
    `threshold`/`Fa`/`Fb`, and `Pipeline.instantiate` raises on a parameter the
    target class does not declare (base pipeline.py:443-444). Handing
    agglomerative VBx's `Fa` would make every non-default selection crash on
    load. Note `threshold` legitimately exists on BOTH VBx and agglomerative with
    DIFFERENT meanings and ranges (`Uniform(0.5, 0.8)` vs `Uniform(0.0, 2.0)`),
    so a shipped value is dropped unless the target class declares that exact
    name -- and the caller's explicit `--clustering-param` is applied afterwards,
    so an operator's value always wins.

    The DEFAULT path does not come through here (see `build_pipeline`): it uses
    the untouched `from_pretrained`, so nothing about this function can move the
    baseline condition or its cache key.
    """
    import yaml
    from pyannote.audio.core.pipeline import expand_subfolders
    from pyannote.audio.pipelines import SpeakerDiarization
    from pyannote.audio.utils.hf_hub import AssetFileName, download_from_hf_hub

    config_yml = download_from_hf_hub(
        str(checkpoint), AssetFileName.Pipeline, token=token
    )
    with open(config_yml, "r") as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)

    config["pipeline"].setdefault("params", {})["clustering"] = class_name

    expand_subfolders(config, str(checkpoint), token=token)

    params = dict(config["pipeline"].get("params", {}))
    params.setdefault("token", token)
    pipeline = SpeakerDiarization(**params)

    shipped = dict(config.get("params", {}) or {})
    clustering_defaults = dict(shipped.pop("clustering", {}) or {})
    declared = set(pipeline.clustering._parameters)
    kept = {
        name: value
        for name, value in clustering_defaults.items()
        if name in declared
    }
    dropped = sorted(set(clustering_defaults) - set(kept))
    if dropped:
        # Printed, not silent. A dropped default means the new class runs at its
        # own defaults for those parameters, which materially changes the
        # condition; an operator comparing methods needs to know that happened.
        print(
            f"note: {class_name} does not declare "
            f"{', '.join(dropped)} -- shipped value(s) not applied"
        )
    if kept:
        shipped["clustering"] = kept
    if shipped:
        pipeline.instantiate(shipped)

    return pipeline


def build_pipeline(
    checkpoint: str,
    token: Optional[str] = None,
    clustering_model: str = DEFAULT_CLUSTERING_MODEL,
    clustering_params: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
):
    """Load the pipeline with a chosen clustering class and hyperparameters.

    TWO DIFFERENT SEAMS, because the class and its values are fixed at
    different moments in `Pipeline.from_pretrained` (core/pipeline.py:275-294).

    1. THE CLASS, at construction. `from_pretrained` reads
       `config["pipeline"]["params"]` and calls `Klass(**params)` (:275-278), so
       the clustering class is decided before the object exists. Passing
       `clustering=<ClassName>` through as a kwarg lets
       `SpeakerDiarization.__init__` (speaker_diarization.py:283-295) do all
       three things it must: the `Clustering` enum lookup, the
       `VBxClustering`-only special case that supplies `self._plda` (VBx's
       `__init__` takes a positional `plda` with no default, so it cannot be
       constructed from a bare name), and setting `_expects_num_speakers` from
       `clustering.expects_num_clusters`.

       NOT done by assigning `pipeline.clustering = ...` on a loaded pipeline.
       That sets one of the three and leaves `klustering` and
       `_expects_num_speakers` describing the previous class. Nothing crashes;
       the pipeline just quietly disagrees with itself, and
       `_expects_num_speakers` governs whether speaker counts are plumbed
       through. A silent, plausible-looking wrong state.

    2. THE HYPERPARAMETERS, via `pipeline.instantiate()` (:290-291). This is the
       sanctioned seam and already how 0.6/0.07/0.8 reach the default VBx
       instance, so the pipeline does not need rebuilding for a sweep point.
       It also fails loudly for free on an unknown name. `VBxClustering`
       overrides `__call__` rather than `cluster` (clustering.py:572), but that
       is irrelevant to this route: `instantiate` acts on the base class's
       descriptor machinery, not on the clustering entry point. Asserted in
       `tests/test_clustering_model_selection.py` rather than assumed.

    Hyperparameters are applied AFTER `from_pretrained` has run the shipped
    config's own `instantiate`, so an override wins over the shipped value while
    every parameter left alone keeps it.
    """
    if clustering_model not in CLUSTERING_MODEL_CLASSES:
        # Rejected here as well as by argparse, because `build_pipeline` is
        # callable directly. The message names the valid values so a typo is
        # recoverable without reading source. Deliberately NOT a fallback to the
        # default: falling back would let the manifest record a model that never
        # ran.
        raise ValueError(
            f"unknown clustering model {clustering_model!r}; expected one of "
            f"{', '.join(VALID_CLUSTERING_MODELS)}"
        )

    class_name = CLUSTERING_MODEL_CLASSES[clustering_model]
    default_class_name = CLUSTERING_MODEL_CLASSES[DEFAULT_CLUSTERING_MODEL]

    if class_name == default_class_name:
        # THE DEFAULT CONDITION TAKES THE UNTOUCHED LIBRARY PATH, deliberately.
        #
        # `pyannote-default` and `vbx` both resolve to the shipped class, so
        # there is nothing to override and no reason to reimplement
        # `from_pretrained`'s loading sequence for them. Keeping this branch
        # byte-for-byte the pre-ticket call is what guarantees the baseline
        # condition's behaviour -- and its cache key -- are unchanged by this
        # ticket. Verified: the default still yields
        # VBxClustering(Fa=0.07, Fb=0.8, threshold=0.6) and final key
        # 190b62c3..., intermediate key c636fc56..., both identical to the
        # pre-ticket values.
        pipeline = Pipeline.from_pretrained(checkpoint, token=token)
    else:
        pipeline = _from_pretrained_with_clustering(
            checkpoint, class_name, token=token
        )

    if device is not None:
        pipeline = pipeline.to(_resolve_device(device))

    if clustering_params:
        # Nested form, matching the shipped config's `params: clustering: {...}`.
        # Raises ValueError on an unknown parameter name (base
        # pyannote/pipeline/pipeline.py:443-444).
        pipeline.instantiate({"clustering": dict(clustering_params)})

    # `is True`, not truthiness. `getattr` on a MagicMock returns a truthy Mock
    # for any attribute name, so a truthiness check would refuse every
    # mock-driven caller (it did: it broke six pre-existing tests). More to the
    # point, `expects_num_clusters` is a declared bool on every clustering class
    # -- anything else here means the attribute is not what we think it is, and
    # guessing from truthiness would be exactly the kind of conflated condition
    # that turns one signal into two meanings.
    if getattr(pipeline, "_expects_num_speakers", False) is True:
        # FAIL AT SELECTION TIME, not 40 minutes into a run.
        #
        # `KMeansClustering.expects_num_clusters` is True, and
        # `SpeakerDiarization.apply()` requires `num_speakers` in that case
        # (speaker_diarization.py:600-607). The harness builds `file` as
        # {"uri", "audio"} and never passes `num_speakers`, and deciding where k
        # comes from -- oracle count, fixed, or estimated -- is a separate
        # ticket ([[clustering-kmeans-estimated-k]]), explicitly out of scope
        # here.
        #
        # Raised HERE rather than left to the library because of the second
        # hazard: `Runner.run()` sets `pipeline.training = True`, and the
        # library's guard falls back to `len(file["annotation"].labels())` when
        # an annotation is present. Today the harness keeps `annotation` off the
        # file dict, so that fallback cannot fire -- but if it ever did, a
        # kmeans run would silently take the speaker count from the ground
        # truth and report an oracle-count result under an ordinary baseline
        # manifest. `harness/segmentation.py` already guards the
        # oracle-segmentation path this way; this covers the baseline path too.
        raise ValueError(
            f"clustering model {clustering_model!r} resolves to "
            f"{class_name}, whose expects_num_clusters is True, but the harness "
            "supplies no num_speakers (run_harness.py builds file as "
            "{'uri', 'audio'}). Choosing where k comes from -- oracle count, "
            "fixed, or estimated -- is out of scope for clustering-model "
            "selection; see the clustering-kmeans-estimated-k ticket. Refusing "
            "to start a run that would either raise mid-corpus or silently take "
            "k from the reference annotation."
        )

    return pipeline


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, float]:
    import argparse

    from pyannote.database.util import load_rttm

    from harness.segmentation import BaselineSegmentation, OracleSegmentation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--condition", default="IHM")
    parser.add_argument("--cache-dir", default=".harness_cache")
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--per-file-csv", default="per_file.csv")
    parser.add_argument("--summary", default="summary.json")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument(
        "--notes", default="",
        help="Free-text description of the experiment (e.g. 'oracle segmentation', "
        "'dbscan clustering', 'random forest overlap detector'), recorded in the run "
        "manifest and shown in runs/comparison.csv.",
    )
    parser.add_argument(
        "--device", default=None,
        help="torch device (e.g. cpu, cuda, cuda:0). Default: cuda if available, else cpu.",
    )
    parser.add_argument(
        "--run-condition", default="baseline",
        choices=list(VALID_CONDITIONS),
        help="Which oracle/ceiling-analysis condition this run belongs to (not to be "
        "confused with --condition, the AMI mic condition). Recorded in the manifest's "
        "'condition' field; T5's cross-condition deltas key on this. "
        "'oracle_segmentation_assignment' is the combined 2x2 cell: pair it with "
        "--refinement-strategy oracle and --oracle-rttm.",
    )
    parser.add_argument(
        "--refinement-strategy", default="identity",
        choices=["identity", "oracle", "nearest_centroid"],
        help="Post-clustering refinement strategy (see harness/refinement.py).",
    )
    parser.add_argument(
        "--oracle-scope", default="all_pairs",
        choices=["all_pairs", "overlap_degraded"],
        help="Scope of pairs the 'oracle' refinement strategy refines (T4). "
        "'overlap_degraded' (primary): only pairs whose support is predominantly "
        "coincident with another speaker. 'all_pairs' (secondary): every pair, the "
        "full post-clustering ceiling. Ignored unless --refinement-strategy oracle.",
    )
    parser.add_argument(
        "--oracle-rttm", default=None,
        help="Path to a reference RTTM (e.g. an only_words RTTM) used to build ground-truth "
        "segmentation when --run-condition oracle_segmentation is selected (T3). Required "
        "in that case; ignored otherwise.",
    )
    parser.add_argument(
        "--clustering-model", default=DEFAULT_CLUSTERING_MODEL,
        choices=list(VALID_CLUSTERING_MODELS),
        help="Which clustering algorithm the pipeline uses. 'pyannote-default' "
        "(the default) is whatever the checkpoint ships -- for community-1 that is "
        "VBxClustering at threshold 0.6 / Fa 0.07 / Fb 0.8, the configuration "
        "behind the recorded baseline DER. 'vbx' is the same class chosen "
        "explicitly, so the two are one condition rather than two. The instantiated "
        "class and every hyperparameter are recorded in the manifest's "
        "'clustering_model' field and are part of the final-hypothesis cache key.",
    )
    parser.add_argument(
        "--clustering-param", action="append", default=None,
        metavar="NAME=VALUE",
        help="Override one clustering hyperparameter, e.g. --clustering-param "
        "threshold=0.7. Repeatable. Applied via the pipeline's own instantiate(), "
        "so a name the selected class does not declare fails loudly rather than "
        "being ignored -- a silently dropped hyperparameter would produce a flat "
        "sweep curve that reads as a real negative result.",
    )
    parser.add_argument("--corpus", default="AMI")
    parser.add_argument(
        "--no-intermediate-cache", action="store_true",
        help="Disable the pre-clustering segmentation/embedding cache "
        "(harness/intermediate_cache.py): neither read nor written. Use for a "
        "genuinely cold reference run. The final-hypothesis RTTM cache is "
        "unaffected by this flag -- to bypass that too, point --cache-dir at "
        "an empty directory.",
    )
    parser.add_argument(
        "--counts-toward-results", action="store_true",
        help="Mark this run as counting toward the oracle/ceiling-analysis results table "
        "(T5). Defaults to False for exploratory/debugging runs.",
    )
    args = parser.parse_args(argv)

    config = HarnessConfig.load(
        data_root=args.data_root,
        split=args.split,
        condition=args.condition,
        cache_dir=args.cache_dir,
        dotenv_path=args.dotenv,
    )

    checkpoint = "pyannote/speaker-diarization-community-1"

    # Clustering selection happens HERE, before any cache identity is derived,
    # so both keys and the manifest describe the pipeline that will actually
    # run rather than the one the flags asked for. Parsing first means a
    # malformed --clustering-param dies before the models load.
    clustering_params = _parse_clustering_params(args.clustering_param)
    pipeline = build_pipeline(
        checkpoint,
        token=config.hf_token,
        clustering_model=args.clustering_model,
        clustering_params=clustering_params,
    )
    pipeline = pipeline.to(_resolve_device(args.device))

    # Two cache identities, built from the LIVE pipeline object so they reflect
    # what was actually instantiated rather than what was requested.
    #
    #   final_config_id -- checkpoint + clustering class + every instantiated
    #     clustering hyperparameter. Keys the RTTM. Clustering MUST be here:
    #     without it a hyperparameter sweep serves point 1's cached RTTM at
    #     every point and reports a flat curve from a clean run.
    #   intermediate_id -- checkpoint only. Keys segmentation + embeddings,
    #     which are computed before clustering and cannot depend on it, so
    #     every sweep point shares one copy.
    #
    # See harness/cache_id.py for the full reasoning and for how to add a new
    # dimension to either key.
    final_config_id = build_pipeline_config_id(checkpoint, pipeline)
    intermediate_id = intermediate_config_id(checkpoint, pipeline)
    clustering_config = clustering_config_description(pipeline)

    # Previously the literal "pyannote-default", which recorded nothing. Now
    # the real instantiated configuration, e.g.
    # "VBxClustering(Fa=0.07, Fb=0.8, threshold=0.6)".
    #
    # Derived from the LIVE pipeline object, never from `args.clustering_model`.
    # That is the guard against this ticket's central failure mode: a flag that
    # is parsed and recorded but never reaches the pipeline would otherwise
    # produce a manifest that looked perfect while the default ran. Reading the
    # instantiated object means the manifest cannot claim a model that did not
    # run.
    clustering_model = clustering_config
    print(f"clustering: {clustering_config} (requested: {args.clustering_model})")
    print(f"pipeline_config_id: {final_config_id}")
    print(f"intermediate_config_id: {intermediate_id}")

    # Membership, not equality: an exact `== "oracle_segmentation"` test let
    # the combined condition fall through to BaselineSegmentation(), which
    # fails silently -- the run completes and writes a manifest claiming
    # oracle segmentation while actually scoring baseline.
    if args.run_condition in ORACLE_SEGMENTATION_CONDITIONS:
        if not args.oracle_rttm:
            parser.error(
                f"--run-condition {args.run_condition} requires --oracle-rttm"
            )
        segmentation_source = OracleSegmentation(
            reference_lookup=load_rttm(args.oracle_rttm)
        )
    else:
        segmentation_source = BaselineSegmentation()

    summary = run_harness(
        config,
        pipeline,
        pipeline_config_id=final_config_id,
        segmentation_source=segmentation_source,
        per_file_csv_path=args.per_file_csv,
        summary_path=args.summary,
        clustering_model=clustering_model,
        runs_dir=args.runs_dir,
        notes=args.notes,
        run_condition=args.run_condition,
        refinement_strategy=args.refinement_strategy,
        oracle_scope=args.oracle_scope,
        corpus=args.corpus,
        counts_toward_results=args.counts_toward_results,
        oracle_rttm=args.oracle_rttm,
        intermediate_config_id_value=intermediate_id,
        clustering_config=clustering_config,
        use_intermediate_cache=not args.no_intermediate_cache,
    )
    print(summary)
    return summary


if __name__ == "__main__":
    main()
