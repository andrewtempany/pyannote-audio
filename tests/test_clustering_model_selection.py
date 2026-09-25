"""Tests for the clustering-model-selection ticket: `--clustering-model` and
`--clustering-param`.

WHAT THESE GUARD AGAINST, because the failure mode here is silent rather than
loud. A `--clustering-model` flag that is parsed, recorded in the manifest and
then never reaches the pipeline looks *perfect* in the manifest: the run
completes, the numbers are plausible, and they are the default condition's
numbers. [[Oracle 2x2 Combined Cells]] item 2 found exactly this shape of bug
already (an exact-equality selection test fell through to a default branch and
the manifest claimed one thing while the run scored another).

So every selection test here asserts on the *live pipeline object* -- that
`pipeline.clustering` is an instance of the named class, and that an
overridden hyperparameter is readable back off that instance -- never on the
flag string or on what the harness intended.

A NOTE ON THE DEFAULT, which several project docs get wrong: `pyannote-default`
is `VBxClustering` with threshold 0.6 / Fa 0.07 / Fb 0.8, NOT
`AgglomerativeClustering`. Those are the values behind the recorded baseline
DER 0.17048543579940637. `vbx` and `pyannote-default` are the same class,
differing only in whether hyperparameters are stated explicitly.

No GPU, no network, no audio. `SpeakerDiarization.__init__` loads real models,
so the two seams are exercised against a stand-in that reproduces the parts of
`__init__` that matter (enum lookup, the VBx PLDA special case, and
`_expects_num_speakers`), plus real `pyannote.pipeline.Pipeline` clustering
objects so the descriptor machinery is the library's own rather than a mock
that assumes the answer.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from harness.cache_id import pipeline_config_id
from harness.run_manifest import CLUSTERING_MODEL_CLASSES, VALID_CLUSTERING_MODELS
from pyannote.audio.pipelines.clustering import (
    AgglomerativeClustering,
    Clustering,
    KMeansClustering,
    VBxClustering,
)
from run_harness import (
    DEFAULT_CLUSTERING_MODEL,
    _parse_clustering_params,
    build_pipeline,
    main,
)

CHECKPOINT = "pyannote/speaker-diarization-community-1"

# The default condition's hyperparameters, from the shipped community-1
# config.yaml (`params.clustering`). Hardcoded here on purpose: if the default
# ever silently changes, the baseline DER stops reproducing, and this constant
# is what makes that a test failure rather than a mystery.
DEFAULT_VBX_PARAMS = {"threshold": 0.6, "Fa": 0.07, "Fb": 0.8}


# --------------------------------------------------------------------------
# A stand-in for SpeakerDiarization that reproduces the three things
# __init__ does with the clustering name (speaker_diarization.py:283-295).
# --------------------------------------------------------------------------


class _FakeSpeakerDiarization:
    """Reproduces `SpeakerDiarization.__init__`'s clustering block exactly.

    The real class loads a segmentation model, an embedding model and a PLDA
    from disk, so it cannot be constructed in a unit test. What this ticket
    needs to exercise is narrower: the enum lookup, the `VBxClustering`-only
    special case that passes `self._plda`, and `_expects_num_speakers` being
    derived from the selected class. Those three are copied here verbatim in
    behaviour, so a selection path that leaves them inconsistent fails here
    the same way it would in production.
    """

    def __init__(self, clustering="VBxClustering", batch_size=32, **kwargs):
        self.klustering = clustering
        self._plda = SimpleNamespace(name="fake-plda")
        self.training = False
        # Real __init__ sets both (speaker_diarization.py:236 and :259).
        # They are part of the intermediate cache key because batch shape
        # changes float reduction order and so the cached arrays' values.
        self.embedding_batch_size = batch_size
        self.segmentation_batch_size = batch_size

        try:
            Klustering = Clustering[clustering]
        except KeyError:
            raise ValueError(
                f"clustering must be one of [{', '.join(list(Clustering.__members__))}]"
            )

        if self.klustering == "VBxClustering":
            self.clustering = Klustering.value(self._plda, metric="cosine")
        else:
            self.clustering = Klustering.value(metric="cosine")

        self._expects_num_speakers = self.clustering.expects_num_clusters

    def instantiate(self, params):
        """Mirrors the nested-dict form `Pipeline.instantiate` takes, and
        delegates the clustering sub-dict to the real clustering object's own
        `instantiate` -- so an unknown parameter name raises the library's
        `ValueError: parameter '<name>' does not exist` rather than a
        harness-invented error that might be weaker."""
        for name, value in params.items():
            if name == "clustering":
                self.clustering.instantiate(value)
            else:
                setattr(self, name, value)
        return self

    def to(self, device):
        return self


def _fake_from_pretrained(checkpoint, token=None):
    """Stands in for `Pipeline.from_pretrained` on the DEFAULT path.

    The signature deliberately takes no `clustering` kwarg, because the real
    `from_pretrained` does not either (`core/pipeline.py:153-161`: a fixed
    signature with no `**kwargs`). An earlier version of this fake accepted one
    and so happily passed while the real call raised
    `TypeError: unexpected keyword argument 'clustering'` -- a fake that is more
    permissive than the thing it stands in for hides exactly the bug it should
    catch. Hence `build_pipeline` routes non-default classes through
    `_from_pretrained_with_clustering` instead, which the fake below stands in
    for separately.
    """
    pipeline = _FakeSpeakerDiarization(clustering="VBxClustering")
    pipeline.instantiate({"clustering": dict(DEFAULT_VBX_PARAMS)})
    return pipeline


def _fake_from_pretrained_with_clustering(checkpoint, class_name, token=None):
    """Stands in for `run_harness._from_pretrained_with_clustering`.

    Mirrors its one real deviation from the library: the shipped config's
    clustering defaults are filtered to the names the target class actually
    declares, because `Pipeline.instantiate` raises on a parameter the class
    does not have, and the shipped config carries VBx's `threshold`/`Fa`/`Fb`.
    """
    pipeline = _FakeSpeakerDiarization(clustering=class_name)
    declared = set(pipeline.clustering._parameters)
    kept = {k: v for k, v in DEFAULT_VBX_PARAMS.items() if k in declared}
    if kept:
        pipeline.instantiate({"clustering": kept})
    return pipeline


@contextmanager
def _patched_loaders():
    """Patch BOTH load paths at once.

    `build_pipeline` deliberately routes the default class through the untouched
    `Pipeline.from_pretrained` and any other class through
    `_from_pretrained_with_clustering`. Patching only one would leave half the
    selection matrix hitting the network, so both are stubbed together.
    """
    with patch("run_harness.Pipeline") as mock_pipeline_cls, patch(
        "run_harness._from_pretrained_with_clustering",
        side_effect=_fake_from_pretrained_with_clustering,
    ):
        mock_pipeline_cls.from_pretrained.side_effect = _fake_from_pretrained
        yield mock_pipeline_cls



# `kmeans` is selectable but NOT RUNNABLE by this harness: KMeansClustering's
# expects_num_clusters is True and the harness supplies no num_speakers, so
# `build_pipeline` refuses it (see its guard, and the finding recorded in the
# ticket). Selection itself must still be proven correct, so these two lists
# separate "can be selected" from "can be run".
RUNNABLE_MODELS = ("pyannote-default", "agglomerative", "vbx")
NUM_SPEAKERS_REQUIRED_MODELS = ("kmeans",)


def _select_without_runnability_guard(model, **kwargs):
    """Select a clustering model and return the pipeline even when the harness
    would refuse to RUN it.

    Needed because `kmeans` is in the vocabulary but unrunnable here, and
    criterion 1 still requires proof that selecting it yields a real
    `KMeansClustering` -- i.e. that the guard fires for the right reason (no
    num_speakers) and not because selection quietly failed."""
    captured = {}

    def _capture(checkpoint, class_name, token=None):
        pipeline = _fake_from_pretrained_with_clustering(checkpoint, class_name, token)
        captured["pipeline"] = pipeline
        return pipeline

    with patch("run_harness.Pipeline") as mock_pipeline_cls, patch(
        "run_harness._from_pretrained_with_clustering", side_effect=_capture
    ):
        mock_pipeline_cls.from_pretrained.side_effect = _fake_from_pretrained
        try:
            return build_pipeline(CHECKPOINT, token=None, clustering_model=model, **kwargs)
        except ValueError as exc:
            if "num_speakers" not in str(exc):
                raise
            return captured["pipeline"]


# --------------------------------------------------------------------------
# Criterion 1: --clustering-model selects the named class
# --------------------------------------------------------------------------


def test_vocabulary_is_the_expected_closed_set():
    """The vocabulary is closed on purpose. `OracleClustering` must NOT be in
    it -- oracle conditions are out of scope for this line of work, and an
    oracle clustering silently available as a sweep point would produce
    impossibly good numbers under a manifest that read like any other run."""
    assert set(VALID_CLUSTERING_MODELS) == {
        "pyannote-default",
        "agglomerative",
        "vbx",
        "kmeans",
    }
    assert "oracle" not in VALID_CLUSTERING_MODELS
    assert "OracleClustering" not in CLUSTERING_MODEL_CLASSES.values()


@pytest.mark.parametrize(
    "model,expected_class",
    [
        ("pyannote-default", VBxClustering),
        ("vbx", VBxClustering),
        ("agglomerative", AgglomerativeClustering),
        ("kmeans", KMeansClustering),
    ],
)
def test_clustering_model_selects_the_named_class(model, expected_class):
    """One case per vocabulary value (criterion 1). Asserts on the LIVE
    pipeline object, because the bug this guards against is a selection that
    falls through to the default while the manifest claims otherwise --
    checking the flag or the mapping table would not catch it."""
    pipeline = _select_without_runnability_guard(model)

    assert isinstance(pipeline.clustering, expected_class)


@pytest.mark.parametrize("model", list(VALID_CLUSTERING_MODELS))
def test_every_vocabulary_value_is_selectable_and_self_consistent(model):
    """Every vocabulary value must not only select its class but leave the
    pipeline INTERNALLY CONSISTENT.

    This is the specific reason class selection goes through construction
    rather than attribute assignment: assigning `pipeline.clustering` alone
    leaves `klustering` and `_expects_num_speakers` describing the previous
    class. Nothing crashes; the pipeline just quietly disagrees with itself,
    and `_expects_num_speakers` in particular changes whether speaker counts
    are passed through. Asserted here for all four values.
    """
    pipeline = _select_without_runnability_guard(model)

    expected_name = CLUSTERING_MODEL_CLASSES[model]
    assert pipeline.klustering == expected_name
    assert type(pipeline.clustering).__name__ == expected_name
    assert pipeline._expects_num_speakers == pipeline.clustering.expects_num_clusters


def test_unknown_clustering_model_fails_loudly_at_the_harness_layer():
    """An unknown value must error, not fall back to the default. Falling back
    is the silent path: the run would complete and score the default while the
    manifest recorded the requested-but-ignored model."""
    with _patched_loaders():
        with pytest.raises(ValueError) as excinfo:
            build_pipeline(CHECKPOINT, token=None, clustering_model="dbscan")

    # The error must name the valid values, or the operator has to read source
    # to recover from a typo.
    message = str(excinfo.value)
    assert "dbscan" in message
    for value in VALID_CLUSTERING_MODELS:
        assert value in message


def test_unknown_clustering_model_is_rejected_by_the_cli():
    """argparse must reject it too (exit code 2), so a typo dies before any
    model is loaded rather than 40 minutes into a run."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--data-root", ".", "--clustering-model", "dbscan"])
    assert excinfo.value.code == 2


def test_harness_vocabulary_is_a_subset_of_the_library_enum():
    """The harness layer must not be WEAKER than the library's own check.

    `SpeakerDiarization.__init__` already raises `ValueError` listing every
    valid `Clustering` enum member. If the harness mapped a vocabulary value to
    a class name the library does not know, the harness would accept it and the
    library would reject it later -- or worse, a future edit could make the
    harness accept something the library silently tolerates. Every mapped name
    must be a real enum member.
    """
    for model, class_name in CLUSTERING_MODEL_CLASSES.items():
        assert class_name in Clustering.__members__, (
            f"{model!r} maps to {class_name!r}, which is not a Clustering enum "
            f"member: {list(Clustering.__members__)}"
        )


def test_library_rejects_an_unknown_class_name_too():
    """Confirms the claim above rather than assuming it: the library's own
    error names the valid members."""
    with pytest.raises(ValueError) as excinfo:
        _FakeSpeakerDiarization(clustering="DBSCANClustering")
    assert "AgglomerativeClustering" in str(excinfo.value)


# --------------------------------------------------------------------------
# Criterion 2: --clustering-param reaches the instantiated object
# --------------------------------------------------------------------------


def test_clustering_param_is_readable_back_off_the_instance():
    """Criterion 2's core assertion: the override must be readable off the
    clustering OBJECT, not merely accepted by the parser.

    Note this is VBx, which overrides `__call__` rather than `cluster`
    (clustering.py:572) -- and is the DEFAULT path, not an exotic one. The
    override route is `Pipeline.instantiate`, which acts on the descriptor
    machinery in the base class, so it is unaffected by which method the
    subclass overrides. Asserted rather than assumed.
    """
    with _patched_loaders():
        pipeline = build_pipeline(
            CHECKPOINT,
            token=None,
            clustering_model="vbx",
            clustering_params={"threshold": 0.72},
        )

    assert pipeline.clustering.threshold == 0.72


def test_clustering_param_override_survives_alongside_untouched_params():
    """Overriding one parameter must not reset the others to descriptors.
    If it did, the pipeline would be left partly uninstantiated and the cache
    key would stop reflecting real values."""
    with _patched_loaders():
        pipeline = build_pipeline(
            CHECKPOINT,
            token=None,
            clustering_model="pyannote-default",
            clustering_params={"Fa": 0.25},
        )

    assert pipeline.clustering.Fa == 0.25
    assert pipeline.clustering.threshold == DEFAULT_VBX_PARAMS["threshold"]
    assert pipeline.clustering.Fb == DEFAULT_VBX_PARAMS["Fb"]


def test_unknown_clustering_param_name_fails_loudly():
    """A silently ignored hyperparameter produces a flat sweep curve that
    reads as a real negative result -- the worst outcome in the list. The
    library's `Pipeline.instantiate` already raises `ValueError: parameter
    '<name>' does not exist`; this confirms the harness routes through it
    rather than around it."""
    with _patched_loaders():
        with pytest.raises(ValueError) as excinfo:
            build_pipeline(
                CHECKPOINT,
                token=None,
                clustering_model="vbx",
                clustering_params={"thresold": 0.72},  # typo, on purpose
            )

    assert "thresold" in str(excinfo.value)


def test_param_valid_for_one_class_is_rejected_for_another():
    """`Fa` exists on VBx and not on agglomerative. Requesting it for
    agglomerative must fail rather than be quietly dropped -- otherwise a
    sweep over the wrong class would silently run at default settings."""
    with _patched_loaders():
        with pytest.raises(ValueError):
            build_pipeline(
                CHECKPOINT,
                token=None,
                clustering_model="agglomerative",
                clustering_params={"Fa": 0.2},
            )


@pytest.mark.parametrize(
    "raw,expected",
    [
        (["threshold=0.7"], {"threshold": 0.7}),
        (["Fa=0.07", "Fb=0.8"], {"Fa": 0.07, "Fb": 0.8}),
        (["min_cluster_size=12"], {"min_cluster_size": 12}),
        (["method=centroid"], {"method": "centroid"}),
        (["constrained_assignment=true"], {"constrained_assignment": True}),
        (["constrained_assignment=False"], {"constrained_assignment": False}),
    ],
)
def test_clustering_param_parsing_preserves_types(raw, expected):
    """`min_cluster_size` is an Integer descriptor and `method` a Categorical,
    so a parser that turned everything into a float or left everything a string
    would break those. Types matter to the cache key too: `0.7` and `"0.7"`
    would hash differently and split a sweep's cache."""
    parsed = _parse_clustering_params(raw)
    assert parsed == expected
    for key in expected:
        assert type(parsed[key]) is type(expected[key])


@pytest.mark.parametrize("raw", [["threshold"], ["=0.7"], ["threshold:0.7"]])
def test_malformed_clustering_param_fails_loudly(raw):
    """Malformed `name=value` input must not be skipped silently."""
    with pytest.raises(ValueError):
        _parse_clustering_params(raw)


def test_no_clustering_params_is_not_an_error():
    """Omitting the flag entirely is the default condition, not a failure."""
    assert _parse_clustering_params(None) == {}
    assert _parse_clustering_params([]) == {}


# --------------------------------------------------------------------------
# Criterion 3: cache keys differ by clustering hyperparameter
#
# The key restructure itself is owned by [[pre-clustering-embedding-cache]],
# which landed first. These tests VERIFY it; they do not reimplement it. The
# sensitivity test below exists because a test that cannot fail is not a
# guard -- it demonstrates these assertions failing against the PRE-CHANGE
# key logic.
# --------------------------------------------------------------------------


def _old_pipeline_config_id(checkpoint, pipeline):
    """The key logic as it was BEFORE the restructure: `run_harness.py` passed
    the bare checkpoint string as `pipeline_config_id` (the ticket's §3). Kept
    here only so the criterion-3 assertions can be shown to have teeth."""
    return checkpoint


def test_final_cache_key_differs_by_clustering_hyperparameter():
    """Criterion 3. Two runs differing only in a clustering hyperparameter
    must not share a final-hypothesis cache key -- otherwise a sweep computes
    point 1, caches it, and serves that same RTTM at every subsequent point:
    identical DER everywhere, a flat curve, valid manifests, no error."""
    with _patched_loaders():
        a = build_pipeline(
            CHECKPOINT, token=None, clustering_model="vbx",
            clustering_params={"threshold": 0.6},
        )
        b = build_pipeline(
            CHECKPOINT, token=None, clustering_model="vbx",
            clustering_params={"threshold": 0.7},
        )

    assert pipeline_config_id(CHECKPOINT, a) != pipeline_config_id(CHECKPOINT, b)


def test_final_cache_key_differs_by_clustering_class():
    """Same checkpoint, different algorithm -> different key."""
    with _patched_loaders():
        vbx = build_pipeline(CHECKPOINT, token=None, clustering_model="vbx")
        agg = build_pipeline(CHECKPOINT, token=None, clustering_model="agglomerative")

    assert pipeline_config_id(CHECKPOINT, vbx) != pipeline_config_id(CHECKPOINT, agg)


def test_criterion_3_assertion_is_sensitive_to_the_key_logic():
    """SENSITIVITY DEMONSTRATION -- the point of this test is that it proves
    the two tests above are not passing vacuously.

    The ticket said criterion 3's test "must fail on the current code". That
    wording predates the key restructure, so it cannot fail now. Instead this
    shows the same assertion failing against the PRE-CHANGE key logic: under
    the old bare-checkpoint key, two different clustering configurations
    collide. If someone reverted the restructure, the tests above would start
    failing -- which is what having teeth means.
    """
    with _patched_loaders():
        a = build_pipeline(
            CHECKPOINT, token=None, clustering_model="vbx",
            clustering_params={"threshold": 0.6},
        )
        b = build_pipeline(
            CHECKPOINT, token=None, clustering_model="vbx",
            clustering_params={"threshold": 0.7},
        )

    # Under the OLD logic the keys collide -- the defect.
    assert _old_pipeline_config_id(CHECKPOINT, a) == _old_pipeline_config_id(
        CHECKPOINT, b
    )
    # Under the CURRENT logic they do not.
    assert pipeline_config_id(CHECKPOINT, a) != pipeline_config_id(CHECKPOINT, b)


def test_intermediate_key_is_shared_across_clustering_configurations():
    """The other half of criterion 7: two clustering configurations must
    differ in the FINAL key while SHARING the intermediate one. Segmentation
    and embeddings run before clustering and cannot depend on it, so a sweep
    reuses one cached copy. If this ever differs, every sweep point pays for
    GPU inference again."""
    from harness.cache_id import intermediate_config_id

    with _patched_loaders():
        a = build_pipeline(
            CHECKPOINT, token=None, clustering_model="vbx",
            clustering_params={"threshold": 0.6},
        )
        b = build_pipeline(CHECKPOINT, token=None, clustering_model="agglomerative")

    assert intermediate_config_id(CHECKPOINT, a) == intermediate_config_id(CHECKPOINT, b)
    assert pipeline_config_id(CHECKPOINT, a) != pipeline_config_id(CHECKPOINT, b)


# --------------------------------------------------------------------------
# Criterion 4: the manifest records class AND hyperparameters, readably
# --------------------------------------------------------------------------


def test_manifest_records_clustering_class_and_every_hyperparameter(tmp_path):
    """Criterion 4, verified by READING BACK a written manifest rather than
    by inspecting the writer -- a writer can look correct while the value
    never reaches the file."""
    import json

    fake_pipeline = MagicMock()

    data_root = tmp_path / "data"
    data_root.mkdir()
    runs_dir = tmp_path / "runs"

    argv = [
        "--data-root", str(data_root),
        "--per-file-csv", str(tmp_path / "per_file.csv"),
        "--summary", str(tmp_path / "summary.json"),
        "--runs-dir", str(runs_dir),
        "--device", "cpu",
        "--clustering-model", "vbx",
        "--clustering-param", "threshold=0.71",
    ]

    with patch("run_harness.Pipeline") as mock_pipeline_cls, \
         patch("run_harness.run_harness") as mock_run_harness:
        mock_pipeline_cls.from_pretrained.side_effect = _fake_from_pretrained
        mock_run_harness.return_value = {"der": 0.0}
        main(argv)

        kwargs = mock_run_harness.call_args.kwargs

    # Write a real manifest through the real writer with what main() passed,
    # then read it off disk.
    from harness.run_manifest import write_manifest

    run_config = {
        "pipeline_config_id": kwargs["pipeline_config_id"],
        "segmentation_source_id": "baseline",
        "clustering_model": kwargs["clustering_model"],
        "clustering_config": kwargs["clustering_config"],
        "extra_pipeline_steps": [],
        "split": "test",
        "condition": "baseline",
        "der_collar": 0.0,
        "der_skip_overlap": False,
        "refinement_strategy": "identity",
        "corpus": "AMI",
        "mic_condition": "IHM",
        "git_commit": "deadbeef",
    }
    manifest_path = write_manifest(run_config, {"der": 0.0}, runs_dir)
    written = json.loads(manifest_path.read_text())["run_config"]

    # The CLASS.
    assert "VBxClustering" in written["clustering_model"]
    # EVERY hyperparameter, in readable form -- not just the overridden one.
    for name in ("threshold", "Fa", "Fb"):
        assert name in written["clustering_model"], (
            f"{name} missing from manifest clustering_model: "
            f"{written['clustering_model']!r}"
        )
    # The overridden VALUE, so the manifest distinguishes sweep points.
    assert "0.71" in written["clustering_model"]


def test_manifest_clustering_model_is_derived_from_the_live_pipeline(tmp_path):
    """Failure mode 1, guarded directly. If `clustering_model` were taken from
    the FLAG rather than from the instantiated pipeline, a flag that never
    reached the pipeline would still produce a manifest that read perfectly.
    So the recorded value must reflect the live object -- here, the default
    VBx values, because no override was requested."""
    data_root = tmp_path / "data"
    data_root.mkdir()

    argv = [
        "--data-root", str(data_root),
        "--per-file-csv", str(tmp_path / "per_file.csv"),
        "--summary", str(tmp_path / "summary.json"),
        "--runs-dir", str(tmp_path / "runs"),
        "--device", "cpu",
    ]

    with patch("run_harness.Pipeline") as mock_pipeline_cls, \
         patch("run_harness.run_harness") as mock_run_harness:
        mock_pipeline_cls.from_pretrained.side_effect = _fake_from_pretrained
        mock_run_harness.return_value = {"der": 0.0}
        main(argv)

        kwargs = mock_run_harness.call_args.kwargs

    recorded = kwargs["clustering_model"]
    assert "VBxClustering" in recorded
    assert "threshold=0.6" in recorded
    assert "Fa=0.07" in recorded
    assert "Fb=0.8" in recorded


# --------------------------------------------------------------------------
# Criterion 5: omitting --clustering-model reproduces current behaviour
# --------------------------------------------------------------------------


def test_default_is_pyannote_default():
    assert DEFAULT_CLUSTERING_MODEL == "pyannote-default"


def test_omitting_the_flag_yields_vbx_with_the_baseline_hyperparameters():
    """Criterion 5, unit half. The coordinator runs the DER check; this asserts
    the default path resolves to VBxClustering with 0.6/0.07/0.8 -- the values
    behind DER 0.17048543579940637. NOT AgglomerativeClustering, which several
    docs wrongly claim; mapping the default to agglomerative would silently
    change the default condition."""
    with _patched_loaders():
        pipeline = build_pipeline(CHECKPOINT, token=None)

    assert isinstance(pipeline.clustering, VBxClustering)
    assert pipeline.klustering == "VBxClustering"
    assert pipeline.clustering.threshold == DEFAULT_VBX_PARAMS["threshold"]
    assert pipeline.clustering.Fa == DEFAULT_VBX_PARAMS["Fa"]
    assert pipeline.clustering.Fb == DEFAULT_VBX_PARAMS["Fb"]
    # The speaker-count leak stays dormant: VBx does not expect a cluster
    # count, so no num_speakers is plumbed through.
    assert pipeline._expects_num_speakers is False


def test_pyannote_default_and_explicit_vbx_are_the_same_condition():
    """`vbx` and `pyannote-default` are the SAME class with the SAME shipped
    hyperparameters, so they must produce the SAME cache key. If they didn't,
    the two would look like two conditions in a results table and the baseline
    would appear to be reproduced by a different setting than it was."""
    with _patched_loaders():
        default = build_pipeline(
            CHECKPOINT, token=None, clustering_model="pyannote-default"
        )
        explicit = build_pipeline(CHECKPOINT, token=None, clustering_model="vbx")

    assert pipeline_config_id(CHECKPOINT, default) == pipeline_config_id(
        CHECKPOINT, explicit
    )


def test_default_pipeline_config_id_is_unchanged_by_this_ticket():
    """THE REGRESSION GUARD FOR THE COORDINATOR'S BASELINE RUN.

    The default configuration's final cache key must be exactly what the
    pre-ticket code produced. If it shifts, the coordinator's cold baseline
    run recomputes and every cached default-condition RTTM is orphaned -- so
    a shift must be a deliberate, noticed decision, not a side effect.

    The expected value is the key for (checkpoint, VBxClustering,
    Fa=0.07, Fb=0.8, threshold=0.6), which is what the code produced before
    `--clustering-model` existed, since the default path is unchanged.
    """
    with _patched_loaders():
        pipeline = build_pipeline(CHECKPOINT, token=None)

    # Recomputed independently of build_pipeline, from a bare VBx instance
    # carrying the shipped values -- so this compares the SELECTION PATH's
    # output against the key formula, rather than against itself.
    reference = _FakeSpeakerDiarization(clustering="VBxClustering")
    reference.instantiate({"clustering": dict(DEFAULT_VBX_PARAMS)})

    assert pipeline_config_id(CHECKPOINT, pipeline) == pipeline_config_id(
        CHECKPOINT, reference
    )


def test_no_flags_takes_the_untouched_library_load_path():
    """Belt-and-braces on criterion 5, and the reason the default is a separate
    branch in `build_pipeline`.

    With no clustering flags the harness must call
    `Pipeline.from_pretrained(checkpoint, token=...)` and nothing else -- the
    same call the pre-ticket code made -- so the shipped config's own class and
    hyperparameter instantiation are untouched and the baseline condition cannot
    have moved. It must NOT go through `_from_pretrained_with_clustering`, which
    reimplements the load sequence and filters shipped defaults.
    """
    with patch("run_harness.Pipeline") as mock_pipeline_cls, patch(
        "run_harness._from_pretrained_with_clustering"
    ) as mock_helper:
        mock_pipeline_cls.from_pretrained.side_effect = _fake_from_pretrained
        build_pipeline(CHECKPOINT, token="tok")

    mock_helper.assert_not_called()
    mock_pipeline_cls.from_pretrained.assert_called_once_with(CHECKPOINT, token="tok")


def test_explicit_vbx_also_takes_the_untouched_path():
    """`vbx` resolves to the shipped class, so it must take the same untouched
    path rather than the reimplemented one -- otherwise the two spellings of one
    condition could diverge."""
    with patch("run_harness.Pipeline") as mock_pipeline_cls, patch(
        "run_harness._from_pretrained_with_clustering"
    ) as mock_helper:
        mock_pipeline_cls.from_pretrained.side_effect = _fake_from_pretrained
        build_pipeline(CHECKPOINT, token="tok", clustering_model="vbx")

    mock_helper.assert_not_called()


def test_non_default_class_takes_the_override_path():
    """The converse: a non-default class MUST go through the override helper,
    since `from_pretrained` has no seam for a construction parameter."""
    with patch("run_harness.Pipeline") as mock_pipeline_cls, patch(
        "run_harness._from_pretrained_with_clustering",
        side_effect=_fake_from_pretrained_with_clustering,
    ) as mock_helper:
        mock_pipeline_cls.from_pretrained.side_effect = _fake_from_pretrained
        build_pipeline(CHECKPOINT, token="tok", clustering_model="agglomerative")

    mock_helper.assert_called_once()
    assert mock_helper.call_args.args[1] == "AgglomerativeClustering"


# --------------------------------------------------------------------------
# Finding: `kmeans` is selectable but not runnable by this harness
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model", NUM_SPEAKERS_REQUIRED_MODELS)
def test_model_requiring_num_speakers_is_refused_at_selection_time(model):
    """`KMeansClustering.expects_num_clusters` is True, and
    `SpeakerDiarization.apply()` requires `num_speakers` in that case
    (speaker_diarization.py:600-607). The harness builds `file` as
    {"uri", "audio"} and passes no count, and deciding where k comes from is a
    separate ticket.

    Refused at SELECTION time rather than left to the library, for two reasons.
    The library would raise part-way through the corpus, after minutes of GPU
    work. And worse: `Runner.run()` sets `pipeline.training = True`, and the
    library's guard falls back to `len(file["annotation"].labels())` when an
    annotation is present -- so if `annotation` ever reached the file dict, a
    kmeans run would silently take its speaker count from the ground truth and
    report an oracle-count result under an ordinary baseline manifest.
    """
    with pytest.raises(ValueError) as excinfo:
        with _patched_loaders():
            build_pipeline(CHECKPOINT, token=None, clustering_model=model)

    message = str(excinfo.value)
    assert "num_speakers" in message
    # Must be refused for the RIGHT reason -- naming the class, not a generic
    # failure that could equally mean selection itself broke.
    assert CLUSTERING_MODEL_CLASSES[model] in message


@pytest.mark.parametrize("model", NUM_SPEAKERS_REQUIRED_MODELS)
def test_the_refusal_is_not_hiding_a_broken_selection(model):
    """Failure mode 3: a raise is only as good as the distinction it draws.
    This refusal must mean "selected correctly, cannot run", not "selection
    failed". Proven by checking the pipeline built before the guard fired really
    holds the requested class."""
    pipeline = _select_without_runnability_guard(model)

    assert type(pipeline.clustering).__name__ == CLUSTERING_MODEL_CLASSES[model]
    assert pipeline.clustering.expects_num_clusters is True


@pytest.mark.parametrize("model", RUNNABLE_MODELS)
def test_runnable_models_are_not_refused(model):
    """The guard must not over-fire. Every model whose clustering does not need
    a speaker count must build cleanly."""
    with _patched_loaders():
        pipeline = build_pipeline(CHECKPOINT, token=None, clustering_model=model)

    assert pipeline._expects_num_speakers is False


def test_num_speakers_guard_does_not_misfire_on_a_mock_pipeline():
    """REGRESSION. The first version of the guard used plain truthiness:
    `if getattr(pipeline, "_expects_num_speakers", False)`. `getattr` on a
    `MagicMock` returns a truthy Mock for ANY attribute name, so the guard
    refused every mock-driven caller and broke six pre-existing tests
    (test_run_harness_cli.py and test_oracle_combined_cells.py).

    The guard now tests `is True`. Kept as a test because the failure was loud
    here but would be worse in reverse: a guard that silently never fires is
    how an unrunnable configuration reaches a real run.
    """
    fake = MagicMock()
    fake.to.return_value = fake

    with patch("run_harness.Pipeline") as mock_pipeline_cls:
        mock_pipeline_cls.from_pretrained.return_value = fake
        # Must not raise, even though getattr(fake, "_expects_num_speakers")
        # is a truthy Mock.
        assert build_pipeline(CHECKPOINT, token=None) is fake


def test_num_speakers_guard_ignores_a_truthy_non_bool():
    """The converse of the above, stated explicitly: only a real `True`
    refuses. Anything else means the attribute is not the declared bool it is
    supposed to be, and inferring intent from truthiness is what caused the
    regression above."""
    with _patched_loaders():
        pipeline = build_pipeline(CHECKPOINT, token=None)
    pipeline._expects_num_speakers = "yes"  # truthy, but not the bool

    with _patched_loaders() as mock_pipeline_cls:
        mock_pipeline_cls.from_pretrained.side_effect = lambda *a, **k: pipeline
        # No raise: the attribute is not True.
        assert build_pipeline(CHECKPOINT, token=None) is pipeline
