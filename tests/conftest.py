# MIT License
#
# Copyright (c) 2020- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """

    from pyannote.database import registry

    registry.load_database("tests/data/database.yml")


# --- Shared fixtures for the eval-harness tickets (TICKET-eval-harness.md) ---
#
# These build a real, tiny, fully offline SpeakerDiarization pipeline -- no
# network, no HF token, no real AMI/community-1 weights -- so harness tests
# exercise the actual pipeline class, not a stand-in for it. Real (but tiny)
# segmentation and embedding models are trained in-process against the
# repo's existing `Debug` protocol fixture, and a real `PLDA` instance is
# built from small, well-conditioned synthetic matrices (SpeakerDiarization
# always loads a PLDA in __init__, regardless of clustering choice -- and
# `apply()` always extracts embeddings before clustering, even under
# OracleClustering, so there's no clustering choice that skips needing a
# real embedding model too). Originally built for TICKET-01's seam tests;
# shared here because TICKET-04 and TICKET-06 need the same fixture.

import numpy as np
import pytest
import torch
from lightning import Trainer
from pyannote.audio.core.plda import PLDA
from pyannote.audio.models.embedding.debug import SimpleEmbeddingModel
from pyannote.audio.models.segmentation.debug import SimpleSegmentationModel
from pyannote.audio.pipelines import SpeakerDiarization as SpeakerDiarizationPipeline
from pyannote.audio.tasks import SpeakerDiarization as SpeakerDiarizationTask
from pyannote.audio.tasks import SupervisedRepresentationLearningWithArcFace
from pyannote.database import FileFinder, registry as _registry


class _MaskAwareEmbeddingModel(SimpleEmbeddingModel):
    """SimpleEmbeddingModel doesn't accept the `weights` mask that
    SpeakerDiarization.get_embeddings() always passes when extracting
    per-speaker embeddings (masked pooling over overlapping speech). Real
    embedding models (e.g. community-1's WeSpeaker) support this; the debug
    model doesn't need to for its own tests, but ours drives it through
    SpeakerDiarization.apply(), which does. Ignoring the mask is fine here --
    this fixture only needs to produce *some* deterministic embedding, not an
    accurate one."""

    def forward(self, waveforms: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        return super().forward(waveforms)


def _fit(model, task):
    """Train `model` on `task` for exactly one step -- just enough to attach
    valid specifications and produce deterministic eval-mode output. These
    are real trained models, not mocks: get_segmentations()/apply() run them
    for real."""
    trainer = Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)
    return model


@pytest.fixture(scope="session")
def protocol():
    return _registry.get_protocol(
        "Debug.SpeakerDiarization.Debug", preprocessors={"audio": FileFinder()}
    )


@pytest.fixture(scope="session")
def trained_segmentation_model(protocol):
    task = SpeakerDiarizationTask(protocol)
    return _fit(SimpleSegmentationModel(task=task), task)


@pytest.fixture(scope="session")
def trained_embedding_model(protocol):
    """apply() always extracts embeddings before clustering -- even under
    OracleClustering -- so every full pipeline() call needs a real embedding
    model, not just the segmentation seam tests."""
    task = SupervisedRepresentationLearningWithArcFace(protocol)
    return _fit(_MaskAwareEmbeddingModel(task=task), task)


@pytest.fixture(scope="session")
def dummy_plda(tmp_path_factory):
    """A real PLDA instance built from small, well-conditioned synthetic
    matrices (identity transforms, distinct positive eigenvalues) so
    vbx_setup's generalized eigenvalue problem is numerically well-behaved.
    SpeakerDiarization.__init__ always loads a PLDA, regardless of clustering
    choice, so this can't be skipped even under OracleClustering."""
    tmpdir = tmp_path_factory.mktemp("plda")
    dim = 4

    transform_path = tmpdir / "transform.npz"
    np.savez(transform_path, mean1=np.zeros(dim), mean2=np.zeros(dim), lda=np.eye(dim))

    plda_path = tmpdir / "plda.npz"
    np.savez(
        plda_path, mu=np.zeros(dim), tr=np.eye(dim), psi=np.array([4.0, 3.0, 2.0, 1.0])
    )

    return PLDA(
        transform_npz=str(transform_path), plda_npz=str(plda_path), lda_dimension=dim
    )


@pytest.fixture()
def pipeline(trained_segmentation_model, dummy_plda):
    """A real, fully offline SpeakerDiarization pipeline instance, sufficient
    for exercising get_segmentations() directly: OracleClustering keeps this
    fixture cheap by skipping SpeakerDiarization.__init__'s embedding-model
    setup, for tests that never invoke embeddings/clustering."""
    built = SpeakerDiarizationPipeline(
        segmentation=trained_segmentation_model,
        plda=dummy_plda,
        clustering="OracleClustering",
    )
    built.training = False
    return built


@pytest.fixture()
def full_pipeline(trained_segmentation_model, trained_embedding_model, dummy_plda):
    """A real, fully offline SpeakerDiarization pipeline instance capable of
    a complete apply() call: AgglomerativeClustering doesn't require
    num_speakers and has a simple, known hyperparameter set to instantiate."""
    built = SpeakerDiarizationPipeline(
        segmentation=trained_segmentation_model,
        embedding=trained_embedding_model,
        plda=dummy_plda,
        clustering="AgglomerativeClustering",
    )
    built.instantiate(
        {
            "segmentation": {"min_duration_off": 0.0},
            "clustering": {
                "threshold": 0.7,
                "method": "centroid",
                "min_cluster_size": 1,
            },
        }
    )
    built.training = False
    return built


@pytest.fixture()
def one_file(protocol):
    return next(protocol.test())
