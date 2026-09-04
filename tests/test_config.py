"""Tests for TICKET-02: the harness config module.

harness/config.py doesn't exist yet at the time these tests are written --
that's deliberate. Every test here calls `harness.config.HarnessConfig`
(and its `.load()` factory) as if it already existed, so the first run of
this suite is expected to fail on import, not on an assertion. Only once
that failure has been shown does harness/config.py get written.
"""

import os

import pytest

from harness.config import ConfigError, HarnessConfig

FAKE_HF_TOKEN = "hf_fake_token_for_tests_only_zzyx"


@pytest.fixture()
def data_root(tmp_path):
    root = tmp_path / "ami_data"
    root.mkdir()
    return root


@pytest.fixture()
def cache_dir(tmp_path):
    # every test that doesn't specifically exercise cache_dir creation still
    # needs to pass one explicitly -- the default argument resolves relative
    # to cwd, and letting it fall through would litter the repo root with a
    # real .harness_cache/ directory as a side effect of running tests.
    return tmp_path / "cache"


def test_missing_data_root_raises_clear_error(tmp_path, cache_dir):
    missing = tmp_path / "does_not_exist"

    with pytest.raises(ConfigError) as exc_info:
        HarnessConfig.load(data_root=missing, cache_dir=cache_dir)

    assert str(missing) in str(exc_info.value)


def test_default_condition_is_ihm(data_root, cache_dir):
    config = HarnessConfig.load(data_root=data_root, cache_dir=cache_dir)
    assert config.condition == "IHM"


def test_condition_overridable_to_sdm(data_root, cache_dir):
    config = HarnessConfig.load(data_root=data_root, condition="SDM", cache_dir=cache_dir)
    assert config.condition == "SDM"


def test_invalid_condition_rejected(data_root, cache_dir):
    with pytest.raises(ConfigError) as exc_info:
        HarnessConfig.load(data_root=data_root, condition="BOGUS", cache_dir=cache_dir)

    assert "BOGUS" in str(exc_info.value)


def test_hf_token_loaded_from_dotenv(data_root, cache_dir, tmp_path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(f"HF_TOKEN={FAKE_HF_TOKEN}\n")

    config = HarnessConfig.load(
        data_root=data_root, cache_dir=cache_dir, dotenv_path=dotenv_path
    )

    assert config.hf_token == FAKE_HF_TOKEN
    # deliberate design: the token lives on the config object, not in the
    # process environment -- loading a harness config must not mutate
    # os.environ with secrets as a side effect.
    assert os.environ.get("HF_TOKEN") != FAKE_HF_TOKEN


def test_cache_dir_created_if_missing(data_root, tmp_path):
    nested_cache_dir = tmp_path / "nested" / "cache"
    assert not nested_cache_dir.exists()

    config = HarnessConfig.load(data_root=data_root, cache_dir=nested_cache_dir)

    assert nested_cache_dir.is_dir()
    assert config.cache_dir == nested_cache_dir


def test_pyannote_skip_dependency_check_env_var_documented(data_root, cache_dir, monkeypatch):
    # start from a clean slate so this test genuinely exercises the
    # behavior rather than riding on whatever the outer shell happened to
    # already have set.
    monkeypatch.delenv("PYANNOTE_SKIP_DEPENDENCY_CHECK", raising=False)

    HarnessConfig.load(data_root=data_root, cache_dir=cache_dir)

    assert os.environ.get("PYANNOTE_SKIP_DEPENDENCY_CHECK") in {"1", "true", "True", "yes", "Yes"}
