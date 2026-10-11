"""Environment sanity checks for the eval-harness dev setup.

See TICKET-00-environment-setup.md. These guard against the two problems
found during harness scoping: this fork's src/pyannote/audio not actually
being what gets imported, and the interpreter/dependency versions not
meeting pyproject.toml's own declared floors.
"""

import os
import re
import sys
from importlib.metadata import version as installed_version
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_pyproject_text() -> str:
    return (REPO_ROOT / "pyproject.toml").read_text()


def _parse_requires_python() -> str:
    text = _read_pyproject_text()
    match = re.search(r'requires-python\s*=\s*"([^"]+)"', text)
    assert match, "requires-python not found in pyproject.toml"
    return match.group(1)


def _parse_dependency_floor(package: str) -> str:
    text = _read_pyproject_text()
    match = re.search(rf'{re.escape(package)}>=([0-9][\w.]*)', text)
    assert match, f"{package} floor not found in pyproject.toml"
    return match.group(1)


def _version_tuple(v: str):
    return tuple(int(p) for p in re.findall(r"\d+", v)[:3])


def test_pyannote_audio_resolves_to_this_fork():
    import pyannote.audio

    resolved = os.path.realpath(pyannote.audio.__file__)
    expected_root = os.path.realpath(REPO_ROOT / "src" / "pyannote" / "audio")
    assert resolved.startswith(expected_root), (
        f"pyannote.audio resolved to {resolved!r}, expected it under "
        f"{expected_root!r} (this fork's src/, not an installed package)"
    )


def test_python_version_meets_pyproject_floor():
    requires = _parse_requires_python()
    match = re.search(r">=\s*([0-9.]+)", requires)
    assert match, f"unsupported requires-python spec: {requires!r}"
    floor = _version_tuple(match.group(1))
    assert sys.version_info[:3] >= floor, (
        f"running Python {sys.version_info[:3]}, "
        f"but pyproject.toml requires >= {floor}"
    )


@pytest.mark.parametrize("package", ["pyannote-core", "pyannote-metrics"])
def test_installed_dependency_versions_satisfy_pyproject_floors(package):
    floor = _version_tuple(_parse_dependency_floor(package))
    installed = _version_tuple(installed_version(package))
    assert installed >= floor, (
        f"{package} {installed} installed, but pyproject.toml requires >= {floor}"
    )
