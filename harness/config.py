"""Harness configuration: data root, split, AMI condition, metric settings,
cache dir, and HF token loading. See TICKET-02-config-module.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from dotenv import dotenv_values

SUPPORTED_CONDITIONS = ("IHM", "SDM")


class ConfigError(ValueError):
    """Raised when harness configuration is invalid or incomplete."""


@dataclass(frozen=True)
class HarnessConfig:
    data_root: Path
    split: str
    condition: str
    cache_dir: Path
    der_collar: float
    der_skip_overlap: bool
    hf_token: Optional[str]

    @classmethod
    def load(
        cls,
        data_root: Union[str, Path],
        split: str = "test",
        condition: str = "IHM",
        cache_dir: Union[str, Path] = Path(".harness_cache"),
        dotenv_path: Optional[Union[str, Path]] = None,
        der_collar: float = 0.0,
        der_skip_overlap: bool = False,
    ) -> "HarnessConfig":
        data_root = Path(data_root)
        if not data_root.exists():
            raise ConfigError(
                f"AMI data root does not exist: {data_root}. "
                "Pass the path to a local AMI checkout via `data_root` -- "
                "this harness does not download AMI itself."
            )

        if condition not in SUPPORTED_CONDITIONS:
            raise ConfigError(
                f"Unsupported AMI condition {condition!r}; "
                f"expected one of {SUPPORTED_CONDITIONS}"
            )

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        hf_token = None
        if dotenv_path is not None:
            hf_token = dotenv_values(dotenv_path).get("HF_TOKEN")

        # pyannote's dependency check rejects this fork's dev version string;
        # this must be a real process env var since pyannote reads it via
        # os.getenv internally (see TICKET-00-environment-setup.md).
        os.environ.setdefault("PYANNOTE_SKIP_DEPENDENCY_CHECK", "1")

        return cls(
            data_root=data_root,
            split=split,
            condition=condition,
            cache_dir=cache_dir,
            der_collar=der_collar,
            der_skip_overlap=der_skip_overlap,
            hf_token=hf_token,
        )
