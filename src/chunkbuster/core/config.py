"""Strict configuration loading shared by both products."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from ..errors import ConfigurationError

ConfigInput = str | Path | Mapping[str, Any] | BaseModel


class StrictConfig(BaseModel):
    """Immutable base for product-specific configurations."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def load_config[ConfigModel: BaseModel](
    source: ConfigInput,
    model: type[ConfigModel],
) -> ConfigModel:
    """Load YAML, JSON, a mapping, or another validated model."""
    try:
        if isinstance(source, BaseModel):
            data: Any = source.model_dump(mode="python")
        elif isinstance(source, Mapping):
            data = deepcopy(dict(source))
        else:
            path = Path(source)
            if path.suffix.lower() in {".yaml", ".yml"}:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
            elif path.suffix.lower() == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
            else:
                raise ConfigurationError(
                    "config path must end in .yaml, .yml, or .json"
                )
        if not isinstance(data, Mapping):
            raise ConfigurationError("config root must be a mapping")
        return model.model_validate(data)
    except ConfigurationError:
        raise
    except (OSError, ValueError, yaml.YAMLError, ValidationError) as exc:
        raise ConfigurationError(f"invalid configuration: {exc}") from exc
