"""Configuration for the first retrieval vertical."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from ..core.config import StrictConfig

Name = Annotated[str, Field(min_length=1)]
PositiveInt = Annotated[int, Field(gt=0)]


class PreprocessorConfig(StrictConfig):
    name: Name
    binding: Name


class RetrieverConfig(StrictConfig):
    name: Name
    binding: Name
    preprocessor: Name
    input: Name | None = None
    top_k: PositiveInt = 20


class RRFFusionConfig(StrictConfig):
    name: Name
    type: Literal["rrf"] = "rrf"
    inputs: Annotated[tuple[Name, ...], Field(min_length=2)]
    k: PositiveInt = 60
    top_k: PositiveInt = 20


class RetrievalConfig(StrictConfig):
    version: Literal[1] = 1
    name: Name
    kind: Literal["retrieve"] = "retrieve"
    preprocessors: tuple[PreprocessorConfig, ...]
    retrievers: tuple[RetrieverConfig, ...]
    fusions: tuple[RRFFusionConfig, ...] = ()
    outputs: dict[Name, Name]
