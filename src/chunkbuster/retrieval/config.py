"""Configuration for the first retrieval vertical."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from ..core.config import StrictConfig
from ..core.normalization import Normalization

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


class FusionConfig(StrictConfig):
    name: Name
    type: Literal["rrf", "weighted_rrf", "weighted_sum", "comb_mnz"] = "weighted_rrf"
    inputs: Annotated[tuple[Name, ...], Field(min_length=2)]
    weights: tuple[Annotated[float, Field(ge=0)], ...] = ()
    normalization: Normalization = "none"
    k: PositiveInt = 60
    top_k: PositiveInt = 20

    @model_validator(mode="after")
    def validate_fusion(self) -> FusionConfig:
        if len(self.inputs) != len(set(self.inputs)):
            raise ValueError("fusion input names must be unique")
        if self.weights and len(self.weights) != len(self.inputs):
            raise ValueError("fusion weights must match inputs")
        if self.weights and not any(self.weights):
            raise ValueError("fusion requires at least one positive weight")
        if self.type in {"rrf", "weighted_rrf"} and self.normalization != "none":
            raise ValueError("weighted_rrf does not accept score normalization")
        return self


class RetrievalConfig(StrictConfig):
    version: Literal[1] = 1
    name: Name
    kind: Literal["retrieve"] = "retrieve"
    preprocessors: tuple[PreprocessorConfig, ...]
    retrievers: tuple[RetrieverConfig, ...]
    fusions: tuple[FusionConfig, ...] = ()
    outputs: dict[Name, Name]
