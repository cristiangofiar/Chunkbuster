"""Configuration for the first tree-classification vertical."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, JsonValue, field_validator

from ..core.config import StrictConfig

Name = Annotated[str, Field(min_length=1)]
PositiveInt = Annotated[int, Field(gt=0)]


class EmbeddingPreprocessorConfig(StrictConfig):
    name: Name
    type: Literal["embedding"] = "embedding"
    binding: Name
    dimensions: PositiveInt


class DenseNodeScorerConfig(StrictConfig):
    name: Name
    type: Literal["dense"] = "dense"
    preprocessor: Name
    similarity: Literal["cosine", "dot_product", "euclidean"] = "cosine"


class MeanPathScorerConfig(StrictConfig):
    name: Name
    type: Literal["mean"] = "mean"
    input: Name
    top_k: PositiveInt = 20


class WeightedSumPathScorerConfig(StrictConfig):
    name: Name
    type: Literal["weighted_sum"]
    input: Name
    terms: Annotated[tuple[dict[str, float], ...], Field(min_length=1)]
    top_k: PositiveInt = 20

    @field_validator("terms")
    @classmethod
    def validate_terms(
        cls, terms: tuple[dict[str, float], ...]
    ) -> tuple[dict[str, float], ...]:
        names = []
        for term in terms:
            if len(term) != 1:
                raise ValueError("each weighted_sum term must contain one key")
            name = next(iter(term))
            suffix = name.removeprefix("level_")
            is_level = name.startswith("level_") and suffix.isdigit()
            if name not in {"root", "mean", "leaf", "lowest", "highest"} and not (
                is_level and int(suffix) > 0
            ):
                raise ValueError(f"unknown weighted_sum term {name!r}")
            names.append(name)
        if len(names) != len(set(names)):
            raise ValueError("weighted_sum term names must be unique")
        return terms


class CustomPathScorerConfig(StrictConfig):
    name: Name
    type: Literal["custom"]
    input: Name
    binding: Name
    top_k: PositiveInt = 20


PathScorerConfig = Annotated[
    MeanPathScorerConfig | WeightedSumPathScorerConfig | CustomPathScorerConfig,
    Field(discriminator="type"),
]


class TopOneDeciderConfig(StrictConfig):
    name: Name
    type: Literal["top_one"]
    input: Name


class TopKDeciderConfig(StrictConfig):
    name: Name
    type: Literal["top_k"]
    input: Name
    count: PositiveInt


class ThresholdDeciderConfig(StrictConfig):
    name: Name
    type: Literal["threshold"]
    input: Name
    min_score: float
    count: PositiveInt | None = None


class LLMDeciderConfig(StrictConfig):
    name: Name
    type: Literal["llm"]
    input: Name
    binding: Name
    count: PositiveInt = 1


DeciderConfig = Annotated[
    TopOneDeciderConfig | TopKDeciderConfig | ThresholdDeciderConfig | LLMDeciderConfig,
    Field(discriminator="type"),
]


class RouterConfig(StrictConfig):
    name: Name
    deciders: Annotated[tuple[Name, ...], Field(min_length=1)]
    binding: Name
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("deciders")
    @classmethod
    def validate_deciders(cls, deciders: tuple[str, ...]) -> tuple[str, ...]:
        if len(deciders) != len(set(deciders)):
            raise ValueError("router decider names must be unique")
        return deciders


class TreeClassificationConfig(StrictConfig):
    version: Literal[1] = 1
    name: Name
    kind: Literal["tree_classification"] = "tree_classification"
    preprocessors: tuple[EmbeddingPreprocessorConfig, ...]
    node_scorers: tuple[DenseNodeScorerConfig, ...]
    path_scorers: tuple[PathScorerConfig, ...]
    deciders: tuple[DeciderConfig, ...]
    routers: tuple[RouterConfig, ...] = ()
    outputs: dict[Name, Name]
