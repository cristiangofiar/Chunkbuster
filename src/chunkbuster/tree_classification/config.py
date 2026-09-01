"""Configuration for the first tree-classification vertical."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from ..core.config import StrictConfig

Name = Annotated[str, Field(min_length=1)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


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


class MeanPathTermConfig(StrictConfig):
    type: Literal["mean"]
    weight: float


class LeafPathTermConfig(StrictConfig):
    type: Literal["leaf"]
    weight: float


class WeakestPathTermConfig(StrictConfig):
    type: Literal["weakest"]
    weight: float


class LevelPathTermConfig(StrictConfig):
    type: Literal["level"]
    index: NonNegativeInt
    weight: float


PathScoreTermConfig = Annotated[
    MeanPathTermConfig
    | LeafPathTermConfig
    | WeakestPathTermConfig
    | LevelPathTermConfig,
    Field(discriminator="type"),
]


class WeightedSumPathScorerConfig(StrictConfig):
    name: Name
    type: Literal["weighted_sum"]
    input: Name
    terms: Annotated[tuple[PathScoreTermConfig, ...], Field(min_length=1)]
    top_k: PositiveInt = 20


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


class TreeClassificationConfig(StrictConfig):
    version: Literal[1] = 1
    name: Name
    kind: Literal["tree_classification"] = "tree_classification"
    preprocessors: tuple[EmbeddingPreprocessorConfig, ...]
    node_scorers: tuple[DenseNodeScorerConfig, ...]
    path_scorers: tuple[PathScorerConfig, ...]
    deciders: tuple[DeciderConfig, ...]
    outputs: dict[Name, Name]
