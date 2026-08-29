from __future__ import annotations

import json
from pathlib import Path

import pytest

from chunkbuster.core.contracts import ComponentBindings
from chunkbuster.core.models import Query
from chunkbuster.errors import BuildError
from chunkbuster.tree_classification.models import Taxonomy, TaxonomyEdge, TaxonomyNode
from chunkbuster.tree_classification.pipeline import TreeClassificationPipeline


class SpyEmbeddingPreprocessor:
    def __init__(self) -> None:
        self.query_calls = 0
        self.document_calls = 0
        self.document_inputs: list[tuple[str, ...]] = []

    async def prepare_query(self, text: str) -> tuple[float, float]:
        self.query_calls += 1
        return (1.0, 0.0)

    async def prepare_documents(
        self, texts: tuple[str, ...]
    ) -> tuple[tuple[float, float], ...]:
        self.document_calls += 1
        self.document_inputs.append(texts)
        vectors = {
            "Root": (1.0, 0.0),
            "A": (1.0, 0.0),
            "B": (0.0, 1.0),
        }
        return tuple(vectors[text] for text in texts)


def _config() -> dict[str, object]:
    return {
        "version": 1,
        "name": "dense_tree",
        "kind": "tree_classification",
        "preprocessors": [
            {
                "name": "semantic",
                "type": "embedding",
                "binding": "embedding",
                "dimensions": 2,
            }
        ],
        "node_scorers": [
            {
                "name": "dense_nodes",
                "type": "dense",
                "preprocessor": "semantic",
                "similarity": "cosine",
            }
        ],
        "path_scorers": [
            {
                "name": "mean_paths",
                "type": "mean",
                "input": "dense_nodes",
                "top_k": 10,
            }
        ],
        "deciders": [
            {"name": "best", "type": "top_one", "input": "mean_paths"},
            {
                "name": "alternatives",
                "type": "top_k",
                "input": "mean_paths",
                "count": 2,
            },
        ],
        "outputs": {"primary": "best", "alternatives": "alternatives"},
    }


def _taxonomy(*, with_embeddings: bool) -> Taxonomy:
    vectors = {
        "root": (1.0, 0.0),
        "a": (1.0, 0.0),
        "b": (0.0, 1.0),
    }
    return Taxonomy(
        "products",
        (
            TaxonomyNode(
                "root",
                "Root",
                embedding=vectors["root"] if with_embeddings else None,
            ),
            TaxonomyNode(
                "a",
                "A",
                embedding=vectors["a"] if with_embeddings else None,
            ),
            TaxonomyNode(
                "b",
                "B",
                embedding=vectors["b"] if with_embeddings else None,
            ),
        ),
        (TaxonomyEdge("root", "a"), TaxonomyEdge("root", "b")),
    )


def _bindings(preprocessor: SpyEmbeddingPreprocessor) -> ComponentBindings:
    return ComponentBindings(preprocessors={"embedding": preprocessor})


@pytest.mark.asyncio
async def test_dense_mean_pipeline_exposes_top_one_and_top_k() -> None:
    preprocessor = SpyEmbeddingPreprocessor()
    pipeline = await TreeClassificationPipeline.build(
        taxonomy=_taxonomy(with_embeddings=True),
        config=_config(),
        bindings=_bindings(preprocessor),
    )

    result = await pipeline.classify(Query("find A", id="query-1"))

    assert result.query_id == "query-1"
    assert result.taxonomy_id == "products"
    assert tuple(result.outputs) == ("primary", "alternatives")
    assert tuple(
        item.item.node_ids for item in result.outputs["primary"].selected
    ) == (("root", "a"),)
    assert tuple(
        item.item.node_ids for item in result.outputs["alternatives"].selected
    ) == (("root", "a"), ("root", "b"))
    assert preprocessor.query_calls == 1
    assert preprocessor.document_calls == 0


@pytest.mark.asyncio
async def test_missing_embeddings_are_generated_once_during_build() -> None:
    source = _taxonomy(with_embeddings=False)
    preprocessor = SpyEmbeddingPreprocessor()

    pipeline = await TreeClassificationPipeline.build(
        taxonomy=source,
        config=_config(),
        bindings=_bindings(preprocessor),
    )

    assert preprocessor.document_calls == 1
    assert preprocessor.document_inputs == [("Root", "A", "B")]
    assert all(node.embedding is None for node in source.nodes)
    assert all(node.embedding is not None for node in pipeline.taxonomy.nodes)

    await pipeline.classify("first")
    await pipeline.classify("second")

    assert preprocessor.document_calls == 1
    assert preprocessor.query_calls == 2


@pytest.mark.asyncio
async def test_partial_taxonomy_embeddings_fail_build() -> None:
    preprocessor = SpyEmbeddingPreprocessor()
    taxonomy = Taxonomy(
        "partial",
        (
            TaxonomyNode("root", "Root", embedding=(1.0, 0.0)),
            TaxonomyNode("leaf", "A"),
        ),
        (TaxonomyEdge("root", "leaf"),),
    )

    with pytest.raises(BuildError, match="complete or entirely absent"):
        await TreeClassificationPipeline.build(
            taxonomy=taxonomy,
            config=_config(),
            bindings=_bindings(preprocessor),
        )

    assert preprocessor.document_calls == 0


@pytest.mark.asyncio
async def test_threshold_decider_can_abstain() -> None:
    config = _config()
    config["deciders"] = [
        {
            "name": "strict",
            "type": "threshold",
            "input": "mean_paths",
            "min_score": 1.1,
        }
    ]
    config["outputs"] = {"strict": "strict"}
    pipeline = await TreeClassificationPipeline.build(
        taxonomy=_taxonomy(with_embeddings=True),
        config=config,
        bindings=_bindings(SpyEmbeddingPreprocessor()),
    )

    decision = (await pipeline.classify("A")).outputs["strict"]

    assert decision.status == "abstained"
    assert decision.selected == ()


@pytest.mark.asyncio
async def test_dict_yaml_and_json_load_the_same_configuration(tmp_path: Path) -> None:
    config = _config()
    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(
        """\
version: 1
name: dense_tree
kind: tree_classification
preprocessors:
  - name: semantic
    type: embedding
    binding: embedding
    dimensions: 2
node_scorers:
  - name: dense_nodes
    type: dense
    preprocessor: semantic
    similarity: cosine
path_scorers:
  - name: mean_paths
    type: mean
    input: dense_nodes
    top_k: 10
deciders:
  - name: best
    type: top_one
    input: mean_paths
  - name: alternatives
    type: top_k
    input: mean_paths
    count: 2
outputs:
  primary: best
  alternatives: alternatives
""",
        encoding="utf-8",
    )
    json_path = tmp_path / "pipeline.json"
    json_path.write_text(json.dumps(config), encoding="utf-8")
    preprocessor = SpyEmbeddingPreprocessor()
    bindings = _bindings(preprocessor)
    taxonomy = _taxonomy(with_embeddings=True)

    pipelines = [
        await TreeClassificationPipeline.build(
            taxonomy=taxonomy,
            config=source,
            bindings=bindings,
        )
        for source in (config, yaml_path, json_path)
    ]

    assert pipelines[0].config == pipelines[1].config == pipelines[2].config
    results = [await pipeline.classify("A") for pipeline in pipelines]
    assert [
        result.outputs["primary"].selected[0].item.node_ids for result in results
    ] == [("root", "a")] * 3
