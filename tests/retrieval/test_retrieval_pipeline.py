"""End-to-end checks for the first retrieval vertical."""

from __future__ import annotations

import pytest

from chunkbuster.core import ComponentBindings, RankedItem, Ranking
from chunkbuster.errors import BuildError, InvalidModelOutputError
from chunkbuster.retrieval import Chunk, RetrievalPipeline


def _ranking(*ids: str) -> Ranking[Chunk]:
    return Ranking(
        tuple(
            RankedItem(item_id, Chunk(item_id, f"chunk {item_id}"), 1.0 / index)
            for index, item_id in enumerate(ids, 1)
        )
    )


class CountingPreprocessor:
    def __init__(self) -> None:
        self.calls = 0

    async def prepare_query(self, text: str) -> str:
        self.calls += 1
        return text.casefold()


class SourceRetriever:
    def __init__(self, ranking: Ranking[Chunk]) -> None:
        self.ranking = ranking

    async def retrieve(self, query: str, *, top_k: int) -> Ranking[Chunk]:
        return self.ranking


class CandidateRetriever:
    async def retrieve_candidates(
        self,
        query: str,
        candidates: Ranking[Chunk],
        *,
        top_k: int,
    ) -> Ranking[Chunk]:
        return Ranking(tuple(item for item in candidates if item.id == "b"))


BASE_CONFIG = {
    "version": 1,
    "name": "products",
    "kind": "retrieve",
    "preprocessors": [{"name": "shared", "binding": "shared"}],
    "retrievers": [
        {
            "name": "dense_a",
            "binding": "source_a",
            "preprocessor": "shared",
            "top_k": 10,
        },
        {
            "name": "dense_b",
            "binding": "source_b",
            "preprocessor": "shared",
            "top_k": 10,
        },
        {
            "name": "filtered",
            "binding": "candidate",
            "preprocessor": "shared",
            "input": "dense_a",
            "top_k": 5,
        },
    ],
    "fusions": [
        {
            "name": "combined",
            "type": "rrf",
            "inputs": ["filtered", "dense_b"],
            "top_k": 3,
        }
    ],
    "outputs": {"primary": "combined", "cascade": "filtered"},
}


@pytest.mark.asyncio
async def test_source_candidate_rrf_outputs_and_shared_preprocessing() -> None:
    preprocessor = CountingPreprocessor()
    pipeline = await RetrievalPipeline.build(
        config=BASE_CONFIG,
        bindings=ComponentBindings(
            preprocessors={"shared": preprocessor},
            retrievers={
                "source_a": SourceRetriever(_ranking("a", "b")),
                "source_b": SourceRetriever(_ranking("b", "c")),
                "candidate": CandidateRetriever(),
            },
        ),
    )

    result = await pipeline.retrieve("Find products")

    assert preprocessor.calls == 1
    assert tuple(result.outputs) == ("primary", "cascade")
    assert result.outputs["primary"].ranking.ids == ("b", "c")
    assert result.outputs["cascade"].ranking.ids == ("b",)
    assert result.outputs["primary"].status == "completed"


class InventingCandidateRetriever:
    async def retrieve_candidates(
        self,
        query: str,
        candidates: Ranking[Chunk],
        *,
        top_k: int,
    ) -> Ranking[Chunk]:
        return _ranking("invented")


@pytest.mark.asyncio
async def test_candidate_retriever_cannot_invent_ids() -> None:
    config = {
        **BASE_CONFIG,
        "retrievers": [BASE_CONFIG["retrievers"][0], BASE_CONFIG["retrievers"][2]],
        "fusions": [],
        "outputs": {"result": "filtered"},
    }
    pipeline = await RetrievalPipeline.build(
        config=config,
        bindings=ComponentBindings(
            preprocessors={"shared": CountingPreprocessor()},
            retrievers={
                "source_a": SourceRetriever(_ranking("a")),
                "candidate": InventingCandidateRetriever(),
            },
        ),
    )

    with pytest.raises(InvalidModelOutputError, match="outside its input"):
        await pipeline.retrieve("Find products")


@pytest.mark.asyncio
async def test_cycle_is_rejected_during_build() -> None:
    config = {
        "version": 1,
        "name": "cycle",
        "kind": "retrieve",
        "preprocessors": [{"name": "shared", "binding": "shared"}],
        "retrievers": [
            {
                "name": "a",
                "binding": "candidate_a",
                "preprocessor": "shared",
                "input": "b",
                "top_k": 5,
            },
            {
                "name": "b",
                "binding": "candidate_b",
                "preprocessor": "shared",
                "input": "a",
                "top_k": 5,
            },
        ],
        "outputs": {"result": "a"},
    }

    with pytest.raises(BuildError, match="cycle"):
        await RetrievalPipeline.build(
            config=config,
            bindings=ComponentBindings(
                preprocessors={"shared": CountingPreprocessor()},
                retrievers={
                    "candidate_a": CandidateRetriever(),
                    "candidate_b": CandidateRetriever(),
                },
            ),
        )
