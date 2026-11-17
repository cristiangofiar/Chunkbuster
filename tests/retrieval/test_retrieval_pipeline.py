"""End-to-end checks for the first retrieval vertical."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from chunkbuster.core import ComponentBindings, Query, RankedItem, Ranking
from chunkbuster.errors import BuildError, ConfigurationError, InvalidModelOutputError
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

    async def prepare_query(self, query: Query) -> str:
        self.calls += 1
        return query.text.casefold()


class SourceRetriever:
    def __init__(self, ranking: Ranking[Chunk]) -> None:
        self.ranking = ranking

    async def retrieve(
        self, query: Query, representation, *, top_k: int
    ) -> Ranking[Chunk]:
        return self.ranking


class CandidateRetriever:
    async def retrieve_candidates(
        self,
        query: Query,
        representation,
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


async def test_query_context_reaches_all_bindings_without_crossing_requests(
    task_factory,
):
    queries = {
        "first": Query("first", id="1", metadata={"market": "EC"}),
        "second": Query("second", id="2", metadata=SimpleNamespace(market="PE")),
    }
    seen = {}

    def market_for(query, stage):
        assert isinstance(query, Query)
        if query.text == "plain":
            queries.setdefault("plain", query)
        assert query is queries[query.text]
        seen.setdefault((query.text, stage), []).append(query)
        metadata = query.metadata
        return (
            metadata.get("market", "default")
            if isinstance(metadata, dict)
            else metadata.market
        )

    class Preprocessor:
        async def prepare_query(self, query):
            market_for(query, "prepare")
            await asyncio.sleep(0)
            return query.text.upper()

    class Source:
        def __init__(self, stage):
            self.stage = stage

        async def retrieve(self, query, representation, *, top_k):
            assert representation == query.text.upper()
            await asyncio.sleep(0)
            return _ranking(market_for(query, self.stage))

    class Candidate:
        async def retrieve_candidates(
            self, query, representation, candidates, *, top_k
        ):
            assert representation == query.text.upper()
            assert candidates.ids == (market_for(query, "candidate"),)
            return candidates

    pipeline = await RetrievalPipeline.build(
        config=BASE_CONFIG,
        bindings=ComponentBindings(
            preprocessors={"shared": Preprocessor()},
            retrievers={
                "source_a": Source("a"),
                "source_b": Source("b"),
                "candidate": Candidate(),
            },
        ),
    )
    results = await asyncio.wait_for(
        asyncio.gather(
            pipeline.retrieve(queries["first"]),
            pipeline.retrieve(queries["second"]),
            pipeline.retrieve("plain"),
        ),
        timeout=2,
    )

    assert [result.query_id for result in results] == ["1", "2", None]
    assert [result.outputs["primary"].ranking.ids for result in results] == [
        ("EC",),
        ("PE",),
        ("default",),
    ]
    assert len(seen) == 12
    assert all(len(calls) == 1 for calls in seen.values())


async def test_independent_retrievers_overlap_and_cascade_does_not_wait_for_sibling(
    task_factory,
):
    sibling_started = asyncio.Event()
    cascade_finished = asyncio.Event()
    preprocessor = CountingPreprocessor()

    class FirstSource:
        async def retrieve(self, query, representation, *, top_k):
            await sibling_started.wait()
            return _ranking("a", "b")

    class SlowSource:
        async def retrieve(self, query, representation, *, top_k):
            sibling_started.set()
            await cascade_finished.wait()
            return _ranking("b", "c")

    class Cascade:
        async def retrieve_candidates(
            self, query, representation, candidates, *, top_k
        ):
            assert candidates.ids == ("a", "b")
            cascade_finished.set()
            return _ranking("b")

    pipeline = await RetrievalPipeline.build(
        config=deepcopy(BASE_CONFIG),
        bindings=ComponentBindings(
            preprocessors={"shared": preprocessor},
            retrievers={
                "source_a": FirstSource(),
                "source_b": SlowSource(),
                "candidate": Cascade(),
            },
        ),
    )

    result = await asyncio.wait_for(pipeline.retrieve("query"), timeout=2)

    assert preprocessor.calls == 1
    assert result.outputs["primary"].ranking.ids == ("b", "c")


@pytest.mark.parametrize("cancel_caller", [False, True])
async def test_retrieval_cancels_other_branches_on_failure_or_caller_cancellation(
    cancel_caller,
    task_factory,
):
    started = asyncio.Event()
    stopped = asyncio.Event()

    class BlockedSource:
        async def retrieve(self, query, representation, *, top_k):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

    class FailingSource:
        async def retrieve(self, query, representation, *, top_k):
            await started.wait()
            raise ValueError("retriever failed")

    retrievers = {"source_a": BlockedSource(), "candidate": CandidateRetriever()}
    retrievers["source_b"] = (
        SourceRetriever(_ranking("b")) if cancel_caller else FailingSource()
    )
    pipeline = await RetrievalPipeline.build(
        config=deepcopy(BASE_CONFIG),
        bindings=ComponentBindings(
            preprocessors={"shared": CountingPreprocessor()}, retrievers=retrievers
        ),
    )
    caller = asyncio.create_task(pipeline.retrieve("query"))
    await asyncio.wait_for(started.wait(), timeout=2)
    if cancel_caller:
        caller.cancel()
    error = asyncio.CancelledError if cancel_caller else ValueError
    with pytest.raises(error):
        await asyncio.wait_for(caller, timeout=2)
    assert stopped.is_set()


@pytest.mark.parametrize(
    ("group", "binding", "method"),
    [
        ("preprocessors", "shared", "prepare_query"),
        ("retrievers", "source_a", "retrieve"),
        ("retrievers", "candidate", "retrieve_candidates"),
    ],
)
async def test_build_requires_async_binding_methods(group, binding, method):
    class InvalidBinding:
        pass

    component = InvalidBinding()
    setattr(component, method, lambda *args, **kwargs: pytest.fail("binding executed"))
    bindings = {
        "preprocessors": {"shared": CountingPreprocessor()},
        "retrievers": {
            "source_a": SourceRetriever(_ranking("a", "b")),
            "source_b": SourceRetriever(_ranking("b", "c")),
            "candidate": CandidateRetriever(),
        },
    }
    bindings[group][binding] = component

    with pytest.raises(BuildError, match=f"async {method}"):
        await RetrievalPipeline.build(
            config=BASE_CONFIG, bindings=ComponentBindings(**bindings)
        )


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "extra"),
    (
        ("weighted_rrf", {"weights": [1, 2], "k": 10}),
        ("weighted_sum", {"weights": [1, 2], "normalization": "min_max"}),
        ("comb_mnz", {"normalization": "min_max"}),
    ),
)
async def test_shared_fusion_methods_are_available_in_retrieval(
    method: str, extra: dict[str, object]
) -> None:
    config = {
        "version": 1,
        "name": "fusion",
        "kind": "retrieve",
        "preprocessors": [{"name": "shared", "binding": "shared"}],
        "retrievers": BASE_CONFIG["retrievers"][:2],
        "fusions": [
            {
                "name": "combined",
                "type": method,
                "inputs": ["dense_a", "dense_b"],
                "top_k": 3,
                **extra,
            }
        ],
        "outputs": {"primary": "combined"},
    }
    pipeline = await RetrievalPipeline.build(
        config=config,
        bindings=ComponentBindings(
            preprocessors={"shared": CountingPreprocessor()},
            retrievers={
                "source_a": SourceRetriever(_ranking("a", "b")),
                "source_b": SourceRetriever(_ranking("b", "c")),
            },
        ),
    )

    result = await pipeline.retrieve("query")

    assert result.outputs["primary"].ranking.ids[0] == "b"


@pytest.mark.asyncio
async def test_rank_fusion_rejects_score_normalization() -> None:
    config = {
        **BASE_CONFIG,
        "fusions": [
            {
                **BASE_CONFIG["fusions"][0],
                "type": "weighted_rrf",
                "normalization": "min_max",
            }
        ],
    }

    with pytest.raises(ConfigurationError, match="does not accept"):
        await RetrievalPipeline.build(config=config)


class InventingCandidateRetriever:
    async def retrieve_candidates(
        self,
        query: Query,
        representation,
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
