"""End-to-end checks for the first retrieval vertical."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest
import yaml

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
        rankings,
        *,
        top_k: int,
    ) -> Ranking[Chunk]:
        return Ranking(tuple(item for item in rankings["dense_a"] if item.id == "b"))


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
            "inputs": "dense_a",
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
        async def retrieve_candidates(self, query, representation, rankings, *, top_k):
            assert representation == query.text.upper()
            candidates = rankings["dense_a"]
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
        async def retrieve_candidates(self, query, representation, rankings, *, top_k):
            assert rankings["dense_a"].ids == ("a", "b")
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


async def test_legacy_preprocessors_start_independently_of_retrievers(task_factory):
    started = asyncio.Event()
    release = asyncio.Event()

    class SlowPreprocessor:
        async def prepare_query(self, query):
            started.set()
            await release.wait()
            return query.text

    class FirstSource:
        async def retrieve(self, query, representation, *, top_k):
            assert started.is_set()
            release.set()
            return _ranking("a", "b")

    config = deepcopy(BASE_CONFIG)
    config["preprocessors"].append({"name": "slow", "binding": "slow"})
    config["retrievers"][1]["preprocessor"] = "slow"
    pipeline = await RetrievalPipeline.build(
        config=config,
        bindings=ComponentBindings(
            preprocessors={
                "shared": CountingPreprocessor(),
                "slow": SlowPreprocessor(),
            },
            retrievers={
                "source_a": FirstSource(),
                "source_b": SourceRetriever(_ranking("b", "c")),
                "candidate": CandidateRetriever(),
            },
        ),
    )

    result = await asyncio.wait_for(pipeline.retrieve("query"), timeout=2)

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
        rankings,
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
                "inputs": "b",
                "top_k": 5,
            },
            {
                "name": "b",
                "binding": "candidate_b",
                "preprocessor": "shared",
                "inputs": "a",
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


class MultiRetriever:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def retrieve_candidates(self, query, representation, rankings, *, top_k):
        self.calls.append((query, representation, rankings, top_k))
        return self.result


class CollectiveRouter:
    async def route(self, query, rankings, *, parameters):
        assert tuple(rankings) == ("hybrid", "fuzzy", "dense")
        assert rankings["dense"].ids == ("a", "b")
        with pytest.raises(TypeError):
            rankings["hybrid"] = Ranking()
        with pytest.raises(TypeError):
            parameters["min_hybrid_score"] = 0
        exact = [hit for hit in rankings["fuzzy"] if hit.item.text == query.text]
        if len(exact) == 1:
            return "fuzzy_exact"
        hybrid = rankings["hybrid"]
        if not exact and hybrid and hybrid[0].score >= parameters["min_hybrid_score"]:
            return "hybrid_top_one"
        return "dm"


class FixedRouter:
    def __init__(self, target):
        self.target = target
        self.calls = 0

    async def route(self, query, rankings, *, parameters):
        self.calls += 1
        return self.target


class ExactRetriever:
    async def retrieve_candidates(self, query, representation, rankings, *, top_k):
        return Ranking(
            tuple(hit for hit in rankings["fuzzy"] if hit.item.text == query.text)
        )


class TopOneRetriever:
    async def retrieve_candidates(self, query, representation, rankings, *, top_k):
        (ranking,) = rankings.values()
        return ranking.top(1)


@pytest.fixture
def routed_config():
    return {
        "name": "collectives",
        "preprocessors": [{"name": "shared", "binding": "shared"}],
        "retrievers": [
            {"name": "dense", "binding": "dense", "preprocessor": "shared"},
            {"name": "fuzzy", "binding": "fuzzy", "preprocessor": "shared"},
            {
                "name": "fuzzy_exact",
                "binding": "exact",
                "preprocessor": "shared",
                "inputs": "fuzzy",
            },
            {
                "name": "hybrid_top_one",
                "binding": "top_one",
                "preprocessor": "shared",
                "inputs": "hybrid",
            },
            {
                "name": "dm",
                "binding": "dm",
                "preprocessor": "shared",
                "inputs": ["hybrid", "fuzzy", "dense"],
                "top_k": 1,
            },
        ],
        "fusions": [
            {
                "name": "hybrid",
                "type": "weighted_sum",
                "inputs": ["dense", "fuzzy"],
                "weights": [0.5, 0.5],
            }
        ],
        "routers": [
            {
                "name": "collective_router",
                "binding": "gate",
                "inputs": ["hybrid", "fuzzy", "dense"],
                "targets": ["fuzzy_exact", "hybrid_top_one", "dm"],
                "parameters": {"min_hybrid_score": 0.80},
            }
        ],
        "outputs": {"primary": "collective_router"},
    }


@pytest.fixture
def routed_bindings():
    return {
        "preprocessors": {"shared": CountingPreprocessor()},
        "retrievers": {
            "dense": SourceRetriever(_ranking("a", "b")),
            "fuzzy": SourceRetriever(_ranking("a", "c")),
            "exact": ExactRetriever(),
            "top_one": TopOneRetriever(),
            "dm": MultiRetriever(_ranking("c", "b")),
        },
        "routers": {"gate": CollectiveRouter()},
    }


@pytest.mark.parametrize(
    ("query_text", "fuzzy_a_score", "fuzzy_c_text", "dm_ids", "expected", "dm_called"),
    [
        ("chunk c", 1.0, "chunk c", ("c",), ("c",), False),
        ("no exact", 0.6, "chunk c", ("c",), ("a",), False),
        ("no exact", 0.5, "chunk c", (), (), True),
        ("chunk a", 1.0, "chunk a", ("c", "b"), ("c",), True),
    ],
)
async def test_collective_router_uses_dm_only_as_fallback(
    routed_config,
    routed_bindings,
    task_factory,
    query_text,
    fuzzy_a_score,
    fuzzy_c_text,
    dm_ids,
    expected,
    dm_called,
):
    fuzzy_a = Chunk("a", "chunk a")
    fuzzy_c = Chunk("c", fuzzy_c_text)
    fuzzy = Ranking(
        (RankedItem("a", fuzzy_a, fuzzy_a_score), RankedItem("c", fuzzy_c, 0.3))
    )
    routed_bindings["retrievers"]["fuzzy"] = SourceRetriever(fuzzy)
    dm_preprocessor = CountingPreprocessor()
    routed_config["preprocessors"].append({"name": "review", "binding": "review"})
    routed_config["retrievers"][-1]["preprocessor"] = "review"
    routed_bindings["preprocessors"]["review"] = dm_preprocessor
    dm = MultiRetriever(_ranking(*dm_ids))
    routed_bindings["retrievers"]["dm"] = dm
    pipeline = await RetrievalPipeline.build(
        config=routed_config, bindings=ComponentBindings(**routed_bindings)
    )
    query = Query(query_text, id="test")

    result = await pipeline.retrieve(query)

    primary = result.outputs["primary"]
    assert primary.terminal_node == "collective_router"
    assert primary.ranking.ids == expected
    assert primary.status == ("completed" if expected else "empty")
    assert len(dm.calls) == int(dm_called)
    assert dm_preprocessor.calls == int(dm_called)
    if dm_called:
        actual_query, representation, rankings, top_k = dm.calls[0]
        assert actual_query is query
        assert representation == query_text.casefold()
        assert top_k == 1
        assert tuple(rankings) == ("hybrid", "fuzzy", "dense")
        assert rankings["fuzzy"].items == fuzzy.items
        assert rankings["hybrid"][0].score == pytest.approx((1 + fuzzy_a_score) / 2)
        assert rankings["dense"].ids == ("a", "b")
        with pytest.raises(TypeError):
            rankings["fuzzy"] = Ranking()


async def test_router_composes_with_ranking_nodes_and_reuses_the_selected_target(
    routed_config,
    routed_bindings,
    task_factory,
):
    retrievers = routed_config["retrievers"]
    retrievers.append({**retrievers[3], "name": "after", "inputs": "collective_router"})
    retrievers.append(
        {**retrievers[4], "name": "cascade", "inputs": ["collective_router"]}
    )
    routed_config["fusions"].append(
        {"name": "final", "inputs": ["after", "cascade", "collective_router"]}
    )
    routed_config["routers"].append(
        {
            "name": "outer",
            "binding": "outer",
            "inputs": ["collective_router"],
            "targets": ["collective_router", "final"],
        }
    )
    routed_config["outputs"].update(composed="outer", direct="dm")
    gate = FixedRouter("dm")
    outer = FixedRouter("final")
    dm = MultiRetriever(_ranking("c"))
    cascade = MultiRetriever(_ranking("c"))
    retrievers[-1]["binding"] = "cascade"
    routed_bindings["routers"].update(gate=gate, outer=outer)
    routed_bindings["retrievers"].update(dm=dm, cascade=cascade)
    pipeline = await RetrievalPipeline.build(
        config=routed_config, bindings=ComponentBindings(**routed_bindings)
    )

    result = await asyncio.wait_for(pipeline.retrieve("query"), timeout=2)
    assert result.outputs["composed"].ranking.ids == ("c",)
    assert result.outputs["primary"].ranking is result.outputs["direct"].ranking
    assert cascade.calls[0][2]["collective_router"] is result.outputs["direct"].ranking
    assert len(dm.calls) == gate.calls == outer.calls == 1
    assert routed_bindings["preprocessors"]["shared"].calls == 1

    direct = await pipeline.retrieve("query", outputs=["direct"])
    assert tuple(direct.outputs) == ("direct",)
    assert len(dm.calls) == 2
    assert gate.calls == outer.calls == 1
    assert (await pipeline.retrieve("query", outputs=[])).outputs == {}
    assert routed_bindings["preprocessors"]["shared"].calls == 2

    outer.target = "collective_router"
    nested = await pipeline.retrieve("query", outputs=["composed"])
    assert nested.outputs["composed"].ranking.ids == ("c",)
    assert len(dm.calls) == 3
    assert gate.calls == outer.calls == 2
    assert len(cascade.calls) == 1


async def test_routed_retrieval_cancels_selected_branch_on_other_output_failure(
    routed_config, routed_bindings, task_factory
):
    started = asyncio.Event()
    stopped = asyncio.Event()

    class BlockedDM:
        async def retrieve_candidates(self, query, representation, rankings, *, top_k):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

    class FailingExactRetriever:
        async def retrieve_candidates(self, query, representation, rankings, *, top_k):
            await started.wait()
            raise ValueError("output branch failed")

    routed_config["outputs"]["other"] = "fuzzy_exact"
    routed_bindings["routers"]["gate"] = FixedRouter("dm")
    routed_bindings["retrievers"].update(dm=BlockedDM(), exact=FailingExactRetriever())
    pipeline = await RetrievalPipeline.build(
        config=routed_config, bindings=ComponentBindings(**routed_bindings)
    )

    with pytest.raises(ValueError, match="output branch failed"):
        await asyncio.wait_for(pipeline.retrieve("query"), timeout=2)
    assert stopped.is_set()


async def test_router_input_failure_aborts_before_routing(
    routed_config, routed_bindings, task_factory
):
    started = asyncio.Event()
    stopped = asyncio.Event()

    class BlockedFuzzy:
        async def retrieve(self, query, representation, *, top_k):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

    class FailingDense:
        async def retrieve(self, query, representation, *, top_k):
            await started.wait()
            raise ValueError("router input failed")

    gate = FixedRouter("dm")
    routed_bindings["routers"]["gate"] = gate
    routed_bindings["retrievers"].update(dense=FailingDense(), fuzzy=BlockedFuzzy())
    pipeline = await RetrievalPipeline.build(
        config=routed_config, bindings=ComponentBindings(**routed_bindings)
    )

    with pytest.raises(ValueError, match="router input failed"):
        await asyncio.wait_for(pipeline.retrieve("query"), timeout=2)
    assert stopped.is_set()
    assert gate.calls == 0
    assert routed_bindings["retrievers"]["dm"].calls == []


@pytest.mark.parametrize("cancel_caller", [False, True])
async def test_aborted_query_cannot_start_a_router_fallback(
    routed_config, routed_bindings, task_factory, cancel_caller
):
    started = asyncio.Event()

    class RecoveringRouter:
        async def route(self, query, rankings, *, parameters):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return "dm"

    class FailingExactRetriever:
        async def retrieve_candidates(self, query, representation, rankings, *, top_k):
            await started.wait()
            raise ValueError("output branch failed")

    if not cancel_caller:
        routed_config["outputs"]["other"] = "fuzzy_exact"
        routed_bindings["retrievers"]["exact"] = FailingExactRetriever()
    routed_bindings["routers"]["gate"] = RecoveringRouter()
    pipeline = await RetrievalPipeline.build(
        config=routed_config, bindings=ComponentBindings(**routed_bindings)
    )

    caller = asyncio.create_task(pipeline.retrieve("query"))
    await asyncio.wait_for(started.wait(), timeout=2)
    if cancel_caller:
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(caller, timeout=2)
    else:
        with pytest.raises(ValueError, match="output branch failed"):
            await asyncio.wait_for(caller, timeout=2)
    assert routed_bindings["retrievers"]["dm"].calls == []


async def test_router_gets_independent_nested_parameters_per_query(
    routed_config, routed_bindings
):
    class Router:
        async def route(self, query, rankings, *, parameters):
            options = dict(parameters)
            return options["routing"]["targets"].pop()

    parameters = {"routing": {"targets": ["hybrid_top_one"]}}
    routed_config["routers"][0]["parameters"] = parameters
    routed_bindings["routers"]["gate"] = Router()
    pipeline = await RetrievalPipeline.build(
        config=routed_config, bindings=ComponentBindings(**routed_bindings)
    )

    results = await asyncio.gather(
        pipeline.retrieve("first"), pipeline.retrieve("second")
    )

    assert [result.outputs["primary"].ranking.ids for result in results] == [
        ("a",),
        ("a",),
    ]
    assert pipeline.config.routers[0].parameters == parameters


async def test_router_can_select_a_long_cascade(
    routed_config, routed_bindings, task_factory
):
    upstream = "dense"
    for index in range(300):
        name = f"stage_{index}"
        routed_config["retrievers"].append(
            {
                "name": name,
                "binding": "top_one",
                "preprocessor": "shared",
                "inputs": upstream,
            }
        )
        upstream = name
    routed_config["routers"][0]["targets"].append(upstream)
    routed_bindings["routers"]["gate"] = FixedRouter(upstream)
    pipeline = await RetrievalPipeline.build(
        config=routed_config, bindings=ComponentBindings(**routed_bindings)
    )

    result = await asyncio.wait_for(pipeline.retrieve("query"), timeout=2)

    assert result.outputs["primary"].ranking.ids == ("a",)


async def test_multi_input_canonicalizes_in_declared_order(
    routed_config, routed_bindings
):
    dense_a = Chunk("a", "dense version")
    fuzzy_a = Chunk("a", "fuzzy version")
    routed_bindings["retrievers"].update(
        dense=SourceRetriever(Ranking((RankedItem("a", dense_a, 0.9),))),
        fuzzy=SourceRetriever(Ranking((RankedItem("a", fuzzy_a, 0.6),))),
        dm=MultiRetriever(_ranking("a")),
    )
    routed_config["retrievers"][-1]["inputs"] = ["fuzzy", "hybrid", "dense"]
    routed_bindings["routers"]["gate"] = FixedRouter("dm")
    pipeline = await RetrievalPipeline.build(
        config=routed_config, bindings=ComponentBindings(**routed_bindings)
    )

    ranking = (await pipeline.retrieve("query")).outputs["primary"].ranking

    assert ranking[0].item is fuzzy_a
    assert ranking[0].score == 1.0
    assert ranking.origins == {"dense", "fuzzy"}


@pytest.mark.parametrize("ranking", [_ranking("a", "invented"), ["a"]])
async def test_multi_input_rejects_invalid_ranking(
    routed_config, routed_bindings, ranking
):
    routed_bindings["routers"]["gate"] = FixedRouter("dm")
    routed_bindings["retrievers"]["dm"] = MultiRetriever(ranking)
    pipeline = await RetrievalPipeline.build(
        config=routed_config, bindings=ComponentBindings(**routed_bindings)
    )

    with pytest.raises(
        InvalidModelOutputError, match="outside its input|return Ranking"
    ):
        await pipeline.retrieve("query")


@pytest.mark.parametrize("target", ["hybrid", "unknown", "", None])
async def test_router_rejects_invalid_selection(routed_config, routed_bindings, target):
    routed_bindings["routers"]["gate"] = FixedRouter(target)
    pipeline = await RetrievalPipeline.build(
        config=routed_config, bindings=ComponentBindings(**routed_bindings)
    )

    with pytest.raises(InvalidModelOutputError, match="target"):
        await pipeline.retrieve("query")


@pytest.mark.parametrize(
    ("group", "index", "field", "value", "error", "message"),
    [
        ("retrievers", -1, "input", "dense", ConfigurationError, "Extra inputs"),
        ("retrievers", -1, "inputs", [], ConfigurationError, "at least 1"),
        ("retrievers", -1, "inputs", "", ConfigurationError, "at least 1"),
        ("retrievers", -1, "inputs", [""], ConfigurationError, "at least 1"),
        (
            "retrievers",
            -1,
            "inputs",
            ["dense", "dense"],
            BuildError,
            "repeats an input",
        ),
        ("retrievers", -1, "inputs", ["missing"], BuildError, "unknown nodes"),
        ("retrievers", -1, "inputs", ["dm"], BuildError, "cycle"),
        (
            "retrievers",
            -1,
            "binding",
            "dense",
            BuildError,
            "async retrieve_candidates",
        ),
        ("routers", 0, "name", "hybrid", BuildError, "names must be unique"),
        ("routers", 0, "inputs", ["dense", "dense"], ConfigurationError, "unique"),
        ("routers", 0, "inputs", ["missing"], BuildError, "unknown nodes"),
        ("routers", 0, "targets", ["dm", "dm"], ConfigurationError, "unique"),
        ("routers", 0, "targets", ["shared"], BuildError, "unknown nodes"),
        ("routers", 0, "targets", ["missing"], BuildError, "unknown nodes"),
        ("routers", 0, "targets", ["dm"], BuildError, "unused nodes"),
        ("routers", 0, "binding", "missing", BuildError, "missing router binding"),
        ("retrievers", -1, "inputs", ["collective_router"], BuildError, "cycle"),
    ],
)
async def test_routed_graph_build_validation(
    routed_config,
    routed_bindings,
    group,
    index,
    field,
    value,
    error,
    message,
):
    routed_config[group][index][field] = value
    with pytest.raises(error, match=message):
        await RetrievalPipeline.build(
            config=routed_config, bindings=ComponentBindings(**routed_bindings)
        )


@pytest.mark.parametrize("inputs", ["dense_a", ["dense_a"]])
async def test_single_input_yaml_evaluates_full_graph_for_requested_outputs(
    tmp_path, inputs
):
    config = deepcopy(BASE_CONFIG)
    config["retrievers"][-1]["inputs"] = inputs
    path = tmp_path / "single-input.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    source_b_calls = []

    class ObservedSource(SourceRetriever):
        async def retrieve(self, query, representation, *, top_k):
            source_b_calls.append(query)
            return await super().retrieve(query, representation, top_k=top_k)

    pipeline = await RetrievalPipeline.build(
        config=path,
        bindings=ComponentBindings(
            preprocessors={"shared": CountingPreprocessor()},
            retrievers={
                "source_a": SourceRetriever(_ranking("a", "b")),
                "source_b": ObservedSource(_ranking("b", "c")),
                "candidate": CandidateRetriever(),
            },
        ),
    )

    result = await pipeline.retrieve("query", outputs=["cascade"])

    assert pipeline.config.retrievers[-1].inputs == ("dense_a",)
    assert tuple(result.outputs) == ("cascade",)
    assert result.outputs["cascade"].ranking.ids == ("b",)
    assert len(source_b_calls) == 1
