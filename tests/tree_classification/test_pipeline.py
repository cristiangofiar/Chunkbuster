from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from chunkbuster import TextTokenizer
from chunkbuster.core.contracts import ComponentBindings
from chunkbuster.core.models import Query
from chunkbuster.errors import BuildError, ConfigurationError, InvalidModelOutputError
from chunkbuster.tree_classification.models import (
    DecisionRoute,
    DecisionSelection,
    Taxonomy,
    TaxonomyEdge,
    TaxonomyNode,
)
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


async def test_tree_branch_can_decide_while_independent_preprocessor_is_blocked(
    task_factory,
):
    lexical_started = asyncio.Event()
    semantic_decided = asyncio.Event()

    class Embedding:
        async def prepare_query(self, text):
            await lexical_started.wait()
            return (1.0, 0.0)

    class Tokenizer:
        async def prepare_documents(self, texts):
            return tuple(("a",) for _ in texts)

        async def prepare_query(self, text):
            lexical_started.set()
            await semantic_decided.wait()
            return ("a",)

    class SemanticDecider:
        async def decide(self, query, candidates, *, count):
            semantic_decided.set()
            return DecisionSelection((candidates.items[0].id,))

    config = _config()
    config["preprocessors"].append(
        {"name": "lexical", "type": "tokenizer", "binding": "tokenizer"}
    )
    config["node_scorers"].append(
        {"name": "bm25_nodes", "type": "bm25", "preprocessor": "lexical"}
    )
    config["path_scorers"].append(
        {"name": "lexical_paths", "type": "mean", "input": "bm25_nodes"}
    )
    config["deciders"] = [
        {"name": "semantic", "type": "llm", "input": "mean_paths", "binding": "llm"},
        {"name": "lexical", "type": "top_one", "input": "lexical_paths"},
    ]
    config["outputs"] = {"primary": "semantic", "secondary": "lexical"}
    pipeline = await TreeClassificationPipeline.build(
        taxonomy=_taxonomy(with_embeddings=True),
        config=config,
        bindings=ComponentBindings(
            preprocessors={"embedding": Embedding(), "tokenizer": Tokenizer()},
            deciders={"llm": SemanticDecider()},
        ),
    )

    result = await asyncio.wait_for(pipeline.classify("query"), timeout=2)

    assert list(result.outputs) == ["primary", "secondary"]
    assert all(output.status == "selected" for output in result.outputs.values())


async def test_custom_path_calls_overlap_and_fusion_waits_for_all_branches(
    task_factory,
):
    started = set()
    all_started = asyncio.Event()
    finished = set()

    class Scorer:
        def __init__(self, name):
            self.name = name

        async def score_path(self, path, scores):
            key = (self.name, path.id)
            started.add(key)
            if len(started) == 4:
                all_started.set()
            await all_started.wait()
            finished.add(key)
            return 1.0

    class Decider:
        async def decide(self, query, candidates, *, count):
            assert len(finished) == 4
            return DecisionSelection((candidates.items[0].id,))

    config = _config()
    config["path_scorers"] = [
        {"name": name, "type": "custom", "binding": name, "input": "dense_nodes"}
        for name in ("first", "second")
    ]
    config["path_fusions"] = [
        {"name": "combined", "type": "weighted_rrf", "inputs": ["first", "second"]}
    ]
    config["deciders"] = [
        {"name": "best", "type": "llm", "binding": "llm", "input": "combined"}
    ]
    config["outputs"] = {"primary": "best"}
    preprocessor = SpyEmbeddingPreprocessor()
    pipeline = await TreeClassificationPipeline.build(
        taxonomy=_taxonomy(with_embeddings=True),
        config=config,
        bindings=ComponentBindings(
            preprocessors={"embedding": preprocessor},
            path_scorers={name: Scorer(name) for name in ("first", "second")},
            deciders={"llm": Decider()},
        ),
    )

    result = await asyncio.wait_for(pipeline.classify("query"), timeout=2)

    assert result.outputs["primary"].status == "selected"
    assert preprocessor.query_calls == 1


async def test_routers_and_deciders_overlap_and_shared_results_are_reused(task_factory):
    routes_started = set()
    decisions_started = set()
    all_routes = asyncio.Event()
    all_decisions = asyncio.Event()

    class Router:
        def __init__(self, name):
            self.name = name
            self.calls = 0

        async def route(self, query, candidates, *, parameters):
            self.calls += 1
            routes_started.add(self.name)
            if len(routes_started) == 2:
                all_routes.set()
            await all_routes.wait()
            return self.name

    class Decider:
        def __init__(self, name):
            self.name = name
            self.calls = 0

        async def decide(self, query, candidates, *, count):
            self.calls += 1
            decisions_started.add(self.name)
            if len(decisions_started) == 2:
                all_decisions.set()
            await all_decisions.wait()
            return DecisionSelection((candidates.items[0].id,))

    config = _config()
    config["deciders"] = [
        {"name": name, "type": "llm", "binding": name, "input": "mean_paths"}
        for name in ("first", "second")
    ]
    config["routers"] = [
        {"name": name + "_router", "binding": name, "deciders": [name]}
        for name in ("first", "second")
    ]
    config["outputs"] = {
        "first": "first_router",
        "second": "second_router",
        "duplicate": "first_router",
        "direct": "first",
    }
    routers = {name: Router(name) for name in ("first", "second")}
    deciders = {name: Decider(name) for name in ("first", "second")}
    pipeline = await TreeClassificationPipeline.build(
        taxonomy=_taxonomy(with_embeddings=True),
        config=config,
        bindings=ComponentBindings(
            preprocessors={"embedding": SpyEmbeddingPreprocessor()},
            routers=routers,
            deciders=deciders,
        ),
    )

    result = await asyncio.wait_for(pipeline.classify("query"), timeout=2)

    assert all(
        binding.calls == 1 for binding in (*routers.values(), *deciders.values())
    )
    assert result.outputs["first"] is result.outputs["duplicate"]
    assert result.outputs["first"] is result.outputs["direct"]


@pytest.mark.parametrize("cancel_caller", [False, True])
async def test_tree_drains_custom_path_tasks_on_failure_or_cancellation(
    cancel_caller,
    task_factory,
):
    started = asyncio.Event()
    stopped = asyncio.Event()

    class Scorer:
        async def score_path(self, path, scores):
            if path.node_ids[-1] == "a":
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped.set()
            await started.wait()
            if not cancel_caller:
                raise ValueError("scorer failed")
            return 1.0

    config = _config()
    config["path_scorers"][0].update(type="custom", binding="custom")
    pipeline = await TreeClassificationPipeline.build(
        taxonomy=_taxonomy(with_embeddings=True),
        config=config,
        bindings=ComponentBindings(
            preprocessors={"embedding": SpyEmbeddingPreprocessor()},
            path_scorers={"custom": Scorer()},
        ),
    )
    caller = asyncio.create_task(pipeline.classify("query"))
    await asyncio.wait_for(started.wait(), timeout=2)
    if cancel_caller:
        caller.cancel()
    error = asyncio.CancelledError if cancel_caller else ValueError
    with pytest.raises(error):
        await asyncio.wait_for(caller, timeout=2)
    assert stopped.is_set()


@pytest.mark.parametrize(
    ("group", "method"),
    [
        ("preprocessors", "prepare_query"),
        ("preprocessors", "prepare_documents"),
        ("path_scorers", "score_path"),
        ("deciders", "decide"),
        ("routers", "route"),
    ],
)
async def test_build_requires_async_binding_methods(group, method):
    class InvalidBinding(SpyEmbeddingPreprocessor):
        pass

    component = InvalidBinding()
    setattr(component, method, lambda *args, **kwargs: pytest.fail("binding executed"))
    config = _config()
    bindings = {"preprocessors": {"embedding": SpyEmbeddingPreprocessor()}}
    if group == "preprocessors":
        bindings[group]["embedding"] = component
    elif group == "path_scorers":
        config["path_scorers"][0].update(type="custom", binding="invalid")
        bindings[group] = {"invalid": component}
    elif group == "deciders":
        config["deciders"][0].update(type="llm", binding="invalid")
        bindings[group] = {"invalid": component}
    else:
        config["routers"] = [
            {"name": "gate", "binding": "invalid", "deciders": ["best"]}
        ]
        config["outputs"]["primary"] = "gate"
        bindings[group] = {"invalid": component}

    with pytest.raises(BuildError, match=f"async {method}"):
        await TreeClassificationPipeline.build(
            taxonomy=_taxonomy(with_embeddings=method != "prepare_documents"),
            config=config,
            bindings=ComponentBindings(**bindings),
        )


async def test_output_subset_skips_unneeded_branch_and_router_skips_unchosen_decider():
    class UnusedScorer:
        async def score_path(self, path, scores):
            pytest.fail("unrequested scoring branch executed")

    class UnchosenDecider:
        async def decide(self, query, candidates, *, count):
            pytest.fail("unchosen decider executed")

    config = _config()
    config["path_scorers"].append(
        {
            "name": "other_paths",
            "type": "custom",
            "binding": "other",
            "input": "dense_nodes",
        }
    )
    config["deciders"].extend(
        [
            {"name": "other", "type": "top_one", "input": "other_paths"},
            {
                "name": "unchosen",
                "type": "llm",
                "binding": "llm",
                "input": "mean_paths",
            },
        ]
    )
    config["routers"] = [
        {"name": "gate", "binding": "gate", "deciders": ["best", "unchosen"]}
    ]
    config["outputs"].update(primary="gate", other="other")
    pipeline = await TreeClassificationPipeline.build(
        taxonomy=_taxonomy(with_embeddings=True),
        config=config,
        bindings=ComponentBindings(
            preprocessors={"embedding": SpyEmbeddingPreprocessor()},
            path_scorers={"other": UnusedScorer()},
            routers={"gate": FakeRouter("best")},
            deciders={"llm": UnchosenDecider()},
        ),
    )

    result = await asyncio.wait_for(
        pipeline.classify("query", outputs=["primary"]), timeout=2
    )

    assert tuple(result.outputs) == ("primary",)
    assert result.outputs["primary"].name == "best"


def _branching_taxonomy() -> Taxonomy:
    return Taxonomy(
        "weighted",
        (
            TaxonomyNode("root", "Root", embedding=(1.0, 0.0)),
            TaxonomyNode("a", "A", embedding=(0.8, 0.0)),
            TaxonomyNode("b", "B", embedding=(0.2, 0.0)),
            TaxonomyNode("c", "C", embedding=(0.0, 1.0)),
            TaxonomyNode("d", "D", embedding=(1.0, 0.0)),
        ),
        (
            TaxonomyEdge("root", "a"),
            TaxonomyEdge("root", "b"),
            TaxonomyEdge("a", "c"),
            TaxonomyEdge("b", "d"),
        ),
    )


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
    assert tuple(item.item.node_ids for item in result.outputs["primary"].selected) == (
        ("root", "a"),
    )
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
async def test_weighted_sum_combines_path_score_features() -> None:
    config = _config()
    config["node_scorers"][0]["similarity"] = "dot_product"
    config["path_scorers"] = [
        {
            "name": "weighted_paths",
            "type": "weighted_sum",
            "input": "dense_nodes",
            "terms": {
                "root": 0.1,
                "level_1": 0.4,
                "level_2": 0.1,
                "mean": 0.1,
                "leaf": 0.1,
                "lowest": 0.1,
                "highest": 0.1,
            },
            "top_k": 10,
        }
    ]
    config["deciders"] = [
        {
            "name": "best",
            "type": "top_one",
            "input": "weighted_paths",
        }
    ]
    config["outputs"] = {"primary": "best"}
    pipeline = await TreeClassificationPipeline.build(
        taxonomy=_branching_taxonomy(),
        config=config,
        bindings=_bindings(SpyEmbeddingPreprocessor()),
    )

    selected = (await pipeline.classify("query")).outputs["primary"].selected[0]

    assert selected.item.node_ids == ("root", "a", "c")
    assert selected.score == pytest.approx(0.58)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terms",
    (
        {"weakest": 1.0},
        {"level_0": 1.0},
        [{"mean": 1.0}],
    ),
)
async def test_weighted_sum_rejects_invalid_terms(terms) -> None:
    config = _config()
    config["path_scorers"] = [
        {
            "name": "weighted_paths",
            "type": "weighted_sum",
            "input": "dense_nodes",
            "terms": terms,
        }
    ]
    for decider in config["deciders"]:
        decider["input"] = "weighted_paths"

    with pytest.raises(ConfigurationError, match="weighted_sum"):
        await TreeClassificationPipeline.build(
            taxonomy=_taxonomy(with_embeddings=True),
            config=config,
            bindings=_bindings(SpyEmbeddingPreprocessor()),
        )


class LeafPathScorer:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], tuple[float, ...]]] = []

    async def score_path(self, path, node_scores):
        self.calls.append((path.node_ids, node_scores))
        return node_scores[-1]


@pytest.mark.asyncio
async def test_custom_path_scorer_binding_defines_arbitrary_scoring() -> None:
    config = _config()
    config["node_scorers"][0]["similarity"] = "dot_product"
    config["path_scorers"] = [
        {
            "name": "custom_paths",
            "type": "custom",
            "binding": "leaf_score",
            "input": "dense_nodes",
            "top_k": 10,
        }
    ]
    for decider in config["deciders"]:
        decider["input"] = "custom_paths"
    scorer = LeafPathScorer()
    pipeline = await TreeClassificationPipeline.build(
        taxonomy=_branching_taxonomy(),
        config=config,
        bindings=ComponentBindings(
            preprocessors={"embedding": SpyEmbeddingPreprocessor()},
            path_scorers={"leaf_score": scorer},
        ),
    )

    decision = (await pipeline.classify("query")).outputs["primary"]

    assert decision.selected[0].item.node_ids == ("root", "b", "d")
    assert len(scorer.calls) == 2


class FakeLLMDecider:
    def __init__(self, *, unknown: bool = False, legacy_output: bool = False) -> None:
        self.unknown = unknown
        self.legacy_output = legacy_output
        self.calls = 0
        self.candidate_ids: tuple[str, ...] = ()

    async def decide(self, query, candidates, *, count):
        self.calls += 1
        assert query.text == "choose a path"
        assert count == 1
        self.candidate_ids = candidates.ids
        path_ids = ("missing",) if self.unknown else (candidates.items[-1].id,)
        if self.legacy_output:
            return path_ids
        return DecisionSelection(
            path_ids=path_ids,
            reason="best semantic match",
            metadata={"validation_used": True, "retry": 0},
        )


class FakeRouter:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls = 0
        self.candidate_ids: dict[str, tuple[str, ...]] = {}
        self.parameters = None

    async def route(self, query, candidates, *, parameters):
        self.calls += 1
        self.candidate_ids = {
            decider: ranking.ids for decider, ranking in candidates.items()
        }
        self.parameters = parameters
        return self.result


class SpyTokenizer:
    def __init__(self) -> None:
        self.tokenizer = TextTokenizer()
        self.document_inputs: list[tuple[str, ...]] = []

    async def prepare_query(self, text: str) -> tuple[str, ...]:
        return await self.tokenizer.prepare_query(text)

    async def prepare_documents(
        self, texts: tuple[str, ...]
    ) -> tuple[tuple[str, ...], ...]:
        self.document_inputs.append(texts)
        return await self.tokenizer.prepare_documents(texts)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ("best", DecisionRoute("best")))
async def test_router_accepts_string_or_typed_route(route) -> None:
    config = _config()
    config["routers"] = [
        {
            "name": "confidence_gate",
            "deciders": ["best", "alternatives"],
            "binding": "gate",
        }
    ]
    config["outputs"] = {"primary": "confidence_gate"}
    router = FakeRouter(route)
    pipeline = await TreeClassificationPipeline.build(
        taxonomy=_taxonomy(with_embeddings=True),
        config=config,
        bindings=ComponentBindings(
            preprocessors={"embedding": SpyEmbeddingPreprocessor()},
            routers={"gate": router},
        ),
    )

    decision = (await pipeline.classify("query")).outputs["primary"]

    assert router.calls == 1
    assert set(router.candidate_ids) == {"best", "alternatives"}
    assert all(len(ids) == 2 for ids in router.candidate_ids.values())
    assert router.parameters == {}
    assert decision.name == "best"
    assert decision.selected[0].item.node_ids == ("root", "a")


@pytest.mark.asyncio
async def test_router_receives_read_only_parameters_from_config() -> None:
    config = _config()
    config["routers"] = [
        {
            "name": "confidence_gate",
            "deciders": ["best", "alternatives"],
            "binding": "gate",
            "parameters": {
                "top_1_threshold": 0.85,
                "rules": ["semantic_only"],
            },
        }
    ]
    config["outputs"] = {"primary": "confidence_gate"}
    router = FakeRouter("best")
    pipeline = await TreeClassificationPipeline.build(
        taxonomy=_taxonomy(with_embeddings=True),
        config=config,
        bindings=ComponentBindings(
            preprocessors={"embedding": SpyEmbeddingPreprocessor()},
            routers={"gate": router},
        ),
    )

    await pipeline.classify("query")

    assert router.parameters == {
        "top_1_threshold": 0.85,
        "rules": ["semantic_only"],
    }
    with pytest.raises(TypeError):
        router.parameters["top_1_threshold"] = 0.5


@pytest.mark.asyncio
async def test_router_parameters_must_be_json_compatible() -> None:
    config = _config()
    config["routers"] = [
        {
            "name": "confidence_gate",
            "deciders": ["best", "alternatives"],
            "binding": "gate",
            "parameters": {"invalid": object()},
        }
    ]
    config["outputs"] = {"primary": "confidence_gate"}

    with pytest.raises(ConfigurationError, match="parameters"):
        await TreeClassificationPipeline.build(
            taxonomy=_taxonomy(with_embeddings=True),
            config=config,
            bindings=_bindings(SpyEmbeddingPreprocessor()),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "error"),
    (
        ("missing", "selected forbidden decider"),
        (None, "must return DecisionRoute or a decider name"),
    ),
)
async def test_router_rejects_invalid_output(route, error) -> None:
    config = _config()
    config["routers"] = [
        {
            "name": "confidence_gate",
            "deciders": ["best", "alternatives"],
            "binding": "gate",
        }
    ]
    config["outputs"] = {"primary": "confidence_gate"}
    pipeline = await TreeClassificationPipeline.build(
        taxonomy=_taxonomy(with_embeddings=True),
        config=config,
        bindings=ComponentBindings(
            preprocessors={"embedding": SpyEmbeddingPreprocessor()},
            routers={"gate": FakeRouter(route)},
        ),
    )

    with pytest.raises(InvalidModelOutputError, match=error):
        await pipeline.classify("query")


@pytest.mark.asyncio
async def test_router_and_decider_results_are_cached_per_query() -> None:
    config = _config()
    config["path_scorers"][0]["top_k"] = 1
    config["deciders"] = [
        {
            "name": "semantic_choice",
            "type": "llm",
            "binding": "llm",
            "input": "mean_paths",
        }
    ]
    config["routers"] = [
        {
            "name": name,
            "deciders": ["semantic_choice"],
            "binding": name,
        }
        for name in ("first_gate", "second_gate")
    ]
    config["outputs"] = {
        "primary": "first_gate",
        "secondary": "second_gate",
    }
    first = FakeRouter("semantic_choice")
    second = FakeRouter(DecisionRoute("semantic_choice"))
    llm = FakeLLMDecider()
    pipeline = await TreeClassificationPipeline.build(
        taxonomy=_taxonomy(with_embeddings=True),
        config=config,
        bindings=ComponentBindings(
            preprocessors={"embedding": SpyEmbeddingPreprocessor()},
            deciders={"llm": llm},
            routers={"first_gate": first, "second_gate": second},
        ),
    )

    result = await pipeline.classify("choose a path")

    assert first.calls == second.calls == 1
    assert set(first.candidate_ids) == set(second.candidate_ids) == {"semantic_choice"}
    assert len(first.candidate_ids["semantic_choice"]) == 1
    assert llm.calls == 1
    assert len(llm.candidate_ids) == 2
    assert result.outputs["primary"] is result.outputs["secondary"]


@pytest.mark.asyncio
async def test_bm25_node_and_path_fusions_feed_different_router_branches() -> None:
    taxonomy = Taxonomy(
        "catalog",
        (
            TaxonomyNode("root", "Catalog", embedding=(1.0, 0.0), tokens=("catalog",)),
            TaxonomyNode("red", "Red shoes", embedding=(0.0, 1.0)),
            TaxonomyNode("blue", "Blue hats", embedding=(1.0, 0.0)),
        ),
        (TaxonomyEdge("root", "red"), TaxonomyEdge("root", "blue")),
    )
    config = {
        "version": 1,
        "name": "hybrid_tree",
        "kind": "tree_classification",
        "preprocessors": [
            {
                "name": "semantic",
                "type": "embedding",
                "binding": "embedding",
                "dimensions": 2,
            },
            {"name": "lexical", "type": "tokenizer", "binding": "tokenizer"},
        ],
        "node_scorers": [
            {
                "name": "dense_nodes",
                "type": "dense",
                "preprocessor": "semantic",
            },
            {
                "name": "bm25_nodes",
                "type": "bm25",
                "preprocessor": "lexical",
            },
        ],
        "node_fusions": [
            {
                "name": "hybrid_nodes",
                "type": "weighted_rrf",
                "inputs": ["dense_nodes", "bm25_nodes"],
                "k": 1,
            }
        ],
        "path_scorers": [
            {"name": "dense_paths", "type": "mean", "input": "dense_nodes"},
            {"name": "bm25_paths", "type": "mean", "input": "bm25_nodes"},
            {"name": "hybrid_paths", "type": "mean", "input": "hybrid_nodes"},
        ],
        "path_fusions": [
            {
                "name": "fused_paths",
                "type": "weighted_rrf",
                "inputs": ["dense_paths", "bm25_paths"],
                "weights": [1, 2],
                "k": 1,
            }
        ],
        "deciders": [
            {"name": "dense", "type": "top_one", "input": "dense_paths"},
            {"name": "bm25", "type": "top_one", "input": "bm25_paths"},
            {"name": "hybrid", "type": "top_one", "input": "hybrid_paths"},
            {"name": "fused", "type": "top_one", "input": "fused_paths"},
        ],
        "routers": [
            {
                "name": "choose",
                "deciders": ["dense", "bm25", "hybrid", "fused"],
                "binding": "choose",
            }
        ],
        "outputs": {"primary": "choose"},
    }
    tokenizer = SpyTokenizer()
    router = FakeRouter("fused")
    pipeline = await TreeClassificationPipeline.build(
        taxonomy=taxonomy,
        config=config,
        bindings=ComponentBindings(
            preprocessors={
                "embedding": SpyEmbeddingPreprocessor(),
                "tokenizer": tokenizer,
            },
            routers={"choose": router},
        ),
    )

    decision = (await pipeline.classify("red shoes")).outputs["primary"]

    assert tokenizer.document_inputs == [("Red shoes", "Blue hats")]
    assert pipeline.taxonomy.nodes[0].tokens == ("catalog",)
    assert pipeline.taxonomy.nodes[1].tokens == ("red", "shoes")
    assert router.candidate_ids["dense"][0] == '["root","blue"]'
    assert router.candidate_ids["bm25"][0] == '["root","red"]'
    assert router.candidate_ids["hybrid"][0] == '["root","red"]'
    assert router.candidate_ids["fused"][0] == '["root","red"]'
    assert decision.name == "fused"


@pytest.mark.asyncio
async def test_unused_tree_scoring_branch_is_rejected() -> None:
    config = _config()
    config["path_scorers"].append(
        {"name": "unused_paths", "type": "mean", "input": "dense_nodes"}
    )

    with pytest.raises(BuildError, match="unused nodes.*unused_paths"):
        await TreeClassificationPipeline.build(
            taxonomy=_taxonomy(with_embeddings=True),
            config=config,
            bindings=_bindings(SpyEmbeddingPreprocessor()),
        )


@pytest.mark.asyncio
async def test_router_references_and_bindings_are_validated_during_build() -> None:
    config = _config()
    config["routers"] = [
        {
            "name": "confidence_gate",
            "deciders": ["missing"],
            "binding": "gate",
        }
    ]
    config["outputs"] = {"primary": "confidence_gate"}

    with pytest.raises(BuildError, match="references unknown deciders"):
        await TreeClassificationPipeline.build(
            taxonomy=_taxonomy(with_embeddings=True),
            config=config,
            bindings=_bindings(SpyEmbeddingPreprocessor()),
        )

    config["routers"][0]["deciders"] = ["best", "alternatives"]
    with pytest.raises(BuildError, match="missing router binding"):
        await TreeClassificationPipeline.build(
            taxonomy=_taxonomy(with_embeddings=True),
            config=config,
            bindings=_bindings(SpyEmbeddingPreprocessor()),
        )


@pytest.mark.asyncio
async def test_llm_decider_receives_all_paths_and_returns_canonical_path() -> None:
    config = _config()
    config["path_scorers"][0]["top_k"] = 1
    config["deciders"] = [
        {
            "name": "semantic_choice",
            "type": "llm",
            "binding": "llm",
            "input": "mean_paths",
            "count": 1,
        }
    ]
    config["outputs"] = {"primary": "semantic_choice"}
    llm = FakeLLMDecider()
    pipeline = await TreeClassificationPipeline.build(
        taxonomy=_taxonomy(with_embeddings=True),
        config=config,
        bindings=ComponentBindings(
            preprocessors={"embedding": SpyEmbeddingPreprocessor()},
            deciders={"llm": llm},
        ),
    )

    decision = (await pipeline.classify("choose a path")).outputs["primary"]

    assert len(llm.candidate_ids) == 2
    assert decision.selected[0].id == llm.candidate_ids[-1]
    assert decision.selected[0].item.node_ids == ("root", "b")
    assert decision.reason == "best semantic match"
    assert decision.metadata == {"validation_used": True, "retry": 0}
    with pytest.raises(TypeError):
        decision.metadata["retry"] = 1


@pytest.mark.asyncio
async def test_llm_decider_cannot_invent_paths() -> None:
    config = _config()
    config["deciders"] = [
        {
            "name": "semantic_choice",
            "type": "llm",
            "binding": "llm",
            "input": "mean_paths",
        }
    ]
    config["outputs"] = {"primary": "semantic_choice"}
    pipeline = await TreeClassificationPipeline.build(
        taxonomy=_taxonomy(with_embeddings=True),
        config=config,
        bindings=ComponentBindings(
            preprocessors={"embedding": SpyEmbeddingPreprocessor()},
            deciders={"llm": FakeLLMDecider(unknown=True)},
        ),
    )

    with pytest.raises(InvalidModelOutputError, match="unknown path IDs"):
        await pipeline.classify("choose a path")


@pytest.mark.asyncio
async def test_llm_decider_must_return_decision_selection() -> None:
    config = _config()
    config["deciders"] = [
        {
            "name": "semantic_choice",
            "type": "llm",
            "binding": "llm",
            "input": "mean_paths",
        }
    ]
    config["outputs"] = {"primary": "semantic_choice"}
    pipeline = await TreeClassificationPipeline.build(
        taxonomy=_taxonomy(with_embeddings=True),
        config=config,
        bindings=ComponentBindings(
            preprocessors={"embedding": SpyEmbeddingPreprocessor()},
            deciders={"llm": FakeLLMDecider(legacy_output=True)},
        ),
    )

    with pytest.raises(InvalidModelOutputError, match="return DecisionSelection"):
        await pipeline.classify("choose a path")


@pytest.mark.asyncio
async def test_dict_yaml_and_json_load_the_same_configuration(tmp_path: Path) -> None:
    config = _config()
    config["routers"] = [
        {
            "name": "confidence_gate",
            "deciders": ["best"],
            "binding": "gate",
            "parameters": {"threshold": 0.85},
        }
    ]
    config["outputs"]["primary"] = "confidence_gate"
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
routers:
  - name: confidence_gate
    deciders:
      - best
    binding: gate
    parameters:
      threshold: 0.85
outputs:
  primary: confidence_gate
  alternatives: alternatives
""",
        encoding="utf-8",
    )
    json_path = tmp_path / "pipeline.json"
    json_path.write_text(json.dumps(config), encoding="utf-8")
    preprocessor = SpyEmbeddingPreprocessor()
    bindings = ComponentBindings(
        preprocessors={"embedding": preprocessor},
        routers={"gate": FakeRouter("best")},
    )
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
