"""Build and run the first configurable retrieval vertical."""

from __future__ import annotations

from asyncio import gather, sleep
from collections.abc import Iterable
from inspect import iscoroutinefunction
from types import MappingProxyType

from ..core._async import task_scope
from ..core.config import ConfigInput, by_name, load_config
from ..core.contracts import ComponentBindings, invoke_router
from ..core.dag import CompiledDAG, compile_dag
from ..core.models import Query, as_query
from ..core.ranking import (
    Ranking,
    fuse_rankings,
    require_ranking,
    require_subset,
)
from ..errors import BuildError, InvalidModelOutputError
from .config import FusionConfig, RetrievalConfig, RetrieverConfig
from .models import Chunk, RetrievalOutput, RetrievalPipelineResult


class RetrievalPipeline:
    """Concurrent DAG of retrievers, ranking fusions, and routers."""

    def __init__(
        self,
        *,
        config: RetrievalConfig,
        dag: CompiledDAG,
        preprocessors: dict[str, object],
        retrievers: dict[str, object],
        routers: dict[str, object],
    ) -> None:
        self._config = config
        self._dag = dag
        self._preprocessors = MappingProxyType(dict(preprocessors))
        self._retrievers = MappingProxyType(dict(retrievers))
        self._routers = MappingProxyType(dict(routers))
        self._retriever_specs = MappingProxyType(
            {spec.name: spec for spec in config.retrievers}
        )
        self._fusion_specs = MappingProxyType(
            {spec.name: spec for spec in config.fusions}
        )
        self._router_specs = MappingProxyType(
            {spec.name: spec for spec in config.routers}
        )

    @classmethod
    async def build(
        cls,
        *,
        config: ConfigInput,
        bindings: ComponentBindings | None = None,
    ) -> RetrievalPipeline:
        parsed = load_config(config, RetrievalConfig)
        bindings = bindings or ComponentBindings()
        preprocessor_specs = by_name(parsed.preprocessors, label="preprocessor")
        retriever_specs = by_name(parsed.retrievers, label="retriever")
        fusion_specs = by_name(parsed.fusions, label="fusion")
        router_specs = by_name(parsed.routers, label="router")

        overlap = (retriever_specs.keys() & fusion_specs.keys()) | (
            router_specs.keys() & (retriever_specs.keys() | fusion_specs.keys())
        )
        if overlap:
            raise BuildError(
                f"executable node names must be unique: {sorted(overlap)!r}"
            )

        referenced_preprocessors = {spec.preprocessor for spec in parsed.retrievers}
        missing = referenced_preprocessors - set(preprocessor_specs)
        if missing:
            raise BuildError(
                f"retrievers reference unknown preprocessors {sorted(missing)!r}"
            )
        unused = set(preprocessor_specs) - referenced_preprocessors
        if unused:
            raise BuildError(f"unused preprocessors: {sorted(unused)!r}")

        preprocessors: dict[str, object] = {}
        for spec in parsed.preprocessors:
            try:
                component = bindings.preprocessors[spec.binding]
            except KeyError as exc:
                raise BuildError(
                    f"missing preprocessor binding {spec.binding!r}"
                ) from exc
            if not iscoroutinefunction(getattr(component, "prepare_query", None)):
                raise BuildError(
                    f"preprocessor binding {spec.binding!r} must define "
                    "async prepare_query(query)"
                )
            preprocessors[spec.name] = component

        retrievers: dict[str, object] = {}
        for spec in parsed.retrievers:
            try:
                component = bindings.retrievers[spec.binding]
            except KeyError as exc:
                raise BuildError(f"missing retriever binding {spec.binding!r}") from exc
            method = "retrieve_candidates" if spec.inputs is not None else "retrieve"
            if not iscoroutinefunction(getattr(component, method, None)):
                raise BuildError(
                    f"retriever binding {spec.binding!r} must define async {method}()"
                )
            retrievers[spec.name] = component

        routers: dict[str, object] = {}
        for spec in parsed.routers:
            try:
                component = bindings.routers[spec.binding]
            except KeyError as exc:
                raise BuildError(f"missing router binding {spec.binding!r}") from exc
            if not iscoroutinefunction(getattr(component, "route", None)):
                raise BuildError(
                    f"router binding {spec.binding!r} must define async route()"
                )
            routers[spec.name] = component

        dependencies = {spec.name: spec.inputs or () for spec in parsed.retrievers}
        dependencies.update({spec.name: spec.inputs for spec in parsed.fusions})
        # All possible routes participate in build-time cycle and reachability checks.
        dependencies.update(
            {
                spec.name: tuple(dict.fromkeys((*spec.inputs, *spec.targets)))
                for spec in parsed.routers
            }
        )
        dag = compile_dag(dependencies, parsed.outputs)
        return cls(
            config=parsed,
            dag=dag,
            preprocessors=preprocessors,
            retrievers=retrievers,
            routers=routers,
        )

    @property
    def config(self) -> RetrievalConfig:
        return self._config

    async def retrieve(
        self,
        query: str | Query,
        *,
        outputs: Iterable[str] | None = None,
    ) -> RetrievalPipelineResult:
        query = as_query(query)
        requested = tuple(self._config.outputs) if outputs is None else tuple(outputs)
        if len(requested) != len(set(requested)):
            raise ValueError("requested output names must be unique")
        unknown = set(requested) - set(self._config.outputs)
        if unknown:
            raise ValueError(f"unknown outputs: {sorted(unknown)!r}")

        prepared = {}
        nodes = {}

        def prepare(name):
            if name not in prepared:
                prepared[name] = start(self._preprocessors[name].prepare_query(query))
            return prepared[name]

        def evaluate(name):
            if name not in nodes:
                nodes[name] = start(run_node(name))
            return nodes[name]

        async def run_node(name: str) -> Ranking[Chunk]:
            if self._routers:
                # Eager task factories would otherwise recurse into dependencies.
                await sleep(0)
            retriever = self._retriever_specs.get(name)
            router = self._router_specs.get(name)
            sources = router.inputs if router else self._dag.dependencies[name]
            pending = [evaluate(source) for source in sources]
            if retriever is not None:
                pending.append(prepare(retriever.preprocessor))
            await gather(*pending)
            rankings = {source: nodes[source].result() for source in sources}

            if retriever is not None:
                return await self._run_retriever(
                    retriever,
                    query,
                    prepared[retriever.preprocessor].result(),
                    rankings,
                )
            if router is not None:
                target = await invoke_router(
                    self._routers[name], query, rankings, parameters=router.parameters
                )
                if not isinstance(target, str) or not target:
                    raise InvalidModelOutputError(
                        f"router {name!r} must return a target name"
                    )
                if target not in router.targets:
                    raise InvalidModelOutputError(
                        f"router {name!r} selected forbidden target {target!r}"
                    )
                return await evaluate(target)
            return self._run_fusion(self._fusion_specs[name], rankings)

        terminals = tuple(self._config.outputs[name] for name in requested)
        roots = terminals if self._routers else self._dag.order
        async with task_scope() as start:
            if not self._routers:
                for name in self._preprocessors:
                    prepare(name)
            await gather(*(evaluate(name) for name in roots))

        result = {
            output_name: RetrievalOutput(terminal, nodes[terminal].result())
            for output_name, terminal in zip(requested, terminals, strict=True)
        }
        return RetrievalPipelineResult(query.id, result)

    async def _run_retriever(
        self,
        spec: RetrieverConfig,
        query: Query,
        query_representation: object,
        rankings: dict[str, Ranking[Chunk]],
    ) -> Ranking[Chunk]:
        component = self._retrievers[spec.name]
        if spec.inputs is None:
            raw = await component.retrieve(
                query, query_representation, top_k=spec.top_k
            )
            ranking = require_ranking(raw, component=spec.name)
            return Ranking(
                ranking.items[: spec.top_k],
                ranking.score_semantics,
                frozenset({spec.name}),
            )

        raw = await component.retrieve_candidates(
            query,
            query_representation,
            MappingProxyType(rankings),
            top_k=spec.top_k,
        )
        ranking = require_ranking(raw, component=spec.name)
        return require_subset(ranking, rankings.values(), component=spec.name).top(
            spec.top_k
        )

    @staticmethod
    def _run_fusion(
        spec: FusionConfig,
        inputs: dict[str, Ranking[Chunk]],
    ) -> Ranking[Chunk]:
        weights = (
            dict(zip(spec.inputs, spec.weights, strict=True)) if spec.weights else None
        )
        return fuse_rankings(
            inputs,
            method="weighted_rrf" if spec.type == "rrf" else spec.type,
            top_k=spec.top_k,
            weights=weights,
            normalization=spec.normalization,
            k=spec.k,
        )
