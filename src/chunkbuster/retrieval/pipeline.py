"""Build and run the first configurable retrieval vertical."""

from __future__ import annotations

from asyncio import gather
from collections.abc import Iterable
from inspect import iscoroutinefunction
from types import MappingProxyType

from ..core._async import task_scope
from ..core.config import ConfigInput, by_name, load_config
from ..core.contracts import ComponentBindings
from ..core.dag import CompiledDAG, compile_dag
from ..core.models import Query, as_query
from ..core.ranking import (
    Ranking,
    fuse_rankings,
    require_ranking,
    require_subset,
)
from ..errors import BuildError
from .config import FusionConfig, RetrievalConfig, RetrieverConfig
from .models import Chunk, RetrievalOutput, RetrievalPipelineResult


class RetrievalPipeline:
    """Concurrent DAG of bound retrievers and built-in ranking fusions."""

    def __init__(
        self,
        *,
        config: RetrievalConfig,
        dag: CompiledDAG,
        preprocessors: dict[str, object],
        retrievers: dict[str, object],
    ) -> None:
        self._config = config
        self._dag = dag
        self._preprocessors = MappingProxyType(dict(preprocessors))
        self._retrievers = MappingProxyType(dict(retrievers))
        self._retriever_specs = MappingProxyType(
            {spec.name: spec for spec in config.retrievers}
        )
        self._fusion_specs = MappingProxyType(
            {spec.name: spec for spec in config.fusions}
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

        overlap = set(retriever_specs) & set(fusion_specs)
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
                    "async prepare_query(text)"
                )
            preprocessors[spec.name] = component

        retrievers: dict[str, object] = {}
        for spec in parsed.retrievers:
            try:
                component = bindings.retrievers[spec.binding]
            except KeyError as exc:
                raise BuildError(f"missing retriever binding {spec.binding!r}") from exc
            method = "retrieve_candidates" if spec.input is not None else "retrieve"
            if not iscoroutinefunction(getattr(component, method, None)):
                raise BuildError(
                    f"retriever binding {spec.binding!r} must define async {method}()"
                )
            retrievers[spec.name] = component

        dependencies = {
            spec.name: (() if spec.input is None else (spec.input,))
            for spec in parsed.retrievers
        }
        dependencies.update({spec.name: tuple(spec.inputs) for spec in parsed.fusions})
        dag = compile_dag(dependencies, parsed.outputs)
        return cls(
            config=parsed,
            dag=dag,
            preprocessors=preprocessors,
            retrievers=retrievers,
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
        rankings: dict[str, Ranking[Chunk]] = {}

        async def run_node(name):
            await gather(*(nodes[source] for source in self._dag.dependencies[name]))
            if name in self._retriever_specs:
                spec = self._retriever_specs[name]
                rankings[name] = await self._run_retriever(
                    spec,
                    await prepared[spec.preprocessor],
                    rankings,
                )
            else:
                spec = self._fusion_specs[name]
                rankings[name] = self._run_fusion(spec, rankings)

        async with task_scope() as start:
            prepared = {
                name: start(component.prepare_query(query.text))
                for name, component in self._preprocessors.items()
            }
            nodes = {}
            for name in self._dag.order:
                nodes[name] = start(run_node(name))
            await gather(*nodes.values())

        requested = tuple(self._config.outputs) if outputs is None else tuple(outputs)
        if len(requested) != len(set(requested)):
            raise ValueError("requested output names must be unique")
        unknown = set(requested) - set(self._config.outputs)
        if unknown:
            raise ValueError(f"unknown outputs: {sorted(unknown)!r}")
        result = {
            output_name: RetrievalOutput(
                self._config.outputs[output_name],
                rankings[self._config.outputs[output_name]],
            )
            for output_name in requested
        }
        return RetrievalPipelineResult(query.id, result)

    async def _run_retriever(
        self,
        spec: RetrieverConfig,
        query_representation: object,
        rankings: dict[str, Ranking[Chunk]],
    ) -> Ranking[Chunk]:
        component = self._retrievers[spec.name]
        if spec.input is None:
            raw = await component.retrieve(query_representation, top_k=spec.top_k)
            ranking = require_ranking(raw, component=spec.name)
            return Ranking(
                ranking.items[: spec.top_k],
                ranking.score_semantics,
                frozenset({spec.name}),
            )

        candidates = rankings[spec.input]
        raw = await component.retrieve_candidates(
            query_representation,
            candidates,
            top_k=spec.top_k,
        )
        ranking = require_ranking(raw, component=spec.name)
        return require_subset(ranking, candidates, component=spec.name).top(spec.top_k)

    @staticmethod
    def _run_fusion(
        spec: FusionConfig,
        rankings: dict[str, Ranking[Chunk]],
    ) -> Ranking[Chunk]:
        inputs = {name: rankings[name] for name in spec.inputs}
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
