"""Build and run the first configurable retrieval vertical."""

from __future__ import annotations

from collections.abc import Iterable
from types import MappingProxyType

from ..core._async import resolve
from ..core.config import ConfigInput, load_config
from ..core.contracts import ComponentBindings
from ..core.dag import CompiledDAG, compile_dag
from ..core.models import Query, as_query
from ..core.ranking import (
    Ranking,
    reciprocal_rank_fusion,
    require_ranking,
    require_subset,
)
from ..errors import BuildError
from .config import RetrievalConfig, RetrieverConfig, RRFFusionConfig
from .models import Chunk, RetrievalOutput, RetrievalPipelineResult


def _by_name(values, *, label: str):
    result = {value.name: value for value in values}
    if len(result) != len(values):
        raise BuildError(f"{label} names must be unique")
    return result


class RetrievalPipeline:
    """Sequential DAG of bound retrievers and built-in RRF fusions."""

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
        preprocessor_specs = _by_name(parsed.preprocessors, label="preprocessor")
        retriever_specs = _by_name(parsed.retrievers, label="retriever")
        fusion_specs = _by_name(parsed.fusions, label="fusion")

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
            if not callable(getattr(component, "prepare_query", None)):
                raise BuildError(
                    f"preprocessor binding {spec.binding!r} must define "
                    "prepare_query(text)"
                )
            preprocessors[spec.name] = component

        retrievers: dict[str, object] = {}
        for spec in parsed.retrievers:
            try:
                component = bindings.retrievers[spec.binding]
            except KeyError as exc:
                raise BuildError(f"missing retriever binding {spec.binding!r}") from exc
            method = "retrieve_candidates" if spec.input is not None else "retrieve"
            if not callable(getattr(component, method, None)):
                raise BuildError(
                    f"retriever binding {spec.binding!r} must define {method}()"
                )
            retrievers[spec.name] = component

        dependencies = {
            spec.name: (() if spec.input is None else (spec.input,))
            for spec in parsed.retrievers
        }
        dependencies.update(
            {spec.name: tuple(spec.inputs) for spec in parsed.fusions}
        )
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
        prepared: dict[str, object] = {}
        rankings: dict[str, Ranking[Chunk]] = {}

        for name in self._dag.order:
            if name in self._retriever_specs:
                spec = self._retriever_specs[name]
                if spec.preprocessor not in prepared:
                    preprocessor = self._preprocessors[spec.preprocessor]
                    prepared[spec.preprocessor] = await resolve(
                        preprocessor.prepare_query(query.text)
                    )
                rankings[name] = await self._run_retriever(
                    spec,
                    prepared[spec.preprocessor],
                    rankings,
                )
            else:
                spec = self._fusion_specs[name]
                rankings[name] = self._run_fusion(spec, rankings)

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
            raw = await resolve(
                component.retrieve(query_representation, top_k=spec.top_k)
            )
            ranking = require_ranking(raw, component=spec.name)
            return Ranking(
                ranking.items[: spec.top_k],
                ranking.score_semantics,
                frozenset({spec.name}),
            )

        candidates = rankings[spec.input]
        raw = await resolve(
            component.retrieve_candidates(
                query_representation,
                candidates,
                top_k=spec.top_k,
            )
        )
        ranking = require_ranking(raw, component=spec.name)
        return require_subset(ranking, candidates, component=spec.name).top(spec.top_k)

    @staticmethod
    def _run_fusion(
        spec: RRFFusionConfig,
        rankings: dict[str, Ranking[Chunk]],
    ) -> Ranking[Chunk]:
        return reciprocal_rank_fusion(
            {name: rankings[name] for name in spec.inputs},
            top_k=spec.top_k,
            k=spec.k,
        )
