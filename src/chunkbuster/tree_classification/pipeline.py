"""Build and run configurable tree-classification graphs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from types import MappingProxyType

from ..core._async import resolve
from ..core.config import ConfigInput, by_name, load_config
from ..core.contracts import ComponentBindings
from ..core.dag import CompiledDAG, compile_dag
from ..core.models import Query, as_query
from ..core.ranking import Ranking, fuse_rankings
from ..errors import BuildError, PreprocessingError
from .config import (
    BM25NodeScorerConfig,
    CustomPathScorerConfig,
    DenseNodeScorerConfig,
    EmbeddingPreprocessorConfig,
    LLMDeciderConfig,
    TokenizerPreprocessorConfig,
    TreeClassificationConfig,
)
from .decisions import decide, decide_with_llm, materialize_decision, route_decision
from .models import Taxonomy, TaxonomyPath, TreeClassificationResult
from .scoring import (
    BM25Index,
    NodeScores,
    build_bm25_index,
    builtin_path_score,
    rank_node_scores,
    rank_paths,
    score_bm25_nodes,
    score_nodes,
    validate_path_score,
    validate_tokens,
    validate_vector,
)
from .taxonomy import TaxonomySnapshot, build_snapshot


def _weights(spec) -> dict[str, float] | None:
    return dict(zip(spec.inputs, spec.weights, strict=True)) if spec.weights else None


class TreeClassificationPipeline:
    """Immutable configurable tree-classification pipeline."""

    def __init__(
        self,
        *,
        snapshot: TaxonomySnapshot,
        config: TreeClassificationConfig,
        dag: CompiledDAG,
        preprocessors: dict[str, object],
        bm25_index: BM25Index | None,
        path_scorers: dict[str, object],
        deciders: dict[str, object],
        routers: dict[str, object],
    ) -> None:
        self._snapshot = snapshot
        self._config = config
        self._dag = dag
        self._preprocessors = MappingProxyType(dict(preprocessors))
        self._bm25_index = bm25_index
        self._path_scorers = MappingProxyType(dict(path_scorers))
        self._deciders = MappingProxyType(dict(deciders))
        self._routers = MappingProxyType(dict(routers))
        self._preprocessor_specs = MappingProxyType(
            {spec.name: spec for spec in config.preprocessors}
        )
        self._node_scorer_specs = MappingProxyType(
            {spec.name: spec for spec in config.node_scorers}
        )
        self._node_fusion_specs = MappingProxyType(
            {spec.name: spec for spec in config.node_fusions}
        )
        self._path_scorer_specs = MappingProxyType(
            {spec.name: spec for spec in config.path_scorers}
        )
        self._path_fusion_specs = MappingProxyType(
            {spec.name: spec for spec in config.path_fusions}
        )

    @classmethod
    async def build(
        cls,
        *,
        taxonomy: Taxonomy,
        config: ConfigInput,
        bindings: ComponentBindings | None = None,
    ) -> TreeClassificationPipeline:
        parsed = load_config(config, TreeClassificationConfig)
        bindings = bindings or ComponentBindings()
        preprocessors = by_name(parsed.preprocessors, label="preprocessor")
        node_scorers = by_name(parsed.node_scorers, label="node scorer")
        node_fusions = by_name(parsed.node_fusions, label="node fusion")
        path_scorers = by_name(parsed.path_scorers, label="path scorer")
        path_fusions = by_name(parsed.path_fusions, label="path fusion")
        deciders = by_name(parsed.deciders, label="decider")
        routers = by_name(parsed.routers, label="router")

        if not node_scorers or not path_scorers or not deciders or not parsed.outputs:
            raise BuildError(
                "the pipeline requires node scorers, path scorers, "
                "deciders, and outputs"
            )
        executable_groups = (
            preprocessors,
            node_scorers,
            node_fusions,
            path_scorers,
            path_fusions,
        )
        executable_names = [name for group in executable_groups for name in group]
        if len(executable_names) != len(set(executable_names)):
            raise BuildError("scoring graph names must be unique")
        if set(deciders) & set(routers):
            raise BuildError("router and decider names must be unique")

        embedding_preprocessors = [
            spec
            for spec in parsed.preprocessors
            if isinstance(spec, EmbeddingPreprocessorConfig)
        ]
        tokenizer_preprocessors = [
            spec
            for spec in parsed.preprocessors
            if isinstance(spec, TokenizerPreprocessorConfig)
        ]
        if len(embedding_preprocessors) > 1 or len(tokenizer_preprocessors) > 1:
            raise BuildError("taxonomy supports one preprocessor per representation")

        for spec in parsed.node_scorers:
            try:
                preprocessor = preprocessors[spec.preprocessor]
            except KeyError as exc:
                raise BuildError(
                    f"node scorer {spec.name!r} references an unknown preprocessor"
                ) from exc
            expected = (
                EmbeddingPreprocessorConfig
                if isinstance(spec, DenseNodeScorerConfig)
                else TokenizerPreprocessorConfig
            )
            if not isinstance(preprocessor, expected):
                raise BuildError(
                    f"node scorer {spec.name!r} references an incompatible preprocessor"
                )
        for spec in parsed.node_fusions:
            missing = set(spec.inputs) - set(node_scorers)
            if missing:
                raise BuildError(
                    f"node fusion {spec.name!r} references unknown node scorers "
                    f"{sorted(missing)!r}"
                )
        node_sources = set(node_scorers) | set(node_fusions)
        for spec in parsed.path_scorers:
            if spec.input not in node_sources:
                raise BuildError(
                    f"path scorer {spec.name!r} references an unknown node source"
                )
        for spec in parsed.path_fusions:
            missing = set(spec.inputs) - set(path_scorers)
            if missing:
                raise BuildError(
                    f"path fusion {spec.name!r} references unknown path scorers "
                    f"{sorted(missing)!r}"
                )
        path_sources = set(path_scorers) | set(path_fusions)
        for spec in parsed.deciders:
            if spec.input not in path_sources:
                raise BuildError(
                    f"decider {spec.name!r} references an unknown path ranking"
                )
        for spec in parsed.routers:
            missing = set(spec.deciders) - set(deciders)
            if missing:
                raise BuildError(
                    f"router {spec.name!r} references unknown deciders "
                    f"{sorted(missing)!r}"
                )

        targets = set(parsed.outputs.values())
        missing_outputs = targets - set(deciders) - set(routers)
        if missing_outputs:
            raise BuildError(
                f"outputs reference unknown targets {sorted(missing_outputs)!r}"
            )
        used_routers = targets & set(routers)
        unused_routers = set(routers) - used_routers
        if unused_routers:
            raise BuildError(f"unused routers: {sorted(unused_routers)!r}")
        used_deciders = targets & set(deciders)
        for router_name in used_routers:
            used_deciders.update(routers[router_name].deciders)
        unused_deciders = set(deciders) - used_deciders
        if unused_deciders:
            raise BuildError(f"unused deciders: {sorted(unused_deciders)!r}")

        dependencies = {name: () for name in preprocessors}
        dependencies.update(
            {spec.name: (spec.preprocessor,) for spec in parsed.node_scorers}
        )
        dependencies.update(
            {spec.name: tuple(spec.inputs) for spec in parsed.node_fusions}
        )
        dependencies.update({spec.name: (spec.input,) for spec in parsed.path_scorers})
        dependencies.update(
            {spec.name: tuple(spec.inputs) for spec in parsed.path_fusions}
        )
        dag = compile_dag(
            dependencies,
            {name: deciders[name].input for name in used_deciders},
        )

        bound_preprocessors = {}
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
            bound_preprocessors[spec.name] = component

        bound_path_scorers = {}
        for spec in parsed.path_scorers:
            if not isinstance(spec, CustomPathScorerConfig):
                continue
            try:
                component = bindings.path_scorers[spec.binding]
            except KeyError as exc:
                raise BuildError(
                    f"missing path scorer binding {spec.binding!r}"
                ) from exc
            if not callable(getattr(component, "score_path", None)):
                raise BuildError(
                    "custom path scorer must define score_path(path, scores)"
                )
            bound_path_scorers[spec.name] = component

        bound_deciders = {}
        for spec in parsed.deciders:
            if not isinstance(spec, LLMDeciderConfig):
                continue
            try:
                component = bindings.deciders[spec.binding]
            except KeyError as exc:
                raise BuildError(f"missing decider binding {spec.binding!r}") from exc
            if not callable(getattr(component, "decide", None)):
                raise BuildError("LLM decider binding must define decide()")
            bound_deciders[spec.name] = component

        bound_routers = {}
        for spec in parsed.routers:
            try:
                component = bindings.routers[spec.binding]
            except KeyError as exc:
                raise BuildError(f"missing router binding {spec.binding!r}") from exc
            if not callable(getattr(component, "route", None)):
                raise BuildError("router binding must define route()")
            bound_routers[spec.name] = component

        snapshot = build_snapshot(taxonomy)
        if embedding_preprocessors:
            spec = embedding_preprocessors[0]
            component = bound_preprocessors[spec.name]
            present = [node.embedding is not None for node in snapshot.taxonomy.nodes]
            if any(present) and not all(present):
                raise BuildError(
                    "taxonomy embeddings must be complete or entirely absent"
                )
            if not any(present):
                vectors = await cls._prepare_documents(
                    component,
                    tuple(node.text for node in snapshot.taxonomy.nodes),
                    label="embedding preprocessor",
                )
                nodes = tuple(
                    replace(
                        node,
                        embedding=validate_vector(
                            vector,
                            dimensions=spec.dimensions,
                            label=f"embedding for node {node.id!r}",
                        ),
                    )
                    for node, vector in zip(
                        snapshot.taxonomy.nodes, vectors, strict=True
                    )
                )
                snapshot = build_snapshot(replace(snapshot.taxonomy, nodes=nodes))
            else:
                for node in snapshot.taxonomy.nodes:
                    validate_vector(
                        node.embedding,
                        dimensions=spec.dimensions,
                        label=f"embedding for node {node.id!r}",
                    )

        if tokenizer_preprocessors:
            spec = tokenizer_preprocessors[0]
            component = bound_preprocessors[spec.name]
            missing = tuple(
                node for node in snapshot.taxonomy.nodes if node.tokens is None
            )
            if missing:
                token_lists = await cls._prepare_documents(
                    component,
                    tuple(node.text for node in missing),
                    label="tokenizer preprocessor",
                )
                generated = {
                    node.id: validate_tokens(
                        tokens, label=f"tokens for node {node.id!r}"
                    )
                    for node, tokens in zip(missing, token_lists, strict=True)
                }
                nodes = tuple(
                    replace(node, tokens=generated[node.id])
                    if node.id in generated
                    else node
                    for node in snapshot.taxonomy.nodes
                )
                snapshot = build_snapshot(replace(snapshot.taxonomy, nodes=nodes))

        bm25_index = (
            build_bm25_index(snapshot)
            if any(
                isinstance(spec, BM25NodeScorerConfig) for spec in parsed.node_scorers
            )
            else None
        )
        return cls(
            snapshot=snapshot,
            config=parsed,
            dag=dag,
            preprocessors=bound_preprocessors,
            bm25_index=bm25_index,
            path_scorers=bound_path_scorers,
            deciders=bound_deciders,
            routers=bound_routers,
        )

    @staticmethod
    async def _prepare_documents(component, texts, *, label: str) -> tuple:
        prepare = getattr(component, "prepare_documents", None)
        if not callable(prepare):
            raise BuildError(f"{label} must define prepare_documents(texts)")
        raw = await resolve(prepare(texts))
        try:
            values = tuple(raw)
        except TypeError as exc:
            raise PreprocessingError(f"{label} must return one value per node") from exc
        if len(values) != len(texts):
            raise PreprocessingError(f"{label} must return one value per node")
        return values

    @property
    def taxonomy(self) -> Taxonomy:
        return self._snapshot.taxonomy

    @property
    def config(self) -> TreeClassificationConfig:
        return self._config

    async def classify(
        self,
        query: str | Query,
        *,
        outputs: Iterable[str] | None = None,
    ) -> TreeClassificationResult:
        query = as_query(query)
        requested = tuple(self._config.outputs) if outputs is None else tuple(outputs)
        if len(requested) != len(set(requested)):
            raise ValueError("requested output names must be unique")
        unknown = set(requested) - set(self._config.outputs)
        if unknown:
            raise ValueError(f"unknown outputs: {sorted(unknown)!r}")

        deciders = {spec.name: spec for spec in self._config.deciders}
        routers = {spec.name: spec for spec in self._config.routers}
        needed_deciders = set()
        for output_name in requested:
            target = self._config.outputs[output_name]
            if target in routers:
                needed_deciders.update(routers[target].deciders)
            else:
                needed_deciders.add(target)
        needed_nodes = self._dependency_closure(
            {deciders[name].input for name in needed_deciders}
        )

        prepared = {}
        node_scores: dict[str, NodeScores] = {}
        path_rankings: dict[str, Ranking[TaxonomyPath]] = {}
        for name in self._dag.order:
            if name not in needed_nodes or name in self._preprocessor_specs:
                continue
            if name in self._node_scorer_specs:
                spec = self._node_scorer_specs[name]
                if spec.preprocessor not in prepared:
                    raw = await resolve(
                        self._preprocessors[spec.preprocessor].prepare_query(query.text)
                    )
                    preprocessor = self._preprocessor_specs[spec.preprocessor]
                    prepared[spec.preprocessor] = (
                        validate_vector(
                            raw,
                            dimensions=preprocessor.dimensions,
                            label="query embedding",
                        )
                        if isinstance(preprocessor, EmbeddingPreprocessorConfig)
                        else validate_tokens(raw, label="query tokens")
                    )
                representation = prepared[spec.preprocessor]
                if isinstance(spec, DenseNodeScorerConfig):
                    node_scores[name] = score_nodes(
                        self._snapshot,
                        representation,
                        similarity=spec.similarity,
                    )
                else:
                    node_scores[name] = score_bm25_nodes(
                        self._snapshot,
                        self._bm25_index,
                        representation,
                        spec,
                    )
            elif name in self._node_fusion_specs:
                spec = self._node_fusion_specs[name]
                rankings = {
                    source: rank_node_scores(
                        self._snapshot,
                        node_scores[source],
                        source=source,
                    )
                    for source in spec.inputs
                }
                fused = fuse_rankings(
                    rankings,
                    method=spec.type,
                    top_k=len(self._snapshot.nodes_by_id),
                    weights=_weights(spec),
                    normalization=spec.normalization,
                    k=spec.k,
                    normalize_missing_as_zero=True,
                )
                node_scores[name] = {item.id: item.score for item in fused}
            elif name in self._path_scorer_specs:
                spec = self._path_scorer_specs[name]
                scores = {}
                for path in self._snapshot.paths:
                    values = tuple(
                        node_scores[spec.input].get(node_id, 0.0)
                        for node_id in path.node_ids
                    )
                    if isinstance(spec, CustomPathScorerConfig):
                        raw = await resolve(
                            self._path_scorers[name].score_path(path, values)
                        )
                        scores[path.id] = validate_path_score(raw, path=path)
                    else:
                        scores[path.id] = builtin_path_score(path, values, spec)
                path_rankings[name] = rank_paths(
                    self._snapshot, scores, source=spec.name
                )
            else:
                spec = self._path_fusion_specs[name]
                path_rankings[name] = fuse_rankings(
                    {source: path_rankings[source] for source in spec.inputs},
                    method=spec.type,
                    top_k=len(self._snapshot.paths),
                    weights=_weights(spec),
                    normalization=spec.normalization,
                    k=spec.k,
                )

        def ranking_for(decider_name: str, *, full: bool = False):
            input_name = deciders[decider_name].input
            ranking = path_rankings[input_name]
            if full:
                return ranking
            producer = self._path_scorer_specs.get(input_name)
            if producer is None:
                producer = self._path_fusion_specs[input_name]
            return ranking.top(producer.top_k)

        decisions = {}
        routes = {}
        result = {}
        for output_name in requested:
            target = self._config.outputs[output_name]
            if target in routers:
                if target not in routes:
                    candidates = {
                        name: ranking_for(name) for name in routers[target].deciders
                    }
                    routes[target] = await route_decision(
                        routers[target], self._routers[target], query, candidates
                    )
                decider_name = routes[target]
            else:
                decider_name = target
            if decider_name not in decisions:
                spec = deciders[decider_name]
                full = isinstance(spec, LLMDeciderConfig)
                ranking = ranking_for(decider_name, full=full)
                selection = (
                    await decide_with_llm(
                        spec, self._deciders[decider_name], query, ranking
                    )
                    if full
                    else decide(spec, ranking)
                )
                decisions[decider_name] = materialize_decision(spec, selection, ranking)
            result[output_name] = decisions[decider_name]
        return TreeClassificationResult(
            query.id,
            self._snapshot.taxonomy.id,
            MappingProxyType(result),
        )

    def _dependency_closure(self, terminals: set[str]) -> set[str]:
        needed = set()

        def visit(name: str) -> None:
            if name in needed:
                return
            needed.add(name)
            for dependency in self._dag.dependencies[name]:
                visit(dependency)

        for terminal in terminals:
            visit(terminal)
        return needed
