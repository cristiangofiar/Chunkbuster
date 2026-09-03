"""Build and run the first complete tree-classification vertical."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from types import MappingProxyType

from ..core._async import resolve
from ..core.config import ConfigInput, load_config
from ..core.contracts import ComponentBindings
from ..core.models import Query, as_query
from ..errors import BuildError, PreprocessingError
from .config import CustomPathScorerConfig, LLMDeciderConfig, TreeClassificationConfig
from .decisions import decide, decide_with_llm, materialize_decision, route_decision
from .models import Taxonomy, TreeClassificationResult
from .scoring import (
    builtin_path_score,
    rank_paths,
    score_nodes,
    validate_path_score,
    validate_vector,
)
from .taxonomy import TaxonomySnapshot, build_snapshot


class TreeClassificationPipeline:
    """Immutable configurable tree-classification pipeline."""

    def __init__(
        self,
        *,
        snapshot: TaxonomySnapshot,
        config: TreeClassificationConfig,
        preprocessor: object,
        path_scorer: object | None,
        deciders: dict[str, object],
        routers: dict[str, object],
    ) -> None:
        self._snapshot = snapshot
        self._config = config
        self._preprocessor = preprocessor
        self._path_scorer = path_scorer
        self._deciders = MappingProxyType(dict(deciders))
        self._routers = MappingProxyType(dict(routers))

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
        if len(parsed.preprocessors) != 1:
            raise BuildError("the first vertical requires exactly one preprocessor")
        if len(parsed.node_scorers) != 1:
            raise BuildError("the first vertical requires exactly one node scorer")
        if len(parsed.path_scorers) != 1:
            raise BuildError("the first vertical requires exactly one path scorer")

        preprocessor_spec = parsed.preprocessors[0]
        node_scorer = parsed.node_scorers[0]
        path_scorer = parsed.path_scorers[0]
        if node_scorer.preprocessor != preprocessor_spec.name:
            raise BuildError("node scorer references an unknown preprocessor")
        if path_scorer.input != node_scorer.name:
            raise BuildError("path scorer must consume the configured node scorer")

        bound_path_scorer = None
        if isinstance(path_scorer, CustomPathScorerConfig):
            try:
                bound_path_scorer = bindings.path_scorers[path_scorer.binding]
            except KeyError as exc:
                raise BuildError(
                    f"missing path scorer binding {path_scorer.binding!r}"
                ) from exc
            if not callable(getattr(bound_path_scorer, "score_path", None)):
                raise BuildError(
                    "custom path scorer must define score_path(path, scores)"
                )

        deciders = {spec.name: spec for spec in parsed.deciders}
        routers = {spec.name: spec for spec in parsed.routers}
        if not deciders or not parsed.outputs:
            raise BuildError("the pipeline requires at least one decider and output")
        if len(deciders) != len(parsed.deciders):
            raise BuildError("decider names must be unique")
        if len(routers) != len(parsed.routers):
            raise BuildError("router names must be unique")
        collisions = set(deciders) & set(routers)
        if collisions:
            raise BuildError(
                f"router and decider names collide: {sorted(collisions)!r}"
            )
        for spec in parsed.deciders:
            if spec.input != path_scorer.name:
                raise BuildError(f"decider {spec.name!r} references an unknown ranking")
        for spec in parsed.routers:
            missing_deciders = set(spec.deciders) - set(deciders)
            if missing_deciders:
                raise BuildError(
                    f"router {spec.name!r} references unknown deciders "
                    f"{sorted(missing_deciders)!r}"
                )
            inputs = {deciders[name].input for name in spec.deciders}
            if len(inputs) != 1:
                raise BuildError(
                    f"router {spec.name!r} deciders must share one input ranking"
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

        bound_deciders: dict[str, object] = {}
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

        bound_routers: dict[str, object] = {}
        for spec in parsed.routers:
            try:
                component = bindings.routers[spec.binding]
            except KeyError as exc:
                raise BuildError(f"missing router binding {spec.binding!r}") from exc
            if not callable(getattr(component, "route", None)):
                raise BuildError("router binding must define route()")
            bound_routers[spec.name] = component

        try:
            preprocessor = bindings.preprocessors[preprocessor_spec.binding]
        except KeyError as exc:
            raise BuildError(
                f"missing preprocessor binding {preprocessor_spec.binding!r}"
            ) from exc
        if not callable(getattr(preprocessor, "prepare_query", None)):
            raise BuildError("embedding preprocessor must define prepare_query(text)")

        snapshot = build_snapshot(taxonomy)
        present = [node.embedding is not None for node in taxonomy.nodes]
        if any(present) and not all(present):
            raise BuildError("taxonomy embeddings must be complete or entirely absent")
        if not any(present):
            prepare_documents = getattr(preprocessor, "prepare_documents", None)
            if not callable(prepare_documents):
                raise BuildError(
                    "missing taxonomy embeddings require prepare_documents(texts)"
                )
            raw_vectors = await resolve(
                prepare_documents(tuple(node.text for node in taxonomy.nodes))
            )
            try:
                vectors = tuple(raw_vectors)
            except TypeError as exc:
                raise PreprocessingError(
                    "prepare_documents must return one vector per node"
                ) from exc
            if len(vectors) != len(taxonomy.nodes):
                raise PreprocessingError(
                    "prepare_documents must return one vector per node"
                )
            nodes = tuple(
                replace(
                    node,
                    embedding=validate_vector(
                        vector,
                        dimensions=preprocessor_spec.dimensions,
                        label=f"embedding for node {node.id!r}",
                    ),
                )
                for node, vector in zip(taxonomy.nodes, vectors, strict=True)
            )
            snapshot = build_snapshot(replace(taxonomy, nodes=nodes))
        else:
            for node in taxonomy.nodes:
                validate_vector(
                    node.embedding,
                    dimensions=preprocessor_spec.dimensions,
                    label=f"embedding for node {node.id!r}",
                )
        return cls(
            snapshot=snapshot,
            config=parsed,
            preprocessor=preprocessor,
            path_scorer=bound_path_scorer,
            deciders=bound_deciders,
            routers=bound_routers,
        )

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
        preprocessor = self._config.preprocessors[0]
        raw_vector = await resolve(self._preprocessor.prepare_query(query.text))
        vector = validate_vector(
            raw_vector,
            dimensions=preprocessor.dimensions,
            label="query embedding",
        )
        node_scorer = self._config.node_scorers[0]
        path_scorer = self._config.path_scorers[0]
        node_scores = score_nodes(
            self._snapshot,
            vector,
            similarity=node_scorer.similarity,
        )
        path_scores: dict[str, float] = {}
        for path in self._snapshot.paths:
            scores = tuple(node_scores[node_id] for node_id in path.node_ids)
            if isinstance(path_scorer, CustomPathScorerConfig):
                raw_score = await resolve(self._path_scorer.score_path(path, scores))
                path_scores[path.id] = validate_path_score(raw_score, path=path)
            else:
                path_scores[path.id] = builtin_path_score(path, scores, path_scorer)
        full_ranking = rank_paths(
            self._snapshot,
            path_scores,
            source=path_scorer.name,
        )
        ranking = full_ranking.top(path_scorer.top_k)
        requested = tuple(self._config.outputs) if outputs is None else tuple(outputs)
        unknown = set(requested) - set(self._config.outputs)
        if unknown:
            raise ValueError(f"unknown outputs: {sorted(unknown)!r}")
        deciders = {spec.name: spec for spec in self._config.deciders}
        routers = {spec.name: spec for spec in self._config.routers}
        decisions = {}
        routes = {}
        result = {}
        for output_name in requested:
            target_name = self._config.outputs[output_name]
            if target_name in routers:
                if target_name not in routes:
                    routes[target_name] = await route_decision(
                        routers[target_name],
                        self._routers[target_name],
                        query,
                        ranking,
                    )
                decider_name = routes[target_name]
            else:
                decider_name = target_name
            if decider_name not in decisions:
                spec = deciders[decider_name]
                if isinstance(spec, LLMDeciderConfig):
                    selection = await decide_with_llm(
                        spec,
                        self._deciders[decider_name],
                        query,
                        full_ranking,
                    )
                    decision_ranking = full_ranking
                else:
                    selection = decide(spec, ranking)
                    decision_ranking = ranking
                decisions[decider_name] = materialize_decision(
                    spec,
                    selection,
                    decision_ranking,
                )
            result[output_name] = decisions[decider_name]
        return TreeClassificationResult(
            query.id,
            self._snapshot.taxonomy.id,
            MappingProxyType(result),
        )
