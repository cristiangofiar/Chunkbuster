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
from .config import TreeClassificationConfig
from .decisions import decide
from .models import Taxonomy, TreeClassificationResult
from .scoring import score_paths, validate_vector
from .taxonomy import TaxonomySnapshot, build_snapshot


class TreeClassificationPipeline:
    """Immutable dense -> mean path -> deterministic decision pipeline."""

    def __init__(
        self,
        *,
        snapshot: TaxonomySnapshot,
        config: TreeClassificationConfig,
        preprocessor: object,
    ) -> None:
        self._snapshot = snapshot
        self._config = config
        self._preprocessor = preprocessor

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

        deciders = {spec.name: spec for spec in parsed.deciders}
        if not deciders or not parsed.outputs:
            raise BuildError("the pipeline requires at least one decider and output")
        if len(deciders) != len(parsed.deciders):
            raise BuildError("decider names must be unique")
        for spec in parsed.deciders:
            if spec.input != path_scorer.name:
                raise BuildError(f"decider {spec.name!r} references an unknown ranking")
        missing_outputs = set(parsed.outputs.values()) - set(deciders)
        if missing_outputs:
            raise BuildError(
                f"outputs reference unknown deciders {sorted(missing_outputs)!r}"
            )
        unused_deciders = set(deciders) - set(parsed.outputs.values())
        if unused_deciders:
            raise BuildError(f"unused deciders: {sorted(unused_deciders)!r}")

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
        return cls(snapshot=snapshot, config=parsed, preprocessor=preprocessor)

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
        ranking = score_paths(
            self._snapshot,
            vector,
            similarity=node_scorer.similarity,
            top_k=path_scorer.top_k,
            source=node_scorer.name,
        )
        requested = tuple(self._config.outputs) if outputs is None else tuple(outputs)
        unknown = set(requested) - set(self._config.outputs)
        if unknown:
            raise ValueError(f"unknown outputs: {sorted(unknown)!r}")
        deciders = {spec.name: spec for spec in self._config.deciders}
        result = {
            output_name: decide(deciders[self._config.outputs[output_name]], ranking)
            for output_name in requested
        }
        return TreeClassificationResult(
            query.id,
            self._snapshot.taxonomy.id,
            MappingProxyType(result),
        )
