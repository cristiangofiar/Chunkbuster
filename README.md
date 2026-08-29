# chunkbuster

`chunkbuster` es una librería Python experimental para construir pipelines de
clasificación jerárquica y retrieval mediante configuración estricta y
componentes Python inyectados explícitamente.

La API todavía puede cambiar. Esta primera entrega implementa dos verticales
pequeños de extremo a extremo:

| Producto | Implementado ahora |
|---|---|
| `TreeClassificationPipeline` | Bosque estricto, embeddings densos, score medio por path y deciders `top_one`, `top_k` y `threshold` |
| `RetrievalPipeline` | Retrievers fuente, retrievers restringidos por candidatos, fusión RRF y múltiples outputs |

No se incluyen SDKs, modelos, vectorstores ni credenciales. Esos objetos se
inyectan mediante `ComponentBindings`; la configuración solo describe cómo se
conectan.

## Instalación local

Requiere Python 3.12 o superior.

Con uv:

```bash
uv sync --dev
uv run pytest
```

Con pip:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
pytest
```

El paquete distribuible vive bajo `src/chunkbuster`.

## Tree classification

Una taxonomía es un bosque: puede tener varias raíces, pero cada nodo no raíz
tiene exactamente un padre. Cada hoja define una clase identificada por su path
completo desde la raíz.

Este ejemplo deja que `build()` genere los embeddings ausentes una sola vez:

```python
import asyncio

from chunkbuster import (
    ComponentBindings,
    Taxonomy,
    TaxonomyEdge,
    TaxonomyNode,
    TreeClassificationPipeline,
)


class FakeEmbeddings:
    vectors = {
        "Productos": (1.0, 0.0),
        "Reembolsos": (1.0, 0.0),
        "Acceso": (0.0, 1.0),
    }

    async def prepare_documents(self, texts):
        return tuple(self.vectors[text] for text in texts)

    async def prepare_query(self, text):
        return (1.0, 0.0)


taxonomy = Taxonomy(
    id="support",
    nodes=(
        TaxonomyNode("products", "Productos"),
        TaxonomyNode("refunds", "Reembolsos"),
        TaxonomyNode("access", "Acceso"),
    ),
    edges=(
        TaxonomyEdge("products", "refunds"),
        TaxonomyEdge("products", "access"),
    ),
)

config = {
    "version": 1,
    "name": "support_classifier",
    "kind": "tree_classification",
    "preprocessors": [
        {
            "name": "semantic",
            "type": "embedding",
            "binding": "fake_embeddings",
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
        {"name": "mean_paths", "type": "mean", "input": "dense_nodes", "top_k": 10}
    ],
    "deciders": [
        {"name": "best", "type": "top_one", "input": "mean_paths"}
    ],
    "outputs": {"primary": "best"},
}


async def main():
    pipeline = await TreeClassificationPipeline.build(
        taxonomy=taxonomy,
        config=config,
        bindings=ComponentBindings(
            preprocessors={"fake_embeddings": FakeEmbeddings()}
        ),
    )
    result = await pipeline.classify("¿Dónde está mi devolución?")
    print(result.outputs["primary"].selected[0].item.node_ids)


asyncio.run(main())  # ('products', 'refunds')
```

Si cada nodo ya trae `embedding`, `prepare_documents()` no se invoca. La
cobertura debe ser completa: mezclar nodos con y sin embedding falla durante
`build()`.

## Retrieval

Un retriever fuente introduce chunks. Un retriever con `input` solo puede
filtrar o reordenar los candidatos recibidos. Las fusiones RRF combinan dos o
más rankings por identidad.

```python
import asyncio

from chunkbuster import ComponentBindings, RankedItem, Ranking
from chunkbuster.retrieval import Chunk, RetrievalPipeline


def ranking(*ids):
    return Ranking(
        tuple(
            RankedItem(item_id, Chunk(item_id, f"chunk {item_id}"), 1.0 / rank)
            for rank, item_id in enumerate(ids, 1)
        )
    )


class FakeQueryPreprocessor:
    async def prepare_query(self, text):
        return text.casefold()


class FakeSource:
    def __init__(self, result):
        self.result = result

    async def retrieve(self, query, *, top_k):
        return self.result.top(top_k)


class FakeCandidateFilter:
    async def retrieve_candidates(self, query, candidates, *, top_k):
        return Ranking(tuple(item for item in candidates if item.id == "b"))


config = {
    "version": 1,
    "name": "products",
    "kind": "retrieve",
    "preprocessors": [{"name": "shared", "binding": "query_text"}],
    "retrievers": [
        {
            "name": "catalog_a",
            "binding": "source_a",
            "preprocessor": "shared",
            "top_k": 10,
        },
        {
            "name": "catalog_b",
            "binding": "source_b",
            "preprocessor": "shared",
            "top_k": 10,
        },
        {
            "name": "filtered_a",
            "binding": "candidate_filter",
            "preprocessor": "shared",
            "input": "catalog_a",
            "top_k": 5,
        },
    ],
    "fusions": [
        {
            "name": "combined",
            "type": "rrf",
            "inputs": ["filtered_a", "catalog_b"],
            "k": 60,
            "top_k": 3,
        }
    ],
    "outputs": {"primary": "combined"},
}


async def main():
    pipeline = await RetrievalPipeline.build(
        config=config,
        bindings=ComponentBindings(
            preprocessors={"query_text": FakeQueryPreprocessor()},
            retrievers={
                "source_a": FakeSource(ranking("a", "b")),
                "source_b": FakeSource(ranking("b", "c")),
                "candidate_filter": FakeCandidateFilter(),
            },
        ),
    )
    result = await pipeline.retrieve("productos")
    print(result.outputs["primary"].ranking.ids)


asyncio.run(main())  # ('b', 'c')
```

## Configuración y bindings

`build(config=...)` acepta el mismo esquema como:

- `dict` o cualquier `Mapping` de Python;
- ruta `.yaml` o `.yml`;
- ruta `.json`;
- modelo Pydantic de configuración ya validado.

Las claves desconocidas fallan. Un archivo describe un pipeline y sus
referencias; no contiene código, clientes ni secretos. El campo `binding`
resuelve el objeto concreto dentro de `ComponentBindings`.

Los adapters pueden ser síncronos o asíncronos. Deben devolver los objetos
tipados de la librería. Un retriever restringido no puede inventar IDs fuera de
su input.

## Limitaciones actuales

Esta entrega es deliberadamente pequeña. Todavía no implementa:

- BM25 ni tokenización;
- node fusions o ranking fusions en Tree;
- rerankers;
- deciders/selectors LLM;
- ejecución concurrente o límites de concurrencia;
- aislamiento de fallos por output;
- integraciones con proveedores o vectorstores;
- persistencia, índices incrementales, CLI o servidor HTTP.

Consulte [ARCHITECTURE.md](ARCHITECTURE.md) para contratos internos, estado del
diseño y próximos verticales.
