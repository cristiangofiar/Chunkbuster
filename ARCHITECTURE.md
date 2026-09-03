# Arquitectura

Este documento describe el paquete bajo `src/chunkbuster`; el repositorio ya no
conserva la implementación del prototipo anterior.

## Estado de esta entrega

La primera entrega implementa dos recorridos completos, deliberadamente
estrechos:

```text
Tree:
query -> embedding -> dense node scores -> configurable path ranking -> optional router -> decider -> output

Retrieval:
query -> preprocessor -> source/candidate retrievers -> optional RRF -> output
```

| Área | Disponible |
|---|---|
| Tree taxonomy | Bosque estricto, múltiples raíces y paths raíz-hoja exactos |
| Tree scoring | Un scorer dense y path scoring `mean`, `weighted_sum` o `custom` |
| Tree decisions | Routers terminales, `top_one`, `top_k`, `threshold`, `llm` y outputs nombrados |
| Retrieval | Retrievers fuente y restringidos, DAG de rankings, RRF y outputs nombrados |
| Configuración | YAML, JSON, mapping o modelo Pydantic |
| Adapters | Objetos Python explícitos, síncronos o asíncronos |

No están implementados todavía BM25, fusiones de nodos/paths en Tree,
rerankers, scheduling concurrente ni aislamiento de fallos por output.
Los diseños futuros no deben documentarse como capacidades actuales.

## Estructura real

```text
src/chunkbuster/
├── __init__.py
├── errors.py
├── py.typed
├── core/
│   ├── _async.py
│   ├── config.py
│   ├── contracts.py
│   ├── dag.py
│   ├── models.py
│   └── ranking.py
├── tree_classification/
│   ├── config.py
│   ├── decisions.py
│   ├── models.py
│   ├── pipeline.py
│   ├── scoring.py
│   └── taxonomy.py
└── retrieval/
    ├── config.py
    ├── models.py
    └── pipeline.py

tests/
├── core/
├── tree_classification/
└── retrieval/
```

`core` contiene solo contratos realmente usados por ambos productos.
`tree_classification` y `retrieval` dependen de `core`, pero no se importan
entre sí. El paquete no importa SDKs de proveedores.

No hay todavía `integrations/`, Registry, sistema de plugins ni runtime
genérico. Se crearán solo cuando exista un segundo consumidor real para la
abstracción correspondiente.

## Invariantes compartidos

1. La configuración es estricta, validada e inmutable.
2. Código, clientes, modelos y secretos se inyectan; nunca se deserializan.
3. Los datos estables se resuelven durante `build()` y cada query usa estado
   privado.
4. Los objetos de dominio son dataclasses frozen; metadata y resultados
   públicos exponen mappings de solo lectura.
5. Un `Ranking` tiene IDs únicos y orden determinista.
6. Una etapa restringida no puede inventar identidades fuera de su input.
7. La configuración se valida antes de ejecutar queries siempre que la
   propiedad pueda conocerse durante `build()`.
8. Se usa stdlib antes de incorporar una dependencia o framework.

## Core

### Query y configuración

`Query` contiene texto, ID opcional y metadata. Tanto `classify()` como
`retrieve()` aceptan también un `str`, que se normaliza a `Query`.

`load_config()` carga:

- un `Mapping`/`dict`;
- una ruta `.yaml`, `.yml` o `.json`;
- un modelo Pydantic ya validado.

Cada producto tiene su propio modelo Pydantic estricto. `extra="forbid"`
evita errores silenciosos y los modelos son frozen. No hay una configuración
universal ni deep-merge de overrides por request.

### Bindings

`ComponentBindings` contiene mappings de objetos concretos:

```text
preprocessors
retrievers
```

La clave del mapping coincide con `binding` en la configuración. El objeto no
repite su nombre y no debe heredar de una clase base: basta con implementar el
método esperado. `_async.resolve()` permite implementaciones sync o async sin
duplicar la API pública.

### Ranking

`RankedItem[T]` conserva:

- `id` canónico;
- item original;
- score finito;
- rank normalizado;
- provenance y metadata.

`Ranking[T]` es una tupla ordenada e inmutable. Al construirla se rechazan IDs
duplicados y se reasignan ranks desde uno. `require_subset()` canonicaliza el
item contra el input y rechaza IDs inventados.

La primera estrategia compartida es RRF:

```text
score(id) = sum(1 / (k + rank_en_cada_input))
```

Opera por identidad, no compara escalas originales y resuelve empates por el
primer orden observado. Requiere al menos dos rankings y aplica su propio
`top_k`.

### DAG

`compile_dag()` recibe `nombre -> dependencias` y outputs terminales. Usa
`graphlib.TopologicalSorter` para:

- resolver referencias;
- rechazar inputs repetidos;
- detectar ciclos;
- calcular alcanzabilidad desde outputs;
- rechazar nodos declarados pero no utilizados.

No ejecuta negocio ni interpreta tipos. Hoy lo usa Retrieval, donde todos los
nodos producen `Ranking[Chunk]`. Tree valida directamente su único recorrido;
generalizar antes de tener fusiones reales añadiría complejidad sin uso.

## TreeClassificationPipeline

### Dominio e identidad

`Taxonomy` contiene nodos y aristas planos. Un `TaxonomyNode` tiene `id`,
`label`, `text`, `embedding`, `tokens` y metadata. En esta entrega solo se usa
`embedding`; `tokens` queda como dato de dominio para el siguiente vertical.

La estructura es un bosque estricto:

- una o más raíces;
- cada nodo no raíz tiene exactamente un padre;
- sin self-loops, ciclos ni endpoints desconocidos;
- una raíz sin hijos es un subárbol válido;
- todos los nodos son alcanzables desde alguna raíz.

`taxonomy.py` usa mapas parent/children y DFS de stdlib. Enumera una vez cada
path raíz-hoja en orden de entrada. La identidad de `TaxonomyPath` es la
secuencia completa de IDs serializada de forma estable, aunque una hoja sea
unívoca en el bosque actual. Esto preserva explicabilidad y no implementa aún
una taxonomía DAG.

### Build

El vertical actual exige exactamente:

- un preprocessor de embeddings;
- un node scorer dense;
- un path scorer `mean`, `weighted_sum` o `custom`;
- uno o más deciders alcanzables directa o indirectamente desde `outputs`;
- cero o más routers terminales que solo pueden seleccionar deciders.

Si todos los nodos traen embeddings, se validan dimensión y finitud. Si ninguno
los trae, `build()` llama una vez a
`preprocessor.prepare_documents(tuple[text, ...])` y guarda los vectores en un
snapshot nuevo. Una cobertura parcial falla para evitar mezclar modelos o
versiones. El `Taxonomy` entregado por el caller nunca se muta.

El preprocessor siempre debe implementar `prepare_query(text)`. El usuario es
responsable de inyectar el mismo modelo, versión, dimensión y normalización que
produjo los embeddings de los nodos; la librería solo puede comprobar sus
propiedades numéricas.

### Classify

Para cada query:

1. se genera un embedding;
2. se puntúa cada nodo con cosine, dot product o euclidean negativo;
3. cada path recibe la media, una suma ponderada configurable o el score de un
   binding `custom`;
4. el ranking se ordena determinísticamente y se recorta;
5. cada output solicitado resuelve su router opcional;
6. se ejecuta el decider terminal elegido.

`top_one` selecciona como máximo uno. `top_k` selecciona hasta `count`.
`threshold` selecciona todos los candidatos con `score >= min_score`, con un
`count` opcional. `llm` llama a `decide(query, ranking, count=...)` sobre todos
los paths puntuados, aunque queden fuera del `top_k` del path scorer. El adapter
devuelve obligatoriamente un `DecisionSelection`; el pipeline valida identidad,
unicidad y límite y materializa los paths canónicos. `reason` y `metadata` se
propagan a `ClassificationDecision`. Una selección vacía produce
`status="abstained"`.

Un router llama a `route(query, ranking)` con el ranking ya recortado por el
path scorer y devuelve un nombre de decider o un `DecisionRoute`. El pipeline
comprueba que el destino pertenezca a `RouterConfig.deciders` antes de
ejecutarlo. Routers y decisiones se cachean por query, por lo que outputs
compartidos no repiten bindings. Los routers no pueden seleccionar otros
routers, ejecutar componentes ni transformar rankings.

`weighted_sum` admite `root`, niveles posteriores a la raíz mediante `level_n`,
`mean`, `leaf`, `lowest` y `highest`; la suma no normaliza pesos. `custom`
resuelve un binding de `ComponentBindings.path_scorers` con
`score_path(path, node_scores)`. Los deciders LLM se resuelven desde
`ComponentBindings.deciders`; los routers, desde `ComponentBindings.routers`.
Ambos pueden ser síncronos o asíncronos.

`classify(query, outputs=...)` puede materializar solo un subconjunto de outputs
públicos, pero no cambia configuración ni scoring.

## RetrievalPipeline

### Grafo de rankings

La configuración declara preprocessors, retrievers, fusiones y outputs. Los
nombres de retrievers y fusiones comparten namespace. Cada retriever referencia
un preprocessor y un binding explícitos.

Un retriever sin `input` es fuente:

```text
retrieve(query_representation, *, top_k) -> Ranking[Chunk]
```

Un retriever con `input` es restringido:

```text
retrieve_candidates(query_representation, candidates, *, top_k)
    -> Ranking[Chunk]
```

El segundo solo puede devolver IDs presentes en `candidates`. El pipeline
reemplaza cualquier copia del chunk por el objeto canónico de entrada y aplica
`top_k`.

Una fusion `type: rrf` consume al menos dos rankings nombrados. “Hybrid” no es
un tipo especial: es simplemente una RRF entre señales diferentes. “Cascade”
es un retriever cuyo `input` referencia otro ranking.

### Build y retrieve

`build()` valida nombres, preprocessors usados, bindings, métodos requeridos,
referencias, outputs, nodos muertos y ciclos. No consulta stores ni ejecuta
retrievers.

En `retrieve()` los nodos se recorren secuencialmente en orden topológico. Cada
preprocessor se memoiza por nombre y se ejecuta como máximo una vez por query,
aunque lo compartan varios retrievers. Los rankings intermedios viven solo en
esa llamada.

Cada output publica `terminal_node`, `ranking` y un status derivado
`completed|empty`. `retrieve(query, outputs=...)` puede seleccionar un subset
de grupos públicos.

Esta ejecución es deliberadamente secuencial. Todavía no hay scheduler de
nodos listos, semáforos ni resultados `failed`: cualquier error aborta la
llamada completa.

## Fronteras externas

Adapters y retrievers son trust boundaries. El pipeline valida forma,
dimensiones, finitud, tipo `Ranking`, unicidad y subconjuntos cuando corresponde.
Las implementaciones del usuario son responsables de timeouts, retries,
credenciales, rate limits y seguridad del cliente externo.

La jerarquía pública de errores se mantiene corta:

```text
ChunkbusterError
├── ConfigurationError
├── BuildError
│   └── InvalidTaxonomyError
└── ExecutionError
    ├── PreprocessingError
    ├── RetrievalError
    └── InvalidModelOutputError
```

No se crean excepciones por cada clase interna.

## Tests y política de cambios

Los tests nuevos cubren contratos, no detalles accidentales:

- unicidad y scores finitos de rankings;
- RRF determinista;
- bosque, múltiples raíces, ciclos y múltiples padres;
- embeddings provistos o generados una sola vez;
- path scoring ponderado/custom y selección LLM canonicalizada;
- carga equivalente desde dict, YAML y JSON;
- source/candidate retrieval, preprocessing compartido y RRF;
- rechazo de IDs inventados y ciclos del DAG.

Una nueva capacidad debe añadir primero el test mínimo que demuestra el
vertical completo. Las abstracciones compartidas se extraen cuando ambos
productos ya exhiben el mismo contrato, no por anticipación.

## Próximos verticales

Orden orientativo, sin promesa de compatibilidad hasta estabilizar la API:

1. Tree lexical: tokenización/BM25, node fusions y ranking fusions.
2. Transformaciones externas: rerankers y canonicalización común.
3. Selector LLM de rankings para Retrieval.
4. Runtime: nodos independientes concurrentes, límites y aislamiento de fallos
   por output.
5. Integraciones concretas con embeddings y vectorstores, una por vez.

Siguen fuera de alcance hasta que exista un caso medido: plugin discovery,
workflow engine genérico, taxonomía DAG, persistencia propia, CLI, servidor
HTTP, cache distribuido y compatibilidad con la API del prototipo.
