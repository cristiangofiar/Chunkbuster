"""Render documentation SVGs from the YAML examples with explicit reading order.

Run with: uv run python scripts/render_doc_diagrams.py
The YAML owns nodes, bindings and edges; only panel/column placement lives here.
"""

from __future__ import annotations

import re
from html import escape
from pathlib import Path
from textwrap import wrap

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs" / "diagrams"
GROUPS = {
    "preprocessors": "Preprocessors",
    "retrievers": "Retrievers",
    "fusions": "Fusions",
    "node_scorers": "Node scorers",
    "node_fusions": "Node fusions",
    "path_scorers": "Path scorers",
    "path_fusions": "Path fusions",
    "deciders": "Deciders",
    "routers": "Routers",
    "outputs": "Outputs",
    "references": "Continuación",
}

# A panel is a title and columns. Names prefixed with @ repeat an upstream
# output as a continuation port, not as a second execution of that stage.
LAYOUTS = {
    "semantic_catalog_search": [
        (
            "Búsqueda semántica",
            [
                ["query_embedder"],
                ["catalog_vector_store"],
                ["primary"],
            ],
        ),
    ],
    "semantic_support_classifier": [
        (
            "Clasificación semántica",
            [
                ["taxonomy_query_embedder"],
                ["dense_node_scorer"],
                ["mean_path_scorer"],
                ["best_path_decider"],
                ["primary"],
            ],
        ),
    ],
    "reranked_catalog_search": [
        (
            "Recuperar y refinar",
            [
                ["query_embedder", "reranker_query_tokenizer"],
                ["catalog_vector_store"],
                ["cross_encoder_reranker"],
                ["primary"],
            ],
        ),
    ],
    "vector_ensemble_with_llm": [
        (
            "Recuperar y fusionar",
            [
                ["query_embedder"],
                ["catalog_vector_store", "local_vector_search"],
                ["vector_rank_fusion"],
            ],
        ),
        (
            "Refinar y seleccionar",
            [
                ["@vector_rank_fusion"],
                ["refinement_query_tokenizer"],
                ["cross_encoder_reranker"],
                ["llm_chunk_selector"],
                ["reranked", "primary"],
            ],
        ),
    ],
    "precise_hybrid_search": [
        (
            "Recuperar con dense y sparse",
            [
                ["query_embedder", "sparse_query_tokenizer"],
                ["catalog_vector_store", "local_vector_search", "bm25_lexical_search"],
                ["hybrid_rank_fusion", "calibrated_score_fusion"],
            ],
        ),
        (
            "Reranking y consenso",
            [
                [
                    "@hybrid_rank_fusion",
                    "@calibrated_score_fusion",
                    "@sparse_query_tokenizer",
                ],
                ["cross_encoder_reranker"],
                ["consensus_fusion"],
                ["reranked"],
            ],
        ),
        (
            "Selección final con LLM",
            [
                ["@consensus_fusion", "@sparse_query_tokenizer"],
                ["llm_chunk_selector"],
                ["primary", "consensus"],
            ],
        ),
    ],
    "semantic_path_strategies": [
        (
            "Dos criterios sobre la misma señal",
            [
                ["taxonomy_query_embedder"],
                ["dense_node_scorer"],
                ["mean_path_scorer", "leaf_weighted_path_scorer"],
                ["best_mean_path_decider", "confident_leaf_path_decider"],
                ["primary", "confident"],
            ],
        ),
    ],
    "hybrid_support_classifier": [
        (
            "Puntuar y fusionar nodos",
            [
                ["taxonomy_query_embedder", "taxonomy_query_tokenizer"],
                ["dense_node_scorer", "bm25_node_scorer"],
                ["hybrid_node_score_fusion"],
            ],
        ),
        (
            "Construir paths y decidir",
            [
                [
                    "@dense_node_scorer",
                    "@bm25_node_scorer",
                    "@hybrid_node_score_fusion",
                ],
                [
                    "dense_mean_path_scorer",
                    "bm25_mean_path_scorer",
                    "hybrid_weighted_path_scorer",
                ],
                ["hybrid_path_rank_fusion"],
                ["best_hybrid_path_decider", "top_hybrid_paths_decider"],
                ["primary", "alternatives"],
            ],
        ),
    ],
    "adaptive_hybrid_classifier": [
        (
            "Scoring de nodos",
            [
                ["taxonomy_query_embedder", "taxonomy_query_tokenizer"],
                ["dense_node_scorer", "bm25_node_scorer"],
                [
                    "weighted_node_score_fusion",
                    "ranked_node_fusion",
                    "consensus_node_fusion",
                ],
            ],
        ),
        (
            "Scoring de paths",
            [
                [
                    "@dense_node_scorer",
                    "@ranked_node_fusion",
                    "@weighted_node_score_fusion",
                    "@consensus_node_fusion",
                ],
                [
                    "semantic_mean_path_scorer",
                    "ranked_mean_path_scorer",
                    "hybrid_weighted_path_scorer",
                    "business_rule_path_scorer",
                ],
                ["path_rank_fusion", "path_score_fusion", "path_consensus_fusion"],
            ],
        ),
        (
            "Deciders, routing y outputs",
            [
                ["@path_rank_fusion", "@path_score_fusion", "@path_consensus_fusion"],
                [
                    "fast_top_one_decider",
                    "shortlist_decider",
                    "confidence_decider",
                    "llm_path_decider",
                ],
                ["adaptive_decision_router"],
                ["primary", "shortlist", "confident"],
            ],
        ),
    ],
}

TITLES = {
    "query_embedder": "Embedder de query",
    "taxonomy_query_embedder": "Embedder de query",
    "reranker_query_tokenizer": "Tokenizer del reranker",
    "refinement_query_tokenizer": "Tokenizer de modelos",
    "sparse_query_tokenizer": "Tokenizer sparse",
    "taxonomy_query_tokenizer": "Tokenizer sparse",
    "catalog_vector_store": "Vector store remota",
    "local_vector_search": "Búsqueda vectorial local",
    "bm25_lexical_search": "Búsqueda BM25",
    "cross_encoder_reranker": "Reranker cross-encoder",
    "llm_chunk_selector": "Selector LLM",
    "dense_node_scorer": "Similitud semántica",
    "bm25_node_scorer": "Relevancia BM25",
    "business_rule_path_scorer": "Regla de negocio",
    "adaptive_decision_router": "Elige un decider",
    "llm_path_decider": "Decisión LLM",
}


def label(spec):
    if spec["name"] in TITLES:
        return TITLES[spec["name"]]
    kind = spec.get("type", "")
    return {
        "weighted_rrf": "Fusión por posición",
        "comb_mnz": "Fusión por consenso",
        "weighted_sum": "Suma ponderada",
        "mean": "Promedio del path",
        "top_one": "Elegir el mejor path",
        "top_k": "Elegir varios paths",
        "threshold": "Aplicar umbral",
    }.get(kind, spec["name"])


def dependencies(config):
    nodes = {}
    edges = []
    for group in GROUPS:
        if group in ("references", "outputs"):
            continue
        for spec in config.get(group, []):
            nodes[spec["name"]] = dict(spec, group=group)
            sources = spec.get("inputs", []) or spec.get("deciders", [])
            sources = [sources] if isinstance(sources, str) else list(sources)
            for key in ("input", "preprocessor"):
                if key in spec:
                    sources.append(spec[key])
            edges.extend((source, spec["name"]) for source in sources)
    for name, source in config["outputs"].items():
        nodes[name] = {"name": name, "group": "outputs"}
        edges.append((source, name))
    return nodes, edges


def render(config):
    nodes, edges = dependencies(config)
    panels = LAYOUTS[config["name"]]
    all_names = [
        name
        for _, columns in panels
        for column in columns
        for name in column
        if not name.startswith("@")
    ]
    assert len(all_names) == len(set(all_names)) and set(all_names) == set(nodes)
    refs = {
        name[1:]
        for _, columns in panels
        for column in columns
        for name in column
        if name.startswith("@")
    }
    ref_ids = {
        name: f"R{i + 1}"
        for i, name in enumerate(name for name in all_names if name in refs)
    }
    width = 1240
    result = []
    drawn_edges = set()
    bound_nodes = set()

    def rect(x, y, w, h, fill="white", stroke="#475569", radius=8, extra=""):
        result.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'rx="{radius}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="1.5" {extra}/>'
        )

    def text(
        x, y, content, size=16, color="#243247", anchor="middle", weight=400, mono=False
    ):
        family = "monospace" if mono else "Arial, sans-serif"
        result.append(
            f'<text x="{x}" y="{y}" font-family="{family}" '
            f'font-size="{size}" fill="{color}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{escape(content)}</text>'
        )

    def line(points, binding=False):
        coords = " ".join(f"{x},{y}" for x, y in points)
        extra = 'stroke-dasharray="6 5"' if binding else ""
        result.append(
            f'<polyline points="{coords}" fill="none" '
            f'stroke="{"#537bad" if binding else "#334155"}" '
            f'stroke-width="{1.6 if binding else 2}" stroke-linejoin="round" '
            f'{extra} marker-end="url(#{"binding" if binding else "arrow"})"/>'
        )

    def identifier(x, y, name, max_chars=25):
        # Break at underscores while preserving the literal YAML name.
        parts = re.findall(r"[^_]+_?", name)
        rows = [""]
        for part in parts:
            if rows[-1] and len(rows[-1]) + len(part) > max_chars:
                rows.append("")
            rows[-1] += part
        for i, row in enumerate(rows):
            text(x, y + i * 16, row, 13, "#475569", mono=True)
        return len(rows)

    top = 8
    for p, (title, columns) in enumerate(panels):
        ncols = len(columns)
        pitch = 1120 / ncols
        card_w = min(228, pitch - 48)
        rows = max(map(len, columns))
        panel_h = 138 + rows * 128
        panel_bottom = top + panel_h
        actual = {
            name for column in columns for name in column if not name.startswith("@")
        }
        preps = [name for name in actual if nodes[name]["group"] == "preprocessors"]
        positions = {}
        rect(104, top, 1128, panel_h, "#fff9df", "#b5a66a", 20)
        product = (
            "RetrievalPipeline"
            if config["kind"] == "retrieve"
            else "TreeClassificationPipeline"
        )
        text(
            132,
            top + 30,
            f"{product}  ·  {p + 1:02d}  {title}",
            20,
            anchor="start",
            weight=600,
        )

        # Group adjacent columns of the same component type into one box.
        group_columns = []
        for c, column in enumerate(columns):
            groups = {
                "references" if name.startswith("@") else nodes[name]["group"]
                for name in column
            }
            assert len(groups) == 1
            group = groups.pop()
            if group_columns and group_columns[-1][0] == group:
                group_columns[-1][2] = c
            else:
                group_columns.append([group, c, c])
            for r, name in enumerate(column):
                x = 120 + c * pitch + (pitch - card_w) / 2
                y = top + 108 + r * 128
                output = not name.startswith("@") and nodes[name]["group"] == "outputs"
                positions[name.lstrip("@")] = (
                    x,
                    y + (16 if output else 0),
                    card_w,
                    60 if output else 100,
                    c,
                    r,
                )
        for group, first, last in group_columns:
            x = 120 + first * pitch + 8
            rect(
                x,
                top + 65,
                pitch * (last - first + 1) - 16,
                panel_h - 85,
                "#ffffff",
                "#d7d9d4",
                10,
            )
            text(x + 16, top + 89, GROUPS[group], 16, anchor="start", weight=600)

        # Query is outside the pipeline, as in the reference sketch.
        if preps:
            qy = top + 145
            rect(8, qy - 21, 76, 42, "#fff", "#334155", 14)
            text(46, qy + 6, "Query", 16)
            for prep in sorted(preps, key=lambda n: positions[n][-1]):
                x, y, _, _, _, row = positions[prep]
                channel = 96 + row * 8
                if positions[prep][4] == 0:
                    line([(84, qy), (channel, qy), (channel, y + 40), (x, y + 40)])
                else:
                    lane = top + 50
                    line(
                        [
                            (84, qy),
                            (96, qy),
                            (96, lane),
                            (x - 10, lane),
                            (x - 10, y + 40),
                            (x, y + 40),
                        ]
                    )

        panel_edges = [(s, t) for s, t in edges if t in actual and s in positions]
        for s, t in panel_edges:
            drawn_edges.add((s, t))
            sx, sy, sw, sh, sc, sr = positions[s]
            tx, ty, tw, th, tc, tr = positions[t]
            assert tc > sc, (title, s, t)
            target_sources = [source for source, target in panel_edges if target == t]
            source_targets = [target for source, target in panel_edges if source == s]
            start_y = sy + 34 + 10 * source_targets.index(t)
            end_y = ty + 30 + 12 * target_sources.index(s)
            if tc == sc + 1:
                mid = sx + sw + 12 + 6 * (sr % 3)
                line([(sx + sw, start_y), (mid, start_y), (mid, end_y), (tx, end_y)])
            else:
                # Long bypasses use the bottom corridor, never intervening cards.
                lane = (
                    panel_bottom
                    - 10
                    - 7
                    * sum(
                        1
                        for a, b in panel_edges[: panel_edges.index((s, t))]
                        if positions[b][4] > positions[a][4] + 1
                    )
                )
                exit_x = sx + sw + 10 + 5 * (sr % 3)
                entry_x = tx - 12 - 5 * (tr % 3)
                line(
                    [
                        (sx + sw, start_y),
                        (exit_x, start_y),
                        (exit_x, lane),
                        (entry_x, lane),
                        (entry_x, end_y),
                        (tx, end_y),
                    ]
                )

        for name, (x, y, w, h, _, _) in positions.items():
            spec = nodes[name]
            reference = name not in actual
            rect(
                x,
                y,
                w,
                h,
                "#f4f7fa" if reference else "#fff",
                "#94a3b8" if reference else "#596579",
                22 if spec["group"] == "outputs" else 9,
                f'data-node="{escape(name)}" data-reference="{str(reference).lower()}"',
            )
            title_text = label(spec)
            lines = wrap(title_text, width=int(w / 8.3), break_long_words=False)
            for i, row in enumerate(lines):
                text(x + w / 2, y + 23 + i * 18, row, 15, weight=600)
            id_y = y + 26 + len(lines) * 18
            if spec["group"] != "outputs":
                identifier(x + w / 2, id_y, name, int(w / 8.1))
            kind = spec.get("type", "")
            if kind:
                norm = spec.get("normalization", "none")
                subtitle = kind + (f" · {norm}" if norm != "none" else "")
                text(x + w / 2, y + h - 9, subtitle, 11, "#64748b")
            if name in refs:
                rect(x + w - 31, y - 10, 36, 21, "#eef2f6", "#94a3b8", 9)
                text(x + w - 13, y + 5, ref_ids[name], 12, weight=600)

        bindings = [
            name
            for column in columns
            for name in column
            if name in actual and "binding" in nodes[name]
        ]
        bottom = panel_bottom
        if bindings:
            by = panel_bottom + 56
            binding_rows = max(
                sum(positions[name][4] == col for name in bindings)
                for col in range(ncols)
            )
            binding_height = 66 + 82 * binding_rows
            rect(104, by, 1128, binding_height, "#e7f0fe", "#6d93c4", 20)
            text(
                132,
                by + binding_height - 22,
                "ComponentBindings  ·  adapters Python",
                17,
                "#315780",
                anchor="start",
                weight=600,
            )
            column_rows = {}
            for name in bindings:
                bound_nodes.add(name)
                x, y, w, h, col, row = positions[name]
                binding_row = column_rows.get(col, 0)
                column_rows[col] = binding_row + 1
                ay = by + 19 + 82 * binding_row
                rect(x, ay, w, 64, "#fff", "#7193be", 8)
                identifier(x + w / 2, ay + 24, nodes[name]["binding"], int(w / 8.1))
                if len(columns[col]) == 1:
                    line([(x + w / 2, ay), (x + w / 2, y + h)], binding=True)
                else:
                    # Stacked preprocessors need separate gutters, so an
                    # adapter never points through a different stage/card.
                    gx = 120 + col * pitch + 5 + row * 4
                    line(
                        [
                            (x, ay + 32),
                            (gx, ay + 32),
                            (gx, y + h - 12),
                            (x, y + h - 12),
                        ],
                        binding=True,
                    )
            bottom = by + binding_height
        if p < len(panels) - 1:
            text(
                126,
                bottom + 29,
                "↓ Continúa en el siguiente tramo. "
                "R1, R2… identifican la misma salida en ambos tramos.",
                14,
                "#536174",
                anchor="start",
            )
        top = bottom + 56

    assert drawn_edges == set(edges), (config["name"], set(edges) - drawn_edges)
    assert bound_nodes == {name for name, spec in nodes.items() if "binding" in spec}
    legend_y = top - 5
    line([(126, legend_y), (170, legend_y)])
    text(182, legend_y + 5, "Flujo / dependencia del YAML", 13, anchor="start")
    line([(535, legend_y), (579, legend_y)], binding=True)
    text(591, legend_y + 5, "Adapter → etapa", 13, anchor="start")
    height = top + 30
    description = (
        f"{config['name']}: {len(panels)} tramos; "
        f"{len(nodes)} etapas y outputs; {len(edges)} dependencias."
    )
    return f'''<svg xmlns="http://www.w3.org/2000/svg"
viewBox="0 0 {width} {height}" width="{width}" height="{height}"
role="img" aria-labelledby="title desc">
<title id="title">{escape(config["name"])}</title>
<desc id="desc">{escape(description)}</desc>
<defs>
<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"
markerHeight="7" orient="auto-start-reverse">
<path d="M 0 1 L 9 5 L 0 9 z" fill="#334155"/></marker>
<marker id="binding" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"
markerHeight="7" orient="auto-start-reverse">
<path d="M 0 1 L 9 5 L 0 9 z" fill="#537bad"/></marker>
</defs>
<rect width="100%" height="100%" fill="white"/>
{chr(10).join(result)}
</svg>'''


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for doc in ("README.md", "EXAMPLES.md"):
        document = (ROOT / doc).read_text(encoding="utf-8")
        for raw in re.findall(r"```yaml\n(.*?)```", document, re.S):
            config = yaml.safe_load(raw)
            if config["name"] not in LAYOUTS:
                continue
            svg = render(config)
            path = DEST / f"{config['name']}.svg"
            path.write_text(svg + "\n", encoding="utf-8")
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
