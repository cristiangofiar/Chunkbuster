# Diagramas de la documentación

Las vistas SVG tienen posiciones fijas para que el grafo principal se lea de
izquierda a derecha y los adapters queden alineados debajo. Los ejemplos grandes
se dividen en tramos; una marca `R1`, `R2`… identifica la misma salida entre tramos.

Los nodos, sus identificadores, las dependencias y los bindings se obtienen de
los bloques YAML de `README.md` y `EXAMPLES.md`. La distribución por columnas se
define en `scripts/render_doc_diagrams.py`. El Mermaid completo permanece en un
desplegable bajo cada vista SVG.

Para regenerar las ocho imágenes, desde la raíz del repositorio:

```bash
uv run python scripts/render_doc_diagrams.py
```

El generador comprueba que cada etapa se dibuje una vez y que se representen
todas las dependencias y todos los bindings del YAML. Después de modificar la
distribución, hay que revisar los renders a ancho de página: la validación del
grafo no comprueba su legibilidad visual.
