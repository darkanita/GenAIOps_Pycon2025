---
name: genaiops-rrhh-triage
description: Investiga y corrige issues de calidad del asistente de RR.HH. (repo GenAIOps_Pycon2025) a partir de trazas de MLflow, y demuestra la corrección con compare_evaluations antes de cerrar el issue. Úsalo cuando te pidan arreglar un issue de MLflow, una regresión de evaluación o una falla vista en producción.
---

# Triage de issues del asistente de RR.HH.

Flujo: **issue → evidencia en trazas → causa → cambio mínimo → evaluación comparada → cerrar issue**.
No declares un issue resuelto sin una comparación que lo demuestre.

## 0. Conexión

```bash
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://localhost:5000}
```
Experimentos: `lab318_produccion` (trazas e issues), `lab318_evaluaciones` (evaluaciones).

## 1. Leer el issue y su evidencia

```bash
python labs/mlflow_318/paso4_issues.py          # lista issues abiertos y genera issues_para_agente.md
```
Para una traza concreta:
```python
import mlflow
t = mlflow.get_trace("<trace_id>")
for s in t.data.spans:
    print(s.span_type, s.name, s.status.status_code, str(s.outputs)[:200])
```

## 2. Ubicar la causa según el span que falla

| Síntoma en la traza | Dónde mirar |
|---|---|
| `RETRIEVER` sin documentos o irrelevantes | `app/rag_pipeline.py` (k, chunking) / `mock_agent.buscar_documentos` |
| `TOOL` con error (timeout, excepción) | la herramienta; agrega manejo de error y mensaje al usuario |
| Respuesta inventada cuando no hay contexto | prompt en `app/prompts/*.txt` / `mock_agent._componer` |
| Respuesta larga, sin fuente o en otro idioma | prompt de la versión activa |
| Muchos tokens de entrada (costo) | contexto enviado al LLM |
| Datos personales repetidos | post-procesamiento de la respuesta / instrucciones del prompt |

## 3. Hacer el cambio mínimo

- Prompts: crea una **nueva** versión en `app/prompts/` y regístrala con
  `python -m app.prompt_registry` (no edites la versión en producción).
- Código: cambia solo el componente que muestra la traza.

## 4. Demostrar la corrección (obligatorio)

```bash
# agente simulado
python labs/mlflow_318/paso2_comparar.py --baseline v1 --candidate <nueva> --gate
# RAG real
python labs/mlflow_318/paso2_comparar.py --agent rag \
  --baseline prompts:/asistente_rrhh@production --candidate prompts:/asistente_rrhh/<N> --gate
```
El gate exige mejora significativa (`assert_improved`) **y** ninguna pregunta peor en
`cubre_palabras_clave` y `abstencion_correcta`. Si alguna fila empeora, revisa esas trazas
antes de seguir: un promedio más alto no basta.

## 5. Cerrar el ciclo

```bash
python labs/mlflow_318/paso4_issues.py --resolver <ISSUE_ID>
python -m app.prompt_registry --promote <N>     # solo si el gate pasó y el humano lo aprueba
```

Reporta al usuario: el issue, la causa (con trace IDs), el cambio, la tabla de
`compare_evaluations` y las filas que cambiaron. Pide confirmación antes de promover a producción.
