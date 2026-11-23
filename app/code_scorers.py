# app/code_scorers.py
"""
Scorers basados en código para el asistente de RR.HH.: deterministas, sin LLM
y sin costo. Se usan en dos lugares:

- app/run_eval.py      → junto a los jueces LLM en la evaluación offline
- app/monitoring.py    → evaluación online de las trazas del chat (MLflow 3.18)

Regla de MLflow 3.18 para la evaluación online: el servidor ejecuta el código
fuente de cada función en un contenedor aislado y sin red, así que cada scorer
importa lo que necesita dentro de su cuerpo y no usa variables del módulo.
"""

from mlflow.genai.scorers import scorer


@scorer
def respuesta_concisa(outputs) -> bool:
    """La respuesta tiene como máximo 150 palabras."""
    return len(str(outputs).split()) <= 150


@scorer
def sin_datos_personales(outputs) -> bool:
    """La respuesta no contiene correos electrónicos ni números de teléfono."""
    import re

    return not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+|\+?\d[\d\s-]{7,}\d", str(outputs))


@scorer
def sin_errores(trace) -> bool:
    """Ningún paso de la traza (retriever, LLM, cadena) terminó en error."""
    return all("ERROR" not in str(span.status.status_code) for span in trace.data.spans)


@scorer
def conversacion_sin_abstenciones(session):
    """En la conversación el asistente nunca tuvo que decir que no tenía información."""
    from mlflow.entities import Feedback

    marcas = ("no tengo información", "no tengo suficiente información")
    sin_respuesta = [t.info.trace_id for t in session
                     if any(m in str(t.data.response or "").lower() for m in marcas)]
    return Feedback(
        value=not sin_respuesta,
        rationale=(f"{len(sin_respuesta)} de {len(session)} turnos sin respuesta: {sin_respuesta}"
                   if sin_respuesta else f"{len(session)} turnos respondidos"),
    )


# Offline (run_eval.py): trazas individuales
EVAL_CODE_SCORERS = [respuesta_concisa, sin_datos_personales, sin_errores]
# Online (monitoring.py): trazas + una métrica por conversación
ONLINE_CODE_SCORERS = EVAL_CODE_SCORERS + [conversacion_sin_abstenciones]
