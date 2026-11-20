# labs/mlflow_318/scorers.py
"""
Scorers basados en código (sin LLM, sin costo, deterministas).

⚠️  Regla para la evaluación online (MLflow 3.18): el servidor ejecuta el
*código fuente* de cada función en un contenedor aislado y sin red. Por eso
cada scorer hace sus `import` dentro del cuerpo y no usa variables del módulo.

    ONLINE_SCORERS   se registran y se activan con .start() sobre trazas nuevas
                     (no pueden recibir `expectations`)
    OFFLINE_SCORERS  se usan en mlflow.genai.evaluate (sí reciben `expectations`)
"""

from mlflow.genai.scorers import scorer


# ---------------------------------------------------------------------------
# Scorers de calidad que funcionan en línea y fuera de línea
# ---------------------------------------------------------------------------
@scorer
def respuesta_concisa(outputs) -> bool:
    """La respuesta tiene como máximo 100 palabras."""
    return len(str(outputs).split()) <= 100


@scorer
def cita_fuente(outputs) -> bool:
    """La respuesta indica el documento de donde sale (p. ej. 'Fuente: PerksPlus.pdf, p. 3')."""
    import re

    texto = str(outputs)
    abstiene = "no tengo información" in texto.lower()
    return abstiene or bool(re.search(r"Fuente:\s*\S+\.pdf", texto))


@scorer
def sin_datos_personales(outputs) -> bool:
    """La respuesta no repite correos ni teléfonos."""
    import re

    return not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+|\+?\d[\d\s-]{7,}\d", str(outputs))


@scorer
def herramientas_sin_error(trace) -> bool:
    """Ningún paso de la traza (retriever, herramienta, LLM) terminó en error."""
    return all("ERROR" not in str(span.status.status_code) for span in trace.data.spans)


@scorer
def longitud_respuesta(outputs) -> int:
    """Número de palabras (menos es mejor: úsalo con greater_is_better=False)."""
    return len(str(outputs).split())


# ---------------------------------------------------------------------------
# Scorer de sesión: evalúa la conversación completa (todas sus trazas)
# ---------------------------------------------------------------------------
@scorer
def sesion_sin_fallas(session):
    """La conversación no tuvo pasos con error ni respuestas de disculpa."""
    from mlflow.entities import Feedback

    fallidas = []
    for t in session:
        error = any("ERROR" in str(s.status.status_code) for s in t.data.spans)
        if error or "lo siento" in str(t.data.response or "").lower():
            fallidas.append(t.info.trace_id)
    return Feedback(
        value=not fallidas,
        rationale=(f"{len(session)} turnos; con fallas: {fallidas}" if fallidas
                   else f"{len(session)} turnos sin fallas"),
    )


# ---------------------------------------------------------------------------
# Scorers que comparan contra la respuesta esperada (solo offline)
# ---------------------------------------------------------------------------
@scorer
def cubre_palabras_clave(outputs, expectations) -> float:
    """Fracción de los datos clave esperados que aparecen en la respuesta."""
    import re
    import unicodedata

    def norm(t):
        t = unicodedata.normalize("NFKD", str(t).lower())
        return "".join(c for c in t if not unicodedata.combining(c))

    claves = expectations.get("palabras_clave") or []
    if not claves:
        return 1.0
    texto = norm(outputs)
    encontradas = [c for c in claves if re.search(rf"\b{re.escape(norm(c))}", texto)]
    return len(encontradas) / len(claves)


@scorer
def abstencion_correcta(outputs, expectations) -> bool:
    """Se abstiene cuando la pregunta está fuera de alcance, y responde cuando no."""
    abstiene = "no tengo información" in str(outputs).lower()
    return abstiene == bool(expectations.get("debe_abstenerse"))


ONLINE_SCORERS = [respuesta_concisa, cita_fuente, sin_datos_personales,
                  herramientas_sin_error, sesion_sin_fallas]
OFFLINE_SCORERS = [respuesta_concisa, cita_fuente, sin_datos_personales,
                   herramientas_sin_error, longitud_respuesta,
                   cubre_palabras_clave, abstencion_correcta]

# Para compare_evaluations: en estos scorers un valor MENOR es mejor
GREATER_IS_BETTER = {"longitud_respuesta": False}
