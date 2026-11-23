# app/tracking.py
"""
Configuración de observabilidad con MLflow 3:

- Tracing: cada llamada a la cadena queda registrada como una traza con sus
  pasos (reformulación de la pregunta, retriever, prompt, LLM), tokens y latencias.
- Versión de la app (LoggedModel): cada combinación proveedor + modelo + prompt + k
  es una "versión" a la que se enlazan las trazas y las evaluaciones, para
  compararlas en la pestaña "Models / Agent versions" de la UI de MLflow.
- Sesiones: cada conversación lleva su session_id y user_id en la traza, para
  ver la conversación completa en la pestaña Sessions, evaluarla con scorers de
  sesión y que la detección automática de issues (MLflow 3.18) la analice.
- Feedback humano: se adjunta a la traza de la respuesta.

Por defecto MLflow guarda todo en ./mlflow.db (SQLite). Para verlo:
    mlflow ui --port 5000
"""

import os
import re

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType, SpanType

from app.llm_provider import describe

CHAT_EXPERIMENT = "chatbot_rrhh"
EVAL_EXPERIMENT = "rag_evaluation"

# Entorno con el que se etiquetan las trazas del chat (filtro de la detección de
# issues programada y de la evaluación online: environment = production)
APP_ENV = os.getenv("APP_ENV", "development")

_autolog_enabled = False
_versions_with_params: set[str] = set()


def enable_tracing():
    """Activa el tracing automático de LangChain (una sola vez por proceso)."""
    global _autolog_enabled
    if not _autolog_enabled:
        mlflow.langchain.autolog(silent=True)
        _autolog_enabled = True


def app_config(prompt_uri: str, k: int) -> dict:
    return {**describe(), "prompt": prompt_uri, "k": str(k)}


def _prompt_label(prompt_uri: str) -> str:
    """prompts:/asistente_rrhh/2 → asistente_rrhh-v2 ; app/prompts/v1_x.txt → local-v1_x"""
    if prompt_uri.startswith("prompts:/"):
        return prompt_uri.replace("prompts:/", "").replace("/", "-v")
    return "local-" + os.path.splitext(os.path.basename(prompt_uri))[0]


def app_version_name(prompt_uri: str, k: int) -> str:
    """
    Nombre legible y estable para la versión de la app.
    MLflow no permite '/', ':', '.', '%', comillas en el nombre: se reemplazan por '-'.
    """
    cfg = describe()
    raw = f"rag-{cfg['llm_provider']}-{cfg['llm_model']}-{_prompt_label(prompt_uri)}-k{k}"
    return re.sub(r"[^A-Za-z0-9_-]+", "-", raw)


def set_app_version(prompt_uri: str, k: int) -> str:
    """
    Marca la versión activa de la app: las trazas generadas a partir de ahora
    quedan enlazadas a ella. Devuelve el model_id.
    """
    active = mlflow.set_active_model(name=app_version_name(prompt_uri, k))
    if active.model_id not in _versions_with_params:
        # Los parámetros de un LoggedModel no se pueden sobrescribir: solo la primera vez
        if not mlflow.get_logged_model(active.model_id).params:
            mlflow.log_model_params(app_config(prompt_uri, k), model_id=active.model_id)
        _versions_with_params.add(active.model_id)
    return active.model_id


def init_tracking(experiment: str, prompt_uri: str, k: int) -> str:
    mlflow.set_experiment(experiment)
    enable_tracing()
    return set_app_version(prompt_uri, k)


def ask(chain, question: str, history: list | None = None, session_id: str | None = None,
        user_id: str | None = None) -> tuple[dict, str]:
    """
    Llama a la cadena RAG dentro de una traza raíz "hr_assistant" que lleva la
    sesión, el usuario y el entorno. Devuelve (resultado de la cadena, trace_id).
    """
    with mlflow.start_span(name="hr_assistant", span_type=SpanType.AGENT) as span:
        span.set_inputs({"question": question})
        metadata = {k: v for k, v in {"mlflow.trace.session": session_id,
                                      "mlflow.trace.user": user_id}.items() if v}
        mlflow.update_current_trace(metadata=metadata or None, tags={"environment": APP_ENV})
        result = chain.invoke({"question": question, "chat_history": history or []})
        span.set_outputs(result["answer"])
    return result, span.trace_id


def log_user_feedback(trace_id: str, thumbs_up: bool, user: str = "streamlit_user",
                      comment: str | None = None):
    """Adjunta el 👍/👎 del usuario a la traza de la respuesta."""
    mlflow.log_feedback(
        trace_id=trace_id,
        name="user_feedback",
        value=thumbs_up,
        rationale=comment,
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id=user),
    )
