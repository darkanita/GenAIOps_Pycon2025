# app/tracking.py
"""
Configuración de observabilidad con MLflow 3:

- Tracing: cada llamada a la cadena queda registrada como una traza con sus
  pasos (reformulación de la pregunta, retriever, prompt, LLM), tokens y latencias.
- Versión de la app (LoggedModel): cada combinación proveedor + modelo + prompt + k
  es una "versión" a la que se enlazan las trazas y las evaluaciones, para
  compararlas en la pestaña "Models / Agent versions" de la UI de MLflow.
- Feedback humano: se adjunta a la traza de la respuesta.

Por defecto MLflow guarda todo en ./mlflow.db (SQLite). Para verlo:
    mlflow ui --port 5000
"""

import re

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType

from app.llm_provider import describe

CHAT_EXPERIMENT = "chatbot_rrhh"
EVAL_EXPERIMENT = "rag_evaluation"

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


def app_version_name(prompt_uri: str, k: int) -> str:
    """Nombre legible y estable para la versión de la app."""
    cfg = describe()
    prompt = prompt_uri.replace("prompts:/", "").replace("/", "_v")
    raw = f"rag-{cfg['llm_provider']}-{cfg['llm_model']}-{prompt}-k{k}"
    return re.sub(r"[^A-Za-z0-9_.@-]", "-", raw)


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
