# app/i18n.py
"""
Bilingual support (Spanish / English) / Soporte bilingüe (español / inglés).

Set APP_LANG=en or APP_LANG=es (default: es) in .env or the environment.
It switches everything a reader sees: the Streamlit UI, CLI messages, the
prompt the assistant uses, the evaluation dataset and the language judge.

    from app.i18n import LANG, t
    st.title(t("chat.title"))
    print(t("eval.experiment", name=EVAL_EXPERIMENT))
"""

import os

from dotenv import load_dotenv

load_dotenv()

SUPPORTED = ("es", "en")
LANG = (os.getenv("APP_LANG") or "es").strip().lower()[:2]
if LANG not in SUPPORTED:
    LANG = "es"

MESSAGES: dict[str, dict[str, str]] = {
    # ---------- Chat UI ----------
    "chat.title": {"es": "🤖 Asistente de Recursos Humanos",
                   "en": "🤖 Human Resources Assistant"},
    "chat.title_contoso": {"es": "🤖 Asistente de Recursos Humanos - Contoso",
                           "en": "🤖 Human Resources Assistant - Contoso"},
    "chat.subtitle": {"es": "Pregunta sobre beneficios, políticas y roles de Contoso Electronics.",
                      "en": "Ask about Contoso Electronics benefits, policies and roles."},
    "chat.loading_index": {"es": "Cargando índice vectorial...", "en": "Loading vector index..."},
    "chat.preparing": {"es": "Preparando el asistente...", "en": "Preparing the assistant..."},
    "chat.no_trace": {"es": "Esta respuesta no tiene traza asociada; no se pudo registrar el feedback.",
                      "en": "This answer has no trace; the feedback could not be logged."},
    "chat.feedback_error": {"es": "No se pudo registrar el feedback en MLflow: {error}",
                            "en": "Could not log the feedback to MLflow: {error}"},
    "chat.references": {"es": "📚 Referencias ({n})", "en": "📚 References ({n})"},
    "chat.page": {"es": " — página {page}", "en": " — page {page}"},
    "chat.unknown_source": {"es": "desconocido", "en": "unknown"},
    "chat.settings": {"es": "### ⚙️ Configuración", "en": "### ⚙️ Settings"},
    "chat.prompt_label": {"es": "Prompt (MLflow Prompt Registry)",
                          "en": "Prompt (MLflow Prompt Registry)"},
    "chat.prompt_help": {"es": "@alias o versión fija. Registra cambios con `python -m app.prompt_registry`",
                         "en": "@alias or pinned version. Register changes with `python -m app.prompt_registry`"},
    "chat.k_label": {"es": "Fragmentos a recuperar (k)", "en": "Chunks to retrieve (k)"},
    "chat.k_help": {"es": "Cuántos fragmentos de los PDFs se envían al LLM como contexto",
                    "en": "How many PDF chunks are sent to the LLM as context"},
    "chat.new_conversation": {"es": "🗑️ Nueva conversación", "en": "🗑️ New conversation"},
    "chat.resolved_prompt": {"es": "Prompt resuelto: `{prompt}`", "en": "Resolved prompt: `{prompt}`"},
    "chat.try_these": {"es": "##### 💡 Prueba con alguna de estas preguntas",
                       "en": "##### 💡 Try one of these questions"},
    "chat.input": {"es": "¿Qué deseas consultar?", "en": "What would you like to ask?"},
    "chat.searching": {"es": "Consultando documentos...", "en": "Searching the documents..."},
    "chat.model_error": {"es": "Error al consultar el modelo: {error}",
                         "en": "Error calling the model: {error}"},
    "chat.setup_error": {"es": "No se pudo preparar el asistente: {error}",
                         "en": "Could not prepare the assistant: {error}"},
    "chat.answer_meta": {"es": "⏱️ {latency:.1f} s · prompt `{prompt}` · k={k}",
                         "en": "⏱️ {latency:.1f} s · prompt `{prompt}` · k={k}"},

    # ---------- Main interface / dashboards ----------
    "ui.page_title": {"es": "Chatbot GenAI + Métricas", "en": "GenAI Chatbot + Metrics"},
    "ui.pick_view": {"es": "Selecciona una vista:", "en": "Choose a view:"},
    "ui.view_chat": {"es": "🤖 Chatbot", "en": "🤖 Chatbot"},
    "ui.view_metrics": {"es": "📊 Métricas", "en": "📊 Metrics"},
    "ui.dashboard_title": {"es": "Dashboard de Evaluación", "en": "Evaluation Dashboard"},
    "ui.chat_page_title": {"es": "Chatbot GenAI RRHH", "en": "GenAI HR Chatbot"},

    # ---------- Metrics UI ----------
    "metrics.title": {"es": "📈 Evaluación y monitoreo del chatbot",
                      "en": "📈 Chatbot evaluation and monitoring"},
    "metrics.caption": {"es": "Datos de MLflow. Para explorar trazas, prompts y versiones: `mlflow ui --port 5000`",
                        "en": "Data from MLflow. To explore traces, prompts and versions: `mlflow ui --port 5000`"},
    "metrics.tab_eval": {"es": "⚖️ Evaluación automática", "en": "⚖️ Automatic evaluation"},
    "metrics.tab_users": {"es": "👥 Feedback de usuarios", "en": "👥 User feedback"},
    "metrics.no_evals": {"es": "Aún no hay evaluaciones. Ejecuta `python app/run_eval.py`.",
                         "en": "No evaluations yet. Run `python app/run_eval.py`."},
    "metrics.config_col": {"es": "configuración", "en": "configuration"},
    "metrics.compare_header": {"es": "⚖️ Comparación de configuraciones",
                               "en": "⚖️ Configuration comparison"},
    "metrics.compare_caption": {"es": "Promedio de cada juez (0 a 1) por evaluación. Cada fila es una ejecución de `run_eval.py`.",
                                "en": "Mean of each judge (0 to 1) per evaluation. Each row is one `run_eval.py` run."},
    "metrics.pick_metrics": {"es": "Métricas a graficar", "en": "Metrics to plot"},
    "metrics.detail_header": {"es": "🔎 Detalle por pregunta", "en": "🔎 Per-question detail"},
    "metrics.evaluation": {"es": "Evaluación", "en": "Evaluation"},
    "metrics.no_traces": {"es": "Este run no tiene trazas.", "en": "This run has no traces."},
    "metrics.question_col": {"es": "pregunta", "en": "question"},
    "metrics.answer_col": {"es": "respuesta", "en": "answer"},
    "metrics.judges_expander": {"es": "Ver respuestas y razonamiento de los jueces",
                                "en": "See answers and judge rationales"},
    "metrics.users_header": {"es": "👥 Feedback de usuarios", "en": "👥 User feedback"},
    "metrics.no_conversations": {"es": "Todavía no hay conversaciones registradas.",
                                 "en": "No conversations logged yet."},
    "metrics.no_feedback": {"es": "Aún no hay 👍/👎 de usuarios.", "en": "No 👍/👎 from users yet."},
    "metrics.app_version_col": {"es": "versión de la app", "en": "app version"},
    "metrics.useful_col": {"es": "útil", "en": "useful"},
    "metrics.comment_col": {"es": "comentario", "en": "comment"},
    "metrics.rated": {"es": "Respuestas calificadas", "en": "Rated answers"},
    "metrics.satisfaction_metric": {"es": "👍 Satisfacción", "en": "👍 Satisfaction"},
    "metrics.negatives_metric": {"es": "👎 Negativas", "en": "👎 Negative"},
    "metrics.satisfaction_col": {"es": "satisfacción", "en": "satisfaction"},
    "metrics.ratings_col": {"es": "calificaciones", "en": "ratings"},
    "metrics.negatives_expander": {"es": "Respuestas con 👎 ({n}) — candidatas para el dataset de evaluación",
                                   "en": "Answers with 👎 ({n}) — candidates for the evaluation dataset"},

    # ---------- run_eval.py ----------
    "eval.app": {"es": "⚙️  App: {info} | prompt={prompt} | k={k}",
                 "en": "⚙️  App: {info} | prompt={prompt} | k={k}"},
    "eval.results": {"es": "📈 Resultados agregados:", "en": "📈 Aggregated results:"},
    "eval.detail": {"es": "🔎 Detalle por pregunta: mlflow ui → experimento '{experiment}' → run '{run}'",
                    "en": "🔎 Per-question detail: mlflow ui → experiment '{experiment}' → run '{run}'"},
    "eval.compare_header": {"es": "📊 Candidata vs línea base (prueba pareada por pregunta):",
                            "en": "📊 Candidate vs baseline (paired test per question):"},
    "eval.regressed_header": {"es": "🔻 Preguntas que empeoraron (el promedio no las muestra):",
                              "en": "🔻 Questions that got worse (the average hides them):"},
    "eval.experiment": {"es": "📊 Experimento MLflow: {name}", "en": "📊 MLflow experiment: {name}"},
    "eval.judge": {"es": "⚖️  Juez: {model} | {n} scorers: {names}",
                   "en": "⚖️  Judge: {model} | {n} scorers: {names}"},
    "eval.slow_mode": {"es": "🐢 Modo pausado para respetar los límites del proveedor (puede tardar varios minutos)",
                       "en": "🐢 Throttled mode to respect the provider's rate limits (may take several minutes)"},
    "eval.arg_desc": {"es": "Evaluación del RAG con jueces de MLflow",
                      "en": "RAG evaluation with MLflow judges"},
    "eval.arg_baseline": {"es": "evalúa también esta versión y compara contra ella (p. ej. {example})",
                          "en": "also evaluate this version and compare against it (e.g. {example})"},
    "eval.unknown_scorers": {"es": "EVAL_SCORERS desconocidos: {unknown}. Disponibles: {available}",
                             "en": "Unknown EVAL_SCORERS: {unknown}. Available: {available}"},
    "eval.guideline_language": {"es": "La respuesta debe estar escrita en español.",
                                "en": "The response must be written in English."},
    "eval.guideline_tone": {"es": "La respuesta debe tener un tono formal y profesional, apropiado "
                                  "para un asistente de Recursos Humanos, y no debe inventar políticas.",
                            "en": "The response must have a formal, professional tone suitable for a "
                                  "Human Resources assistant, and must not make up policies."},

    # ---------- prompt_registry.py ----------
    "prompts.registered": {"es": "📝 {file} → {name} v{version} (@{alias})",
                           "en": "📝 {file} → {name} v{version} (@{alias})"},
    "prompts.production": {"es": "🚀 @{alias} → v{version}", "en": "🚀 @{alias} → v{version}"},
    "prompts.up_to_date": {"es": "✔️  Prompts al día, no hay versiones nuevas.",
                           "en": "✔️  Prompts up to date, no new versions."},
    "prompts.not_found": {"es": "Prompt no encontrado: {path}", "en": "Prompt not found: {path}"},
    "prompts.arg_desc": {"es": "Registro de prompts en MLflow", "en": "MLflow prompt registration"},
    "prompts.arg_promote": {"es": "Mueve el alias @production a esta versión",
                            "en": "Move the @production alias to this version"},
    "prompts.commit": {"es": "Desde {file}", "en": "From {file}"},
    "prompts.updated": {"es": " (actualizado)", "en": " (updated)"},

    # ---------- rag_pipeline.py / llm_provider.py ----------
    "rag.no_index": {"es": "No existe el índice en '{path}'. Genéralo primero con: "
                           'python -c "from app.rag_pipeline import save_vectorstore; save_vectorstore()"',
                     "en": "No index found at '{path}'. Build it first with: "
                           'python -c "from app.rag_pipeline import save_vectorstore; save_vectorstore()"'},
    "llm.unsupported": {"es": "{var} no soportado: '{value}'. Usa uno de: {options}",
                        "en": "Unsupported {var}: '{value}'. Use one of: {options}"},
    "llm.no_groq_key": {"es": "Falta GROQ_API_KEY en el entorno o en .env",
                        "en": "GROQ_API_KEY is missing from the environment or .env"},

    # ---------- monitoring.py ----------
    "mon.desc": {"es": "Evaluación online del chat con scorers de código",
                 "en": "Online evaluation of the chat with code-based scorers"},
    "mon.need_server": {"es": "❌ La evaluación online necesita un servidor MLflow.\n"
                              "   ./labs/mlflow_318/start_server.sh && export MLFLOW_TRACKING_URI=http://localhost:5000",
                        "en": "❌ Online evaluation needs an MLflow server.\n"
                              "   ./labs/mlflow_318/start_server.sh && export MLFLOW_TRACKING_URI=http://localhost:5000"},
    "mon.stopped": {"es": "⏹️  {name} detenido", "en": "⏹️  {name} stopped"},
    "mon.active": {"es": "▶️  {name} activo", "en": "▶️  {name} active"},
    "mon.start_error": {"es": "❌ {name}: {error}\n   Reinicia el servidor con ./labs/mlflow_318/start_server.sh (con Docker).",
                        "en": "❌ {name}: {error}\n   Restart the server with ./labs/mlflow_318/start_server.sh (with Docker)."},
    "mon.only_production": {"es": "ℹ️  Solo se evalúan trazas con environment=production: ejecuta la app con APP_ENV=production",
                            "en": "ℹ️  Only traces with environment=production are scored: run the app with APP_ENV=production"},
    "mon.arg_all": {"es": "evaluar todas las trazas (por defecto solo environment=production)",
                    "en": "score every trace (default: only environment=production)"},
}


def add_messages(messages: dict[str, dict[str, str]]) -> None:
    """Lets other modules (e.g. the lab) register their own texts."""
    MESSAGES.update(messages)


def t(key: str, **kwargs) -> str:
    """Text for `key` in the active language, formatted with kwargs."""
    entry = MESSAGES.get(key)
    if entry is None:
        return key
    text = entry.get(LANG) or entry.get("es") or key
    return text.format(**kwargs) if kwargs else text


def by_lang(es, en):
    """Picks a value by language: by_lang('tests/eval_dataset.json', 'tests/eval_dataset.en.json')."""
    return en if LANG == "en" else es
