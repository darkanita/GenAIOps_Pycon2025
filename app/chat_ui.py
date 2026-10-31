# app/chat_ui.py
"""
Interfaz de chat del asistente (usada por ui_streamlit.py y main_interface.py).

Cómo funciona una pregunta, de punta a punta
--------------------------------------------
1. render_sidebar()   → el usuario elige el prompt (Prompt Registry) y k.
2. get_chain()        → se arma la cadena RAG (índice FAISS + prompt + LLM).
                        Queda en caché: no se reconstruye en cada clic.
3. init_tracking()    → las trazas se enlazan a la "versión de la app"
                        (proveedor + modelo + prompt + k) en MLflow.
4. chain.invoke()     → recupera fragmentos, llama al LLM y, si el tracing
                        está activo, MLflow guarda una traza con todos los pasos.
5. _render_answer()   → muestra la respuesta, sus referencias y el feedback.
6. Feedback           → 👍 se registra de inmediato; 👎 pide un comentario.
                        Ambos se adjuntan a la traza con mlflow.log_feedback.

Nota sobre Streamlit: el script completo se vuelve a ejecutar en cada
interacción. Por eso el historial vive en st.session_state.messages y los
recursos costosos se guardan con @st.cache_resource / @st.cache_data.
"""

import json
import os
import time

import mlflow
import streamlit as st

from app.llm_provider import describe
from app.prompt_registry import DEFAULT_PROMPT, resolve_prompt_ref
from app.rag_pipeline import build_chain, list_prompt_versions, load_vectorstore_from_disk
from app.tracking import CHAT_EXPERIMENT, CHAT_TRACING, init_tracking, log_user_feedback

EVAL_DATASET = "tests/eval_dataset.json"

# Preguntas de ejemplo si no se puede leer el dataset de evaluación
FALLBACK_EXAMPLES = [
    "¿Qué beneficios de salud ofrece la empresa?",
    "¿Qué incluye el programa PerksPlus?",
    "¿Cuáles son las responsabilidades de un analista de datos?",
]

# Cada mensaje del historial (st.session_state.messages) es un dict con:
#   question, answer      → texto de la pregunta y la respuesta
#   sources               → [{document, page, snippet}] para "📚 Referencias"
#   latency               → segundos que tardó la cadena
#   prompt_version, k     → configuración con la que se respondió
#   trace_id              → id de la traza en MLflow (None si no hay tracing)
#   feedback              → None (sin calificar), 1 (👍) o 0 (👎) ya registrado
#   comment               → comentario enviado con un 👎
#   awaiting_comment      → True mientras se muestra el formulario del 👎


# ============================================================
# 1. Recursos en caché
# ============================================================

@st.cache_resource(show_spinner="Cargando índice vectorial...")
def get_vectordb():
    """Índice FAISS en disco. Se carga una sola vez por proceso."""
    return load_vectorstore_from_disk()


@st.cache_resource(show_spinner="Preparando el asistente...")
def get_chain(prompt_version: str, k: int):
    """
    Cadena RAG para una versión de prompt concreta y un k.
    Devuelve (cadena, uri_del_prompt). La caché se indexa por los argumentos:
    cambiar de prompt o de k crea una cadena nueva; repetirlos la reutiliza.
    """
    chain = build_chain(get_vectordb(), prompt_version=prompt_version, k=k)
    prompt_uri = chain.combine_docs_chain.llm_chain.prompt.metadata["prompt_uri"]
    return chain, prompt_uri


@st.cache_data(ttl=30, show_spinner=False)
def resolve_prompt(ref: str) -> str:
    """
    Convierte un alias (@production) en su versión concreta (/2).
    Se revisa cada 30 s: así una promoción con --promote llega sin reiniciar.
    """
    return resolve_prompt_ref(ref)


@st.cache_data(ttl=30, show_spinner=False)
def get_prompt_options():
    """Alias y versiones del prompt para el selector de la barra lateral."""
    return list_prompt_versions()


@st.cache_data
def get_example_questions(max_n: int = 4):
    """Toma las preguntas del dataset de evaluación: si cambias de dominio, cambian solas."""
    try:
        with open(EVAL_DATASET, encoding="utf-8") as f:
            questions = [item["question"] for item in json.load(f)]
        return questions[:max_n] or FALLBACK_EXAMPLES
    except Exception:
        return FALLBACK_EXAMPLES


# ============================================================
# 2. Barra lateral
# ============================================================

def _new_conversation():
    """Callback del botón "Nueva conversación": borra el historial."""
    st.session_state.messages = []


def render_sidebar() -> dict:
    """Muestra la configuración activa y devuelve {"prompt_version", "k"} elegidos."""
    info = describe()
    st.sidebar.markdown("### ⚙️ Configuración")
    st.sidebar.markdown(
        f"**LLM:** `{info['llm_provider']}` · `{info['llm_model']}`  \n"
        f"**Embeddings:** `{info['embeddings_provider']}` · `{info['embeddings_model'].split('/')[-1]}`"
    )

    # El valor por defecto viene de PROMPT_VERSION (.env) o de @production
    options = get_prompt_options()
    default = os.getenv("PROMPT_VERSION") or DEFAULT_PROMPT
    if default not in options:
        options = [default, *options]
    prompt_version = st.sidebar.selectbox(
        "Prompt (MLflow Prompt Registry)", options, index=options.index(default),
        format_func=lambda o: o.replace("prompts:/", ""),
        help="@alias o versión fija. Registra cambios con `python -m app.prompt_registry`",
    )
    k = st.sidebar.slider("Fragmentos a recuperar (k)", 1, 10, 4,
                          help="Cuántos fragmentos de los PDFs se envían al LLM como contexto")

    st.sidebar.button("🗑️ Nueva conversación", width="stretch", on_click=_new_conversation)
    return {"prompt_version": prompt_version, "k": k}


# ============================================================
# 3. Referencias (documentos usados para responder)
# ============================================================

def _format_sources(source_documents) -> list[dict]:
    """
    Convierte los Document de LangChain en [{document, page, snippet}],
    agrupando por (documento, página) para no repetir referencias.
    """
    refs = {}
    for doc in source_documents or []:
        meta = doc.metadata or {}
        name = os.path.basename(meta.get("source", "desconocido"))
        page = meta.get("page")
        page = page + 1 if isinstance(page, int) else None  # PyPDFLoader cuenta desde 0
        if (name, page) not in refs:
            refs[(name, page)] = doc.page_content.strip()
    return [{"document": n, "page": p, "snippet": t} for (n, p), t in refs.items()]


def _render_sources(sources):
    """Desplegable "📚 Referencias" con documento, página y un extracto de cada fragmento."""
    if not sources:
        return
    with st.expander(f"📚 Referencias ({len(sources)})"):
        for i, ref in enumerate(sources, 1):
            page = f" — página {ref['page']}" if ref["page"] else ""
            st.markdown(f"**{i}. {ref['document']}**{page}")
            snippet = ref["snippet"]
            st.caption(snippet[:400] + ("…" if len(snippet) > 400 else ""))


# ============================================================
# 4. Feedback del usuario (👍 / 👎 + comentario)
# ============================================================
# Flujo:
#   👍 → _on_thumbs registra el feedback de inmediato.
#   👎 → _on_thumbs muestra un formulario; al enviarlo, _on_submit_negative
#        registra el 👎 junto con el comentario (o sin él, si el usuario lo omite).
# Se usan callbacks (on_change / on_click) porque Streamlit los ejecuta antes
# de volver a dibujar la página, así la interfaz queda actualizada al instante.

def _log_feedback(msg: dict, thumbs_up: bool, comment: str | None = None) -> bool:
    """Adjunta el feedback a la traza de la respuesta. Nunca tumba la app."""
    if not msg.get("trace_id"):
        st.warning("Esta respuesta no tiene traza asociada; no se pudo registrar el feedback.")
        return False
    try:
        log_user_feedback(msg["trace_id"], thumbs_up, comment=comment)
        return True
    except Exception as e:
        st.warning(f"No se pudo registrar el feedback en MLflow: {e}")
        return False


def _on_thumbs(idx: int):
    """Se ejecuta al tocar 👍 o 👎 en la respuesta número idx."""
    value = st.session_state.get(f"fb_{idx}")  # 1 = 👍, 0 = 👎, None = deseleccionado
    msg = st.session_state.messages[idx]
    if value == 1:
        if _log_feedback(msg, thumbs_up=True):
            msg["feedback"] = 1
        msg["awaiting_comment"] = False
    elif value == 0:
        msg["awaiting_comment"] = True  # se registra al enviar el formulario
    else:
        msg["awaiting_comment"] = False


def _on_submit_negative(idx: int, include_comment: bool):
    """Registra el 👎 de la respuesta idx, con el comentario escrito si se pidió."""
    msg = st.session_state.messages[idx]
    comment = None
    if include_comment:
        comment = (st.session_state.get(f"fb_comment_{idx}") or "").strip() or None
    if _log_feedback(msg, thumbs_up=False, comment=comment):
        msg.update(feedback=0, comment=comment, awaiting_comment=False)


def _render_feedback(idx: int, msg: dict):
    """Muestra 👍/👎, el formulario de comentario tras un 👎, o el agradecimiento si ya se calificó."""
    # Sin traza no hay dónde guardar el feedback
    if not msg.get("trace_id"):
        st.caption("👍/👎 desactivado: requiere tracing (ENABLE_TRACING=true)")
        return

    # Ya calificada: solo agradecer
    if msg.get("feedback") == 1:
        st.caption("👍 Gracias por tu feedback")
        return
    if msg.get("feedback") == 0:
        detail = f": “{msg['comment']}”" if msg.get("comment") else ""
        st.caption(f"👎 Gracias, lo usaremos para mejorar{detail}")
        return

    # Sin calificar: mostrar 👍/👎 y, tras un 👎, el formulario del comentario
    st.feedback("thumbs", key=f"fb_{idx}", on_change=_on_thumbs, args=(idx,))
    if msg.get("awaiting_comment"):
        with st.form(f"fb_form_{idx}"):
            st.text_area(
                "¿Qué estuvo mal en esta respuesta?",
                key=f"fb_comment_{idx}",
                placeholder="Ej.: la información es incorrecta, está incompleta o citó el documento equivocado.",
            )
            c1, c2 = st.columns(2)
            c1.form_submit_button("Enviar comentario", type="primary", width="stretch",
                                  on_click=_on_submit_negative, args=(idx, True))
            c2.form_submit_button("Enviar sin comentario", width="stretch",
                                  on_click=_on_submit_negative, args=(idx, False))


# ============================================================
# 5. Mensajes del chat
# ============================================================

def _render_answer(idx: int, msg: dict):
    """Respuesta del asistente: texto, referencias, latencia/configuración y feedback."""
    st.markdown(msg["answer"])
    _render_sources(msg["sources"])
    st.caption(f"⏱️ {msg['latency']:.1f} s · prompt `{msg['prompt_version']}` · k={msg['k']}")
    _render_feedback(idx, msg)


def _render_message(idx: int, msg: dict):
    """Un turno completo del historial: pregunta del usuario + respuesta del asistente."""
    with st.chat_message("user", avatar="👤"):
        st.markdown(msg["question"])
    with st.chat_message("assistant", avatar="🤖"):
        _render_answer(idx, msg)


def _ask(chain, question: str) -> tuple[dict, float, str | None]:
    """
    Envía la pregunta a la cadena RAG con el historial de la conversación.
    Devuelve (resultado, latencia_en_segundos, trace_id).
    """
    history = [(m["question"], m["answer"]) for m in st.session_state.messages]
    start = time.perf_counter()
    result = chain.invoke({"question": question, "chat_history": history})
    latency = time.perf_counter() - start
    # Con autolog activo, MLflow acaba de crear una traza para esta llamada
    trace_id = mlflow.get_last_active_trace_id(thread_local=True) if CHAT_TRACING else None
    return result, latency, trace_id


# ============================================================
# 6. Página completa
# ============================================================

def render_chat(title="🤖 Asistente de Recursos Humanos",
                subtitle="Pregunta sobre beneficios, políticas y roles de Contoso Electronics."):
    """Dibuja la página del chat. Ver el flujo completo al inicio de este archivo."""
    config = render_sidebar()
    st.title(title)
    st.caption(subtitle)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Cadena RAG para el prompt y k elegidos (desde caché si ya existe)
    try:
        chain, prompt_uri = get_chain(resolve_prompt(config["prompt_version"]), config["k"])
    except FileNotFoundError as e:  # todavía no se construyó el índice
        st.error(str(e))
        st.stop()
    init_tracking(CHAT_EXPERIMENT, prompt_uri, config["k"], tracing=CHAT_TRACING)
    st.sidebar.caption(f"Prompt resuelto: `{prompt_uri.replace('prompts:/', '')}`")

    # Preguntas de ejemplo: solo antes del primer mensaje
    pending = None
    examples_box = st.empty()
    if not st.session_state.messages:
        with examples_box.container():
            st.markdown("##### 💡 Prueba con alguna de estas preguntas")
            cols = st.columns(2)
            for i, q in enumerate(get_example_questions()):
                if cols[i % 2].button(q, key=f"ex_{i}", width="stretch"):
                    pending = q

    # Historial de la conversación
    for idx, msg in enumerate(st.session_state.messages):
        _render_message(idx, msg)

    # Nueva pregunta: escrita o elegida de los ejemplos
    question = st.chat_input("¿Qué deseas consultar?") or pending
    if not question:
        return
    examples_box.empty()

    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Consultando documentos..."):
            try:
                result, latency, trace_id = _ask(chain, question)
            except Exception as e:
                st.error(f"Error al consultar el modelo: {e}")
                st.stop()

        msg = {
            "question": question,
            "answer": result["answer"],
            "sources": _format_sources(result.get("source_documents")),
            "latency": latency,
            "prompt_version": prompt_uri.replace("prompts:/", ""),
            "k": config["k"],
            "trace_id": trace_id,
            "feedback": None,
        }
        st.session_state.messages.append(msg)
        _render_answer(len(st.session_state.messages) - 1, msg)
