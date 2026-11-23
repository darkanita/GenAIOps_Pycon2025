# app/chat_ui.py
"""
Componente de chat reutilizable para ui_streamlit.py y main_interface.py.

Incluye:
- Preguntas de ejemplo (tomadas de tests/eval_dataset.json)
- Referencias: documento, página y fragmento usado para responder
- Panel lateral con proveedor/modelo, versión de prompt (Prompt Registry) y nº de fragmentos (k)
- Caché del vectorstore y de la cadena (no se recargan en cada interacción)
- Tracing con MLflow: cada respuesta es una traza enlazada a la versión de la app,
  agrupada por conversación (session_id) para la vista Sessions de MLflow
- Feedback 👍/👎 adjunto a la traza de la respuesta (mlflow.log_feedback)
- Tiempo de respuesta
"""

import json
import os
import time
import uuid

import streamlit as st

from app.i18n import by_lang, t
from app.llm_provider import describe
from app.prompt_registry import DEFAULT_PROMPT, resolve_prompt_ref
from app.rag_pipeline import (
    build_chain,
    list_prompt_versions,
    load_vectorstore_from_disk,
)
from app.tracking import CHAT_EXPERIMENT, ask, init_tracking, log_user_feedback

EVAL_DATASET = by_lang("tests/eval_dataset.json", "tests/eval_dataset.en.json")
FALLBACK_EXAMPLES = by_lang(
    ["¿Qué beneficios de salud ofrece la empresa?",
     "¿Qué incluye el programa PerksPlus?",
     "¿Cuáles son las responsabilidades de un analista de datos?"],
    ["What health benefits does the company offer?",
     "What does the PerksPlus program include?",
     "What are the responsibilities of a data analyst?"],
)


# ---------- Recursos cacheados ----------

@st.cache_resource(show_spinner=t("chat.loading_index"))
def get_vectordb():
    return load_vectorstore_from_disk()


@st.cache_resource(show_spinner=t("chat.preparing"))
def get_chain(prompt_version: str, k: int):
    chain = build_chain(get_vectordb(), prompt_version=prompt_version, k=k)
    prompt_uri = chain.combine_docs_chain.llm_chain.prompt.metadata["prompt_uri"]
    return chain, prompt_uri


@st.cache_data(ttl=30, show_spinner=False)
def resolve_prompt(ref: str) -> str:
    """Alias → versión concreta. Se revisa cada 30 s: una promoción llega sin reiniciar."""
    return resolve_prompt_ref(ref)


@st.cache_data(ttl=30, show_spinner=False)
def get_prompt_options():
    return list_prompt_versions()


@st.cache_data
def get_example_questions(max_n: int = 4):
    """Usa las preguntas del dataset de evaluación: si cambias de dominio, cambian solas."""
    try:
        with open(EVAL_DATASET, encoding="utf-8") as f:
            questions = [item["question"] for item in json.load(f)]
        return questions[:max_n] or FALLBACK_EXAMPLES
    except Exception:
        return FALLBACK_EXAMPLES


# ---------- Utilidades ----------

def _format_sources(source_documents):
    """Agrupa fragmentos por (documento, página) y elimina duplicados."""
    refs = {}
    for doc in source_documents or []:
        meta = doc.metadata or {}
        name = os.path.basename(meta.get("source", t("chat.unknown_source")))
        page = meta.get("page")
        page = page + 1 if isinstance(page, int) else None  # PyPDFLoader cuenta desde 0
        key = (name, page)
        if key not in refs:
            refs[key] = doc.page_content.strip()
    return [{"document": n, "page": p, "snippet": t} for (n, p), t in refs.items()]


def _log_feedback(msg, thumbs_up: bool):
    if not msg.get("trace_id"):
        st.warning(t("chat.no_trace"))
        return False
    try:
        log_user_feedback(msg["trace_id"], thumbs_up)
        return True
    except Exception as e:  # el feedback nunca debe tumbar la app
        st.warning(t("chat.feedback_error", error=e))
        return False


def _render_sources(sources):
    if not sources:
        return
    with st.expander(t("chat.references", n=len(sources))):
        for i, ref in enumerate(sources, 1):
            page = t("chat.page", page=ref["page"]) if ref["page"] else ""
            st.markdown(f"**{i}. {ref['document']}**{page}")
            snippet = ref["snippet"]
            st.caption(snippet[:400] + ("…" if len(snippet) > 400 else ""))


def _render_feedback(idx, msg):
    key = f"fb_{idx}"
    if msg.get("feedback") is not None:
        st.session_state.setdefault(key, msg["feedback"])
    st.feedback("thumbs", key=key, on_change=_on_feedback, args=(idx,))


def _on_feedback(idx):
    value = st.session_state.get(f"fb_{idx}")  # 1 = 👍, 0 = 👎
    msg = st.session_state.messages[idx]
    if value is not None and value != msg.get("feedback") and _log_feedback(msg, bool(value)):
        msg["feedback"] = value


def _new_conversation():
    st.session_state.messages = []
    st.session_state.pop("session_id", None)


# ---------- Sidebar ----------

def render_sidebar():
    info = describe()
    st.sidebar.markdown(t("chat.settings"))
    st.sidebar.markdown(
        f"**LLM:** `{info['llm_provider']}` · `{info['llm_model']}`  \n"
        f"**Embeddings:** `{info['embeddings_provider']}` · `{info['embeddings_model'].split('/')[-1]}`"
    )

    options = get_prompt_options()
    default = os.getenv("PROMPT_VERSION") or DEFAULT_PROMPT
    if default not in options:
        options = [default, *options]
    prompt_version = st.sidebar.selectbox(
        t("chat.prompt_label"), options, index=options.index(default),
        format_func=lambda o: o.replace("prompts:/", ""),
        help=t("chat.prompt_help"),
    )
    k = st.sidebar.slider(t("chat.k_label"), 1, 10, 4,
                          help=t("chat.k_help"))

    st.sidebar.button(t("chat.new_conversation"), width="stretch",
                      on_click=_new_conversation)

    return {"prompt_version": prompt_version, "k": k}


# ---------- Chat ----------

def render_chat(title=None, subtitle=None):
    title, subtitle = title or t("chat.title"), subtitle or t("chat.subtitle")
    config = render_sidebar()

    st.title(title)
    st.caption(subtitle)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Una sesión de MLflow por conversación (se renueva al limpiar el chat)
    st.session_state.setdefault("session_id", f"chat-{uuid.uuid4().hex[:12]}")

    try:
        chain, prompt_uri = get_chain(resolve_prompt(config["prompt_version"]), config["k"])
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    # Trazas en el experimento del chat, enlazadas a esta versión de la app
    init_tracking(CHAT_EXPERIMENT, prompt_uri, config["k"])
    st.sidebar.caption(t("chat.resolved_prompt", prompt=prompt_uri.replace("prompts:/", "")))

    # Preguntas de ejemplo (solo al inicio de la conversación)
    pending = None
    examples_box = st.empty()
    if not st.session_state.messages:
        with examples_box.container():
            st.markdown(t("chat.try_these"))
            cols = st.columns(2)
            for i, q in enumerate(get_example_questions()):
                if cols[i % 2].button(q, key=f"ex_{i}", width="stretch"):
                    pending = q

    # Historial
    for idx, msg in enumerate(st.session_state.messages):
        _render_message(idx, msg)

    question = st.chat_input(t("chat.input")) or pending
    if not question:
        return

    examples_box.empty()  # oculta los ejemplos al empezar la conversación

    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner(t("chat.searching")):
            history = [(m["question"], m["answer"]) for m in st.session_state.messages]
            start = time.perf_counter()
            try:
                result, trace_id = ask(chain, question, history,
                                       session_id=st.session_state.session_id,
                                       user_id=os.getenv("APP_USER", "streamlit_user"))
            except Exception as e:
                st.error(t("chat.model_error", error=e))
                st.stop()
            latency = time.perf_counter() - start

        msg = {
            "question": question,
            "answer": result["answer"],
            "sources": _format_sources(result.get("source_documents")),
            "latency": latency,
            "prompt_version": prompt_uri.replace("prompts:/", ""),
            "k": config["k"],
            "trace_id": trace_id,
        }
        st.session_state.messages.append(msg)
        _render_answer(len(st.session_state.messages) - 1, msg)


def _render_answer(idx, msg):
    st.markdown(msg["answer"])
    _render_sources(msg["sources"])
    st.caption(t("chat.answer_meta", latency=msg["latency"], prompt=msg["prompt_version"], k=msg["k"]))
    _render_feedback(idx, msg)


def _render_message(idx, msg):
    with st.chat_message("user", avatar="👤"):
        st.markdown(msg["question"])
    with st.chat_message("assistant", avatar="🤖"):
        _render_answer(idx, msg)
