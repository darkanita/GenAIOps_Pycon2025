# app/chat_ui.py
"""
Componente de chat reutilizable para ui_streamlit.py y main_interface.py.

Incluye:
- Preguntas de ejemplo (tomadas de tests/eval_dataset.json)
- Referencias: documento, página y fragmento usado para responder
- Panel lateral con proveedor/modelo, versión de prompt y nº de fragmentos (k)
- Caché del vectorstore y de la cadena (no se recargan en cada interacción)
- Feedback 👍/👎 registrado en MLflow (experimento "chat_feedback")
- Tiempo de respuesta
"""

import json
import os
import time

import mlflow
import streamlit as st

from app.llm_provider import describe
from app.rag_pipeline import (
    build_chain,
    list_prompt_versions,
    load_vectorstore_from_disk,
)

EVAL_DATASET = "tests/eval_dataset.json"
FEEDBACK_EXPERIMENT = "chat_feedback"
FALLBACK_EXAMPLES = [
    "¿Qué beneficios de salud ofrece la empresa?",
    "¿Qué incluye el programa PerksPlus?",
    "¿Cuáles son las responsabilidades de un analista de datos?",
]


# ---------- Recursos cacheados ----------

@st.cache_resource(show_spinner="Cargando índice vectorial...")
def get_vectordb():
    return load_vectorstore_from_disk()


@st.cache_resource(show_spinner="Preparando el asistente...")
def get_chain(prompt_version: str, k: int):
    return build_chain(get_vectordb(), prompt_version=prompt_version, k=k)


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
        name = os.path.basename(meta.get("source", "desconocido"))
        page = meta.get("page")
        page = page + 1 if isinstance(page, int) else None  # PyPDFLoader cuenta desde 0
        key = (name, page)
        if key not in refs:
            refs[key] = doc.page_content.strip()
    return [{"document": n, "page": p, "snippet": t} for (n, p), t in refs.items()]


def _log_feedback(msg, rating):
    try:
        mlflow.set_experiment(FEEDBACK_EXPERIMENT)
        with mlflow.start_run(run_name="feedback"):
            mlflow.log_params({
                "question": msg["question"][:500],
                "prompt_version": msg["prompt_version"],
                "k": msg["k"],
                **describe(),
            })
            mlflow.log_metric("thumbs_up", 1 if rating == "up" else 0)
            mlflow.log_metric("latency_s", msg["latency"])
            mlflow.log_text(msg["answer"], "answer.txt")
            mlflow.log_dict({"sources": msg["sources"]}, "sources.json")
        return True
    except Exception as e:  # el feedback nunca debe tumbar la app
        st.warning(f"No se pudo registrar el feedback en MLflow: {e}")
        return False


def _render_sources(sources):
    if not sources:
        return
    with st.expander(f"📚 Referencias ({len(sources)})"):
        for i, ref in enumerate(sources, 1):
            page = f" — página {ref['page']}" if ref["page"] else ""
            st.markdown(f"**{i}. {ref['document']}**{page}")
            snippet = ref["snippet"]
            st.caption(snippet[:400] + ("…" if len(snippet) > 400 else ""))


def _render_feedback(idx, msg):
    if msg.get("feedback"):
        st.caption("✅ Gracias por tu feedback" if msg["feedback"] == "up"
                   else "📝 Gracias, lo usaremos para mejorar")
        return
    c1, c2, _ = st.columns([1, 1, 10])
    for col, rating, icon in ((c1, "up", "👍"), (c2, "down", "👎")):
        col.button(icon, key=f"fb_{rating}_{idx}", help="Útil" if rating == "up" else "No útil",
                   on_click=_on_feedback, args=(idx, rating))


def _on_feedback(idx, rating):
    msg = st.session_state.messages[idx]
    if _log_feedback(msg, rating):
        msg["feedback"] = rating


def _new_conversation():
    st.session_state.messages = []


# ---------- Sidebar ----------

def render_sidebar():
    info = describe()
    st.sidebar.markdown("### ⚙️ Configuración")
    st.sidebar.markdown(
        f"**LLM:** `{info['llm_provider']}` · `{info['llm_model']}`  \n"
        f"**Embeddings:** `{info['embeddings_provider']}` · `{info['embeddings_model'].split('/')[-1]}`"
    )

    versions = list_prompt_versions()
    default = os.getenv("PROMPT_VERSION", "v1_asistente_rrhh")
    prompt_version = st.sidebar.selectbox(
        "Versión de prompt", versions,
        index=versions.index(default) if default in versions else 0,
    )
    k = st.sidebar.slider("Fragmentos a recuperar (k)", 1, 10, 4,
                          help="Cuántos fragmentos de los PDFs se envían al LLM como contexto")

    st.sidebar.button("🗑️ Nueva conversación", use_container_width=True,
                      on_click=_new_conversation)

    return {"prompt_version": prompt_version, "k": k}


# ---------- Chat ----------

def render_chat(title="🤖 Asistente de Recursos Humanos",
                subtitle="Pregunta sobre beneficios, políticas y roles de Contoso Electronics."):
    config = render_sidebar()

    st.title(title)
    st.caption(subtitle)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    try:
        chain = get_chain(config["prompt_version"], config["k"])
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    # Preguntas de ejemplo (solo al inicio de la conversación)
    pending = None
    examples_box = st.empty()
    if not st.session_state.messages:
        with examples_box.container():
            st.markdown("##### 💡 Prueba con alguna de estas preguntas")
            cols = st.columns(2)
            for i, q in enumerate(get_example_questions()):
                if cols[i % 2].button(q, key=f"ex_{i}", use_container_width=True):
                    pending = q

    # Historial
    for idx, msg in enumerate(st.session_state.messages):
        _render_message(idx, msg)

    question = st.chat_input("¿Qué deseas consultar?") or pending
    if not question:
        return

    examples_box.empty()  # oculta los ejemplos al empezar la conversación

    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Consultando documentos..."):
            history = [(m["question"], m["answer"]) for m in st.session_state.messages]
            start = time.perf_counter()
            try:
                result = chain.invoke({"question": question, "chat_history": history})
            except Exception as e:
                st.error(f"Error al consultar el modelo: {e}")
                st.stop()
            latency = time.perf_counter() - start

        msg = {
            "question": question,
            "answer": result["answer"],
            "sources": _format_sources(result.get("source_documents")),
            "latency": latency,
            "prompt_version": config["prompt_version"],
            "k": config["k"],
        }
        st.session_state.messages.append(msg)
        _render_answer(len(st.session_state.messages) - 1, msg)


def _render_answer(idx, msg):
    st.markdown(msg["answer"])
    _render_sources(msg["sources"])
    st.caption(f"⏱️ {msg['latency']:.1f} s · prompt `{msg['prompt_version']}` · k={msg['k']}")
    _render_feedback(idx, msg)


def _render_message(idx, msg):
    with st.chat_message("user", avatar="👤"):
        st.markdown(msg["question"])
    with st.chat_message("assistant", avatar="🤖"):
        _render_answer(idx, msg)
