# app/metrics_ui.py
"""
Vista de métricas compartida por dashboard.py y main_interface.py.

1. Comparación de configuraciones: un run por evaluación (mlflow.genai.evaluate)
   con las métricas agregadas de cada juez.
2. Detalle por pregunta: trazas del run con el veredicto y razonamiento de cada juez.
3. Feedback de usuarios: 👍/👎 registrados sobre las trazas del chatbot.
"""

import json

import mlflow
import pandas as pd
import streamlit as st

from app.tracking import CHAT_EXPERIMENT, EVAL_EXPERIMENT

CONFIG_PARAMS = ["llm_provider", "llm_model", "prompt", "k", "chunk_size", "embeddings_provider", "judge_model"]
EXPECTATION_NAMES = {"expected_response", "expected_facts", "guidelines"}


# ---------- Utilidades ----------

def _to_score(value):
    """Convierte el veredicto de un juez a número (yes/no, True/False, numérico)."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("yes", "true", "pass"):
            return 1.0
        if v in ("no", "false", "fail"):
            return 0.0
    return None


def _question(request):
    if isinstance(request, str):
        try:
            request = json.loads(request)
        except Exception:
            return request
    if isinstance(request, dict):
        return request.get("question") or request.get("inputs", {}).get("question") or str(request)
    return str(request)


def _answer(response):
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except Exception:
            return response
    if isinstance(response, dict):
        return response.get("answer") or str(response)
    return str(response)


def _feedback_items(trace):
    for a in trace.info.assessments or []:
        if a.name in EXPECTATION_NAMES or getattr(a, "feedback", None) is None:
            continue
        yield a


# ---------- Secciones ----------

def _render_comparison(experiment):
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id],
                              order_by=["start_time DESC"])
    if runs.empty:
        st.info("Aún no hay evaluaciones. Ejecuta `python app/run_eval.py`.")
        return None

    metric_cols = [c for c in runs.columns if c.startswith("metrics.") and c.endswith("/mean")]
    param_cols = [f"params.{p}" for p in CONFIG_PARAMS if f"params.{p}" in runs.columns]

    table = runs[["run_id", "tags.mlflow.runName", "start_time", *param_cols, *metric_cols]].copy()
    table.columns = [c.replace("params.", "").replace("metrics.", "").replace("/mean", "")
                     .replace("tags.mlflow.runName", "configuración") for c in table.columns]
    metrics = [c.replace("metrics.", "").replace("/mean", "") for c in metric_cols]

    st.subheader("⚖️ Comparación de configuraciones")
    st.caption("Promedio de cada juez (0 a 1) por evaluación. Cada fila es una ejecución de `run_eval.py`.")
    st.dataframe(
        table.drop(columns=["run_id"]),
        width="stretch", hide_index=True,
        column_config={m: st.column_config.ProgressColumn(m, min_value=0, max_value=1, format="%.2f")
                       for m in metrics},
    )

    selected = st.multiselect("Métricas a graficar", metrics, default=metrics[:4])
    if selected:
        chart = table.drop_duplicates("configuración").set_index("configuración")[selected]
        st.bar_chart(chart, stack=False)

    return table


def _render_question_detail(table, experiment):
    st.subheader("🔎 Detalle por pregunta")
    options = {f"{r['configuración']} · {r['start_time']:%Y-%m-%d %H:%M}": r["run_id"]
               for _, r in table.iterrows()}
    run_id = options[st.selectbox("Evaluación", list(options))]

    traces = mlflow.search_traces(locations=[experiment.experiment_id], run_id=run_id, return_type="list")
    if not traces:
        st.info("Este run no tiene trazas.")
        return

    rows, details = [], []
    for t in traces:
        q = _question(t.data.request)
        row = {"pregunta": q}
        for a in _feedback_items(t):
            score = _to_score(a.feedback.value)
            row[a.name] = "⚠️" if a.feedback.error else ("✅" if score == 1 else "❌" if score == 0 else a.feedback.value)
        rows.append(row)
        details.append((q, _answer(t.data.response), list(_feedback_items(t))))

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    with st.expander("Ver respuestas y razonamiento de los jueces"):
        for q, ans, items in details:
            st.markdown(f"**❓ {q}**")
            st.markdown(f"🤖 {ans}")
            for a in items:
                if a.feedback.error:
                    st.caption(f"⚠️ **{a.name}**: {a.feedback.error.error_message[:200]}")
                else:
                    st.caption(f"**{a.name}** → `{a.feedback.value}` — {a.rationale or ''}")
            st.divider()


def _render_user_feedback():
    st.subheader("👥 Feedback de usuarios")
    exp = mlflow.get_experiment_by_name(CHAT_EXPERIMENT)
    if exp is None:
        st.info("Todavía no hay conversaciones registradas.")
        return
    traces = mlflow.search_traces(locations=[exp.experiment_id], max_results=500, return_type="list")

    rows = []
    for t in traces:
        for a in t.info.assessments or []:
            if a.name == "user_feedback" and a.feedback is not None:
                rows.append({
                    "versión de la app": (t.info.trace_metadata or {}).get("mlflow.modelId", "—"),
                    "pregunta": _question(t.data.request),
                    "respuesta": _answer(t.data.response),
                    "útil": _to_score(a.feedback.value),
                    "comentario": a.rationale or "",
                })
    if not rows:
        st.info("Aún no hay 👍/👎 de usuarios.")
        return

    df = pd.DataFrame(rows)
    c1, c2, c3 = st.columns(3)
    c1.metric("Respuestas calificadas", len(df))
    c2.metric("👍 Satisfacción", f"{df['útil'].mean():.0%}")
    c3.metric("👎 Negativas", int((df["útil"] == 0).sum()))

    names = {m.model_id: m.name for m in mlflow.search_logged_models(
        experiment_ids=[exp.experiment_id], output_format="list")}
    df["versión de la app"] = df["versión de la app"].map(lambda m: names.get(m, m))
    st.dataframe(df.groupby("versión de la app")["útil"].agg(["mean", "count"])
                 .rename(columns={"mean": "satisfacción", "count": "calificaciones"}),
                 width="stretch",
                 column_config={"satisfacción": st.column_config.ProgressColumn(
                     "satisfacción", min_value=0, max_value=1, format="percent")})

    negatives = df[df["útil"] == 0]
    if not negatives.empty:
        with st.expander(f"Respuestas con 👎 ({len(negatives)}) — candidatas para el dataset de evaluación"):
            st.dataframe(negatives[["pregunta", "respuesta", "comentario"]],
                         width="stretch", hide_index=True)


def render_metrics():
    st.title("📈 Evaluación y monitoreo del chatbot")
    st.caption("Datos de MLflow. Para explorar trazas, prompts y versiones: `mlflow ui --port 5000`")

    experiment = mlflow.get_experiment_by_name(EVAL_EXPERIMENT)
    tab_eval, tab_users = st.tabs(["⚖️ Evaluación automática", "👥 Feedback de usuarios"])
    with tab_eval:
        if experiment is None:
            st.info("Aún no hay evaluaciones. Ejecuta `python app/run_eval.py`.")
        else:
            table = _render_comparison(experiment)
            if table is not None:
                st.divider()
                _render_question_detail(table, experiment)
    with tab_users:
        _render_user_feedback()
