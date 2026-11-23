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

from app.i18n import t
from app.tracking import CHAT_EXPERIMENT, EVAL_EXPERIMENT

CONFIG_PARAMS = ["llm_provider", "llm_model", "prompt", "k", "chunk_size", "embeddings_provider", "judge_model"]
EXPECTATION_NAMES = {"expected_response", "expected_facts", "guidelines"}
CONFIG_COL = t("metrics.config_col")
QUESTION_COL, ANSWER_COL = t("metrics.question_col"), t("metrics.answer_col")
APP_COL, USEFUL_COL, COMMENT_COL = (t("metrics.app_version_col"), t("metrics.useful_col"),
                                    t("metrics.comment_col"))


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
        st.info(t("metrics.no_evals"))
        return None

    metric_cols = [c for c in runs.columns if c.startswith("metrics.") and c.endswith("/mean")]
    param_cols = [f"params.{p}" for p in CONFIG_PARAMS if f"params.{p}" in runs.columns]

    table = runs[["run_id", "tags.mlflow.runName", "start_time", *param_cols, *metric_cols]].copy()
    table.columns = [c.replace("params.", "").replace("metrics.", "").replace("/mean", "")
                     .replace("tags.mlflow.runName", CONFIG_COL) for c in table.columns]
    metrics = [c.replace("metrics.", "").replace("/mean", "") for c in metric_cols]

    st.subheader(t("metrics.compare_header"))
    st.caption(t("metrics.compare_caption"))
    st.dataframe(
        table.drop(columns=["run_id"]),
        width="stretch", hide_index=True,
        column_config={m: st.column_config.ProgressColumn(m, min_value=0, max_value=1, format="%.2f")
                       for m in metrics},
    )

    selected = st.multiselect(t("metrics.pick_metrics"), metrics, default=metrics[:4])
    if selected:
        chart = table.drop_duplicates(CONFIG_COL).set_index(CONFIG_COL)[selected]
        st.bar_chart(chart, stack=False)

    return table


def _render_question_detail(table, experiment):
    st.subheader(t("metrics.detail_header"))
    options = {f"{r[CONFIG_COL]} · {r['start_time']:%Y-%m-%d %H:%M}": r["run_id"]
               for _, r in table.iterrows()}
    run_id = options[st.selectbox(t("metrics.evaluation"), list(options))]

    traces = mlflow.search_traces(locations=[experiment.experiment_id], run_id=run_id, return_type="list")
    if not traces:
        st.info(t("metrics.no_traces"))
        return

    rows, details = [], []
    for tr in traces:
        q = _question(tr.data.request)
        row = {t("metrics.question_col"): q}
        for a in _feedback_items(tr):
            score = _to_score(a.feedback.value)
            row[a.name] = "⚠️" if a.feedback.error else ("✅" if score == 1 else "❌" if score == 0 else a.feedback.value)
        rows.append(row)
        details.append((q, _answer(tr.data.response), list(_feedback_items(tr))))

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    with st.expander(t("metrics.judges_expander")):
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
    st.subheader(t("metrics.users_header"))
    exp = mlflow.get_experiment_by_name(CHAT_EXPERIMENT)
    if exp is None:
        st.info(t("metrics.no_conversations"))
        return
    traces = mlflow.search_traces(locations=[exp.experiment_id], max_results=500, return_type="list")

    rows = []
    for tr in traces:
        for a in tr.info.assessments or []:
            if a.name == "user_feedback" and a.feedback is not None:
                rows.append({
                    APP_COL: (tr.info.trace_metadata or {}).get("mlflow.modelId", "—"),
                    QUESTION_COL: _question(tr.data.request),
                    ANSWER_COL: _answer(tr.data.response),
                    USEFUL_COL: _to_score(a.feedback.value),
                    COMMENT_COL: a.rationale or "",
                })
    if not rows:
        st.info(t("metrics.no_feedback"))
        return

    df = pd.DataFrame(rows)
    c1, c2, c3 = st.columns(3)
    c1.metric(t("metrics.rated"), len(df))
    c2.metric(t("metrics.satisfaction_metric"), f"{df[USEFUL_COL].mean():.0%}")
    c3.metric(t("metrics.negatives_metric"), int((df[USEFUL_COL] == 0).sum()))

    names = {m.model_id: m.name for m in mlflow.search_logged_models(
        experiment_ids=[exp.experiment_id], output_format="list")}
    df[APP_COL] = df[APP_COL].map(lambda m: names.get(m, m))
    st.dataframe(df.groupby(APP_COL)[USEFUL_COL].agg(["mean", "count"])
                 .rename(columns={"mean": t("metrics.satisfaction_col"), "count": t("metrics.ratings_col")}),
                 width="stretch",
                 column_config={t("metrics.satisfaction_col"): st.column_config.ProgressColumn(
                     t("metrics.satisfaction_col"), min_value=0, max_value=1, format="percent")})

    negatives = df[df[USEFUL_COL] == 0]
    if not negatives.empty:
        with st.expander(t("metrics.negatives_expander", n=len(negatives))):
            st.dataframe(negatives[[QUESTION_COL, ANSWER_COL, COMMENT_COL]],
                         width="stretch", hide_index=True)


def render_metrics():
    st.title(t("metrics.title"))
    st.caption(t("metrics.caption"))

    experiment = mlflow.get_experiment_by_name(EVAL_EXPERIMENT)
    tab_eval, tab_users = st.tabs([t("metrics.tab_eval"), t("metrics.tab_users")])
    with tab_eval:
        if experiment is None:
            st.info(t("metrics.no_evals"))
        else:
            table = _render_comparison(experiment)
            if table is not None:
                st.divider()
                _render_question_detail(table, experiment)
    with tab_users:
        _render_user_feedback()
