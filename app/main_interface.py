# app/main_interface.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
st.set_page_config(page_title="Chatbot GenAI + Métricas", page_icon="📚", layout="wide")

import pandas as pd
import mlflow
import json
from app.chat_ui import render_chat


modo = st.sidebar.radio("Selecciona una vista:", ["🤖 Chatbot", "📊 Métricas"])
st.sidebar.divider()

if modo == "🤖 Chatbot":
    render_chat()

elif modo == "📊 Métricas":
    st.title("📈 Resultados de Evaluación")

    client = mlflow.tracking.MlflowClient()
    experiments = [exp for exp in client.search_experiments() if exp.name.startswith("eval_")]

    if not experiments:
        st.warning("No se encontraron experimentos de evaluación.")
        st.stop()

    exp_names = [exp.name for exp in experiments]
    selected_exp = st.selectbox("Selecciona un experimento:", exp_names)

    experiment = client.get_experiment_by_name(selected_exp)
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])

    if not runs:
        st.warning("No hay ejecuciones registradas.")
        st.stop()

    # Armar dataframe
    data = []
    for run in runs:
        params = run.data.params
        metrics = run.data.metrics
        data.append({
            "Pregunta": params.get("question"),
            "Prompt": params.get("prompt_version"),
            "Chunk Size": int(params.get("chunk_size", 0)),
            "Correcto (LC)": metrics.get("lc_is_correct", 0)
        })

    df = pd.DataFrame(data)
    st.dataframe(df)

    # Agrupado
    st.subheader("📊 Promedio por configuración")
    grouped = df.groupby(["Prompt", "Chunk Size"]).agg({"Correcto (LC)": "mean"}).reset_index()
    grouped.rename(columns={"Correcto (LC)": "Precisión"}, inplace=True)
    grouped["config"] = grouped["Prompt"] + " | " + grouped["Chunk Size"].astype(str)
    st.bar_chart(grouped.set_index("config")["Precisión"])
