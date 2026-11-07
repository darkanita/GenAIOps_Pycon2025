# app/main_interface.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
st.set_page_config(page_title="Ritual Seg · Gestión del Gasto", page_icon="🛡️", layout="wide")

from app.chat_ui import render_chat
from app.metrics_ui import render_metrics

modo = st.sidebar.radio("Selecciona una vista:", ["💼 Asistente de gastos", "📊 Métricas"])
st.sidebar.divider()

if modo == "💼 Asistente de gastos":
    render_chat()
else:
    render_metrics()
