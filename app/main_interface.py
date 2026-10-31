# app/main_interface.py
"""
Punto de entrada: chat + métricas en una sola app, con selector en la barra lateral.

    streamlit run app/main_interface.py

- "🤖 Chatbot" → app/chat_ui.py
- "📊 Métricas" → app/metrics_ui.py
"""
import sys, os
# Permite "from app..." al ejecutar con streamlit desde la raíz del repo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
# set_page_config debe ser la primera llamada a Streamlit del script
st.set_page_config(page_title="Chatbot GenAI + Métricas", page_icon="📚", layout="wide")

from app.chat_ui import render_chat
from app.metrics_ui import render_metrics

modo = st.sidebar.radio("Selecciona una vista:", ["🤖 Chatbot", "📊 Métricas"])
st.sidebar.divider()

if modo == "🤖 Chatbot":
    render_chat()
else:
    render_metrics()
