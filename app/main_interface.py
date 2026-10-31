# app/main_interface.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
st.set_page_config(page_title="Chatbot GenAI + Métricas", page_icon="📚", layout="wide")

from app.chat_ui import render_chat
from app.metrics_ui import render_metrics

modo = st.sidebar.radio("Selecciona una vista:", ["🤖 Chatbot", "📊 Métricas"])
st.sidebar.divider()

if modo == "🤖 Chatbot":
    render_chat()
else:
    render_metrics()
