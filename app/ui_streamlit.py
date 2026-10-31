# app/ui_streamlit.py
"""
Punto de entrada: solo el chat.

    streamlit run app/ui_streamlit.py

Toda la lógica está en app/chat_ui.py.
"""
import sys, os
# Permite "from app..." al ejecutar con streamlit desde la raíz del repo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
st.set_page_config(page_title="Chatbot GenAI RRHH", page_icon="🤖", layout="centered")

from app.chat_ui import render_chat

render_chat(title="🤖 Asistente de Recursos Humanos - Contoso")
