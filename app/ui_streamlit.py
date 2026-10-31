# app/ui_streamlit.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
st.set_page_config(page_title="Chatbot GenAI RRHH", page_icon="🤖", layout="centered")

from app.chat_ui import render_chat

render_chat(title="🤖 Asistente de Recursos Humanos - Contoso")
