# app/ui_streamlit.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from app.i18n import t
st.set_page_config(page_title=t("ui.chat_page_title"), page_icon="🤖", layout="centered")

from app.chat_ui import render_chat

render_chat(title=t("chat.title_contoso"))
