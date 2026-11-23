# app/main_interface.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from app.i18n import t
st.set_page_config(page_title=t("ui.page_title"), page_icon="📚", layout="wide")

from app.chat_ui import render_chat
from app.metrics_ui import render_metrics

modo = st.sidebar.radio(t("ui.pick_view"), [t("ui.view_chat"), t("ui.view_metrics")])
st.sidebar.divider()

if modo == t("ui.view_chat"):
    render_chat()
else:
    render_metrics()
