# app/dashboard.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from app.i18n import t
st.set_page_config(page_title=t("ui.dashboard_title"), page_icon="📊", layout="wide")

from app.metrics_ui import render_metrics

render_metrics()
