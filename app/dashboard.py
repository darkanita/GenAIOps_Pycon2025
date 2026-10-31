# app/dashboard.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
st.set_page_config(page_title="Dashboard de Evaluación", page_icon="📊", layout="wide")

from app.metrics_ui import render_metrics

render_metrics()
