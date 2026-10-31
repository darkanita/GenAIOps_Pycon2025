# app/dashboard.py
"""
Punto de entrada: solo el dashboard de evaluación y feedback.

    streamlit run app/dashboard.py

Toda la lógica está en app/metrics_ui.py.
"""
import sys, os
# Permite "from app..." al ejecutar con streamlit desde la raíz del repo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
st.set_page_config(page_title="Dashboard de Evaluación", page_icon="📊", layout="wide")

from app.metrics_ui import render_metrics

render_metrics()
