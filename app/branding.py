# app/branding.py
"""
Identidad visual de Ritual Seg (azul y blanco).
Los colores base están en .streamlit/config.toml; aquí solo se agrega el
encabezado corporativo y algunos ajustes de estilo.
"""

import streamlit as st

COMPANY = "Ritual Seg"
COMPANY_LEGAL = "Ritual Seg S.A."
TAGLINE = "Entidad financiera aseguradora"
BLUE = "#0047AB"
NAVY = "#0A2E6E"

_CSS = f"""
<style>
.rs-header {{
    background: linear-gradient(90deg, {NAVY} 0%, {BLUE} 100%);
    color: #FFFFFF;
    padding: 1.1rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 1rem;
}}
.rs-header .rs-brand {{
    font-size: 0.8rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    opacity: 0.85;
}}
.rs-header h1 {{
    color: #FFFFFF;
    font-size: 1.7rem;
    margin: 0.15rem 0 0.25rem 0;
    padding: 0;
}}
.rs-header p {{
    color: #DCE8FF;
    margin: 0;
    font-size: 0.95rem;
}}
section[data-testid="stSidebar"] .rs-side-brand {{
    font-weight: 700;
    font-size: 1.15rem;
    letter-spacing: 0.06em;
    color: #FFFFFF;
}}
section[data-testid="stSidebar"] .rs-side-tag {{
    color: #BFD4FF;
    font-size: 0.8rem;
}}
</style>
"""


def apply_brand():
    """Inyecta el CSS corporativo y la marca en la barra lateral."""
    st.markdown(_CSS, unsafe_allow_html=True)
    st.sidebar.markdown(
        f'<div class="rs-side-brand">🛡️ {COMPANY.upper()}</div>'
        f'<div class="rs-side-tag">{TAGLINE}</div>',
        unsafe_allow_html=True,
    )


def render_header(title: str, subtitle: str = ""):
    """Encabezado azul con el nombre de la compañía."""
    st.markdown(
        f'<div class="rs-header"><div class="rs-brand">{COMPANY_LEGAL} · {TAGLINE}</div>'
        f"<h1>{title}</h1><p>{subtitle}</p></div>",
        unsafe_allow_html=True,
    )
