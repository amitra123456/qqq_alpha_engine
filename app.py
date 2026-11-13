"""
Standalone Streamlit Entrypoint for QQQ Alpha Engine
==================================================
Launch from repo root:
    python -m streamlit run app.py --server.port 8501 --server.headless true
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ui.terminal import render_pit_ma200_terminal

st.set_page_config(
    page_title="PIT Cap-Weight + QQQ 200d MA Terminal",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .stApp {
            background-color: #0B0E14;
            color: #F1F5F9;
        }
        .block-container {
            padding-top: 0.75rem !important;
            padding-bottom: 1rem !important;
            padding-left: 1.25rem !important;
            padding-right: 1.25rem !important;
            max-width: 100% !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

render_pit_ma200_terminal(data_outputs_dir=REPO_ROOT / "data" / "outputs", key_prefix="standalone_term")
