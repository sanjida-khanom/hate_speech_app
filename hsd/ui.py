"""Streamlit helpers shared by every page."""

from __future__ import annotations

import time

import streamlit as st

from . import config
from .model import WeightsNotFound, load_classifier
from .translate import load_translator

NAVY, RED, GREEN, GOLD, SLATE = "#1F3864", "#B42318", "#1E7A46", "#B08D57", "#64748B"

CSS = """
<style>
  .block-container{ padding-top:2rem; max-width:1100px; }
  h1, h2, h3{ letter-spacing:-.01em; }
  .hsd-sub{ color:#5B6577; font-size:.95rem; margin-top:-.6rem; }
  .verdict{ border-radius:14px; padding:18px 22px; font-size:1.35rem; font-weight:700; margin:.4rem 0 1rem; }
  .verdict.hate{ background:#FBEAEA; color:#B42318; border:1px solid #F3C3C0; }
  .verdict.ok{ background:#E9F5EE; color:#1E7A46; border:1px solid #BFE3CC; }
  .tok{ display:inline-block; padding:6px 10px; margin:3px; border-radius:8px; font-size:.98rem; }
  .stub{ background:#B42318; color:#fff; padding:10px 14px; border-radius:10px; font-weight:600; margin-bottom:1rem; }
  .note{ font-size:.85rem; color:#5B6577; }
  [data-testid="stMetricValue"]{ font-size:1.45rem; }
</style>
"""


def setup(title: str, icon: str = "🛡️"):
    st.set_page_config(page_title=f"{title} · Hate Speech Detector", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    if config.BACKEND == "stub":
        st.markdown(
            '<div class="stub">DEV STUB ACTIVE: predictions come from a keyword heuristic, '
            "not the research model. Never deploy with HSD_BACKEND=stub.</div>",
            unsafe_allow_html=True,
        )
    sidebar()


def sidebar():
    with st.sidebar:
        st.markdown("### 🛡️ Hate Speech Detector")
        st.caption("Distilled BERT-base student · Bangla/English input")
        st.markdown(
            f"**Threshold:** {config.THRESHOLD}  \n"
            f"**Max input:** {config.MAX_CHARS} characters  \n"
            f"**MC-dropout passes:** {config.MC_PASSES}"
        )
        st.divider()
        st.caption(
            "Research and educational use only. Texts are processed in memory for your "
            "session and are never stored on the server."
        )
        st.caption(
            "Authors (equal contribution): Sanjida Khanom · Mohammod Ashikur Rahman"
        )


@st.cache_resource(show_spinner="Loading the model (first visit only)…")
def get_classifier():
    return load_classifier()


@st.cache_resource(show_spinner="Loading the Bangla→English translator (first Bangla input only)…")
def get_translator():
    return load_translator()


def classifier_or_stop():
    """Load the model, or explain clearly why it cannot be loaded."""
    try:
        return get_classifier()
    except WeightsNotFound as e:
        st.error(
            "**Model weights are not available.** "
            f"{e} Set `HSD_MODEL_REPO` (and `HF_TOKEN` for a private repo) as "
            "Space secrets, or place `student_best.pt` next to `app.py`."
        )
    except Exception as e:  # network, corrupted file, version mismatch…
        st.error(f"**The model could not be loaded:** {type(e).__name__}: {e}")
    st.stop()


# ---- per-session log used by the Live Analytics page ---------------------
def log_prediction(a, source: str = "detector"):
    st.session_state.setdefault("log", []).append({
        "time": time.strftime("%H:%M:%S"),
        "source": source,
        "language": a.lang,
        "chars": len(a.text),
        "words": len(a.text.split()),
        "p_hate": a.p_hate,
        "label": a.label,
        "margin": a.margin,
        "end_to_end_ms": a.timings_ms.get("end_to_end"),
        "inference_ms": a.timings_ms.get("inference"),
        "translation_ms": a.timings_ms.get("translation"),
        "mc_std": (a.mc or {}).get("std"),
    })


def session_log():
    return st.session_state.get("log", [])
