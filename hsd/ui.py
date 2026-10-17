"""Shared look and feel: academic navy-and-gold theme, header banner,
section titles, KPI cards, author block, chart styling and cached models."""

from __future__ import annotations

import html
import time

import streamlit as st

from . import config
from .model import WeightsNotFound, load_classifier
from .translate import load_translator

NAVY, NAVY_2, GOLD = "#1F3864", "#2E5E8E", "#B08D57"
RED, GREEN, SLATE, INK = "#B42318", "#1E7A46", "#64748B", "#0E1A2E"
FONT = "Inter, system-ui, sans-serif"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea { font-family: 'Inter', system-ui, sans-serif; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }
h1, h2, h3 { font-family: 'Source Serif 4', Georgia, serif !important; letter-spacing: -.01em; color: #0E1A2E; }
[data-testid="stSidebar"] { background: #F3F5F9; border-right: 1px solid #E3E7EE; }

/* header banner */
.hero { position: relative; overflow: hidden; border-radius: 18px; padding: 28px 32px 26px;
        background: linear-gradient(125deg, #0B1A30 0%, #1F3864 55%, #2E5E8E 100%); color: #fff; margin-bottom: 1.4rem;
        box-shadow: 0 18px 40px -22px rgba(14,26,46,.55); }
.hero:before { content: ""; position: absolute; right: -120px; top: -140px; width: 380px; height: 380px; border-radius: 50%;
               background: radial-gradient(circle, rgba(176,141,87,.35), transparent 65%); }
.hero .kicker { font-size: .74rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: #E6D3AE; }
.hero h1 { color: #fff !important; font-size: 2.05rem; margin: .35rem 0 .45rem; line-height: 1.15; padding: 0; }
.hero p { color: rgba(255,255,255,.82); font-size: .98rem; margin: 0; max-width: 72ch; }
.hero .chips { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; }
.hero .chip { font-size: .78rem; font-weight: 600; color: #fff; background: rgba(255,255,255,.12);
              border: 1px solid rgba(255,255,255,.22); padding: 4px 11px; border-radius: 999px; }
.hero .chip.gold { background: rgba(176,141,87,.28); border-color: rgba(230,211,174,.5); }

/* section titles */
.sec { margin: 1.6rem 0 .7rem; }
.sec h3 { font-size: 1.32rem; margin: 0; padding: 0; }
.sec .rule { width: 46px; height: 2px; background: linear-gradient(90deg, #B08D57, transparent); margin-top: 7px; }
.sec .sub { color: #5B6577; font-size: .9rem; margin-top: 6px; }

/* KPI cards */
.kpis { display: grid; grid-template-columns: repeat(var(--n), minmax(0,1fr)); gap: 12px; margin: .4rem 0 .6rem; }
.kpi { background: #fff; border: 1px solid #E3E7EE; border-radius: 14px; padding: 14px 16px; box-shadow: 0 1px 2px rgba(14,26,46,.04); }
.kpi .l { font-size: .74rem; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; color: #7C879B; }
.kpi .v { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.55rem; font-weight: 700; color: #0E1A2E; margin-top: 2px; line-height: 1.2; }
.kpi .h { font-size: .78rem; color: #5B6577; margin-top: 2px; }
@media (max-width: 760px) { .kpis { grid-template-columns: repeat(2, minmax(0,1fr)); } }

/* verdict */
.verdict { border-radius: 16px; padding: 20px 22px; display: flex; align-items: center; gap: 14px; }
.verdict .ic { font-size: 1.9rem; }
.verdict .t { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.6rem; font-weight: 700; line-height: 1.1; }
.verdict .s { font-size: .86rem; margin-top: 3px; opacity: .85; }
.verdict.hate { background: #FBEAEA; color: #8F1C13; border: 1px solid #F1C2BE; }
.verdict.ok { background: #E9F5EE; color: #155C34; border: 1px solid #BFE3CC; }
.tok { display: inline-block; padding: 6px 10px; margin: 3px; border-radius: 8px; font-size: .97rem; color: #0E1A2E; }

/* author block, notes */
.who { font-size: .86rem; line-height: 1.45; padding-bottom: 8px; }
.who .n { font-weight: 700; color: #0E1A2E; }
.who .r { color: #5B6577; font-size: .8rem; }
.who .lab { font-size: .7rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; color: #B08D57; margin: 12px 0 6px; }
.stub { background: #B42318; color: #fff; padding: 10px 14px; border-radius: 10px; font-weight: 600; margin-bottom: 1rem; }
.note { font-size: .84rem; color: #5B6577; }
.foot { margin-top: 2.6rem; padding-top: 14px; border-top: 1px solid #E3E7EE; font-size: .8rem; color: #7C879B; }

/* widgets */
.stButton > button[kind="primary"], .stDownloadButton > button { background: #1F3864; border-color: #1F3864; border-radius: 10px; font-weight: 600; }
.stButton > button[kind="primary"]:hover { background: #2E5E8E; border-color: #2E5E8E; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #E3E7EE; }
.stTabs [data-baseweb="tab"] { font-weight: 600; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; }
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


def people_html(compact: bool = True) -> str:
    rows = []
    for a in config.AUTHORS:
        rows.append(f'<div><span class="n">{html.escape(a["name"])}</span><br>'
                    f'<span class="r">{html.escape(a["role"])}, {html.escape(a["org"])}</span></div>')
    sup = config.SUPERVISOR
    note = '<div class="r" style="margin-top:4px">Equal contribution</div>' if config.EQUAL_CONTRIBUTION else ""
    return (
        '<div class="who"><div class="lab">Authors</div>'
        + '<div style="display:grid;gap:8px">' + "".join(rows) + "</div>" + note
        + '<div class="lab">Supervisor</div>'
        + f'<div><span class="n">{html.escape(sup["name"])}</span><br>'
          f'<span class="r">{html.escape(sup["role"])}, {html.escape(sup["org"])}</span></div></div>'
    )


def sidebar():
    with st.sidebar:
        st.markdown("#### 🛡️ Hate Speech Detector")
        st.caption("Distilled BERT-base · Bangla & English input")
        st.markdown(people_html(), unsafe_allow_html=True)
        st.divider()
        st.caption(
            f"Threshold {config.THRESHOLD} · up to {config.MAX_CHARS} characters · "
            f"{config.MC_PASSES} MC-dropout passes"
        )
        st.caption("Research and educational use only. Texts are processed in memory for your session and never stored.")


def hero(kicker: str, title: str, subtitle: str, chips: list[str] | None = None, gold: int = 0):
    chip_html = "".join(
        f'<span class="chip{" gold" if i < gold else ""}">{html.escape(c)}</span>' for i, c in enumerate(chips or [])
    )
    st.markdown(
        f'<div class="hero"><div class="kicker">{html.escape(kicker)}</div><h1>{html.escape(title)}</h1>'
        f'<p>{subtitle}</p><div class="chips">{chip_html}</div></div>',
        unsafe_allow_html=True,
    )


def section(title: str, sub: str | None = None):
    st.markdown(
        f'<div class="sec"><h3>{html.escape(title)}</h3><div class="rule"></div>'
        + (f'<div class="sub">{sub}</div>' if sub else "") + "</div>",
        unsafe_allow_html=True,
    )


def kpis(items: list[tuple]):
    """items: (label, value) or (label, value, hint)."""
    cells = []
    for it in items:
        label, value, hint = (list(it) + [""])[:3]
        cells.append(f'<div class="kpi"><div class="l">{html.escape(str(label))}</div>'
                     f'<div class="v">{html.escape(str(value))}</div>'
                     + (f'<div class="h">{html.escape(str(hint))}</div>' if hint else "") + "</div>")
    st.markdown(f'<div class="kpis" style="--n:{len(items)}">{"".join(cells)}</div>', unsafe_allow_html=True)


def footer():
    names = " · ".join(a["name"] for a in config.AUTHORS)
    st.markdown(
        f'<div class="foot">{names} · Supervised by {config.SUPERVISOR["name"]}, '
        f'{config.SUPERVISOR["org"]}. Research and educational use only.</div>',
        unsafe_allow_html=True,
    )


def style_fig(fig, height: int = 340, title: str | None = None):
    """One consistent chart style for every page."""
    fig.update_layout(
        height=height, template="plotly_white", font=dict(family=FONT, size=13, color=INK),
        title=dict(text=title, font=dict(size=15, family=FONT, color=INK), x=0, xanchor="left") if title else None,
        margin=dict(l=10, r=16, t=52 if title else 16, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        # legend sits under the chart so it never collides with the title
        legend=dict(orientation="h", yanchor="top", y=-0.3, x=0, title=None),
        hoverlabel=dict(font_family=FONT),
    )
    fig.update_xaxes(gridcolor="#EEF1F5", zerolinecolor="#DDE2EA")
    fig.update_yaxes(gridcolor="#EEF1F5", zerolinecolor="#DDE2EA")
    return fig


PLOT_CFG = {"displayModeBar": False}


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
            f"{e} Add `HSD_MODEL_REPO` and `HF_TOKEN` to the app's secrets "
            "(Streamlit Cloud: Manage app → Settings → Secrets), then reboot the app."
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
        "band": a.band.split(":")[0],
        "end_to_end_ms": a.timings_ms.get("end_to_end"),
        "inference_ms": a.timings_ms.get("inference"),
        "translation_ms": a.timings_ms.get("translation"),
        "mc_std": (a.mc or {}).get("std"),
    })


def session_log():
    return st.session_state.get("log", [])
