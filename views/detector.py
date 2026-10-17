# ============================================================
# Bilingual Hate Speech Detection — Detector page
# Other pages live in views/; app.py wires them together.
# ============================================================

import html

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from hsd import config
from hsd.pipeline import HATE, analyze
from hsd.ui import (GOLD, GREEN, NAVY, NAVY_2, PLOT_CFG, RED, SLATE, classifier_or_stop, footer,
                    get_translator, hero, kpis, log_prediction, section, setup, style_fig)

setup("Detector")

EXAMPLES = {
    "— choose an example —": "",
    "Neutral (English)": "The workshop on responsible AI was really useful for our students.",
    "Neutral (Bangla)": "আজকের সেমিনারটা খুব শিক্ষণীয় ছিল।",
    "Criticism, not hate (English)": "I strongly disagree with this policy and I think the minister is wrong.",
    "Hostile (English)": "People like you are disgusting trash and should go back where you came from.",
}

hero(
    "DynamicFusionNet · live research demo",
    "Bilingual Hate Speech Detector",
    "A BERT + XLM-R + DeBERTa ensemble distilled into one BERT-base student. Bangla input is translated "
    "to English, and every prediction is explained and checked for uncertainty.",
    ["Accuracy 91.04%", "ROC-AUC 0.9652", "4.8× smaller than the ensemble", "Bangla & English"],
    gold=2,
)

# ---------------------------------------------------------------- input
with st.container(border=True):
    left, right = st.columns([3, 1.25], gap="large")
    with right:
        ex = st.selectbox("Try an example", list(EXAMPLES))
        show_mc = st.toggle("Uncertainty (MC-dropout)", value=True, help=f"{config.MC_PASSES} stochastic forward passes")
        show_attr = st.toggle("Explain prediction", value=True,
                              help="Word occlusion: how much P(hate) changes when each word is removed")
    with left:
        text = st.text_area("Text to analyse (English or Bangla)", value=EXAMPLES[ex], height=128,
                            max_chars=config.MAX_CHARS, placeholder="Type or paste a sentence…")
        go_btn = st.button("Analyse", type="primary")

if not go_btn:
    section("How it works")
    c = st.columns(4)
    steps = [("1 · Detect language", "Bangla script is recognised automatically."),
             ("2 · Translate", "Bangla is translated to English with MarianMT (opus-mt-bn-en)."),
             ("3 · Classify", f"The distilled student scores the text; above {config.THRESHOLD} is hate."),
             ("4 · Explain & check", "Word-level attribution and MC-dropout uncertainty for every result.")]
    for col, (h, b) in zip(c, steps):
        with col.container(border=True):
            st.markdown(f"**{h}**")
            st.caption(b)
    footer()
    st.stop()

if not text.strip():
    st.warning("Please enter some text first.")
    st.stop()

clf = classifier_or_stop()
try:
    with st.spinner("Analysing…"):
        a = analyze(text, clf, get_translator, mc=show_mc, explain=show_attr)
except Exception as e:
    st.error(f"Analysis failed: {type(e).__name__}: {e}")
    st.stop()
log_prediction(a)

if a.truncated:
    st.info(f"Input was longer than {config.MAX_CHARS} characters and was truncated.")

# ---------------------------------------------------------------- verdict
section("Result")
v1, v2 = st.columns([1.05, 1], gap="large")
with v1:
    cls, icon = ("hate", "⚠️") if a.label == HATE else ("ok", "✅")
    st.markdown(
        f'<div class="verdict {cls}"><div class="ic">{icon}</div><div><div class="t">{a.label}</div>'
        f'<div class="s">{a.band} · P(hate) {a.p_hate*100:.1f}% · threshold {config.THRESHOLD}</div></div></div>',
        unsafe_allow_html=True,
    )
    kpis([
        ("Language", "Bangla" if a.lang == "bn" else "English"),
        ("Words", len(a.text.split())),
        ("Decision margin", f"{a.margin:.3f}"),
    ])
with v2:
    t = config.THRESHOLD * 100
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=a.p_hate * 100, number={"suffix": "%", "font": {"size": 40, "color": NAVY}},
        title={"text": "Probability of hate speech", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": SLATE},
            "bar": {"color": RED if a.label == HATE else GREEN, "thickness": 0.28},
            "steps": [{"range": [0, t - 10], "color": "#E9F5EE"},
                      {"range": [t - 10, t + 10], "color": "#F6EEDF"},
                      {"range": [t + 10, 100], "color": "#FBEAEA"}],
            "threshold": {"line": {"color": NAVY, "width": 3}, "thickness": 0.85, "value": t},
        },
    ))
    style_fig(gauge, 250).update_layout(margin=dict(l=36, r=44, t=40, b=6))
    st.plotly_chart(gauge, use_container_width=True, config=PLOT_CFG)
    st.caption("Green: clearly non-hate · gold: borderline zone (±0.10 around the threshold) · red: clearly hate.")

# ---------------------------------------------------------------- detail tabs
tabs = ["Explanation", "Uncertainty", "Performance"]
if a.lang == "bn":
    tabs.insert(0, "Translation")
T = dict(zip(tabs, st.tabs(tabs)))

if "Translation" in T:
    with T["Translation"]:
        c1, c2 = st.columns(2)
        with c1.container(border=True):
            st.caption("Original (Bangla)")
            st.markdown(html.escape(a.text))
        with c2.container(border=True):
            st.caption("Translated (English) — what the classifier saw")
            st.markdown(html.escape(a.classified_text))
        st.caption("Translation can soften slang or sarcasm; treat Bangla results with extra care.")

with T["Explanation"]:
    if not a.attribution:
        st.info("Turn on **Explain prediction** to see which words drove the decision.")
    else:
        st.caption("Each word's impact = drop in P(hate) when that word is removed. Red pushes toward hate, blue away from it.")
        peak = max((abs(v) for _, v in a.attribution), default=0) or 1
        chips = []
        for word, val in a.attribution:
            alpha = min(abs(val) / peak, 1) * 0.72 + 0.06
            rgb = "180,35,24" if val > 0 else "29,78,216"
            chips.append(f'<span class="tok" style="background:rgba({rgb},{alpha:.2f})" '
                         f'title="{val:+.3f}">{html.escape(word)}</span>')
        st.markdown("".join(chips), unsafe_allow_html=True)

        top = [x for x in sorted(a.attribution, key=lambda x: abs(x[1]), reverse=True) if abs(x[1]) >= 0.005][:12][::-1]
        if not top:
            st.caption("No single word changes the score by more than 0.005: the decision rests on the sentence as a whole.")
        bar = go.Figure(go.Bar(
            x=[v for _, v in top], y=[f"{w}  " for w, _ in top], orientation="h",
            marker_color=[RED if v > 0 else "#1D4ED8" for _, v in top],
            text=[f"{v:+.3f}" for _, v in top], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
        ))
        bar.add_vline(x=0, line_color=SLATE, line_width=1)
        if top:
            st.plotly_chart(style_fig(bar, max(240, 34 * len(top) + 80), "Most influential words"),
                            use_container_width=True, config=PLOT_CFG)

with T["Uncertainty"]:
    kpis([
        ("Decision margin", f"{a.margin:.3f}", "|P(hate) − threshold|"),
        ("Prediction entropy", f"{a.entropy_bits:.3f} bits", "0 = certain, 1 = coin flip"),
        ("MC mean ± std", f"{a.mc['mean']*100:.1f}% ± {a.mc['std']*100:.1f}%" if a.mc else "—"),
        ("Label stability", f"{a.mc['stability']*100:.0f}%" if a.mc else "—", "MC passes agreeing"),
    ])
    if a.mc:
        s = np.array(a.mc["samples"]) * 100
        hist = go.Figure()
        hist.add_vrect(x0=a.mc["lo"] * 100, x1=a.mc["hi"] * 100, fillcolor=GOLD, opacity=0.14, line_width=0,
                       annotation_text="95% interval", annotation_position="top left")
        hist.add_trace(go.Histogram(x=s, nbinsx=min(20, max(5, len(s))), marker_color=NAVY_2, opacity=0.85,
                                    name="MC samples", hovertemplate="%{x:.1f}%: %{y}<extra></extra>"))
        hist.add_vline(x=config.THRESHOLD * 100, line_dash="dot", line_color=RED, annotation_text="threshold")
        hist.add_vline(x=a.p_hate * 100, line_color=NAVY, line_width=3, annotation_text="prediction",
                       annotation_position="bottom right")
        hist.update_xaxes(title="P(hate) across dropout samples (%)", range=[0, 100])
        hist.update_yaxes(title="Passes")
        st.plotly_chart(style_fig(hist, 320, f"Prediction stability over {a.mc['passes']} MC-dropout passes"),
                        use_container_width=True, config=PLOT_CFG)
        if a.mc["stability"] < 0.8:
            st.warning("The label flips across dropout samples: this input is genuinely ambiguous for the model.")
    else:
        st.info("Turn on **Uncertainty (MC-dropout)** to see how stable this prediction is.")

with T["Performance"]:
    tm = a.timings_ms
    fp = clf.footprint_mb()
    kpis([
        ("End-to-end", f"{tm['end_to_end']:.0f} ms"),
        ("Device", clf.device.upper()),
        ("Parameters", f"{clf.n_params/1e6:.1f} M" if clf.n_params else "—"),
        ("Model size", f"{fp:.0f} MB" if fp else "—"),
    ])
    stages = [("Translation", tm.get("translation"), GOLD), ("Inference", tm.get("inference"), NAVY),
              ("Uncertainty", tm.get("mc_dropout"), NAVY_2), ("Explanation", tm.get("attribution"), SLATE)]
    stages = [(n, v, c) for n, v, c in stages if v is not None]
    lat = go.Figure()
    for n, v, c in stages:
        lat.add_trace(go.Bar(x=[v], y=["This request"], orientation="h", name=n, marker_color=c,
                             text=[f"{n} {v:.0f} ms"], textposition="inside", insidetextanchor="middle",
                             hovertemplate=f"{n}: %{{x:.0f}} ms<extra></extra>"))
    lat.update_layout(barmode="stack", showlegend=False)
    lat.update_xaxes(title="Milliseconds")
    st.plotly_chart(style_fig(lat, 190, "Where the time went"), use_container_width=True, config=PLOT_CFG)
    st.caption("Measured live on this server for this request. The first request after a restart also loads the model.")

st.markdown(
    '<p class="note" style="margin-top:1.2rem">⚠️ For research and education only. Automated predictions can be wrong, '
    "especially for sarcasm, dialect and reclaimed language; a human should review any real moderation decision.</p>",
    unsafe_allow_html=True,
)
footer()
