"""Live analytics for this session. Every number is measured from real
predictions made in the browser session; nothing here is simulated."""

import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from hsd import config
from hsd.pipeline import HATE, analyze
from hsd.ui import GOLD, GREEN, NAVY, RED, SLATE, classifier_or_stop, get_translator, log_prediction, session_log, setup

setup("Live Analytics", "📈")
st.title("📈 Live analytics")
st.markdown(
    '<p class="hsd-sub">Measured from the predictions made in your session. '
    "Run the latency benchmark to profile the deployed model on this server.</p>",
    unsafe_allow_html=True,
)

BENCH = "results/benchmark_texts.txt"

c1, c2, _ = st.columns([1.3, 1, 3])
if c1.button("Run latency benchmark", type="primary"):
    clf = classifier_or_stop()
    with open(BENCH, encoding="utf-8") as f:
        bench = [l.strip() for l in f if l.strip()]
    bar = st.progress(0.0, text="Benchmarking…")
    for i, t in enumerate(bench):
        log_prediction(analyze(t, clf, get_translator, mc=False, explain=False), source="benchmark")
        bar.progress((i + 1) / len(bench), text=f"Benchmarking… {i+1}/{len(bench)}")
    bar.empty()
if c2.button("Clear session"):
    st.session_state["log"] = []
    st.session_state.pop("batch", None)

log = session_log()
if not log:
    st.info("No predictions yet. Use the Detector page, run a batch, or click **Run latency benchmark**.")
    st.stop()

df = pd.DataFrame(log)
df["request"] = np.arange(1, len(df) + 1)
lat = df["end_to_end_ms"].dropna()

k = st.columns(5)
k[0].metric("Predictions", len(df))
k[1].metric("Flagged as hate", f"{(df.label == HATE).mean()*100:.0f}%")
k[2].metric("Median latency", f"{lat.median():.0f} ms" if len(lat) else "—")
k[3].metric("p95 latency", f"{np.percentile(lat, 95):.0f} ms" if len(lat) >= 2 else "—")
k[4].metric("Borderline share", f"{(df.margin < 0.1).mean()*100:.0f}%",
            help="Predictions within 0.1 of the threshold")

row1 = st.columns(2)
with row1[0]:
    fig = go.Figure()
    for lab, colr in ((HATE, RED), ("Non-Hate", GREEN)):
        d = df[df.label == lab]
        fig.add_trace(go.Scatter(x=d.request, y=d.p_hate, mode="markers", name=lab,
                                 marker=dict(color=colr, size=9, line=dict(width=1, color="white")),
                                 customdata=d[["source", "language"]],
                                 hovertemplate="#%{x} · P(hate) %{y:.3f}<br>%{customdata[0]} · %{customdata[1]}<extra></extra>"))
    fig.add_hrect(y0=config.THRESHOLD - 0.1, y1=config.THRESHOLD + 0.1, fillcolor=GOLD, opacity=0.12,
                  line_width=0, annotation_text="borderline zone", annotation_position="top left")
    fig.add_hline(y=config.THRESHOLD, line_dash="dot", line_color=NAVY)
    fig.update_layout(title="P(hate) per request", height=360, yaxis=dict(range=[-0.02, 1.02], title="P(hate)"),
                      xaxis_title="Request", legend=dict(orientation="h", y=-0.2), margin=dict(t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
with row1[1]:
    timed = df.dropna(subset=["end_to_end_ms"])
    if len(timed):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=timed.request, y=timed.inference_ms, name="Inference", marker_color=NAVY))
        fig.add_trace(go.Bar(x=timed.request, y=timed.translation_ms.fillna(0), name="Translation", marker_color=GOLD))
        other = (timed.end_to_end_ms - timed.inference_ms - timed.translation_ms.fillna(0)).clip(lower=0)
        fig.add_trace(go.Bar(x=timed.request, y=other, name="Uncertainty + explanation", marker_color=SLATE))
        fig.update_layout(barmode="stack", title="Latency breakdown per request", height=360,
                          yaxis_title="ms", xaxis_title="Request", legend=dict(orientation="h", y=-0.2),
                          margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Latency appears after single-text predictions or the benchmark.")

row2 = st.columns(2)
with row2[0]:
    fig = px.histogram(df, x="margin", nbins=15, color_discrete_sequence=[NAVY],
                       labels={"margin": "Decision margin |P(hate) − threshold|"}, title="How decisive were the predictions?")
    fig.add_vline(x=0.1, line_dash="dot", line_color=RED, annotation_text="borderline")
    fig.update_layout(height=330, bargap=0.05, margin=dict(t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
with row2[1]:
    if len(timed) >= 3:
        fig = px.scatter(timed, x="words", y="inference_ms", color="language", trendline=None,
                         color_discrete_map={"en": NAVY, "bn": GOLD},
                         labels={"words": "Words in input", "inference_ms": "Inference (ms)"},
                         title="Input length vs inference time")
        fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
        fig.update_layout(height=330, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        lang = df.language.map({"en": "English", "bn": "Bangla"}).value_counts()
        fig = px.pie(values=lang.values, names=lang.index, hole=0.55, title="Input language",
                     color_discrete_sequence=[NAVY, GOLD])
        fig.update_layout(height=330, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

with st.expander("Raw session log"):
    st.dataframe(df.drop(columns=["request"]), use_container_width=True, hide_index=True)
st.caption("Texts are not shown or stored here: the log keeps only lengths, scores and timings, and it is cleared when you close the tab.")
