"""Live analytics for this session. Every number is measured from real
predictions made in the browser session; nothing here is simulated."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from hsd import config
from hsd.pipeline import HATE, analyze
from hsd.ui import (GOLD, GREEN, NAVY, NAVY_2, PLOT_CFG, RED, SLATE, classifier_or_stop, footer, get_translator,
                    hero, kpis, log_prediction, section, session_log, setup, style_fig)

setup("Live Analytics", "📈")
hero("Live analytics", "How the model behaves on this server",
     "Every chart is measured from real predictions in your session. Run the benchmark to profile latency "
     "on a fixed set of texts, then compare it with your own inputs.",
     ["Measured, not simulated", "Latency p50 / p95", "Uncertainty analysis"])

BENCH = "results/benchmark_texts.txt"

with st.container(border=True):
    c1, c2, c3 = st.columns([1.3, 1, 3])
    run = c1.button("Run latency benchmark", type="primary")
    if c2.button("Clear session"):
        st.session_state["log"] = []
        st.session_state.pop("batch", None)
    c3.caption("The benchmark classifies 12 fixed English and Bangla texts with uncertainty estimation on, "
               "so the numbers are comparable between runs and servers.")
if run:
    clf = classifier_or_stop()
    with open(BENCH, encoding="utf-8") as f:
        bench = [l.strip() for l in f if l.strip()]
    bar = st.progress(0.0, text="Benchmarking…")
    for i, t in enumerate(bench):
        log_prediction(analyze(t, clf, get_translator, mc=True, explain=False), source="benchmark")
        bar.progress((i + 1) / len(bench), text=f"Benchmarking… {i+1}/{len(bench)}")
    bar.empty()

log = session_log()
if not log:
    st.info("No predictions yet. Use the Detector page, run a batch, or click **Run latency benchmark**.")
    footer()
    st.stop()

df = pd.DataFrame(log)
# Older log entries (from before an app update) may lack newer fields.
for col in ("source", "language", "chars", "words", "p_hate", "label", "margin", "band",
            "end_to_end_ms", "inference_ms", "translation_ms", "mc_std"):
    if col not in df.columns:
        df[col] = None
for col in ("end_to_end_ms", "inference_ms", "translation_ms", "mc_std", "margin", "p_hate"):
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["request"] = np.arange(1, len(df) + 1)
lat = df["end_to_end_ms"].dropna()
timed = df.dropna(subset=["end_to_end_ms"])

section("Session summary")
kpis([
    ("Predictions", len(df), f"{(df.source == 'benchmark').sum()} from the benchmark"),
    ("Flagged as hate", f"{(df.label == HATE).mean()*100:.0f}%"),
    ("Median latency", f"{lat.median():.0f} ms" if len(lat) else "—", "end to end"),
    ("p95 latency", f"{np.percentile(lat, 95):.0f} ms" if len(lat) >= 2 else "—", "95% of requests faster"),
    ("Borderline share", f"{(df.margin < 0.1).mean()*100:.0f}%", "margin < 0.10"),
])

section("Predictions")
r1 = st.columns(2, gap="large")
with r1[0]:
    fig = go.Figure()
    for lab, colr in ((HATE, RED), ("Non-Hate", GREEN)):
        d = df[df.label == lab]
        fig.add_trace(go.Scatter(x=d.request, y=d.p_hate, mode="markers", name=lab,
                                 marker=dict(color=colr, size=9, line=dict(width=1, color="white")),
                                 customdata=d[["source", "language"]],
                                 hovertemplate="#%{x} · P(hate) %{y:.3f}<br>%{customdata[0]} · %{customdata[1]}<extra></extra>"))
    fig.add_hrect(y0=config.THRESHOLD - 0.1, y1=config.THRESHOLD + 0.1, fillcolor=GOLD, opacity=0.12, line_width=0)
    fig.add_hline(y=config.THRESHOLD, line_dash="dot", line_color=NAVY)
    fig.update_yaxes(range=[-0.02, 1.02], title="P(hate)")
    fig.update_xaxes(title="Request")
    st.plotly_chart(style_fig(fig, 340, "P(hate) per request (gold = borderline zone)"), use_container_width=True, config=PLOT_CFG)
with r1[1]:
    bands = df["band"].fillna("Batch (no band)").value_counts()
    colors = {"High confidence": NAVY, "Moderate confidence": NAVY_2, "Borderline": GOLD, "Batch (no band)": SLATE}
    fig = go.Figure(go.Pie(labels=bands.index, values=bands.values, hole=0.6, sort=False,
                           marker=dict(colors=[colors.get(b, SLATE) for b in bands.index]),
                           textinfo="percent", hovertemplate="%{label}: %{value}<extra></extra>"))
    fig.update_layout(annotations=[dict(text=f"<b>{len(df)}</b><br>predictions", showarrow=False, font=dict(size=14))])
    st.plotly_chart(style_fig(fig, 340, "Confidence bands"), use_container_width=True, config=PLOT_CFG)

section("Uncertainty", "Do the model's own uncertainty estimates agree with how close it is to the threshold?")
unc = df.dropna(subset=["mc_std"])
r2 = st.columns(2, gap="large")
with r2[0]:
    if len(unc) >= 2:
        fig = px.scatter(unc, x="margin", y="mc_std", color="label", color_discrete_map={HATE: RED, "Non-Hate": GREEN},
                         labels={"margin": "Decision margin |P(hate) − threshold|", "mc_std": "MC-dropout std"})
        fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
        fig.add_vline(x=0.1, line_dash="dot", line_color=GOLD, annotation_text="borderline")
        st.plotly_chart(style_fig(fig, 330, "Margin vs MC-dropout spread"), use_container_width=True, config=PLOT_CFG)
        if len(unc) >= 3 and unc.margin.std() > 0 and unc.mc_std.std() > 0:
            r = np.corrcoef(unc.margin, unc.mc_std)[0, 1]
            st.caption(f"Pearson r = {r:+.2f}. A negative value means the model is less stable exactly where it is closest to the threshold, as expected.")
    else:
        st.info("Needs at least two predictions with uncertainty on (the benchmark provides 12).")
with r2[1]:
    fig = px.histogram(df, x="margin", nbins=15, color_discrete_sequence=[NAVY],
                       labels={"margin": "Decision margin"})
    fig.add_vline(x=0.1, line_dash="dot", line_color=RED, annotation_text="borderline")
    fig.update_layout(bargap=0.05)
    fig.update_yaxes(title="Predictions")
    st.plotly_chart(style_fig(fig, 330, "How decisive were the predictions?"), use_container_width=True, config=PLOT_CFG)

section("Latency", "Where the time goes, and how consistent it is.")
r3 = st.columns(2, gap="large")
with r3[0]:
    if len(timed):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=timed.request, y=timed.inference_ms, name="Inference", marker_color=NAVY))
        fig.add_trace(go.Bar(x=timed.request, y=timed.translation_ms.fillna(0), name="Translation", marker_color=GOLD))
        other = (timed.end_to_end_ms - timed.inference_ms - timed.translation_ms.fillna(0)).clip(lower=0)
        fig.add_trace(go.Bar(x=timed.request, y=other, name="Uncertainty + explanation", marker_color=SLATE))
        fig.update_layout(barmode="stack")
        fig.update_xaxes(title="Request")
        fig.update_yaxes(title="ms")
        st.plotly_chart(style_fig(fig, 330, "Latency breakdown per request"), use_container_width=True, config=PLOT_CFG)
    else:
        st.info("Latency appears after single-text predictions or the benchmark.")
with r3[1]:
    if len(lat) >= 2:
        p50, p95 = lat.median(), np.percentile(lat, 95)
        fig = go.Figure(go.Histogram(x=lat, nbinsx=15, marker_color=NAVY_2, opacity=0.9,
                                     hovertemplate="%{x:.0f} ms: %{y}<extra></extra>"))
        fig.add_vline(x=p50, line_color=NAVY, line_width=2, annotation_text=f"p50 {p50:.0f} ms")
        fig.add_vline(x=p95, line_color=RED, line_dash="dot", annotation_text=f"p95 {p95:.0f} ms",
                      annotation_position="top left")
        fig.update_xaxes(title="End-to-end latency (ms)")
        fig.update_yaxes(title="Requests")
        st.plotly_chart(style_fig(fig, 330, "Latency distribution"), use_container_width=True, config=PLOT_CFG)
    if len(timed) >= 3:
        lang_lat = timed.groupby("language")["end_to_end_ms"].median().rename({"en": "English", "bn": "Bangla"})
        st.caption(" · ".join(f"{k}: median {v:.0f} ms" for k, v in lang_lat.items())
                   + ". Bangla includes the translation step.")

with st.expander("Raw session log"):
    st.dataframe(df.drop(columns=["request"]), use_container_width=True, hide_index=True)
st.caption("The log keeps only lengths, scores and timings, never the texts, and it is cleared when you close the tab.")
footer()
