import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from hsd import config
from hsd.pipeline import HATE, analyze_batch
from hsd.ui import (GOLD, GREEN, NAVY, PLOT_CFG, RED, classifier_or_stop, footer, get_translator, hero, kpis,
                    section, setup, style_fig)

setup("Batch Analysis", "📄")
hero("Batch analysis", "Score many texts at once",
     f"Upload a CSV with a text column, or paste one text per line (up to {config.MAX_BATCH_ROWS} rows). "
     "You get the score distribution, a language breakdown, a review queue of borderline cases and a downloadable file.",
     ["CSV in, CSV out", "Bangla & English", "Human-review queue"])

with st.container(border=True):
    tab_csv, tab_paste = st.tabs(["Upload CSV", "Paste lines"])
    texts: list[str] = []
    with tab_csv:
        up = st.file_uploader("CSV file", type=["csv"])
        if up is not None:
            try:
                df_in = pd.read_csv(up)
            except Exception as e:
                st.error(f"Could not read the CSV: {e}")
                st.stop()
            cols = list(df_in.columns)
            col = st.selectbox("Text column", cols, index=cols.index("text") if "text" in cols else 0)
            texts = df_in[col].dropna().astype(str).tolist()
    with tab_paste:
        pasted = st.text_area("One text per line", height=150)
        if pasted.strip() and not texts:
            texts = [l for l in pasted.splitlines() if l.strip()]

    if texts and len(texts) > config.MAX_BATCH_ROWS:
        st.info(f"Only the first {config.MAX_BATCH_ROWS} of {len(texts)} rows will be analysed.")
        texts = texts[: config.MAX_BATCH_ROWS]
    run = st.button("Run batch", type="primary", disabled=not texts)

if run:
    clf = classifier_or_stop()
    t0 = time.perf_counter()
    with st.spinner(f"Classifying {len(texts)} texts…"):
        rows = analyze_batch(texts, clf, get_translator)
    total_ms = (time.perf_counter() - t0) * 1000
    df = pd.DataFrame(rows)
    st.session_state["batch"] = df
    st.session_state["batch_ms"] = total_ms
    for r in rows:
        st.session_state.setdefault("log", []).append({
            "time": time.strftime("%H:%M:%S"), "source": "batch", "language": r["language"],
            "chars": len(r["text"]), "words": len(r["text"].split()), "p_hate": r["p_hate"],
            "label": r["label"], "margin": r["margin"], "band": None, "end_to_end_ms": None,
            "inference_ms": r["ms_per_item"], "translation_ms": None, "mc_std": None,
        })

df = st.session_state.get("batch")
if df is None or not len(df):
    footer()
    st.stop()

total_ms = st.session_state.get("batch_ms", 0)
hate = df.label == HATE
border = df.margin < 0.1

section("Overview")
kpis([
    ("Texts analysed", len(df), f"{total_ms/1000:.1f} s total"),
    ("Flagged as hate", f"{hate.sum()}", f"{hate.mean()*100:.0f}% of the batch"),
    ("Borderline", int(border.sum()), "within 0.10 of the threshold"),
    ("Bangla inputs", int((df.language == "bn").sum()), "translated first"),
    ("Mean P(hate)", f"{df.p_hate.mean():.3f}", f"median {df.p_hate.median():.3f}"),
])

c1, c2 = st.columns(2, gap="large")
with c1:
    fig = px.histogram(df, x="p_hate", nbins=20, color="label",
                       color_discrete_map={HATE: RED, "Non-Hate": GREEN}, labels={"p_hate": "P(hate)"})
    fig.add_vrect(x0=config.THRESHOLD - 0.1, x1=config.THRESHOLD + 0.1, fillcolor=GOLD, opacity=0.12, line_width=0)
    fig.add_vline(x=config.THRESHOLD, line_dash="dot", line_color=NAVY, annotation_text="threshold")
    fig.update_layout(bargap=0.05)
    fig.update_yaxes(title="Texts")
    st.plotly_chart(style_fig(fig, 340, "Score distribution"), use_container_width=True, config=PLOT_CFG)
with c2:
    ordered = df.sort_values("p_hate").reset_index(drop=True)
    fig = go.Figure(go.Scatter(
        x=list(range(1, len(ordered) + 1)), y=ordered.p_hate, mode="lines+markers",
        line=dict(color=NAVY, width=2), marker=dict(size=6, color=[RED if l == HATE else GREEN for l in ordered.label]),
        hovertemplate="rank %{x} · P(hate) %{y:.3f}<extra></extra>"))
    fig.add_hline(y=config.THRESHOLD, line_dash="dot", line_color=RED)
    fig.update_xaxes(title="Texts ranked by score")
    fig.update_yaxes(title="P(hate)", range=[-0.02, 1.02])
    st.plotly_chart(style_fig(fig, 340, "Ranked scores (how sharply the batch separates)"),
                    use_container_width=True, config=PLOT_CFG)

if (df.language == "bn").any() and (df.language == "en").any():
    counts = df.groupby(["language", "label"]).size().reset_index(name="n")
    counts["language"] = counts.language.map({"en": "English", "bn": "Bangla"})
    fig = px.bar(counts, x="n", y="language", color="label", orientation="h", barmode="stack",
                 color_discrete_map={HATE: RED, "Non-Hate": GREEN}, labels={"n": "Texts", "language": ""})
    st.plotly_chart(style_fig(fig, 220, "Labels by input language"), use_container_width=True, config=PLOT_CFG)

section("Human-review queue", "The borderline texts a moderator should check first, closest to the threshold at the top.")
queue = df[border].sort_values("margin")
if len(queue):
    st.dataframe(queue[["text", "language", "label", "p_hate", "margin"]], use_container_width=True, hide_index=True,
                 column_config={"p_hate": st.column_config.ProgressColumn("P(hate)", min_value=0, max_value=1, format="%.3f"),
                                "margin": st.column_config.NumberColumn("Margin", format="%.3f")})
else:
    st.success("No borderline texts: every prediction is at least 0.10 away from the threshold.")

section("All results")
st.dataframe(
    df[["text", "language", "label", "p_hate", "margin"]].sort_values("p_hate", ascending=False),
    use_container_width=True, hide_index=True,
    column_config={"p_hate": st.column_config.ProgressColumn("P(hate)", min_value=0, max_value=1, format="%.3f"),
                   "margin": st.column_config.NumberColumn("Margin", format="%.3f")},
)
st.download_button("Download results (CSV)", df.to_csv(index=False).encode("utf-8"),
                   "hate_speech_predictions.csv", "text/csv")
footer()
