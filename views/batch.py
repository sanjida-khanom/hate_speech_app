import time

import pandas as pd
import plotly.express as px
import streamlit as st

from hsd import config
from hsd.pipeline import HATE, analyze_batch
from hsd.ui import GREEN, NAVY, RED, classifier_or_stop, get_translator, setup

setup("Batch Analysis", "📄")
st.title("📄 Batch analysis")
st.markdown(
    f'<p class="hsd-sub">Upload a CSV with a <code>text</code> column, or paste one text per line. '
    f"Up to {config.MAX_BATCH_ROWS} rows per run.</p>",
    unsafe_allow_html=True,
)

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
    pasted = st.text_area("One text per line", height=160)
    if pasted.strip() and not texts:
        texts = [l for l in pasted.splitlines() if l.strip()]

if texts and len(texts) > config.MAX_BATCH_ROWS:
    st.info(f"Only the first {config.MAX_BATCH_ROWS} of {len(texts)} rows will be analysed.")
    texts = texts[: config.MAX_BATCH_ROWS]

if st.button("Run batch", type="primary", disabled=not texts):
    clf = classifier_or_stop()
    t0 = time.perf_counter()
    with st.spinner(f"Classifying {len(texts)} texts…"):
        rows = analyze_batch(texts, clf, get_translator)
    total_ms = (time.perf_counter() - t0) * 1000
    df = pd.DataFrame(rows)
    st.session_state["batch"] = df
    for r in rows:
        st.session_state.setdefault("log", []).append({
            "time": time.strftime("%H:%M:%S"), "source": "batch", "language": r["language"],
            "chars": len(r["text"]), "words": len(r["text"].split()), "p_hate": r["p_hate"],
            "label": r["label"], "margin": r["margin"], "end_to_end_ms": None,
            "inference_ms": r["ms_per_item"], "translation_ms": None, "mc_std": None,
        })
    st.success(f"Done in {total_ms/1000:.1f} s ({total_ms/len(rows):.0f} ms per text on average).")

df = st.session_state.get("batch")
if df is not None and len(df):
    k = st.columns(4)
    k[0].metric("Texts", len(df))
    k[1].metric("Flagged as hate", f"{(df.label == HATE).sum()} ({(df.label == HATE).mean()*100:.0f}%)")
    k[2].metric("Bangla inputs", int((df.language == "bn").sum()))
    k[3].metric("Borderline (margin < 0.1)", int((df.margin < 0.1).sum()))

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="p_hate", nbins=20, color="label",
                           color_discrete_map={HATE: RED, "Non-Hate": GREEN},
                           labels={"p_hate": "P(hate)"}, title="Score distribution")
        fig.add_vline(x=config.THRESHOLD, line_dash="dot", line_color=NAVY, annotation_text="threshold")
        fig.update_layout(height=340, bargap=0.05, legend_title=None, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        counts = df.groupby(["language", "label"]).size().reset_index(name="n")
        counts["language"] = counts.language.map({"en": "English", "bn": "Bangla"})
        fig = px.bar(counts, x="language", y="n", color="label", barmode="group",
                     color_discrete_map={HATE: RED, "Non-Hate": GREEN},
                     title="Labels by input language", labels={"n": "Texts", "language": ""})
        fig.update_layout(height=340, legend_title=None, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        df[["text", "language", "label", "p_hate", "margin"]].sort_values("p_hate", ascending=False),
        use_container_width=True, hide_index=True,
        column_config={"p_hate": st.column_config.ProgressColumn("P(hate)", min_value=0, max_value=1, format="%.3f"),
                       "margin": st.column_config.NumberColumn("Margin", format="%.3f")},
    )
    st.download_button("Download results (CSV)", df.to_csv(index=False).encode("utf-8"),
                       "hate_speech_predictions.csv", "text/csv")
