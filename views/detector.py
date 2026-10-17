# ============================================================
# Bilingual Hate Speech Detection — Detector page
# Distilled BERT-base student from a BERT + XLM-R + DeBERTa ensemble.
# Bangla input is translated to English (accessibility layer) first.
# Other pages live in views/; app.py wires them together.
# ============================================================

import html

import plotly.graph_objects as go
import streamlit as st

from hsd import config
from hsd.pipeline import HATE, analyze
from hsd.ui import (GREEN, NAVY, RED, classifier_or_stop, get_translator,
                    log_prediction, setup)

setup("Detector")

EXAMPLES = {
    "— choose an example —": "",
    "Neutral (English)": "The workshop on responsible AI was really useful for our students.",
    "Neutral (Bangla)": "আজকের সেমিনারটা খুব শিক্ষণীয় ছিল।",
    "Hostile (English)": "People like you are disgusting trash and should go back where you came from.",
}

st.title("🛡️ Bilingual Hate Speech Detector")
st.markdown(
    '<p class="hsd-sub">Distilled BERT-base student · English classifier · '
    "optional Bangla accessibility layer · Accuracy 91.04% · AUC 0.9652 · 4.8× compression</p>",
    unsafe_allow_html=True,
)

left, right = st.columns([3, 1.2])
with right:
    ex = st.selectbox("Try an example", list(EXAMPLES))
    show_mc = st.toggle("Uncertainty (MC-dropout)", value=True,
                        help=f"{config.MC_PASSES} stochastic forward passes")
    show_attr = st.toggle("Explain prediction", value=True,
                          help="Word occlusion: how much P(hate) drops when each word is removed")
with left:
    text = st.text_area(
        "Enter text in English or Bangla",
        value=EXAMPLES[ex], height=130, max_chars=config.MAX_CHARS,
        placeholder="Type or paste a sentence…",
    )
    go_btn = st.button("Analyze", type="primary", use_container_width=False)

if go_btn:
    if not text.strip():
        st.warning("Please enter some text first.")
        st.stop()

    clf = classifier_or_stop()
    try:
        with st.spinner("Analyzing…"):
            a = analyze(text, clf, get_translator, mc=show_mc, explain=show_attr)
    except Exception as e:
        st.error(f"Analysis failed: {type(e).__name__}: {e}")
        st.stop()
    log_prediction(a)

    st.divider()
    if a.truncated:
        st.info(f"Input was longer than {config.MAX_CHARS} characters and was truncated.")

    # ---- translation layer --------------------------------------------
    if a.lang == "bn":
        st.subheader("Accessibility layer: translation")
        c1, c2 = st.columns(2)
        c1.markdown(f"**Original (Bangla)**  \n{html.escape(a.text)}")
        c2.markdown(f"**Translated (English)**  \n{html.escape(a.classified_text)}")

    # ---- prediction ----------------------------------------------------
    st.subheader("Prediction")
    pc1, pc2 = st.columns([1, 1.6])
    with pc1:
        cls = "hate" if a.label == HATE else "ok"
        icon = "⚠️" if a.label == HATE else "✅"
        st.markdown(f'<div class="verdict {cls}">{icon} {a.label}</div>', unsafe_allow_html=True)
        st.caption(f"{a.band} · threshold {config.THRESHOLD}")
    with pc2:
        fig = go.Figure(go.Bar(
            x=[a.p_hate * 100, (1 - a.p_hate) * 100], y=["Hate", "Non-Hate"], orientation="h",
            marker_color=[RED, GREEN], text=[f"{a.p_hate*100:.1f}%", f"{(1-a.p_hate)*100:.1f}%"],
            textposition="outside", cliponaxis=False,
        ))
        fig.add_vline(x=config.THRESHOLD * 100, line_dash="dot", line_color=NAVY,
                      annotation_text="threshold", annotation_position="top")
        fig.update_layout(height=170, margin=dict(l=0, r=40, t=20, b=0),
                          xaxis=dict(range=[0, 105], title="Probability (%)"), showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ---- explanation ---------------------------------------------------
    if a.attribution:
        st.subheader("Why the model decided this")
        st.caption("Red words push toward Hate, blue toward Non-Hate (word occlusion on the text the model saw).")
        peak = max((abs(v) for _, v in a.attribution), default=0) or 1
        chips = []
        for word, v in a.attribution:
            alpha = min(abs(v) / peak, 1) * 0.75 + 0.06
            rgb = "180,35,24" if v > 0 else "29,78,216"
            chips.append(f'<span class="tok" style="background:rgba({rgb},{alpha:.2f})">{html.escape(word)}</span>')
        st.markdown("".join(chips), unsafe_allow_html=True)
        top_h = sorted([x for x in a.attribution if x[1] > 0], key=lambda x: -x[1])[:5]
        top_n = sorted([x for x in a.attribution if x[1] < 0], key=lambda x: x[1])[:5]
        if top_h:
            st.markdown(f":red[Top → Hate: {', '.join(f'{w} (+{v:.2f})' for w, v in top_h)}]")
        if top_n:
            st.markdown(f":blue[Top → Non-Hate: {', '.join(f'{w} ({v:.2f})' for w, v in top_n)}]")

    # ---- confidence ----------------------------------------------------
    st.subheader("Confidence and validity")
    m = st.columns(4)
    m[0].metric("Decision margin", f"{a.margin:.3f}")
    m[1].metric("Confidence band", a.band.split(":")[0])
    m[2].metric("Prediction entropy", f"{a.entropy_bits:.3f} bits")
    if a.mc:
        m[3].metric("MC-dropout P(hate)", f"{a.mc['mean']*100:.1f}% ± {a.mc['std']*100:.1f}%")
        m2 = st.columns(4)
        m2[0].metric("95% MC interval", f"[{a.mc['lo']*100:.1f}%, {a.mc['hi']*100:.1f}%]")
        m2[1].metric("MC label stability", f"{a.mc['stability']*100:.0f}%")
        if a.mc["stability"] < 0.8:
            st.warning("The label changes across dropout samples. This input is genuinely ambiguous for the model.")

    # ---- deployment metrics (measured live, this request) ---------------
    st.subheader("Deployment metrics (this request)")
    t = a.timings_ms
    d = st.columns(4)
    d[0].metric("End-to-end latency", f"{t['end_to_end']:.0f} ms")
    d[1].metric("Inference only", f"{t['inference']:.0f} ms")
    d[2].metric("Translation step", f"{t['translation']:.0f} ms" if "translation" in t else "—")
    d[3].metric("Device", clf.device.upper())
    d2 = st.columns(4)
    d2[0].metric("Parameters", f"{clf.n_params/1e6:.1f} M" if clf.n_params else "—")
    fp = clf.footprint_mb()
    d2[1].metric("Student footprint", f"{fp:.0f} MB" if fp else "—")
    d2[2].metric("Uncertainty step", f"{t['mc_dropout']:.0f} ms" if "mc_dropout" in t else "—")
    d2[3].metric("Explanation step", f"{t['attribution']:.0f} ms" if "attribution" in t else "—")

st.divider()
st.markdown(
    '<p class="note">⚠️ Part of an ongoing research project, for research and educational use only. '
    "Do not use it to generate, target or amplify hateful content. Automated predictions can be wrong, "
    "especially for sarcasm, dialect and reclaimed language; a human should review any real moderation decision.</p>",
    unsafe_allow_html=True,
)
