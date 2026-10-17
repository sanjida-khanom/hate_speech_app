import json

import plotly.graph_objects as go
import streamlit as st

from hsd import config
from hsd.ui import GOLD, GREEN, NAVY, RED, SLATE, setup

setup("Research", "🔬")

try:
    with open(config.METRICS_PATH, encoding="utf-8") as f:
        M = json.load(f)
except (OSError, json.JSONDecodeError) as e:
    st.error(f"Could not read {config.METRICS_PATH}: {e}")
    st.stop()

paper, S = M["paper"], M["student"]

st.title("🔬 The research behind this app")
st.markdown(f"**{paper['title']}**  \n{paper['authors']}  \n*{paper['venue']} · {paper['status']}*")

tab_over, tab_method, tab_results, tab_limits = st.tabs(
    ["Overview", "Method", "Results", "Limitations & responsible use"])

# --------------------------------------------------------------------------
with tab_over:
    st.markdown(
        """
Large transformer ensembles detect hate speech well, but they are too slow and heavy to run
in real time. This work **distils** an ensemble of three transformers into a single BERT-base
**student** that keeps most of the accuracy at a fraction of the size, so it can run on an ordinary CPU.

Much hateful content in Bangladesh is written in **Bangla**. The classifier itself is English, so
Bangla input goes through a **translation-mediated** accessibility layer first. Every prediction comes
with an explanation and an uncertainty estimate, so a human moderator can judge how far to trust it.
"""
    )
    k = st.columns(5)
    k[0].metric("Accuracy", f"{S['accuracy']*100:.2f}%")
    k[1].metric("ROC-AUC", f"{S['roc_auc']:.4f}")
    k[2].metric("Parameters", f"{S['params_millions']} M")
    k[3].metric("Compression", f"{S['compression_vs_ensemble']}×")
    k[4].metric("Model size", f"{S['size_mb']} MB")

# --------------------------------------------------------------------------
with tab_method:
    teachers = M.get("teachers", [])
    t_nodes = "\n".join(f'    t{i} [label="{t}"];' for i, t in enumerate(teachers))
    t_edges = "\n".join(f"    t{i} -> fuse;" for i in range(len(teachers)))
    st.graphviz_chart(f"""
digraph G {{
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="{NAVY}", fillcolor="#EEF2F8"];
  edge [color="{SLATE}"];
  subgraph cluster_train {{
    label="Training (offline)"; fontname="Helvetica"; color="#C9D1DC"; style="rounded";
{t_nodes}
    fuse [label="Cross-attention fusion\\n(teacher ensemble)", fillcolor="#F6EEDF", color="{GOLD}"];
{t_edges}
    fuse -> student [label="soft labels\\n(knowledge distillation)", fontsize=9];
    student [label="BERT-base student\\n{S['params_millions']}M params", fillcolor="{NAVY}", fontcolor="white"];
  }}
  subgraph cluster_serve {{
    label="Serving (this app)"; fontname="Helvetica"; color="#C9D1DC"; style="rounded";
    inp [label="Input text"]; lang [label="Bangla?", shape=diamond, fillcolor="white"];
    mt [label="Bangla → English\\ntranslation"]; clf [label="Student classifier\\nthreshold {S['threshold']}"];
    out [label="Label + explanation\\n+ uncertainty", fillcolor="#E9F5EE", color="{GREEN}"];
    inp -> lang; lang -> mt [label="yes", fontsize=9]; lang -> clf [label="no", fontsize=9]; mt -> clf; clf -> out;
  }}
  student -> clf [style=dashed, label="deployed", fontsize=9];
}}""", use_container_width=True)

    st.markdown(
        f"""
**1. Teacher ensemble.** {", ".join(teachers)} are fine-tuned and their representations are fused
with cross-attention, giving a strong but heavy ensemble.

**2. Knowledge distillation.** A single BERT-base student learns from the ensemble's soft labels as
well as the true labels, which transfers much of the ensemble's knowledge into one model.

**3. Threshold tuning.** The decision threshold is tuned on validation data to **{S['threshold']}**
rather than left at 0.5.

**4. Serving.** Bangla text is translated to English, classified, then explained by word occlusion and
checked with Monte-Carlo dropout ({config.MC_PASSES} stochastic passes) to estimate uncertainty.
"""
    )

# --------------------------------------------------------------------------
with tab_results:
    c1, c2 = st.columns(2)
    with c1:
        ens = S["params_millions"] * S["compression_vs_ensemble"]
        fig = go.Figure(go.Bar(
            x=["Teacher ensemble", "Distilled student"], y=[ens, S["params_millions"]],
            marker_color=[SLATE, NAVY], text=[f"≈{ens:.0f} M", f"{S['params_millions']} M"], textposition="outside"))
        fig.update_layout(title=f"Model size: {S['compression_vs_ensemble']}× smaller", yaxis_title="Parameters (millions)",
                          height=340, margin=dict(t=50, b=10), yaxis=dict(range=[0, ens * 1.18]))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Ensemble size derived from the reported 4.8× compression.")
    with c2:
        fig = go.Figure()
        for name, val, colr in (("Accuracy", S["accuracy"], NAVY), ("ROC-AUC", S["roc_auc"], GOLD)):
            fig.add_trace(go.Bar(x=[name], y=[val * 100], marker_color=colr, text=[f"{val*100:.2f}"],
                                 textposition="outside", showlegend=False))
        fig.update_layout(title="Student performance (test set)", yaxis=dict(range=[80, 100], title="%"),
                          height=340, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    shown = 0
    cm = M.get("confusion_matrix")
    roc = M.get("roc_curve")
    if cm or roc:
        r = st.columns(2)
        if cm:
            z = cm["matrix"]
            fig = go.Figure(go.Heatmap(z=z, x=cm["labels"], y=cm["labels"], colorscale=[[0, "#EEF2F8"], [1, NAVY]],
                                       text=z, texttemplate="%{text}", showscale=False))
            fig.update_layout(title="Confusion matrix", xaxis_title="Predicted", yaxis_title="True",
                              yaxis_autorange="reversed", height=360, margin=dict(t=50, b=10))
            r[0].plotly_chart(fig, use_container_width=True); shown += 1
        if roc:
            fig = go.Figure(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", line=dict(color=NAVY, width=3),
                                       name=f"Student (AUC {S['roc_auc']:.3f})", fill="tozeroy", fillcolor="rgba(31,56,100,.08)"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color=SLATE), name="Chance"))
            fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate",
                              height=360, margin=dict(t=50, b=10), legend=dict(x=0.45, y=0.08))
            r[1].plotly_chart(fig, use_container_width=True); shown += 1

    comp = M.get("model_comparison")
    if comp:
        names = [c["model"] for c in comp]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Accuracy", x=names, y=[c["accuracy"] * 100 for c in comp], marker_color=NAVY))
        fig.add_trace(go.Bar(name="ROC-AUC", x=names, y=[c["roc_auc"] * 100 for c in comp], marker_color=GOLD))
        fig.update_layout(barmode="group", title="Teachers vs ensemble vs student", yaxis_title="%",
                          height=360, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True); shown += 1

    rel = M.get("reliability")
    if rel:
        fig = go.Figure(go.Scatter(x=rel["bin_confidence"], y=rel["bin_accuracy"], mode="lines+markers",
                                   line=dict(color=NAVY, width=3), name="Student"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color=SLATE), name="Perfect calibration"))
        fig.update_layout(title=f"Calibration (ECE {rel.get('ece', 0):.3f})", xaxis_title="Confidence",
                          yaxis_title="Accuracy", height=360, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True); shown += 1

    if not shown:
        st.info("The confusion matrix, ROC curve, model comparison and calibration plots appear here "
                "automatically once their values from the paper are added to `results/metrics.json`.")

# --------------------------------------------------------------------------
with tab_limits:
    st.markdown(
        """
- **Translation can change meaning.** Slang, sarcasm and code-mixed Bangla can be softened or lost
  in translation, which affects the prediction.
- **Context is missing.** The model sees one message, not the conversation around it.
- **Reclaimed and in-group language** can be misread as hateful.
- **Use it to support people, not replace them.** Predictions should help human moderators prioritise,
  not make final decisions about real users.
- **Privacy.** This demo keeps texts in memory for your session only and never writes them to disk.
- **Prohibited uses:** generating hateful content, harassment, targeting individuals or communities,
  or evading platform rules.
"""
    )
