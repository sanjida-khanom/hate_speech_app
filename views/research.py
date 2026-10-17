import json

import plotly.graph_objects as go
import streamlit as st

from hsd import config
from hsd.ui import (GOLD, GREEN, NAVY, NAVY_2, PLOT_CFG, SLATE, footer, hero, kpis, people_html, section,
                    setup, style_fig)

setup("Research", "🔬")

try:
    with open(config.METRICS_PATH, encoding="utf-8") as f:
        M = json.load(f)
except (OSError, json.JSONDecodeError) as e:
    st.error(f"Could not read {config.METRICS_PATH}: {e}")
    st.stop()

paper, S = M["paper"], M["student"]
teachers = M.get("teachers", [])

hero("The research behind this app", paper["title"],
     f"{paper['authors']} · <i>{paper['venue']}</i> · {paper['status']}",
     [f"Accuracy {S['accuracy']*100:.2f}%", f"ROC-AUC {S['roc_auc']:.4f}", f"{S['params_millions']} M parameters",
      f"{S['compression_vs_ensemble']}× compression"], gold=2)

tab_over, tab_method, tab_results, tab_card, tab_people, tab_limits = st.tabs(
    ["Overview", "Method", "Results", "Model card", "App team & citation", "Limitations"])

# --------------------------------------------------------------------------
with tab_over:
    c1, c2 = st.columns([1.5, 1], gap="large")
    with c1:
        section("Problem and contribution")
        st.markdown(
            """
**The problem.** Transformer ensembles detect hate speech well, but they are too slow and heavy for
real-time moderation, and most hateful content in Bangladesh is written in **Bangla**, which strong
English classifiers do not read.

**The contribution.**
1. A **cross-attention fusion** of three transformer teachers.
2. **Knowledge distillation** of that ensemble into one BERT-base student that runs on a CPU.
3. A **translation-mediated** layer so Bangla text can be screened with the English model.
4. **Explainable, uncertainty-aware** predictions, so a human moderator can judge how far to trust each one.
"""
        )
    with c2:
        section("At a glance")
        kpis([("Accuracy", f"{S['accuracy']*100:.2f}%"), ("ROC-AUC", f"{S['roc_auc']:.4f}")])
        kpis([("Parameters", f"{S['params_millions']} M"), ("Compression", f"{S['compression_vs_ensemble']}×")])
        kpis([("Model size", f"{S['size_mb']} MB"), ("Threshold", S["threshold"])])

# --------------------------------------------------------------------------
with tab_method:
    t_nodes = "\n".join(f'    t{i} [label="{t}"];' for i, t in enumerate(teachers))
    t_edges = "\n".join(f"    t{i} -> fuse;" for i in range(len(teachers)))
    st.graphviz_chart(f"""
digraph G {{
  rankdir=LR; bgcolor="transparent"; pad=0.2;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="{NAVY}", fillcolor="#EEF2F8", margin="0.18,0.08"];
  edge [color="{SLATE}", fontname="Helvetica"];
  subgraph cluster_train {{
    label="Training (offline)"; fontname="Helvetica"; fontcolor="{GOLD}"; color="#C9D1DC"; style="rounded";
{t_nodes}
    fuse [label="Cross-attention fusion\\n(teacher ensemble)", fillcolor="#F6EEDF", color="{GOLD}"];
{t_edges}
    fuse -> student [label="soft labels\\n(distillation)", fontsize=9];
    student [label="BERT-base student\\n{S['params_millions']}M params", fillcolor="{NAVY}", fontcolor="white"];
  }}
  subgraph cluster_serve {{
    label="Serving (this app)"; fontname="Helvetica"; fontcolor="{GOLD}"; color="#C9D1DC"; style="rounded";
    inp [label="Input text"]; lang [label="Bangla?", shape=diamond, fillcolor="white"];
    mt [label="Bangla → English\\ntranslation"]; clf [label="Student classifier\\nthreshold {S['threshold']}"];
    out [label="Label + explanation\\n+ uncertainty", fillcolor="#E9F5EE", color="{GREEN}"];
    inp -> lang; lang -> mt [label="yes", fontsize=9]; lang -> clf [label="no", fontsize=9]; mt -> clf; clf -> out;
  }}
  student -> clf [style=dashed, label="deployed", fontsize=9];
}}""", use_container_width=True)

    steps = [
        ("1 · Teacher ensemble", f"{', '.join(teachers)} are fine-tuned and their representations are fused with cross-attention."),
        ("2 · Knowledge distillation", "One BERT-base student learns from the ensemble's soft labels and the true labels."),
        ("3 · Threshold tuning", f"The decision threshold is tuned on validation data to {S['threshold']} instead of 0.5."),
        ("4 · Serving", f"Bangla is translated, the student classifies, word occlusion explains, and {config.MC_PASSES} MC-dropout passes estimate uncertainty."),
    ]
    cols = st.columns(4)
    for col, (h, b) in zip(cols, steps):
        with col.container(border=True):
            st.markdown(f"**{h}**")
            st.caption(b)

# --------------------------------------------------------------------------
with tab_results:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        ens = S["params_millions"] * S["compression_vs_ensemble"]
        fig = go.Figure(go.Bar(x=["Teacher ensemble", "Distilled student"], y=[ens, S["params_millions"]],
                               marker_color=[SLATE, NAVY], text=[f"≈{ens:.0f} M", f"{S['params_millions']} M"],
                               textposition="outside", width=0.5))
        fig.update_yaxes(title="Parameters (millions)", range=[0, ens * 1.18])
        st.plotly_chart(style_fig(fig, 340, f"Model size: {S['compression_vs_ensemble']}× smaller"),
                        use_container_width=True, config=PLOT_CFG)
        st.caption("Ensemble size derived from the reported 4.8× compression.")
    with c2:
        fig = go.Figure()
        for name, val, colr in (("Accuracy", S["accuracy"], NAVY), ("ROC-AUC", S["roc_auc"], GOLD)):
            fig.add_trace(go.Bar(x=[name], y=[val * 100], marker_color=colr, text=[f"{val*100:.2f}"],
                                 textposition="outside", showlegend=False, width=0.5))
        fig.update_yaxes(range=[80, 100], title="%")
        st.plotly_chart(style_fig(fig, 340, "Student performance (test set)"), use_container_width=True, config=PLOT_CFG)

    shown = 0
    cm, roc = M.get("confusion_matrix"), M.get("roc_curve")
    if cm or roc:
        r = st.columns(2, gap="large")
        if cm:
            z = cm["matrix"]
            fig = go.Figure(go.Heatmap(z=z, x=cm["labels"], y=cm["labels"], colorscale=[[0, "#EEF2F8"], [1, NAVY]],
                                       text=z, texttemplate="%{text}", showscale=False))
            fig.update_xaxes(title="Predicted")
            fig.update_yaxes(title="True", autorange="reversed")
            r[0].plotly_chart(style_fig(fig, 360, "Confusion matrix"), use_container_width=True, config=PLOT_CFG); shown += 1
        if roc:
            fig = go.Figure(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", line=dict(color=NAVY, width=3),
                                       name=f"Student (AUC {S['roc_auc']:.3f})", fill="tozeroy", fillcolor="rgba(31,56,100,.08)"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color=SLATE), name="Chance"))
            fig.update_xaxes(title="False positive rate")
            fig.update_yaxes(title="True positive rate")
            r[1].plotly_chart(style_fig(fig, 360, "ROC curve"), use_container_width=True, config=PLOT_CFG); shown += 1

    comp = M.get("model_comparison")
    if comp:
        names = [c["model"] for c in comp]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Accuracy", x=names, y=[c["accuracy"] * 100 for c in comp], marker_color=NAVY))
        fig.add_trace(go.Bar(name="ROC-AUC", x=names, y=[c["roc_auc"] * 100 for c in comp], marker_color=GOLD))
        fig.update_layout(barmode="group")
        fig.update_yaxes(title="%")
        st.plotly_chart(style_fig(fig, 360, "Teachers vs ensemble vs student"), use_container_width=True, config=PLOT_CFG); shown += 1

    rel = M.get("reliability")
    if rel:
        fig = go.Figure(go.Scatter(x=rel["bin_confidence"], y=rel["bin_accuracy"], mode="lines+markers",
                                   line=dict(color=NAVY, width=3), name="Student"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color=SLATE), name="Perfect calibration"))
        fig.update_xaxes(title="Confidence")
        fig.update_yaxes(title="Accuracy")
        st.plotly_chart(style_fig(fig, 360, f"Calibration (ECE {rel.get('ece', 0):.3f})"), use_container_width=True, config=PLOT_CFG); shown += 1

    if not shown:
        st.info("The confusion matrix, ROC curve, model comparison and calibration plots appear here automatically "
                "once their values from the paper are added to `results/metrics.json`.")

# --------------------------------------------------------------------------
with tab_card:
    section("Model card", "A short, standard summary of what this model is and how it should be used.")
    rows = [
        ("Model", f"{S['name']}, distilled from a {' + '.join(teachers)} ensemble"),
        ("Base architecture", f"{config.MODEL_NAME}, {S['params_millions']} M parameters, {S['size_mb']} MB (fp32)"),
        ("Task", "Binary classification: Hate Speech vs Non-Hate"),
        ("Input", f"English text (Bangla via translation), up to {S['max_tokens']} tokens"),
        ("Decision threshold", f"{S['threshold']} (tuned on validation data)"),
        ("Test performance", f"Accuracy {S['accuracy']*100:.2f}%, ROC-AUC {S['roc_auc']:.4f}"),
        ("Intended use", "Research, education and assisting human moderators in prioritising content"),
        ("Out of scope", "Automated removal or punishment of users without human review; "
                         "generating or targeting hateful content"),
        ("Explanations", "Word occlusion (live); SHAP analysis in the paper"),
        ("Uncertainty", f"Monte-Carlo dropout, {config.MC_PASSES} passes"),
    ]
    st.markdown("| | |\n|---|---|\n" + "\n".join(f"| **{k}** | {v} |" for k, v in rows))

# --------------------------------------------------------------------------
with tab_people:
    c1, c2 = st.columns([1, 1.3], gap="large")
    with c1:
        section("Who built this web app", "The paper's authors are listed in the header above; this is the team that built and deployed the live demo.")
        with st.container(border=True):
            st.markdown(people_html(compact=False), unsafe_allow_html=True)
    with c2:
        section("How to cite the paper")
        st.code(
            "@article{rahman2026dynamicfusionnet,\n"
            f"  title   = {{{paper['title']}}},\n"
            "  author  = {M. A. Rahman and S. Khanom and M. A. Rahman and F. S. Richi and M. A. Rahaman and A. B. M. S. Ali},\n"
            f"  journal = {{{paper['venue'].split(' (')[0]}}},\n"
            "  year    = {2026},\n"
            f"  note    = {{{paper['status']}}}\n"
            "}", language="bibtex")
        st.caption("Under review: please check the final citation after publication.")

# --------------------------------------------------------------------------
with tab_limits:
    section("Limitations and responsible use")
    st.markdown(
        """
- **Translation can change meaning.** Slang, sarcasm and code-mixed Bangla can be softened or lost.
- **Context is missing.** The model sees one message, not the conversation around it.
- **Reclaimed and in-group language** can be misread as hateful.
- **Support people, don't replace them.** Predictions should help moderators prioritise, not make final decisions.
- **Privacy.** Texts stay in memory for your session only and are never written to disk.
- **Prohibited uses:** generating hateful content, harassment, targeting individuals or communities, or evading platform rules.
"""
    )
footer()
