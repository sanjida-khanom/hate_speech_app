# ============================================================
# Bilingual Hate Speech Detection — Streamlit App
# Model: Distilled BERT-base (Student model from KD)
# Accuracy: 91.51% | AUC: 0.9669 | ECE: 0.034
# ============================================================

import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="Hate Speech Detector",
    page_icon="🛡️",
    layout="centered"
)

# ============================================================
# Model Loading (cached — loads once)
# ============================================================

MODEL_NAME  = "bert-base-uncased"
MODEL_PATH  = "student_best.pt"   # place this file in the same folder
MAX_LEN     = 128
THRESHOLD   = 0.51   # from threshold tuning (s_best_t)

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    model.dropout = nn.Dropout(0.3)  # matches training architecture

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return tokenizer, model, device

# ============================================================
# Inference Function
# ============================================================

def predict(text, tokenizer, model, device):
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    hate_prob = probs[1]
    label     = "Hate Speech" if hate_prob > THRESHOLD else "Non-Hate"
    return label, hate_prob, probs

# ============================================================
# UI
# ============================================================

st.title("🛡️ Monolingual Hate Speech Detector")
st.caption("Distilled BERT model — 91.51% accuracy | AUC 0.97 | 109M params")

st.markdown("---")

text_input = st.text_area(
    "Enter text to analyze:",
    placeholder="Type or paste a sentence here...",
    height=120
)

col1, col2 = st.columns([1, 4])
with col1:
    analyze_btn = st.button("Analyze", type="primary")

if analyze_btn:
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            tokenizer, model, device = load_model()
            label, hate_prob, probs = predict(text_input, tokenizer, model, device)

        st.markdown("---")

        if label == "Hate Speech":
            st.error(f"⚠️ **{label}**")
        else:
            st.success(f"✅ **{label}**")

        # Probability bars
        st.markdown("**Confidence:**")
        st.progress(float(hate_prob), text=f"Hate: {hate_prob*100:.1f}%")
        st.progress(float(probs[0]), text=f"Non-Hate: {probs[0]*100:.1f}%")

        with st.expander("Details"):
            st.write(f"Threshold used: {THRESHOLD}")
            st.write(f"Raw probabilities — Non-Hate: {probs[0]:.4f} | Hate: {probs[1]:.4f}")

st.markdown("---")
st.caption(
    "Model: BERT-base distilled from a 3-model ensemble (BERT+XLM-R+DeBERTa) "
    "via Knowledge Distillation. Trained on bilingual (Bangla-English) hate speech data."
)