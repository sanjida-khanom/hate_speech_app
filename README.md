---
title: Bilingual Hate Speech Detector
emoji: 🛡️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Distilled BERT hate speech detection for Bangla and English
---

# 🛡️ Bilingual Hate Speech Detector (Bangla / English)

A real-time web app for the research paper **"DynamicFusionNet: Distilling a Cross-Attention
Transformer Ensemble for Efficient and Explainable Translation-Mediated Hate Speech Detection"**
(under review, *Expert Systems with Applications*).

A BERT + XLM-RoBERTa + DeBERTa ensemble is distilled into one BERT-base student that runs on a CPU.
Bangla text is translated to English first, and every prediction comes with an explanation and an
uncertainty estimate.

**Live demo:** https://bilingual-hate-speech.streamlit.app/

| Metric | Value |
|---|---|
| Accuracy | 91.04% |
| ROC-AUC | 0.9652 |
| Parameters | 109.5 M |
| Compression vs ensemble | 4.8× |
| Decision threshold | 0.51 |

## ⚠️ Responsible use

Part of an ongoing research project, for **research and educational use only**.
Do not use it to generate hateful content, harass people, target individuals or communities,
or evade platform rules. Predictions can be wrong; a human should review real moderation decisions.

## 👥 Web app developed and deployed by

**Mohammod Ashikur Rahman** — Software Engineer L1, bKash Limited ([@ashikbuilds](https://github.com/ashikbuilds))
**Sanjida Khanom** — Research Assistant, Department of CSE, BUBT ([@sanjida-khanom](https://github.com/sanjida-khanom))
*(equal contribution)*

**Under the supervision of:** Md. Ashiqur Rahman, Assistant Professor, Department of CSE,
Bangladesh University of Business and Technology (BUBT)

The research paper's own author list is shown on the app's Research page.

## What the app does

| Page | What it shows |
|---|---|
| **Detector** | Label, probability, translation (for Bangla), word-level explanation, MC-dropout uncertainty, live latency |
| **Batch Analysis** | Upload a CSV or paste lines; score distribution, labels by language, downloadable results |
| **Live Analytics** | Real measurements from your session: latency breakdown and p95, score timeline, borderline share; one-click latency benchmark |
| **Research** | Method diagram, model card, compression and performance plots, limitations |

Definitions follow Fig. 8 of the paper: decision margin = |P(hate) − 0.51|, entropy in bits,
MC-dropout mean ± std with a 95% interval and label stability.

## Project structure

```
app.py                     entry point and page navigation
views/                     Detector, Batch Analysis, Live Analytics, Research
hsd/
  config.py                every setting, overridable by environment variables
  model.py                 BERT student (and a test-only stub)
  translate.py             Bangla → English (Helsinki-NLP/opus-mt-bn-en)
  pipeline.py              prediction, uncertainty, explanation, timings
  text.py, ui.py           language detection, limits, shared UI
results/metrics.json       published numbers for the Research page
tests/                     unit tests and page smoke tests
Dockerfile                 Hugging Face Space image
.github/workflows/deploy.yml   test, then auto-deploy on every push
```

## Run locally

```bash
git clone https://github.com/sanjida-khanom/hate_speech_app.git
cd hate_speech_app
python -m venv venv
venv\Scripts\activate          # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
# put student_best.pt in this folder, OR set HSD_MODEL_REPO to the Hub repo
streamlit run app.py
```

To work on the interface without the weights, use the test-only stub (a red banner makes it obvious):
`HSD_BACKEND=stub streamlit run app.py`

## Model weights

The trained weights (`student_best.pt`, ~418 MB) are hosted on the Hugging Face Hub and downloaded
on first start. They are available from the authors on request.

## Deployment

Hosted free on Streamlit Community Cloud. Every push to `main` runs the tests and redeploys the app automatically. See [DEPLOY.md](DEPLOY.md).

## Research status

Under review at *Expert Systems with Applications* (Elsevier).
