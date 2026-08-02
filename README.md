# 🛡️ Hate Speech Detection App

A Streamlit web app that detects hate speech in text using a lightweight, distilled BERT model. It takes a research model all the way to a working app anyone can use.

---

## ⚠️ Important Notice

**This app is part of an ongoing research project.** It is for **research and educational use only.**

**Do not use it for:**
- ❌ Generating hate speech or toxic content
- ❌ Harassment or abuse
- ❌ Harming individuals or communities
- ❌ Getting around platform content rules

**Misuse is strictly prohibited.**

---

## 👥 Authors

**Equal contribution:**
- Sanjida Khanom ([@sanjida-khanom](https://github.com/sanjida-khanom))
- Mohammod Ashikur Rahman ([@ashikbuilds](https://github.com/ashikbuilds))
---

## 🎯 Why we Built This

- Big language models detect hate speech well, but they are too heavy to run in real time.
- I compressed a large model into a small, fast one that keeps almost the same accuracy.
- Then I turned it into a simple web app so anyone can try it — not just a notebook.
- Hate speech detection is a real problem, so it felt worth solving.

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Accuracy | 91.04% |
| ROC-AUC | 0.9652 |
| Model Size | 109M parameters |

---

## 🧠 Skills we Used

- **Machine Learning:** knowledge distillation (making a small model learn from a big one)
- **NLP:** fine-tuning a BERT model with Hugging Face
- **Model compression:** keeping accuracy while making the model smaller and faster
- **Evaluation:** measuring accuracy, ROC-AUC, and tuning the threshold
- **Deployment:** building the app with Streamlit
- **Good practices:** clean code, clear setup, and documentation

---

## 📈 What we've Learned

- How to take a model from research all the way to a real, working app.
- How model distillation actually works, by building it myself.
- How to judge a model properly — not just by accuracy, but ROC-AUC too.
- How to think about responsible AI and prevent misuse.
- How to finish a full project end to end, on my own.

---

## 📁 Setup

**Prerequisites:** Python 3.9+ and pip

**Windows:**
```powershell
git clone https://github.com/sanjida-khanom/hate_speech_app.git
cd hate_speech_app
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

**Linux/macOS:**
```bash
git clone https://github.com/sanjida-khanom/hate_speech_app.git
cd hate_speech_app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the app at: **http://localhost:8501**

---

## 📦 Model Weights

The trained weights (`student_best.pt`, ~418 MB) are available on request from the authors. Place the file in the project root folder.

---

## 🖥️ How to Use

1. Type or paste text in the input box
2. Click **Analyze**
3. See the result (Hate Speech / Non-Hate) and the confidence score

---

## ⚙️ Configuration

Key settings in `app.py`:

| Parameter | Default |
|---|---|
| Model | bert-base-uncased |
| Max Length | 128 tokens |
| Threshold | 0.51 |

---

## 📋 Requirements

```
streamlit>=1.28.0
torch>=2.0.0
transformers>=4.30.0
sentencepiece>=0.1.99
sacremoses>=0.0.53
```

---

## 🔬 Research Status

This project is part of ongoing research on model compression and ensemble learning. **Publication is pending.**

---
