# 🛡️ Hate Speech Detection App

A Streamlit web application for detecting hate speech in text using a distilled BERT-base model.

---

## ⚠️ Important Notice

**This application is part of an ongoing research project.** The model and code are provided for **research and educational purposes only**. 

**Prohibited Uses:**
- ❌ Generating hate speech or toxic content
- ❌ Using for harassment or abuse
- ❌ Misusing to harm individuals or communities
- ❌ Circumventing content policies on platforms

**Misuse of this model is strictly prohibited.**

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Accuracy | 91.04% |
| ROC-AUC | 0.9652 |
| Model Size | 109M parameters |

---

## 📁 Setup

### Prerequisites
- Python 3.9+
- pip

### Installation

**Windows:**
```powershell
git clone https://github.com/sanjida-khanom/hate-speech-detection.git
cd hate-speech-detection

python -m venv venv
venv\Scripts\Activate.ps1

pip install -r requirements.txt
streamlit run app.py
```

**Linux/macOS:**
```bash
git clone https://github.com/sanjida-khanom/hate-speech-detection.git
cd hate-speech-detection

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Access the app at: **http://localhost:8501**

---

## 📦 Getting Model Weights

The trained model weights (`student_best.pt`, ~418 MB) are available upon request from the authors:

- **Sanjida Khanom** — [@sanjida-khanom](https://github.com/sanjida-khanom)
- **Mohammod Ashikur Rahman** — [@ashikbuilds](https://github.com/ashikbuilds)

Place `student_best.pt` in the project root folder.

---

## 🖥️ Using the App

1. Enter text in the input field
2. Click "Analyze"
3. View prediction (Hate Speech / Non-Hate)
4. See confidence score

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

For academic use, please cite appropriately and contact the authors for more details.

---

## 👥 Authors

**Equal contribution:**
- Sanjida Khanom ([@sanjida-khanom](https://github.com/sanjida-khanom))
- Mohammod Ashikur Rahman ([@ashikbuilds](https://github.com/ashikbuilds))

---


**Last Updated:** August 2024
