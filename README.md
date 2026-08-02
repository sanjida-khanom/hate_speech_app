# 🛡️ Bilingual Hate Speech Detector(Bangla/English)

A Streamlit web app for real-time hate speech detection, powered by a **distilled BERT-base** student model trained via **Knowledge Distillation** from a three-model teacher ensemble (BERT + XLM-RoBERTa + DeBERTa).

| Metric | Value |
|---|---|
| Accuracy | 91.04% |
| AUC | 0.9652 |
| ECE (calibration) | 0.034 |
| Parameters | 109M |
| Decision threshold | 0.48 (from threshold tuning) |

---

## 📋 Overview

This app loads a fine-tuned BERT-base classifier and predicts whether a given piece of text is **Hate Speech** or **Non-Hate**, along with calibrated confidence scores. The model was distilled from a 3-model ensemble to achieve strong accuracy at a fraction of the size (≈4.8× compression vs. the teacher).

---

## 📁 Project Structure

```
hate_speech_app/
├── app.py              # Streamlit application
├── requirements.txt    # Python dependencies
├── .gitignore          # excludes *.pt model weights
├── README.md           # this file
└── student_best.pt     # model weights — NOT in repo, see below (~418 MB)
```

> ⚠️ **Note:** `student_best.pt` is **not** included in this repository (it exceeds GitHub's 100 MB file limit). The weights are **available upon request** — see [Getting the Model Weights](#-getting-the-model-weights). Place the file in the project root before running.

---

## 📦 Getting the Model Weights

The trained weights file `student_best.pt` (~418 MB) is **not** included in this repository and is **available upon request**. Please reach out to the authors (see [Authors](#-authors)) to obtain it.

Once you have the file, place it in the **same folder as `app.py`**. The app expects it at:
```
hate_speech_app/student_best.pt
```

---

## 🚀 Setup & Run

Choose your platform below.

### 🪟 Windows (PowerShell)

```powershell
# 1. Clone the repository
git clone https://github.com/sanjida-khanom/hate_speech_app.git
cd hate_speech_app

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\Activate.ps1
# If activation is blocked, run once (as needed):
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Place student_best.pt in this folder (see "Getting the Model Weights")

# 5. Run the app
streamlit run app.py
```

### 🐧 Linux / macOS (Bash)

```bash
# 1. Clone the repository
git clone https://github.com/sanjida-khanom/hate_speech_app.git
cd hate_speech_app

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
python3 -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Place student_best.pt in this folder (see "Getting the Model Weights")

# 5. Run the app
streamlit run app.py
```

Once running, Streamlit opens the app at **http://localhost:8501** in your browser.

---

## 🖥️ GPU Support (Optional)

The app automatically uses a CUDA GPU if one is available, otherwise it falls back to CPU. CPU inference works fine for single-sentence prediction.

To use GPU, install a CUDA-enabled build of PyTorch **instead of** the default one in `requirements.txt`. For example (adjust the CUDA version to match your system):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Check availability inside Python:
```python
import torch; print(torch.cuda.is_available())
```

---

## 🧪 Usage

1. Launch the app with `streamlit run app.py`.
2. Type or paste text into the input box.
3. Click **Analyze**.
4. View the predicted label (**Hate Speech** / **Non-Hate**) and confidence bars.
5. Expand **Details** to see the raw probabilities and threshold used.

---

## ⚙️ Configuration

Key settings are defined at the top of `app.py`:

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `bert-base-uncased` | Base tokenizer/architecture |
| `MODEL_PATH` | `student_best.pt` | Path to the weights file |
| `MAX_LEN` | `128` | Max token sequence length |
| `THRESHOLD` | `0.51` | Hate-class decision threshold |

---

## 🧠 Model Details

The student model is a BERT-base classifier distilled from a teacher ensemble of **BERT + XLM-RoBERTa + DeBERTa** using knowledge distillation, trained on monolingual (English) hate speech data. Distillation transfers the ensemble's performance into a single compact model suitable for real-time deployment.

---

## 📝 Requirements

```
streamlit
torch
transformers
sentencepiece
sacremoses
```

Python 3.9+ recommended.

---

## 👥 Authors

This project was developed as a **team effort**, with full and equal credit shared between:

- **Sanjida Khanom** — [@sanjida-khanom](https://github.com/sanjida-khanom)
- **Ashik** — [@ashikbuilds]((https://github.com/ashikbuilds )

For access to the model weights or any questions, feel free to reach out to either author.

---

## 📄 License

<!-- Add a license if you intend to share this publicly, e.g. MIT. -->
Specify a license here (e.g., MIT) before publishing.
