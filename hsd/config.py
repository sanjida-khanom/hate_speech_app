"""Central configuration. Every value can be overridden with an environment
variable, so the same code runs locally, in CI and on Hugging Face Spaces."""

import os


def _setting(name: str, default: str = "") -> str:
    """Environment variable first (Docker, Hugging Face, local), then
    Streamlit secrets (Streamlit Community Cloud's Secrets box)."""
    if os.getenv(name):
        return os.environ[name]
    try:
        import streamlit as st

        value = st.secrets.get(name)
        if value:
            os.environ.setdefault(name, str(value))  # visible to hf_hub_download too
            return str(value)
    except Exception:  # no secrets file locally, or not running under Streamlit
        pass
    return default


# ---- model -------------------------------------------------------------
MODEL_NAME = os.getenv("HSD_BASE_MODEL", "bert-base-uncased")
MAX_LEN = int(os.getenv("HSD_MAX_LEN", "128"))
THRESHOLD = float(os.getenv("HSD_THRESHOLD", "0.51"))  # from threshold tuning (s_best_t)
DROPOUT = 0.3  # matches the training architecture of the student

# Weights: a local file wins; otherwise they are downloaded once from the
# Hugging Face Hub and cached. A private model repo needs HF_TOKEN set as a
# Space secret.
LOCAL_WEIGHTS = os.getenv("HSD_WEIGHTS_PATH", "student_best.pt")
HUB_REPO = _setting("HSD_MODEL_REPO")          # e.g. "sanjida-khanom/hate-speech-distilled-bert"
HUB_FILE = os.getenv("HSD_MODEL_FILE", "student_best.pt")

# ---- Bangla accessibility layer -----------------------------------------
TRANSLATOR = os.getenv("HSD_TRANSLATOR", "Helsinki-NLP/opus-mt-bn-en")

# ---- real-time safety limits --------------------------------------------
MAX_CHARS = int(os.getenv("HSD_MAX_CHARS", "1000"))       # per text
MAX_BATCH_ROWS = int(os.getenv("HSD_MAX_BATCH_ROWS", "200"))
MC_PASSES = int(os.getenv("HSD_MC_PASSES", "10"))         # MC-dropout samples
MAX_ATTR_WORDS = int(os.getenv("HSD_MAX_ATTR_WORDS", "40"))
CPU_THREADS = int(os.getenv("HSD_CPU_THREADS", "2"))

# "stub" is a keyword heuristic for local UI work and CI only. The app shows
# a red banner whenever it is active, and the Space never sets it.
BACKEND = os.getenv("HSD_BACKEND", "model")

# ---- published results (shown on the Research page) ---------------------
METRICS_PATH = os.getenv("HSD_METRICS_PATH", "results/metrics.json")

# ---- people (shown in the sidebar, header and Research page) --------------
AUTHORS = [
    {"name": "Mohammod Ashikur Rahman", "role": "Software Engineer L1", "org": "bKash Limited",
     "github": "https://github.com/ashikbuilds"},
    {"name": "Sanjida Khanom", "role": "Research Assistant", "org": "Department of CSE, BUBT",
     "github": "https://github.com/sanjida-khanom"},
]
EQUAL_CONTRIBUTION = True
SUPERVISOR = {"name": "Md. Ashiqur Rahman", "role": "Assistant Professor",
              "org": "Department of CSE, Bangladesh University of Business and Technology (BUBT)"}
