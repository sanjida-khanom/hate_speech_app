"""Central configuration. Every value can be overridden with an environment
variable, so the same code runs locally, in CI and on Hugging Face Spaces."""

import os

# ---- model -------------------------------------------------------------
MODEL_NAME = os.getenv("HSD_BASE_MODEL", "bert-base-uncased")
MAX_LEN = int(os.getenv("HSD_MAX_LEN", "128"))
THRESHOLD = float(os.getenv("HSD_THRESHOLD", "0.51"))  # from threshold tuning (s_best_t)
DROPOUT = 0.3  # matches the training architecture of the student

# Weights: a local file wins; otherwise they are downloaded once from the
# Hugging Face Hub and cached. A private model repo needs HF_TOKEN set as a
# Space secret.
LOCAL_WEIGHTS = os.getenv("HSD_WEIGHTS_PATH", "student_best.pt")
HUB_REPO = os.getenv("HSD_MODEL_REPO", "")          # e.g. "sanjida-khanom/hate-speech-distilled-bert"
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
