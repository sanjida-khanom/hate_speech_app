"""Classifier backends.

`BertClassifier` is the distilled BERT-base student used in the paper.
`StubClassifier` is a tiny keyword heuristic used only for UI development
and CI, where the 418 MB weights are not available. The app shows a red
warning banner whenever the stub is active.
"""

from __future__ import annotations

import math
import os

import numpy as np

from . import config


class WeightsNotFound(RuntimeError):
    """Raised when no local weights exist and no Hub repo is configured."""


# --------------------------------------------------------------------------
# Shared interface
# --------------------------------------------------------------------------
class BaseClassifier:
    name = "base"
    is_stub = False
    device = "cpu"
    n_params = 0

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """P(hate) for each text."""
        raise NotImplementedError

    def mc_dropout(self, text: str, passes: int) -> np.ndarray:
        """P(hate) under `passes` stochastic forward passes."""
        raise NotImplementedError

    def footprint_mb(self) -> float | None:
        return None


# --------------------------------------------------------------------------
# Real model
# --------------------------------------------------------------------------
def resolve_weights() -> str:
    """Local file first, then a one-time cached download from the Hub."""
    if os.path.isfile(config.LOCAL_WEIGHTS):
        return config.LOCAL_WEIGHTS
    # read at call time, so secrets added after start-up are picked up
    hub_repo = config._setting("HSD_MODEL_REPO")
    if hub_repo:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=hub_repo,
            filename=config.HUB_FILE,
            token=config._setting("HF_TOKEN") or None,
        )
    raise WeightsNotFound(
        f"No weights at '{config.LOCAL_WEIGHTS}' and HSD_MODEL_REPO is not set."
    )


class BertClassifier(BaseClassifier):
    name = "Distilled BERT-base student"

    def __init__(self):
        import torch
        import torch.nn as nn
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._nn = nn
        torch.set_num_threads(config.CPU_THREADS)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.MODEL_NAME, num_labels=2
        )
        model.dropout = nn.Dropout(config.DROPOUT)  # matches training architecture

        self.weights_path = resolve_weights()
        state = torch.load(self.weights_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        model.to(self.device).eval()
        self.model = model
        self.n_params = sum(p.numel() for p in model.parameters())

    def footprint_mb(self) -> float | None:
        try:
            return os.path.getsize(self.weights_path) / 1e6
        except OSError:
            return None

    def _encode(self, texts):
        # Dynamic padding: pad to the longest text in the batch, not to 128,
        # which is several times faster for short real-time inputs.
        enc = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=config.MAX_LEN, return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}

    def predict_proba(self, texts, batch_size: int = 32):
        torch = self._torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                logits = self.model(**self._encode(texts[i : i + batch_size])).logits
                out.append(torch.softmax(logits, dim=1)[:, 1].float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def mc_dropout(self, text, passes):
        torch = self._torch
        dropouts = [m for m in self.model.modules() if isinstance(m, self._nn.Dropout)]
        enc = self._encode([text] * passes)
        try:
            for m in dropouts:
                m.train()  # dropout on, everything else stays in eval mode
            with torch.inference_mode():
                logits = self.model(**enc).logits
                return torch.softmax(logits, dim=1)[:, 1].float().cpu().numpy()
        finally:
            for m in dropouts:
                m.eval()


# --------------------------------------------------------------------------
# Stub (UI development and CI only)
# --------------------------------------------------------------------------
_STUB_TERMS = {"hate": 2.2, "stupid": 1.6, "idiot": 1.8, "kill": 2.4, "trash": 1.4,
               "disgusting": 1.6, "bitch": 2.6, "go back": 1.5, "vermin": 2.4}


class StubClassifier(BaseClassifier):
    name = "DEV STUB (keyword heuristic, not the research model)"
    is_stub = True
    n_params = 0

    def _score(self, text):
        t = text.lower()
        z = -2.2 + sum(w for k, w in _STUB_TERMS.items() if k in t)
        return 1 / (1 + math.exp(-z))

    def predict_proba(self, texts, batch_size: int = 32):
        return np.array([self._score(t) for t in texts], dtype=float)

    def mc_dropout(self, text, passes):
        p = self._score(text)
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return np.clip(p + rng.normal(0, 0.03, passes), 0, 1)


def load_classifier() -> BaseClassifier:
    if config.BACKEND == "stub":
        return StubClassifier()
    return BertClassifier()
