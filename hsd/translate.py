"""Bangla -> English accessibility layer (MarianMT, opus-mt-bn-en).

The classifier is English-only; Bangla input is translated first. The
translator is loaded lazily, so English-only traffic never pays for it.
"""

from __future__ import annotations

from . import config


class Translator:
    is_stub = False

    def __init__(self):
        from transformers import MarianMTModel, MarianTokenizer

        self.tokenizer = MarianTokenizer.from_pretrained(config.TRANSLATOR)
        self.model = MarianMTModel.from_pretrained(config.TRANSLATOR).eval()

    def __call__(self, text: str) -> str:
        import torch

        enc = self.tokenizer([text], return_tensors="pt", truncation=True, max_length=256)
        with torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=256, num_beams=4)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


class StubTranslator:
    """Used with the stub backend only: returns the text unchanged."""

    is_stub = True

    def __call__(self, text: str) -> str:
        return text


def load_translator():
    if config.BACKEND == "stub":
        return StubTranslator()
    return Translator()
