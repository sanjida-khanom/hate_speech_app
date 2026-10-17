"""One call that turns raw text into everything the UI shows.

Definitions follow the paper's Fig. 8:
  decision margin  = |P(hate) - threshold|
  entropy          = binary entropy of P(hate), in bits
  MC-dropout       = mean, std and 95% interval over stochastic passes
  label stability  = share of MC passes that agree with the final label
Token attribution here is word occlusion (drop one word, measure the change
in P(hate)). It is fast enough for real-time CPU serving; the paper's SHAP
analysis was run offline.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import config
from .text import clean, is_bangla, was_truncated, words

HATE, NON_HATE = "Hate Speech", "Non-Hate"


@dataclass
class Analysis:
    raw: str
    text: str                    # cleaned input
    lang: str                    # "bn" or "en"
    classified_text: str         # what the model actually saw (translated if bn)
    p_hate: float
    label: str
    truncated: bool
    margin: float
    band: str
    entropy_bits: float
    mc: dict | None = None
    attribution: list[tuple[str, float]] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)


def label_for(p: float) -> str:
    return HATE if p > config.THRESHOLD else NON_HATE


def confidence_band(margin: float) -> str:
    if margin >= 0.30:
        return "High confidence"
    if margin >= 0.10:
        return "Moderate confidence"
    return "Borderline: treat with caution"


def binary_entropy_bits(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def occlusion(classifier, text: str) -> list[tuple[str, float]]:
    toks = words(text)[: config.MAX_ATTR_WORDS]
    if len(toks) < 2:
        return [(t, 0.0) for t in toks]
    base = float(classifier.predict_proba([" ".join(toks)])[0])
    variants = [" ".join(toks[:i] + toks[i + 1 :]) for i in range(len(toks))]
    probs = classifier.predict_proba(variants)
    return [(t, base - float(p)) for t, p in zip(toks, probs)]


def analyze(raw: str, classifier, get_translator, *, mc: bool = True, explain: bool = True) -> Analysis:
    t0 = time.perf_counter()
    text = clean(raw)
    lang = "bn" if is_bangla(text) else "en"
    timings = {}

    classified = text
    if lang == "bn":
        t = time.perf_counter()
        classified = get_translator()(text)
        timings["translation"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    p = float(classifier.predict_proba([classified])[0])
    timings["inference"] = (time.perf_counter() - t) * 1000

    label = label_for(p)
    margin = abs(p - config.THRESHOLD)
    result = Analysis(
        raw=raw, text=text, lang=lang, classified_text=classified, p_hate=p,
        label=label, truncated=was_truncated(raw), margin=margin,
        band=confidence_band(margin), entropy_bits=binary_entropy_bits(p),
    )

    if mc and config.MC_PASSES > 1:
        t = time.perf_counter()
        samples = classifier.mc_dropout(classified, config.MC_PASSES)
        timings["mc_dropout"] = (time.perf_counter() - t) * 1000
        agree = np.mean([label_for(s) == label for s in samples])
        result.mc = {
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples)),
            "lo": float(np.percentile(samples, 2.5)),
            "hi": float(np.percentile(samples, 97.5)),
            "stability": float(agree),
            "passes": int(len(samples)),
            "samples": [float(x) for x in samples],
        }

    if explain:
        t = time.perf_counter()
        result.attribution = occlusion(classifier, classified)
        timings["attribution"] = (time.perf_counter() - t) * 1000

    timings["end_to_end"] = (time.perf_counter() - t0) * 1000
    result.timings_ms = timings
    return result


def analyze_batch(texts: list[str], classifier, get_translator) -> list[dict]:
    """Fast path for CSV uploads: translation where needed, one batched pass."""
    cleaned = [clean(t) for t in texts]
    langs = ["bn" if is_bangla(t) else "en" for t in cleaned]
    to_model = [get_translator()(t) if l == "bn" else t for t, l in zip(cleaned, langs)]
    t = time.perf_counter()
    probs = classifier.predict_proba(to_model) if to_model else []
    per_item_ms = (time.perf_counter() - t) * 1000 / max(len(to_model), 1)
    return [
        {"text": c, "language": l, "p_hate": float(p), "label": label_for(float(p)),
         "margin": abs(float(p) - config.THRESHOLD), "ms_per_item": per_item_ms}
        for c, l, p in zip(cleaned, langs, probs)
    ]
