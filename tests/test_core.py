import json
import math

from hsd import config
from hsd.model import StubClassifier
from hsd.pipeline import (HATE, NON_HATE, analyze, analyze_batch, binary_entropy_bits,
                          confidence_band, label_for)
from hsd.text import clean, is_bangla, was_truncated
from hsd.translate import StubTranslator

clf = StubClassifier()
tr = lambda: StubTranslator()


def test_language_detection():
    assert is_bangla("আজকের সেমিনারটা খুব শিক্ষণীয় ছিল।")
    assert not is_bangla("The seminar was useful.")
    assert not is_bangla("1234 !!")


def test_clean_and_truncate():
    assert clean("  a   b \n c ") == "a b c"
    long = "x" * (config.MAX_CHARS + 50)
    assert len(clean(long)) == config.MAX_CHARS and was_truncated(long)


def test_threshold_and_definitions():
    assert label_for(config.THRESHOLD + 0.01) == HATE
    assert label_for(config.THRESHOLD) == NON_HATE
    assert math.isclose(binary_entropy_bits(0.5), 1.0)
    # matches the paper's Fig. 8: P=0.951 -> 0.284 bits
    assert abs(binary_entropy_bits(0.951) - 0.283) < 0.002
    assert confidence_band(0.44).startswith("High")
    assert confidence_band(0.02).startswith("Borderline")


def test_analyze_full():
    a = analyze("People like you are disgusting trash", clf, tr)
    assert a.label == HATE and 0 <= a.p_hate <= 1
    assert a.mc and a.mc["passes"] == config.MC_PASSES and 0 <= a.mc["stability"] <= 1
    assert [w for w, _ in a.attribution][:2] == ["People", "like"]
    assert {"inference", "end_to_end"} <= set(a.timings_ms)


def test_analyze_bangla_goes_through_translator():
    a = analyze("আজকের সেমিনারটা খুব শিক্ষণীয় ছিল।", clf, tr, mc=False, explain=False)
    assert a.lang == "bn" and "translation" in a.timings_ms


def test_batch():
    rows = analyze_batch(["hello", "you idiot", "ভালো"], clf, tr)
    assert len(rows) == 3 and rows[2]["language"] == "bn"


def test_metrics_file_matches_published_numbers():
    m = json.load(open(config.METRICS_PATH, encoding="utf-8"))
    assert m["student"]["accuracy"] == 0.9104 and m["student"]["roc_auc"] == 0.9652
    assert m["student"]["threshold"] == config.THRESHOLD
