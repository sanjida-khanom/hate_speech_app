"""Smoke tests: every page renders, and the detector works end to end."""

import pytest
from streamlit.testing.v1 import AppTest

ROOT = "../"  # AppTest resolves paths relative to this file


@pytest.mark.parametrize("page", ["app.py", "views/detector.py", "views/batch.py",
                                  "views/analytics.py", "views/research.py"])
def test_page_renders(page):
    at = AppTest.from_file(ROOT + page, default_timeout=60).run()
    assert not at.exception


def test_detector_end_to_end():
    at = AppTest.from_file(ROOT + "views/detector.py", default_timeout=60).run()
    at.text_area[0].input("You are an idiot and nobody wants you here.").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("Hate Speech" in m.value for m in at.markdown)
    assert len(at.session_state["log"]) == 1


def test_benchmark_fills_analytics():
    at = AppTest.from_file(ROOT + "views/analytics.py", default_timeout=60).run()
    at.button[0].click().run()
    assert not at.exception
    assert len(at.session_state["log"]) >= 10
