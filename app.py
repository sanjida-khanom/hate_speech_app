# ============================================================
# Bilingual Hate Speech Detection — Streamlit entry point.
# Each page lives in views/; shared logic lives in hsd/.
# ============================================================

import streamlit as st

pages = [
    st.Page("views/detector.py", title="Detector", icon="🛡️", default=True),
    st.Page("views/batch.py", title="Batch Analysis", icon="📄"),
    st.Page("views/analytics.py", title="Live Analytics", icon="📈"),
    st.Page("views/research.py", title="Research", icon="🔬"),
]
st.navigation(pages).run()
