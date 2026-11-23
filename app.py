"""Audio deepfake detector: Streamlit app (v2, PyTorch).

    streamlit run app.py

Pages
  Use       Overview · Detect (one recording) · Compare A/B · Batch (many files or a ZIP) · History (this session)
  Research  Results · Error analysis
  About     Model card · Help

The default detector is the merged model (adfd.ensemble: WavLM E7a + VoiceGuard improved +
VoiceGuard v2): one REAL / FAKE verdict, a trust lookup from the held-out error analysis
(adfd.trust), a time-local timeline with playable flagged segments, a forensic view and an
HTML report. The research pages read experiments/ and docs/ only and never load PyTorch.
Page code: ui/pages.py; detector views: ui/detector.py; theme: ui/theme.py.
"""

from __future__ import annotations

import logging

import streamlit as st

st.set_page_config(page_title="VoiceGuard · Audio forensics", page_icon="ui/assets/mark.svg", layout="wide")

from adfd.paths import configure_caches  # noqa: E402
from ui import nav, theme  # noqa: E402

configure_caches()
logging.getLogger("streamlit").setLevel(logging.ERROR)
theme.inject()
st.logo("ui/assets/logo.svg", icon_image="ui/assets/mark.svg", size="large")

nav.navigation().run()
