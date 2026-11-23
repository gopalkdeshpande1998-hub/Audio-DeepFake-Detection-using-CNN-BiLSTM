"""The app's pages: built once per script run by build() and looked up by key for st.page_link."""

from __future__ import annotations

import streamlit as st

PAGES: dict = {}
SECTIONS = {"Use": ["overview", "detect", "compare", "batch", "history"],
            "Research": ["results", "errors"],
            "About": ["card", "help"]}


def build() -> dict:
    from ui import pages

    spec = {"overview": (pages.overview, "Overview", "home", "overview"),
            "detect": (pages.detect, "Detect", "graphic_eq", "detect"),
            "compare": (pages.compare, "Compare A/B", "compare_arrows", "compare"),
            "batch": (pages.batch, "Batch", "library_music", "batch"),
            "history": (pages.history, "History", "history", "history"),
            "results": (pages.results, "Results", "table_chart", "results"),
            "errors": (pages.errors, "Error analysis", "troubleshoot", "errors"),
            "card": (pages.model_card, "Model card", "description", "model-card"),
            "help": (pages.help_page, "Help", "help", "help")}
    PAGES.clear()
    PAGES.update({k: st.Page(f, title=t, icon=f":material/{i}:", url_path=u, default=(k == "overview"))
                  for k, (f, t, i, u) in spec.items()})
    return PAGES


def navigation():
    build()
    return st.navigation({s: [PAGES[k] for k in keys] for s, keys in SECTIONS.items()})


def get(key: str):
    return PAGES[key]
