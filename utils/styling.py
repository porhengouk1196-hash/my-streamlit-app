"""
Small HTML-rendering helpers so every page uses the same card styles
instead of re-writing inline HTML everywhere. Keeps the 3-color theme
(navy / white / light grey / sky-blue accent) consistent across pages.
"""

import streamlit as st


def load_css(path: str) -> None:
    """Inject the shared stylesheet. Fails silently with a warning if the
    file is missing, rather than crashing the whole app."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Stylesheet not found at '{path}'. The app will still run, "
                   f"but without the custom banking theme.")


def render_card(title: str, body_html: str) -> None:
    """A simple bordered content card used on Home / Business Insight /
    Recommendation pages."""
    st.markdown(
        f"""
        <div class="card">
            <strong>{title}</strong>
            <div style="margin-top:0.4rem;">{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(label: str, value: str, tone: str = "low") -> None:
    """Result card for the Prediction page.

    tone: "low" / "medium" use the fixed navy-white-grey-sky-blue palette
    (visual weight only). tone="high" is the one deliberate exception: it
    renders in red so a High risk result is impossible to miss.
    """
    tone_class = {
        "low": "result-low",
        "medium": "result-medium",
        "high": "result-high",
    }.get(tone, "result-low")

    st.markdown(
        f"""
        <div class="result-card {tone_class}">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_divider() -> None:
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
