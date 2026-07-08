"""Streamlit demo UI. Run: streamlit run src/clauseiq/ui_streamlit.py"""
from __future__ import annotations

import streamlit as st

from clauseiq.pipeline import ClauseIQ


@st.cache_resource
def _engine() -> ClauseIQ:
    eng = ClauseIQ()
    eng.ingest_dir()
    return eng


def main() -> None:
    st.set_page_config(page_title="ClauseIQ", page_icon="📄", layout="wide")
    st.title("📄 ClauseIQ — Contract Intelligence")
    eng = _engine()
    st.caption(
        f"Embedder: {type(eng.embedder).__name__} · LLM: {eng.llm.name} · "
        f"{eng.store.stats()}"
    )

    q = st.text_input(
        "Ask about the contracts",
        value="What is the termination notice period and what happens on breach?",
    )
    use_agent = st.checkbox("Agentic decomposition", value=True)
    if st.button("Ask") and q:
        resp = eng.query(q, use_agent=use_agent)
        st.subheader("Answer")
        st.write(resp.answer)
        if resp.sub_questions:
            st.caption("Sub-questions: " + " · ".join(resp.sub_questions))
        st.subheader("Citations")
        for c in resp.citations:
            with st.expander(f"{c.chunk_id}  (score={c.score})"):
                st.write(c.snippet)


if __name__ == "__main__":
    main()
