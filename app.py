"""
app.py – Streamlit chat UI for the Debales AI assistant.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import sys
sys.path.insert(0, "src")

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from agent import chat, get_graph

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Debales AI Assistant",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Debales AI Assistant")
st.caption("Ask anything about Debales AI – or any general question!")

# ── Load graph once ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base …")
def load_graph():
    return get_graph()

try:
    load_graph()
except FileNotFoundError:
    st.error(
        "⚠️ FAISS index not found. "
        "Run `python scripts/ingest.py` first, then restart the app."
    )
    st.stop()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []      # list of {"role": str, "content": str}
if "history" not in st.session_state:
    st.session_state.history = []       # list of BaseMessage

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input ─────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask me anything …"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):
            answer = chat(prompt, st.session_state.history)
        st.markdown(answer)

    # Persist
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.history.append(HumanMessage(content=prompt))
    st.session_state.history.append(AIMessage(content=answer))
    # Keep last 10 turns
    st.session_state.history = st.session_state.history[-20:]
