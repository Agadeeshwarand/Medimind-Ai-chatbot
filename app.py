from __future__ import annotations

from pathlib import Path

import streamlit as st

from ollama_client import OLLAMA_MODEL, generate_with_ollama, is_ollama_enabled
from rag import HealthcareRAG


APP_DIR = Path(__file__).resolve().parent
DOCS_DIR = APP_DIR / "data" / "health_docs"


@st.cache_resource
def load_rag() -> HealthcareRAG:
    return HealthcareRAG(DOCS_DIR)


def answer_question(rag: HealthcareRAG, question: str) -> tuple[str, list]:
    retrieved = rag.retrieve(question, top_k=3)
    prompt = rag.build_prompt(question, retrieved)
    answer = generate_with_ollama(prompt)

    if not answer:
        answer = rag.retrieval_only_answer(question, retrieved)

    return answer, retrieved


st.set_page_config(page_title="Medimind AI Chatbot", page_icon="H", layout="centered")

st.title("Medimind AI Chatbot")
st.caption(
    "Educational assistant using local healthcare documents. "
    "Not for diagnosis, treatment, or emergencies."
)

with st.sidebar:
    st.header("Knowledge Base")
    docs = sorted(path.name for path in DOCS_DIR.glob("*.txt"))
    st.write(f"{len(docs)} documents loaded")
    for doc in docs:
        st.write(f"- {doc}")

    st.header("Generation")
    if is_ollama_enabled():
        st.success(f"Ollama enabled: {OLLAMA_MODEL}")
    else:
        st.info("Ollama disabled. Using retrieval-only answers.")

    st.warning("For urgent symptoms, contact emergency services immediately.")

rag = load_rag()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi, I can answer general healthcare education questions from the local "
                "documents. What would you like to know?"
            ),
            "sources": [],
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"**{source.source}** - relevance {source.score:.2f}\n\n"
                        f"{source.text[:500]}..."
                    )

question = st.chat_input("Ask a healthcare question")

if question:
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching local health documents..."):
            answer, sources = answer_question(rag, question)
        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for source in sources:
                    st.markdown(
                        f"**{source.source}** - relevance {source.score:.2f}\n\n"
                        f"{source.text[:500]}..."
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
