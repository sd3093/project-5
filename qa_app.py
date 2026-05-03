from pathlib import Path

import streamlit as st

from document_qa import DocumentQA

# Resolve the data path relative to this file
DATA_DIR = str(Path(__file__).parent / "data" / "policies")

# Configure the page
st.set_page_config(
    page_title="Company Policy Q&A",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Company Policy Q&A Chatbot")
st.write("Ask questions about company policies with AI-powered semantic search.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the Q&A engine once per session
if "qa_system" not in st.session_state:
    with st.spinner("Loading policy documents and building search index..."):
        try:
            st.session_state.qa_system = DocumentQA(DATA_DIR)
        except Exception as e:
            st.error(f"Error loading documents: {str(e)}")
            st.stop()

qa_system = st.session_state.qa_system

# Sidebar with system info and controls
with st.sidebar:
    st.header("📖 System Info")

    try:
        stats = qa_system.get_stats()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", stats["num_documents"])
        with col2:
            st.metric("Chunks", stats["num_chunks"])

        st.subheader("📄 Loaded Documents")
        for doc in stats.get("documents", []):
            st.caption(f"• {doc['filename']} ({doc['num_chunks']} chunks)")

        st.divider()

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    except Exception as e:
        st.error(f"Error loading system info: {str(e)}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📎 Sources"):
                for source in message["sources"]:
                    st.caption(f"• {source}")

# Process user input and generate answer
user_input = st.chat_input("Ask a question about company policies...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                result = qa_system.answer_question(user_input)
                answer = result.get("answer", "No answer found.")
                sources = result.get("sources", [])
                nodes = result.get("nodes", [])

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

                if nodes:
                    with st.expander("📎 Sources & Relevance"):
                        for node in nodes:
                            source = node.node.metadata.get("file_name", "Unknown")
                            score = node.score
                            st.caption(f"**{source}** (relevance: {score:.3f})")
                            st.text(node.node.text[:200] + "...")
                            st.divider()

            except Exception as e:
                error_msg = f"Error generating answer: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                })
