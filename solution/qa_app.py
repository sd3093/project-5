import streamlit as st
import os

# Set working directory to project root (project-5/)
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from document_qa import DocumentQA

# Configure the page
st.set_page_config(
    page_title="Company Policy Q&A",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Company Policy Q&A Chatbot")
st.write("Ask questions about company policies with AI-powered semantic search.")

# Initialize session state for messages if not exists
if "messages" not in st.session_state:
    st.session_state.messages = []


# TODO 8: Initialize DocumentQA in session_state
if "qa_system" not in st.session_state:
    with st.spinner("Loading policy documents and building search index..."):
        try:
            st.session_state.qa_system = DocumentQA("data/policies")
        except Exception as e:
            st.error(f"Error loading documents: {str(e)}")
            st.stop()

qa_system = st.session_state.qa_system


# Display sidebar with system information
with st.sidebar:
    st.header("📖 System Info")

    # TODO 12: Sidebar with system info and controls
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


# TODO 9: Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display sources for assistant messages if available
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📎 Sources"):
                for source in message["sources"]:
                    st.caption(f"• {source}")


# TODO 10 + 11: Process user input and generate answer
user_input = st.chat_input("Ask a question about company policies...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and display answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                result = qa_system.answer_question(user_input)
                answer = result.get("answer", "No answer found.")
                sources = result.get("sources", [])
                nodes = result.get("nodes", [])

                st.markdown(answer)

                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

                # Display sources with relevance scores
                if nodes:
                    with st.expander("📎 Sources & Relevance"):
                        for node in nodes:
                            source = node.node.metadata.get('file_name', 'Unknown')
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
                    "sources": []
                })
