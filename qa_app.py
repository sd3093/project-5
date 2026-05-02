import streamlit as st
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
# Check if 'qa_system' exists in st.session_state
# If not, create DocumentQA('data/policies') and store it
# Use st.spinner("Loading policy documents and building search index...")
# Handle any errors with st.error() and st.stop()
# After initialization, assign: qa_system = st.session_state.qa_system
pass


# Display sidebar with system information
with st.sidebar:
    st.header("📖 System Info")

    # TODO 12: Sidebar with system info and controls
    # 1. Call qa_system.get_stats() to get document and chunk counts
    # 2. Display metrics using st.metric() (Documents, Chunks)
    # 3. List loaded document filenames from stats['documents']
    # 4. Add a "Clear Chat History" button that resets st.session_state.messages
    #    and calls st.rerun()
    pass


# TODO 9: Display chat history
# Loop through st.session_state.messages
# Use st.chat_message(msg['role']) to display each message
# For assistant messages with 'sources', show sources in an expander
pass


# TODO 10: Process user input
# Use st.chat_input("Ask a question about company policies...")
# When user sends a message:
# 1. Add it to st.session_state.messages with role='user'
# 2. Display it with st.chat_message("user")
pass


# TODO 11: Generate and display answer with sources
# Inside an st.chat_message("assistant") block:
# 1. Show a spinner while searching
# 2. Call qa_system.answer_question(user_input)
# 3. Display the answer with st.markdown()
# 4. Add to messages history with role='assistant', content, and sources
# 5. Show sources in an st.expander with relevance scores
# 6. Handle errors gracefully
pass
