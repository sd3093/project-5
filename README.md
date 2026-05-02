# Project 5: Company Policy Chatbot — Document Q&A System

Build an AI-powered Q&A system that answers employee questions about company policies using **Retrieval-Augmented Generation (RAG)**. The system uses semantic search with embeddings to find relevant policy sections, sends them to Google Gemini, and displays answers with source citations — all wrapped in a Streamlit chat interface.

## How It Works

```
User Question → Embed → Semantic Search → Retrieve Top Chunks → Prompt + Context → Gemini → Answer + Citations
```

1. **Load** 6 company policy .txt files using LlamaIndex's `SimpleDirectoryReader`
2. **Chunk** each document into searchable pieces using `SentenceSplitter`
3. **Embed** all chunks into 384-dimensional vectors using HuggingFace `bge-small-en-v1.5`
4. **Index** all vectors in a `VectorStoreIndex` for fast similarity search
5. **Search** for the most relevant chunks when a user asks a question (cosine similarity)
6. **Generate** an answer using Google Gemini with the retrieved context
7. **Cite** which policy document(s) the answer came from

## Architecture

See `architecture.svg` for the full RAG pipeline diagram.

## Dataset

6 company policy documents for Nexus Technologies Inc. (fictional):

| Document | Topics Covered |
|----------|---------------|
| `remote_work_policy.txt` | Eligibility, schedule, workspace, equipment, in-office requirements |
| `expense_policy.txt` | Travel, meals, approval limits, submission process, non-reimbursable items |
| `pto_policy.txt` | Vacation, sick days, personal days, holidays, parental leave, bereavement |
| `benefits_policy.txt` | Health/dental/vision insurance, 401(k), life insurance, ESPP, wellness |
| `code_of_conduct.txt` | Workplace behavior, harassment, conflicts of interest, confidentiality |
| `data_security_policy.txt` | Passwords, device security, VPN, email security, incident reporting |

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

This installs LlamaIndex (core + HuggingFace embeddings + Gemini LLM), Streamlit, and python-dotenv. The HuggingFace embedding model (`bge-small-en-v1.5`) downloads automatically on first run (~130MB).

### 2. Set up your API key

```bash
cp .env.example .env
# Edit .env and add your Google Gemini API key
# Get a free key at: https://aistudio.google.com/apikey
```

The search/retrieval pipeline works **without** an API key — you only need it for Gemini answer generation.

### 3. Work through the teaching notebook

Review the Session 23 teaching notebook to learn the core concepts: document loading, chunking, embeddings, vector search, and RAG with LlamaIndex.

### 4. Complete the engine

Open `document_qa.py` and implement TODOs 1–7.

### 5. Build the Streamlit app

Open `qa_app.py` and implement TODOs 8–12.

### 6. Run the chatbot

```bash
streamlit run qa_app.py

# Or test the reference solution
cd solution && streamlit run qa_app.py
```

## Project Structure

```
project-5/
├── data/
│   └── policies/              ← 6 company policy .txt files
├── document_qa.py             ← Student engine (TODOs 1-7)
├── qa_app.py                  ← Student Streamlit app (TODOs 8-12)
├── solution/
│   ├── document_qa.py         ← Reference engine solution
│   └── qa_app.py              ← Reference Streamlit solution
├── architecture.svg           ← RAG pipeline diagram
├── README.md
├── requirements.txt
├── .env.example
└── .gitignore
```

## TODOs

### Engine (`document_qa.py`) — TODOs 1–7

| TODO | Function | What to Implement |
|------|----------|-------------------|
| 1 | `load_documents(directory)` | Use `SimpleDirectoryReader` to load .txt files |
| 2 | `configure_splitter(...)` | Create `SentenceSplitter` with chunk_size and overlap |
| 3 | `setup_embedding_model(...)` | Initialize `HuggingFaceEmbedding` (bge-small-en-v1.5) |
| 4 | `build_index(...)` | Create `VectorStoreIndex` from documents with embeddings |
| 5 | `create_retriever(...)` | Get a retriever from the index for similarity search |
| 6 | `format_prompt(...)` | Build RAG prompt with context chunks + user question |
| 7 | `DocumentQA.answer_question()` | Orchestrate: search → format → generate → return |

### Streamlit App (`qa_app.py`) — TODOs 8–12

| TODO | What to Implement |
|------|-------------------|
| 8 | Initialize DocumentQA in session_state |
| 9 | Display chat history |
| 10 | Process user input |
| 11 | Generate and display answer with sources |
| 12 | Sidebar with stats and clear button |

## Key Concepts

- **Embeddings**: Numerical vectors that capture text meaning. Similar texts → similar vectors. The `bge-small-en-v1.5` model produces 384-dimensional vectors.
- **Semantic search**: Unlike keyword search (TF-IDF), embedding-based search finds results by *meaning*. Searching "guarantee" can find text about "warranty."
- **Chunking**: Documents are split into smaller pieces so the search can find specific, relevant sections rather than returning entire documents.
- **VectorStoreIndex**: Stores all chunks with their embedding vectors for fast cosine similarity search.
- **RAG**: Retrieval-Augmented Generation — retrieve relevant context first, then generate an answer using an LLM. This keeps the LLM grounded in actual documents.

## Example Questions to Test

- "How many vacation days do I get as a new employee?"
- "Can I expense a conference registration fee?"
- "What is the policy on working from home?"
- "What happens if I lose my company laptop?"
- "How do I report harassment?"
- "What does the 401(k) match look like?"
- "What is the password policy?"
- "Can I use my personal phone for work?"

## Git Workflow

Follow the commit-per-part pattern:

```bash
git init
git add .
git commit -m "Initial commit: project starter code and policy documents"

# After completing TODOs 1-3 (loading + chunking + embeddings)
git add document_qa.py
git commit -m "Implement document loading, chunking, and embedding setup"

# After completing TODOs 4-5 (index + retrieval)
git add document_qa.py
git commit -m "Build vector index and semantic search retriever"

# After completing TODOs 6-7 (prompt + generation)
git add document_qa.py
git commit -m "Add RAG prompt formatting and Gemini answer generation"

# After completing TODOs 8-12 (Streamlit app)
git add qa_app.py
git commit -m "Build Streamlit chat interface"
```
