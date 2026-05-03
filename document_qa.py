"""
Company Policy Chatbot - Document Q&A System

A RAG-powered Q&A engine that loads company policy files, chunks them into
searchable pieces, embeds them using HuggingFace embeddings, stores them
in a vector index, and generates answers using Google Gemini.
"""

import os
from typing import Dict, Any

from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter

# Graceful fallback if optional embedding dependency is not installed
try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    HuggingFaceEmbedding = None

# Graceful fallback if optional LLM dependency is not installed
try:
    from llama_index.llms.gemini import Gemini
except ImportError:
    Gemini = None

load_dotenv()


def load_documents(directory: str):
    """Load all .txt files from a directory using SimpleDirectoryReader."""
    reader = SimpleDirectoryReader(input_dir=directory, required_exts=[".txt"])
    return reader.load_data()


def configure_splitter(chunk_size: int = 512, chunk_overlap: int = 50):
    """Create a SentenceSplitter for chunking documents at sentence boundaries."""
    return SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def setup_embedding_model(model_name: str = "BAAI/bge-small-en-v1.5"):
    """Initialize the HuggingFace embedding model for converting text to 384-dim vectors."""
    if HuggingFaceEmbedding is None:
        raise ImportError(
            "llama-index-embeddings-huggingface is not installed. "
            "Run: pip install llama-index-embeddings-huggingface"
        )
    return HuggingFaceEmbedding(model_name=model_name)


def build_index(documents, embed_model, splitter):
    """Build a VectorStoreIndex by chunking and embedding all documents."""
    Settings.embed_model = embed_model
    return VectorStoreIndex.from_documents(
        documents,
        transformations=[splitter],
        show_progress=True,
    )


def create_retriever(index, top_k: int = 3):
    """Create a cosine-similarity retriever from the vector index."""
    return index.as_retriever(similarity_top_k=top_k)


def format_prompt(question: str, retrieved_nodes) -> str:
    """Build a RAG prompt combining retrieved policy context with the user question."""
    prompt = (
        "You are a company policy assistant for Nexus Technologies Inc. "
        "Answer the following question ONLY using the provided policy document "
        "excerpts. Cite the source document for each fact you mention. "
        "If the answer is not found in the provided context, say so clearly.\n\n"
        "Context from policy documents:\n"
    )

    for i, node in enumerate(retrieved_nodes, 1):
        source = node.node.metadata.get("file_name", "Unknown")
        score = node.score
        text = node.node.text
        prompt += f"\n--- Excerpt {i} (Source: {source}, Relevance: {score:.3f}) ---\n"
        prompt += f"{text}\n"

    prompt += f"\nQuestion: {question}\n\nAnswer:"
    return prompt


class DocumentQA:
    """Main Q&A engine: loads documents, builds a vector index, and answers questions."""

    def __init__(
        self,
        doc_directory: str,
        api_key: str = None,
        model: str = "gemini-2.5-flash",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 3,
    ):
        """Initialize the Q&A engine."""
        self.doc_directory = doc_directory
        self.model_name = model
        self.top_k = top_k

        self.documents = load_documents(doc_directory)
        self.splitter = configure_splitter(chunk_size, chunk_overlap)
        self.embed_model = setup_embedding_model()
        self.index = build_index(self.documents, self.embed_model, self.splitter)
        self.retriever = create_retriever(self.index, top_k)

        # Initialize Gemini LLM for answer generation
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if api_key and Gemini is not None:
            self.llm = Gemini(model=f"models/{model}", api_key=api_key)
        else:
            self.llm = None

    def search(self, query: str, top_k: int = None):
        """Search for relevant document chunks using semantic similarity."""
        if top_k and top_k != self.top_k:
            retriever = create_retriever(self.index, top_k)
            return retriever.retrieve(query)
        return self.retriever.retrieve(query)

    def answer_question(self, question: str, top_k: int = None) -> Dict[str, Any]:
        """Answer a question using the full RAG pipeline: search → prompt → generate."""
        nodes = self.search(question, top_k)
        prompt = format_prompt(question, nodes)

        if self.llm is not None:
            try:
                response = self.llm.complete(prompt)
                answer = response.text
            except Exception as e:
                answer = f"Error generating answer: {str(e)}"
        else:
            answer = (
                "Gemini API not available. Set GOOGLE_API_KEY environment variable "
                "to enable answer generation.\n\n"
                "Retrieved context (search still works without API key):\n"
            )
            for node in nodes:
                source = node.node.metadata.get("file_name", "Unknown")
                answer += f"\n[{source}]: {node.node.text[:200]}...\n"

        sources = sorted(set(
            node.node.metadata.get("file_name", "Unknown") for node in nodes
        ))

        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": len(nodes),
            "nodes": nodes,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded documents and index."""
        all_nodes = list(self.index.docstore.docs.values())

        source_counts: Dict[str, int] = {}
        for node in all_nodes:
            source = node.metadata.get("file_name", "Unknown")
            source_counts[source] = source_counts.get(source, 0) + 1

        return {
            "num_documents": len(self.documents),
            "num_chunks": len(all_nodes),
            "documents": [
                {"filename": name, "num_chunks": count}
                for name, count in sorted(source_counts.items())
            ],
        }
