"""
Project 5: Company Policy Chatbot - Document Q&A System (Solution)

Uses LlamaIndex with HuggingFace embeddings for semantic search
and Google Gemini for answer generation (RAG pipeline).
"""

import os
from typing import List, Dict, Any

from dotenv import load_dotenv

# LlamaIndex core
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter

# Embedding model (runs locally, no API key needed)
try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    HuggingFaceEmbedding = None

# LLM for answer generation (requires API key)
try:
    from llama_index.llms.gemini import Gemini
except ImportError:
    Gemini = None

load_dotenv()


# ---------------------------------------------------------------------------
# TODO 1: Load documents
# ---------------------------------------------------------------------------

def load_documents(directory: str):
    """
    Load all .txt files from a directory using LlamaIndex's SimpleDirectoryReader.
    """
    reader = SimpleDirectoryReader(
        input_dir=directory,
        required_exts=[".txt"]
    )
    documents = reader.load_data()
    print(f"Loaded {len(documents)} documents")
    return documents


# ---------------------------------------------------------------------------
# TODO 2: Configure chunking
# ---------------------------------------------------------------------------

def configure_splitter(chunk_size: int = 512, chunk_overlap: int = 50):
    """
    Create a SentenceSplitter for chunking documents into smaller pieces.
    """
    return SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )


# ---------------------------------------------------------------------------
# TODO 3: Set up embedding model
# ---------------------------------------------------------------------------

def setup_embedding_model(model_name: str = "BAAI/bge-small-en-v1.5"):
    """
    Initialize the HuggingFace embedding model for converting text to vectors.
    """
    if HuggingFaceEmbedding is None:
        raise ImportError(
            "llama-index-embeddings-huggingface is not installed. "
            "Run: pip install llama-index-embeddings-huggingface"
        )
    return HuggingFaceEmbedding(model_name=model_name)


# ---------------------------------------------------------------------------
# TODO 4: Build vector index
# ---------------------------------------------------------------------------

def build_index(documents, embed_model, splitter):
    """
    Build a VectorStoreIndex from documents.
    """
    Settings.embed_model = embed_model
    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[splitter],
        show_progress=True
    )
    return index


# ---------------------------------------------------------------------------
# TODO 5: Create retriever
# ---------------------------------------------------------------------------

def create_retriever(index, top_k: int = 3):
    """
    Create a retriever from the vector index for semantic search.
    """
    return index.as_retriever(similarity_top_k=top_k)


# ---------------------------------------------------------------------------
# TODO 6: Format RAG prompt
# ---------------------------------------------------------------------------

def format_prompt(question: str, retrieved_nodes) -> str:
    """
    Build a RAG prompt combining retrieved context with the user question.
    """
    prompt = (
        "You are a company policy assistant for Nexus Technologies Inc. "
        "Answer the following question ONLY using the provided policy document "
        "excerpts. Cite the source document for each fact you mention. "
        "If the answer is not found in the provided context, say so clearly.\n\n"
        "Context from policy documents:\n"
    )

    for i, node in enumerate(retrieved_nodes, 1):
        source = node.node.metadata.get('file_name', 'Unknown')
        score = node.score
        text = node.node.text
        prompt += f"\n--- Excerpt {i} (Source: {source}, Relevance: {score:.3f}) ---\n"
        prompt += f"{text}\n"

    prompt += f"\nQuestion: {question}\n\nAnswer:"
    return prompt


# ---------------------------------------------------------------------------
# DocumentQA class (orchestrates everything)
# ---------------------------------------------------------------------------

class DocumentQA:
    """
    Main Q&A engine: loads documents, builds vector index, answers questions.
    """

    def __init__(self,
                 doc_directory: str,
                 api_key: str = None,
                 model: str = 'gemini-2.5-flash',
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 top_k: int = 3):
        """Initialize the Q&A engine."""
        self.doc_directory = doc_directory
        self.model_name = model
        self.top_k = top_k

        # Step 1: Load documents
        self.documents = load_documents(doc_directory)

        # Step 2: Set up components
        self.splitter = configure_splitter(chunk_size, chunk_overlap)
        self.embed_model = setup_embedding_model()

        # Step 3: Build vector index
        self.index = build_index(self.documents, self.embed_model, self.splitter)
        print(f"Index built with {len(self.index.docstore.docs)} chunks")

        # Step 4: Create retriever
        self.retriever = create_retriever(self.index, top_k)

        # Step 5: Initialize Gemini LLM
        api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if api_key and Gemini is not None:
            self.llm = Gemini(model=f"models/{model}", api_key=api_key)
        else:
            self.llm = None
            print("Warning: Gemini not available. Set GOOGLE_API_KEY for answer generation.")

    def search(self, query: str, top_k: int = None):
        """Search for relevant document chunks."""
        if top_k and top_k != self.top_k:
            retriever = create_retriever(self.index, top_k)
            return retriever.retrieve(query)
        return self.retriever.retrieve(query)

    def answer_question(self, question: str, top_k: int = None) -> Dict[str, Any]:
        """
        Answer a question using the full RAG pipeline.

        TODO 7: Orchestrate search → format → generate → return
        """
        # 1. Retrieve relevant chunks
        nodes = self.search(question, top_k)

        # 2. Format prompt with context
        prompt = format_prompt(question, nodes)

        # 3. Generate answer
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
                source = node.node.metadata.get('file_name', 'Unknown')
                answer += f"\n[{source}]: {node.node.text[:200]}...\n"

        # 4. Extract unique sources
        sources = list(set(
            node.node.metadata.get('file_name', 'Unknown')
            for node in nodes
        ))

        # 5. Return result
        return {
            'answer': answer,
            'sources': sorted(sources),
            'chunks_used': len(nodes),
            'nodes': nodes
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded documents and index."""
        docstore = self.index.docstore
        all_nodes = list(docstore.docs.values())

        source_counts = {}
        for node in all_nodes:
            source = node.metadata.get('file_name', 'Unknown')
            source_counts[source] = source_counts.get(source, 0) + 1

        return {
            'num_documents': len(self.documents),
            'num_chunks': len(all_nodes),
            'documents': [
                {'filename': name, 'num_chunks': count}
                for name, count in sorted(source_counts.items())
            ]
        }


if __name__ == '__main__':
    qa = DocumentQA('data/policies')
    stats = qa.get_stats()
    print(f"\nIndex Stats:")
    print(f"  Documents: {stats['num_documents']}")
    print(f"  Chunks: {stats['num_chunks']}")
    for doc in stats['documents']:
        print(f"    {doc['filename']}: {doc['num_chunks']} chunks")

    # Test semantic search (works without API key)
    print("\n--- Search Test ---")
    test_queries = [
        "How many vacation days do I get?",
        "What is the password policy?",
        "Can I work from home?",
    ]
    for query in test_queries:
        print(f"\nQ: {query}")
        results = qa.search(query)
        for r in results:
            print(f"  [{r.score:.3f}] {r.node.metadata.get('file_name')}: {r.node.text[:80]}...")

    # Test full pipeline (requires GOOGLE_API_KEY)
    print("\n--- Full Pipeline Test ---")
    result = qa.answer_question("How many vacation days do new employees get?")
    print(f"Answer: {result['answer'][:300]}...")
    print(f"Sources: {result['sources']}")
    print(f"Chunks used: {result['chunks_used']}")
