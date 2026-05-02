"""
Project 5: Company Policy Chatbot - Document Q&A System

A Document Q&A engine that loads company policy files, chunks them into
searchable pieces, embeds them using HuggingFace embeddings, stores them
in a vector index, and generates answers using Google Gemini.

Uses LlamaIndex as the RAG framework with:
- HuggingFace BAAI/bge-small-en-v1.5 for embeddings (runs locally, free)
- Google Gemini gemini-2.0-flash for answer generation

Students implement 7 functions to build a complete RAG pipeline:
1. Load documents using SimpleDirectoryReader
2. Configure SentenceSplitter for chunking
3. Set up HuggingFace embedding model
4. Build VectorStoreIndex from documents
5. Create a retriever for semantic search
6. Format a RAG prompt with retrieved context
7. Orchestrate the full Q&A pipeline in DocumentQA class
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

    Args:
        directory: Path to folder containing .txt policy documents

    Returns:
        List of LlamaIndex Document objects

    TODO Steps:
    1. Create a SimpleDirectoryReader with:
       - input_dir=directory
       - required_exts=[".txt"]  (only load .txt files)
    2. Call .load_data() to read all files into Document objects
    3. Print how many documents were loaded
    4. Return the list of documents

    Example:
        docs = load_documents('./data/policies/')
        print(len(docs))  # 6
        print(docs[0].metadata['file_name'])  # 'remote_work_policy.txt'
    """
    pass  # Replace with your implementation


# ---------------------------------------------------------------------------
# TODO 2: Configure chunking
# ---------------------------------------------------------------------------

def configure_splitter(chunk_size: int = 512, chunk_overlap: int = 50):
    """
    Create a SentenceSplitter for chunking documents into smaller pieces.

    The SentenceSplitter breaks text at sentence boundaries, respecting
    the chunk_size limit while maintaining overlap between chunks for
    context continuity.

    Args:
        chunk_size: Maximum number of characters per chunk
        chunk_overlap: Number of overlapping characters between consecutive chunks

    Returns:
        Configured SentenceSplitter object

    TODO Steps:
    1. Create a SentenceSplitter with chunk_size and chunk_overlap
    2. Return the splitter

    Example:
        splitter = configure_splitter(chunk_size=512, chunk_overlap=50)
    """
    pass  # Replace with your implementation


# ---------------------------------------------------------------------------
# TODO 3: Set up embedding model
# ---------------------------------------------------------------------------

def setup_embedding_model(model_name: str = "BAAI/bge-small-en-v1.5"):
    """
    Initialize the HuggingFace embedding model for converting text to vectors.

    This model runs locally (no API key needed) and produces 384-dimensional
    vectors. Similar texts will have similar vectors, enabling semantic search.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        HuggingFaceEmbedding object

    TODO Steps:
    1. Check if HuggingFaceEmbedding is available (not None)
    2. Create a HuggingFaceEmbedding with model_name=model_name
    3. Return the embedding model

    Example:
        embed_model = setup_embedding_model()
        # Now you can embed text:
        # vector = embed_model.get_text_embedding("Hello world")
        # len(vector) == 384
    """
    pass  # Replace with your implementation


# ---------------------------------------------------------------------------
# TODO 4: Build vector index
# ---------------------------------------------------------------------------

def build_index(documents, embed_model, splitter):
    """
    Build a VectorStoreIndex from documents.

    This is the core of the RAG system: documents are chunked by the splitter,
    each chunk is embedded into a vector, and all vectors are stored in an
    in-memory index for fast similarity search.

    Args:
        documents: List of LlamaIndex Document objects from load_documents()
        embed_model: HuggingFaceEmbedding from setup_embedding_model()
        splitter: SentenceSplitter from configure_splitter()

    Returns:
        VectorStoreIndex with embedded document chunks

    TODO Steps:
    1. Set the global embedding model: Settings.embed_model = embed_model
    2. Create a VectorStoreIndex using VectorStoreIndex.from_documents():
       - Pass the documents list
       - Pass transformations=[splitter] to chunk during indexing
       - Set show_progress=True for a progress bar
    3. Return the index

    Example:
        index = build_index(documents, embed_model, splitter)
        print(len(index.docstore.docs))  # number of chunks created
    """
    pass  # Replace with your implementation


# ---------------------------------------------------------------------------
# TODO 5: Create retriever
# ---------------------------------------------------------------------------

def create_retriever(index, top_k: int = 3):
    """
    Create a retriever from the vector index for semantic search.

    The retriever takes a text query, embeds it using the same model,
    and finds the top_k most similar chunks using cosine similarity.

    Args:
        index: VectorStoreIndex from build_index()
        top_k: Number of most similar chunks to retrieve

    Returns:
        VectorIndexRetriever object

    TODO Steps:
    1. Call index.as_retriever(similarity_top_k=top_k)
    2. Return the retriever

    Example:
        retriever = create_retriever(index, top_k=3)
        results = retriever.retrieve("What is the PTO policy?")
        for node in results:
            print(f"Score: {node.score:.3f}")
            print(f"Source: {node.node.metadata['file_name']}")
            print(f"Text: {node.node.text[:100]}...")
    """
    pass  # Replace with your implementation


# ---------------------------------------------------------------------------
# TODO 6: Format RAG prompt
# ---------------------------------------------------------------------------

def format_prompt(question: str, retrieved_nodes) -> str:
    """
    Build a RAG prompt combining retrieved context with the user question.

    This is the prompt sent to Gemini. It includes:
    - Instructions telling the LLM to only use provided context
    - The retrieved document chunks with source info
    - The user's question

    Args:
        question: User's question
        retrieved_nodes: List of NodeWithScore objects from retriever.retrieve()

    Returns:
        Formatted prompt string ready to send to Gemini

    TODO Steps:
    1. Start with a system instruction, e.g.:
       "You are a company policy assistant for Nexus Technologies Inc.
        Answer ONLY using the provided policy excerpts.
        Cite the source document for each fact.
        If the answer is not in the context, say so clearly."
    2. Add a "Context:" section. For each node in retrieved_nodes:
       a. Get source filename: node.node.metadata.get('file_name', 'Unknown')
       b. Get relevance score: node.score
       c. Get chunk text: node.node.text
       d. Format as: "Source: {filename} (relevance: {score:.3f})\n{text}\n"
    3. Add the question: "Question: {question}\n\nAnswer:"
    4. Return the complete prompt string

    Example:
        prompt = format_prompt("How many vacation days?", nodes)
        # prompt starts with instructions, then context, then question
    """
    pass  # Replace with your implementation


# ---------------------------------------------------------------------------
# DocumentQA class (orchestrates everything)
# ---------------------------------------------------------------------------

class DocumentQA:
    """
    Main Q&A engine: loads documents, builds vector index, answers questions.

    Workflow:
    1. __init__: Load docs, chunk, embed, build index, init Gemini
    2. search: Retrieve relevant chunks for a query
    3. answer_question: Full RAG pipeline (search → prompt → generate)
    4. get_stats: Return index statistics
    """

    def __init__(self,
                 doc_directory: str,
                 api_key: str = None,
                 model: str = 'gemini-2.0-flash',
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 top_k: int = 3):
        """
        Initialize the Q&A engine.

        Args:
            doc_directory: Path to folder with .txt policy files
            api_key: Google Gemini API key (reads GOOGLE_API_KEY env var if None)
            model: Gemini model name
            chunk_size: Characters per chunk for SentenceSplitter
            chunk_overlap: Overlap between chunks
            top_k: Default number of chunks to retrieve per query

        (Pre-written; no TODO)
        """
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
        """
        Search for relevant document chunks.

        Args:
            query: Search query
            top_k: Override default top_k (optional)

        Returns:
            List of NodeWithScore objects

        (Pre-written; no TODO)
        """
        if top_k and top_k != self.top_k:
            retriever = create_retriever(self.index, top_k)
            return retriever.retrieve(query)
        return self.retriever.retrieve(query)

    def answer_question(self, question: str, top_k: int = None) -> Dict[str, Any]:
        """
        Answer a question using the full RAG pipeline.

        Args:
            question: User's question about company policies
            top_k: Number of chunks to retrieve (optional override)

        Returns:
            Dict with keys:
            - 'answer': Generated answer text
            - 'sources': List of source filenames
            - 'chunks_used': Number of chunks retrieved
            - 'nodes': The retrieved NodeWithScore objects

        TODO 7 Steps:
        1. Call self.search(question, top_k) to get relevant chunks
        2. Call format_prompt(question, retrieved_nodes) to build the prompt
        3. Check if self.llm is available (not None):
           - If yes: call self.llm.complete(prompt) and get response.text
           - If no: set answer to a message about needing an API key
        4. Extract unique source filenames from the nodes:
           - Use node.node.metadata.get('file_name', 'Unknown') for each node
        5. Return a dict with: answer, sources, chunks_used, nodes

        Example:
            qa = DocumentQA('data/policies')
            result = qa.answer_question('How many vacation days do I get?')
            print(result['answer'])
            print(result['sources'])  # ['pto_policy.txt']
        """
        pass  # Replace with your implementation

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded documents and index.

        Returns:
            Dict with keys: num_documents, num_chunks, documents (list of per-doc stats)

        (Pre-written; no TODO)
        """
        docstore = self.index.docstore
        all_nodes = list(docstore.docs.values())

        # Count chunks per source
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
    # Quick test (uncomment to run after implementing TODOs)
    # qa = DocumentQA('data/policies')
    # print(qa.get_stats())
    #
    # # Test search (works without API key)
    # results = qa.search("How many vacation days do I get?")
    # for r in results:
    #     print(f"  [{r.score:.3f}] {r.node.metadata.get('file_name')}")
    #     print(f"    {r.node.text[:100]}...")
    #
    # # Test full pipeline (requires GOOGLE_API_KEY)
    # result = qa.answer_question("What is the remote work policy?")
    # print(f"Answer: {result['answer']}")
    # print(f"Sources: {result['sources']}")
    pass
