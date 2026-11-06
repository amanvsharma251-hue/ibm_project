"""
rag_engine.py
=============
RAG (Retrieval-Augmented Generation) core for the College Admission Agent.

Responsibilities:
  - Load and split knowledge base documents
  - Generate embeddings using sentence-transformers (runs locally, no API cost)
  - Store and persist vectors in ChromaDB
  - Retrieve relevant chunks for a given user query
"""

import os
import logging
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------
load_dotenv("config.env")   # loads config.env; rename to .env if desired

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (from env or sensible defaults)
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE_DIR = Path(__file__).parent.parent / "knowledge_base"
CHROMA_PERSIST_DIR  = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE          = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP       = int(os.getenv("CHUNK_OVERLAP", "50"))
RAG_TOP_K           = int(os.getenv("RAG_TOP_K", "5"))


# ---------------------------------------------------------------------------
# Document Loading
# ---------------------------------------------------------------------------

def load_documents(kb_dir: Path = KNOWLEDGE_BASE_DIR) -> List[Document]:
    """
    Load all .txt (and optionally .pdf / .docx) files from the knowledge base.
    Returns a flat list of LangChain Document objects.
    """
    if not kb_dir.exists():
        raise FileNotFoundError(
            f"Knowledge base directory not found: {kb_dir}\n"
            "Place your .txt / .pdf files in the 'knowledge_base/' folder."
        )

    docs: List[Document] = []

    # Load .txt files
    txt_loader = DirectoryLoader(
        str(kb_dir),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    txt_docs = txt_loader.load()
    logger.info("Loaded %d text document(s) from %s", len(txt_docs), kb_dir)
    docs.extend(txt_docs)

    # Optionally load PDFs if pypdf is installed
    try:
        from langchain_community.document_loaders import PyPDFDirectoryLoader
        pdf_loader = PyPDFDirectoryLoader(str(kb_dir))
        pdf_docs = pdf_loader.load()
        if pdf_docs:
            logger.info("Loaded %d PDF page(s)", len(pdf_docs))
            docs.extend(pdf_docs)
    except Exception:
        pass   # pypdf not installed or no PDFs — silently skip

    if not docs:
        raise ValueError(
            "No documents found in knowledge base. "
            "Add .txt or .pdf files to 'knowledge_base/'."
        )

    return docs


# ---------------------------------------------------------------------------
# Text Splitting
# ---------------------------------------------------------------------------

def split_documents(docs: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks suitable for embedding.
    Uses RecursiveCharacterTextSplitter which respects paragraph/sentence boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    logger.info(
        "Split %d document(s) into %d chunk(s) "
        "(chunk_size=%d, overlap=%d)",
        len(docs), len(chunks), CHUNK_SIZE, CHUNK_OVERLAP,
    )
    return chunks


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Returns a locally-running HuggingFace sentence-transformer embedding model.
    No API key required.  First run downloads the model (~90 MB).
    """
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------

def build_vector_store(chunks: List[Document], embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Create a new ChromaDB vector store from document chunks.
    Persists to CHROMA_PERSIST_DIR for reuse across sessions.
    """
    logger.info("Building vector store at '%s' …", CHROMA_PERSIST_DIR)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name="college_admission_kb",
    )
    logger.info("Vector store built with %d vectors.", vector_store._collection.count())
    return vector_store


def load_vector_store(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Load an existing persisted ChromaDB vector store.
    Raises FileNotFoundError if the store has not been built yet.
    """
    if not Path(CHROMA_PERSIST_DIR).exists():
        raise FileNotFoundError(
            f"Vector store not found at '{CHROMA_PERSIST_DIR}'. "
            "Run build_knowledge_base() first."
        )
    logger.info("Loading existing vector store from '%s' …", CHROMA_PERSIST_DIR)
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="college_admission_kb",
    )


def get_or_build_vector_store(force_rebuild: bool = False) -> Chroma:
    """
    Convenience function:
      - If the vector store already exists (and force_rebuild is False), load it.
      - Otherwise, load documents, split, embed, and persist.
    """
    embeddings = get_embedding_model()

    if not force_rebuild and Path(CHROMA_PERSIST_DIR).exists():
        try:
            return load_vector_store(embeddings)
        except Exception as e:
            logger.warning("Could not load existing store (%s). Rebuilding …", e)

    docs   = load_documents()
    chunks = split_documents(docs)
    return build_vector_store(chunks, embeddings)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_relevant_chunks(query: str, vector_store: Chroma, top_k: int = RAG_TOP_K) -> List[Document]:
    """
    Retrieve the top-k most semantically similar document chunks for a query.

    Args:
        query: The user's natural-language question.
        vector_store: An initialised ChromaDB Chroma instance.
        top_k: Number of chunks to return.

    Returns:
        List of LangChain Document objects with page_content and metadata.
    """
    retriever = vector_store.as_retriever(
        search_type="mmr",          # Maximal Marginal Relevance — diverse results
        search_kwargs={"k": top_k, "fetch_k": top_k * 3},
    )
    results = retriever.invoke(query)
    logger.debug("Retrieved %d chunk(s) for query: %s", len(results), query[:80])
    return results


def format_context(docs: List[Document]) -> str:
    """
    Format retrieved document chunks into a single context string for the LLM prompt.
    Each chunk is prefixed with its source filename for transparency.
    """
    if not docs:
        return "No relevant information found in the knowledge base."

    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        source_name = Path(source).name if source != "unknown" else "knowledge base"
        parts.append(f"[Source {i}: {source_name}]\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# CLI: rebuild knowledge base
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import shutil
    print("Rebuilding knowledge base vector store …")
    if Path(CHROMA_PERSIST_DIR).exists():
        shutil.rmtree(CHROMA_PERSIST_DIR)
        print(f"Deleted old store at '{CHROMA_PERSIST_DIR}'")
    store = get_or_build_vector_store(force_rebuild=True)
    print(f"Done! Vector store ready with {store._collection.count()} vectors.")
