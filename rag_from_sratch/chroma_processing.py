import logging
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api.types import EmbeddingFunction
from chromadb.utils import embedding_functions

from .read_docs import DocumentReaderFactory
from .text_splitter import split_text

logger = logging.getLogger(__name__)


class DocumentProcessingError(Exception):
    """Custom exception for document processing errors"""

    pass


def process_document(file_path: str | Path) -> tuple[list[str], list[str], list[dict]]:
    """Process a single document into chunks with ids and metadata"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise DocumentProcessingError(f"File not found: {file_path}")

    try:
        # Read the document
        content = DocumentReaderFactory.read_document(file_path)
    except ValueError as e:
        raise DocumentProcessingError(f"Unsupported file format: {e}")
    except Exception as e:
        raise DocumentProcessingError(f"Error reading file: {e}")

    # Split into chunks
    chunks = split_text(content)
    if not chunks:
        logger.warning(f"No content chunks generated from {file_path}")
        return [], [], []

    # Prepare metadata
    file_name = file_path.name
    metadatas = [{"source": file_name, "chunk": i} for i in range(len(chunks))]
    ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]

    return ids, chunks, metadatas


def add_to_collection(
    collection: Collection, ids: list[str], texts: list[str], metadatas: list[dict]
) -> None:
    """Add documents to collection in batches"""
    if not texts:
        return

    if not (len(ids) == len(texts) == len(metadatas)):
        raise ValueError("Mismatched lengths for ids, texts, and metadatas")

    batch_size = 100
    for i in range(0, len(texts), batch_size):
        end_idx = min(i + batch_size, len(texts))
        try:
            collection.add(
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx],
            )
        except Exception as e:
            logger.error(f"Error adding batch to collection: {e}")
            raise


def get_supported_extensions() -> set[str]:
    """Get set of supported file extensions"""
    return set(DocumentReaderFactory._readers.keys())


def process_and_add_documents(collection: Collection, folder_path: str | Path) -> None:
    """Process all documents in a folder and add to collection"""
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        raise ValueError(f"Invalid folder path: {folder_path}")

    supported_extensions = get_supported_extensions()
    files = [
        file
        for file in folder_path.iterdir()
        if file.is_file() and file.suffix.lower() in supported_extensions
    ]

    if not files:
        logger.warning(f"No supported files found in {folder_path}")
        return

    for file_path in files:
        try:
            logger.info(f"Processing {file_path.name}...")
            ids, texts, metadatas = process_document(file_path)
            add_to_collection(collection, ids, texts, metadatas)
            logger.info(f"Added {len(texts)} chunks to collection")
        except DocumentProcessingError as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error processing {file_path}: {e}")
            raise


def get_embedding_function(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingFunction:
    """Create and return a sentence transformer embedding function.

    Args:
        model_name: Name of the sentence transformer model to use

    Returns:
        SentenceTransformerEmbeddingFunction instance
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )


def create_collection(
    path: str | Path = "./chroma",
    model_name: str = "all-MiniLM-L6-v2",
    collection_name: str = "documents_collection",
) -> Collection:
    """Create or get existing collection with sentence transformer embeddings"""
    # Initialize ChromaDB client: persistent client for dev
    client = chromadb.PersistentClient(
        path=str(Path(path))
    )  # ChromaDB expects string path

    # Get embedding function
    sentence_transformer_ef = get_embedding_function(model_name)

    # Create or get existing collection
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=sentence_transformer_ef
    )
    return collection


def semantic_search(collection: Collection, query: str, n_results: int = 2) -> dict:
    """Perform semantic search on the collection"""
    results = collection.query(query_texts=[query], n_results=n_results)
    return results


def get_context_with_sources(results: dict) -> tuple[str, list[str]]:
    """Extract context and source information from search results"""
    # Combine document chunks into a single context
    context = "\n\n".join(results["documents"][0])

    # Format sources with metadata and distances
    sources = [
        f"{meta['source']} (chunk {meta['chunk']}, dist {round(dist, 4)})"
        for meta, dist in zip(results["metadatas"][0], results["distances"][0])
    ]

    return context, sources
