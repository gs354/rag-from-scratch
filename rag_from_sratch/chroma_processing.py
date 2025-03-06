import logging
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api.types import EmbeddingFunction
from chromadb.errors import InvalidCollectionException
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


def get_processed_files(collection: Collection) -> set[str]:
    """Get set of files that have already been processed"""
    # Get all metadatas from collection
    results = collection.get()
    if not results["metadatas"]:
        return set()

    # Extract unique source filenames
    return {meta["source"] for meta in results["metadatas"]}


def process_and_add_documents(collection: Collection, folder_path: str | Path) -> None:
    """Process all documents in a folder and add to collection"""
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        raise ValueError(f"Invalid folder path: {folder_path}")

    # Get already processed files
    processed_files = get_processed_files(collection)

    # Get list of files to process
    supported_extensions = get_supported_extensions()
    files = [
        file
        for file in folder_path.iterdir()
        if file.is_file()
        and file.suffix.lower() in supported_extensions
        and file.name not in processed_files  # Only process new files
    ]

    if not files:
        logger.info("No new files to process")
        return

    for file_path in files:
        try:
            logger.info(f"Processing new file: {file_path.name}...")
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
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=str(Path(path)))

    # Check if collection exists
    try:
        # Try to get existing collection
        collection = client.get_collection(name=collection_name)
        logger.info(f"Retrieved existing collection: {collection_name}")
    except InvalidCollectionException:
        # Collection doesn't exist, create new one with embedding function
        sentence_transformer_ef = get_embedding_function(model_name)
        collection = client.create_collection(
            name=collection_name, embedding_function=sentence_transformer_ef
        )
        logger.info(f"Created new collection: {collection_name}")

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
