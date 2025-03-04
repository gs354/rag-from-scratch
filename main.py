import logging
from pathlib import Path

from chromadb.api import Collection
from openai import APIError

from rag_from_sratch.chroma_processing import (
    create_collection,
    process_and_add_documents,
)
from rag_from_sratch.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DOCS_DIR,
    EMBEDDING_MODEL,
)
from rag_from_sratch.logging_config import setup_logging
from rag_from_sratch.openai_interface import rag_query
from rag_from_sratch.save_results import save_rag_results

logger = logging.getLogger(__name__)


def get_collection(
    path: str | Path = CHROMA_DIR,
    model_name: str = EMBEDDING_MODEL,
    collection_name: str = COLLECTION_NAME,
) -> Collection:
    """Get or create a collection with sentence transformer embeddings"""
    collection = create_collection(
        path=path,
        model_name=model_name,
        collection_name=collection_name,
    )
    return collection


def process_query(collection: Collection, query: str) -> tuple[str, list[str]]:
    """Process a query and return response with sources"""
    logger.info(f"Processing query: {query}")
    try:
        response, sources, semantic_search_results = rag_query(collection, query)
        logger.info("Query processed successfully")
        return response, sources, semantic_search_results
    except APIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise


def main():
    """Main function to process and add documents to ChromaDB collection"""
    # Set up logging
    setup_logging()
    logger.info("Starting document processing")

    try:
        # Initialize collection and process documents
        collection = get_collection()
        process_and_add_documents(collection=collection, folder_path=DOCS_DIR)
        logger.info("Document processing completed successfully")

        # Process query
        query = "What are the main recommendations given by the contractors?"
        response, sources, semantic_search_results = process_query(collection, query)

        # Save results
        results = {
            "documents": [[response]],
            "metadatas": [[{"source": "OpenAI Response"}]],
            "distances": semantic_search_results["distances"],
        }
        save_rag_results(results, query, include_sources=sources)
        logger.info("Results saved successfully")

    except FileNotFoundError as e:
        logger.error(f"File or directory not found: {e}")
        raise
    except APIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to process documents: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
