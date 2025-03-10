import logging

from openai import APIError

from ..core.rag_pipeline import process_query
from ..services.chroma_service import (
    get_collection,
    process_and_add_documents,
)
from ..utils.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DOCS_DIR,
    EMBEDDING_MODEL,
)
from ..utils.logging_config import setup_logging
from ..utils.save_results import save_rag_results

logger = logging.getLogger(__name__)


def main():
    """Main function for basic question-answer RAG"""
    # Set up logging
    setup_logging()
    logger.info("Starting document processing")

    try:
        # Initialize collection and process documents
        collection = get_collection(
            path=CHROMA_DIR, model_name=EMBEDDING_MODEL, collection_name=COLLECTION_NAME
        )
        process_and_add_documents(collection=collection, folder_path=DOCS_DIR)
        logger.info("Document processing completed successfully")

        # Process query
        query = "What are the main recommendations given by the contractors?"
        response, sources, _ = process_query(collection, query)

        # Save results
        results = {
            "query": query,
            "response": response,
            "sources": "\n".join(sources),
        }
        save_rag_results(results)
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
