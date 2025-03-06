import csv
import logging
from datetime import datetime

from .config import RESULTS_DIR

logger = logging.getLogger(__name__)


def save_rag_results(results: dict) -> None:
    """Save RAG query results to a CSV file.

    Args:
        results: Dictionary containing:
            - query: The search query
            - response: The OpenAI response
            - sources: Sources, chunks and distances
    """
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = RESULTS_DIR / f"rag_results_{timestamp}.csv"

    try:
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(["Query", "Response", "Sources"])

            # Write results
            writer.writerow(
                [
                    results["query"],
                    results["response"],
                    results["sources"],
                ]
            )

        logger.info(f"RAG results saved to {filename}")

    except Exception as e:
        logger.error(f"Error saving RAG results: {e}")
        raise
