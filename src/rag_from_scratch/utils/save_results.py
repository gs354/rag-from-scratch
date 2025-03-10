import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def save_rag_results(filepath: Path | str, results: dict) -> None:
    """Save RAG query results to a CSV file.

    Args:
        filename: File path to save
        results: Dictionary containing:
            - query: The search query
            - response: The OpenAI response
            - sources: Sources, chunks and distances
    """

    try:
        with open(filepath, "a", newline="", encoding="utf-8") as f:
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

        logger.info(f"RAG results saved to {filepath}")

    except Exception as e:
        logger.error(f"Error saving RAG results: {e}")
        raise
