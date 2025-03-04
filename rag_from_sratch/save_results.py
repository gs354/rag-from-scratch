import csv
import logging
from datetime import datetime

from .config import RESULTS_DIR

logger = logging.getLogger(__name__)


def save_rag_results(
    results: dict, query: str, include_sources: list[str] | None = None
) -> None:
    """Save RAG results to a CSV file in the results directory.

    Args:
        results: Search results from ChromaDB
        query: The search query that produced these results
        include_sources: Optional list of sources to include in results
    """
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = RESULTS_DIR / f"search_results_{timestamp}.csv"

    try:
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Write header
            header = ["Query", "Rank", "Source", "Distance", "Content"]
            if include_sources:
                header.extend(["Referenced Sources"])
            writer.writerow(header)

            # Write results
            for i in range(len(results["documents"][0])):
                doc = results["documents"][0][i]
                meta = results["metadatas"][0][i]
                distance = results["distances"][0][i]

                row = [
                    query,
                    i + 1,
                    meta.get("source", "Unknown"),
                    distance,
                    doc,
                ]

                # Add sources if provided and this is the first row
                if include_sources and i == 0:
                    row.append("\n".join(include_sources))

                writer.writerow(row)

        logger.info(f"Search results saved to {filename}")

    except Exception as e:
        logger.error(f"Error saving search results: {e}")
        raise
