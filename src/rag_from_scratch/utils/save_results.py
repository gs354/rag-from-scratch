import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def save_rag_results(filepath: Path | str, results: dict) -> None:
    """Save RAG query results to a CSV file.

    Args:
        filepath: File path to save
        results: Dictionary containing:
            - query: The search query
            - response: The OpenAI response
            - sources: Sources, chunks and distances
    """
    try:
        # Convert filepath to Path object if it's a string
        filepath = Path(filepath) if isinstance(filepath, str) else filepath

        # Check if the file exists
        write_header = not filepath.exists()

        # Open the file in append mode
        with filepath.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Write header only if the file doesn't exist
            if write_header:
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

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except PermissionError as e:
        logger.error(f"Permission denied: {e}")
        raise
    except OSError as e:
        logger.error(f"OS error occurred: {e}")
        raise
    except csv.Error as e:
        logger.error(f"CSV writing error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
