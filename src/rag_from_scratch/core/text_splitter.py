import logging
import re

from .abbreviations import get_common_abbreviations

logger = logging.getLogger(__name__)


def normalize_whitespace(text: str) -> str:
    """Replace multiple whitespace characters with a single space"""
    return re.sub(r"\s+", " ", text)


def split_into_potential_sentences(text: str) -> list[str]:
    """Split text into potential sentences based on punctuation and capitalization.
    The output list of strings is constructed as follows:
    - Even indices (0, 2, 4...) contain the text before punctuation.
    - Odd indices (1, 3, 5...) contain the punctuation.

    Args:
        text (str): The text to split into potential sentences

    Returns:
        list[str]: A list of potential sentences split from their punctuation.
    """
    # Look for: .!? followed by space and capital letter, or end of string
    return re.split(r"([.!?]+(?=\s+[A-Z]|\s*$))", text)


def is_abbreviation_end(text: str, abbreviations: set[str]) -> bool:
    """Check if the text ends with a known abbreviation"""
    return any(text.lower().endswith(abbr) for abbr in abbreviations)


def reconstruct_sentences(
    potential_sentences: list[str], abbreviations: set[str]
) -> list[str]:
    """Reconstruct proper sentences, handling abbreviations.
    Potential sentences are split from their punctuation, so we need to reconstruct.
    The input list of strings is constructed as follows:
    - Even indices (0, 2, 4...) contain the text before punctuation.
    - Odd indices (1, 3, 5...) contain the punctuation.

    Args:
        potential_sentences (list[str]): A list of potential sentences split from their punctuation.
        abbreviations (set[str]): A set of common abbreviations.

    Returns:
        list[str]: A list of reconstructed sentences.
    """
    sentences = []
    current_sentence = ""

    for i in range(0, len(potential_sentences) - 1, 2):
        current_part = potential_sentences[i].strip()
        punctuation = (
            potential_sentences[i + 1] if i + 1 < len(potential_sentences) else ""
        )

        if is_abbreviation_end(current_part + punctuation, abbreviations):
            current_sentence += current_part + punctuation + " "
        else:
            sentences.append((current_sentence + current_part + punctuation).strip())
            current_sentence = ""

    # Handle any remaining text
    if current_sentence or (
        len(potential_sentences) % 2 == 1 and potential_sentences[-1].strip()
    ):
        sentences.append((current_sentence + potential_sentences[-1]).strip())

    return sentences


def create_chunks(sentences: list[str], chunk_size: int) -> list[str]:
    """Create chunks of text from sentences, respecting maximum chunk size"""
    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_size = len(sentence)

        # Check if adding this sentence would exceed chunk size
        if current_size + sentence_size > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size + 1  # +1 for the space between sentences

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def split_text(text: str, chunk_size: int = 500) -> list[str]:
    """Split text into chunks while preserving sentence boundaries"""
    text = normalize_whitespace(text)
    abbreviations = get_common_abbreviations()
    potential_sentences = split_into_potential_sentences(text)
    sentences = reconstruct_sentences(potential_sentences, abbreviations)
    return create_chunks(sentences, chunk_size)
