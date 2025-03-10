from rag_from_scratch.core.abbreviations import get_common_abbreviations
from rag_from_scratch.core.text_splitter import (
    normalize_whitespace,
    reconstruct_sentences,
    split_into_potential_sentences,
)

ABBREVIATIONS = get_common_abbreviations()


def test_normalize_whitespace():
    text = "   Testing  whitespace   normalization.  "
    normalized = normalize_whitespace(text)
    assert normalized == " Testing whitespace normalization. "


def test_split_into_potential_sentences():
    text = "Dr. Smith went home. He was tired!"
    potential_sentences = split_into_potential_sentences(text)
    expected = ["Dr", ".", " Smith went home", ".", " He was tired", "!", ""]
    assert potential_sentences == expected


def test_reconstruct_sentences():
    input = ["Dr", ".", " Smith went home", ".", " He was tired", "!", ""]
    reconstructed = reconstruct_sentences(
        potential_sentences=input, abbreviations=ABBREVIATIONS
    )
    expected = ["Dr. Smith went home.", "He was tired!"]
    assert reconstructed == expected
