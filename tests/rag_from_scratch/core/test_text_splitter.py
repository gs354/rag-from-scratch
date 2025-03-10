from rag_from_scratch.core.text_splitter import (
    create_chunks,
    normalize_whitespace,
    reconstruct_sentences,
    split_into_potential_sentences,
)


def test_normalize_whitespace():
    assert normalize_whitespace("Hello   World") == "Hello World"
    assert normalize_whitespace("This   is   a   test.") == "This is a test."
    assert normalize_whitespace("NoSpacesHere") == "NoSpacesHere"
    assert (
        normalize_whitespace("   Multiple leading and trailing spaces   ")
        == " Multiple leading and trailing spaces "
    )
    assert normalize_whitespace("") == ""


def test_split_into_potential_sentences():
    assert split_into_potential_sentences(
        "Hello Dr. Smith! How are you? I am fine."
    ) == ["Hello Dr", ".", " Smith", "!", " How are you", "?", " I am fine", ".", ""]
    assert split_into_potential_sentences("No punctuation here") == [
        "No punctuation here"
    ]
    assert split_into_potential_sentences("End with emphasis!") == [
        "End with emphasis",
        "!",
        "",
    ]
    assert split_into_potential_sentences("What about... ellipses? Yes!") == [
        "What about... ellipses",
        "?",
        " Yes",
        "!",
        "",
    ]
    assert split_into_potential_sentences("") == [""]


def test_reconstruct_sentences():
    abbreviations = {"mr.", "dr.", "etc.", "e.g."}

    # Simple case
    potential_sentences = ["Hello", "!", " How are you", "?", " I am fine", ".", ""]
    assert reconstruct_sentences(potential_sentences, abbreviations) == [
        "Hello!",
        "How are you?",
        "I am fine.",
    ]

    # Case with abbreviations
    potential_sentences = [
        "This is Mr",
        ".",
        "Smith",
        ".",
        " He is Dr",
        ".",
        "Smith",
        ".",
        "",
    ]
    assert reconstruct_sentences(potential_sentences, abbreviations) == [
        "This is Mr. Smith.",
        "He is Dr. Smith.",
    ]

    # Case with ellipses and abbreviations
    potential_sentences = ["What about... ellipses", "?", " Yes", "!", ""]
    assert reconstruct_sentences(potential_sentences, abbreviations) == [
        "What about... ellipses?",
        "Yes!",
    ]

    # Case with no punctuation
    potential_sentences = ["No punctuation here"]
    assert reconstruct_sentences(potential_sentences, abbreviations) == [
        "No punctuation here"
    ]

    # Case with empty input
    potential_sentences = [""]
    assert reconstruct_sentences(potential_sentences, abbreviations) == []


# Test create_chunks
def test_create_chunks_basic():
    sentences = [
        "This is sentence one.",
        "This is sentence two.",
        "This is sentence three.",
    ]
    chunk_size = 50
    expected_output = [
        "This is sentence one. This is sentence two.",
        "This is sentence three.",
    ]
    assert create_chunks(sentences, chunk_size) == expected_output


def test_create_chunks_exact_size():
    sentences = [
        "These are very good sentences.",
        "This is another good sentence.",
    ]
    chunk_size = 30
    expected_output = [
        "These are very good sentences.",
        "This is another good sentence.",
    ]
    assert create_chunks(sentences, chunk_size) == expected_output


def test_create_chunks_single_sentence():
    sentences = ["This is a single sentence that is longer than the chunk size."]
    chunk_size = 20
    expected_output = ["This is a single sentence that is longer than the chunk size."]
    assert create_chunks(sentences, chunk_size) == expected_output


def test_create_chunks_empty_sentences():
    sentences = ["", " ", "This is a sentence.", "", "Another sentence."]
    chunk_size = 30
    expected_output = [
        "This is a sentence.",
        "Another sentence.",
    ]
    assert create_chunks(sentences, chunk_size) == expected_output
