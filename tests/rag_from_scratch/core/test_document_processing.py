import tempfile
from pathlib import Path

import docx
import pypdf
import pytest

from rag_from_scratch.core.document_processing import (
    DocumentReaderFactory,
    DocxReader,
    PDFReader,
    TextReader,
)


@pytest.fixture
def text_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(b"Hello, this is a text file.")
        tmp_path = tmp.name
    yield tmp_path
    Path(tmp_path).unlink()


@pytest.fixture
def pdf_file():
    pdf = pypdf.PdfWriter()
    pdf.add_blank_page(72, 72)
    pdf.write
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.write(tmp)
        tmp_path = tmp.name
    yield tmp_path
    Path(tmp_path).unlink()


@pytest.fixture
def docx_file():
    doc = docx.Document()
    doc.add_paragraph("Hello, this is a Word document.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        doc.save(tmp.name)
        tmp_path = tmp.name
    yield tmp_path
    Path(tmp_path).unlink()


def test_factory_text_reader(text_file):
    reader = DocumentReaderFactory.get_reader(file_path=text_file)
    assert isinstance(reader, TextReader)


def test_factory_pdf_reader(pdf_file):
    reader = DocumentReaderFactory.get_reader(file_path=pdf_file)
    assert isinstance(reader, PDFReader)


def test_factory_docx_reader(docx_file):
    reader = DocumentReaderFactory.get_reader(file_path=docx_file)
    assert isinstance(reader, DocxReader)


def test_factory_unsupported_format():
    with pytest.raises(ValueError, match="Unsupported file format: .unsupported"):
        DocumentReaderFactory.get_reader("test.unsupported")


def test_text_reader(text_file):
    reader = TextReader()
    content = reader.read(text_file)
    assert content == "Hello, this is a text file."


def test_docx_reader(docx_file):
    reader = DocxReader()
    content = reader.read(docx_file)
    assert content == "Hello, this is a Word document."


def test_read_document_text(text_file):
    content = DocumentReaderFactory.read_document(text_file)
    assert content == "Hello, this is a text file."


def test_read_document_docx(docx_file):
    content = DocumentReaderFactory.read_document(docx_file)
    assert content == "Hello, this is a Word document."
