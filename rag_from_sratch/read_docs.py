import logging
from pathlib import Path
from typing import Protocol

import docx
import PyPDF2

logger = logging.getLogger(__name__)


class DocumentReader(Protocol):
    """Protocol defining the interface for document readers"""

    def read(self, file_path: str | Path) -> str:
        """Read and return the content of a document"""
        ...


class TextReader:
    """Reader for text files"""

    def read(self, file_path: str | Path) -> str:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()


class PDFReader(DocumentReader):
    """Reader for PDF files"""

    def read(self, file_path: str | Path) -> str:
        text = ""
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text


class DocxReader(DocumentReader):
    """Reader for Word documents"""

    def read(self, file_path: str | Path) -> str:
        doc = docx.Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])


class DocumentReaderFactory:
    """Factory class for creating document readers"""

    _readers = {
        ".txt": TextReader,
        ".pdf": PDFReader,
        ".docx": DocxReader,
    }

    @classmethod
    def get_reader(cls, file_path: str | Path) -> DocumentReader:
        """Get appropriate reader based on file extension"""
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()

        reader_class = cls._readers.get(file_extension)
        if reader_class is None:
            raise ValueError(f"Unsupported file format: {file_extension}")

        return reader_class()

    @classmethod
    def read_document(cls, file_path: str | Path) -> str:
        """Convenience method to read a document"""
        reader = cls.get_reader(file_path)
        return reader.read(file_path)
