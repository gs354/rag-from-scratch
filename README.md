# RAG from scratch study project

A Retrieval-Augmented Generation (RAG) implementation using ChromaDB and OpenAI.

## Installation

1. Create and activate a virtual environment, e.g. on MacOs/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Clone the package and move into the directory:
```bash
git clone https://github.com/gs354/rag-from-scratch.git
cd rag-from-scratch
```

3. Install the package for users (no development dependencies):
```bash
pip install -e .
```

4. Or install the package with development dependencies (required to run tests):
```bash
pip install -e ".[dev]"
```

## Project Structure

```
rag-project/
├── src/
│   └── rag_from_scratch/
│       ├── core/                      # Core processing logic
│       │   ├── document_processing.py # Document readers
│       │   ├── text_splitter.py       # Text chunking
│       │   ├── rag_pipeline.py        # RAG conversation management
│       │   └── abbreviations.py       # Text processing helpers
│       ├── services/                  # External service integrations
│       │   ├── chroma_service.py      # Vector DB operations
│       │   └── openai_service.py      # LLM operations
│       ├── utils/                     # Utilities
│       │   └── save_results.py        # Results handling
│       ├── config/                    # Config
│       │   ├── config.py              # Variables, keys and paths setup
│       │   ├── logging_config.py      # Logging setup
│       │   └── config.toml            # Default settings
│       └── cli/                       # Command-line interface
│           └── main.py                # Entry point
├── tests/                    # Test files
├── data/                     # Data directories
│   ├── raw/                 # Input documents
│   └── processed/           # Vector DB storage
|
├── results/                 # Query results  
└── pyproject.toml           # Project metadata
```

## Usage
- Place documents to be embedded in the `data/raw` directory.
- To start the app, run:

```bash
rag-from-scratch
```
- You will be prompted to ask questions in a loop, until you type 'exit'.

### Running tests

```bash
pytest
```


### Core Dependencies
- **[ChromaDB](https://www.trychroma.com/)** >=0.6.3
  - Vector database for document storage and retrieval
  - Handles embedding storage and similarity search

- **[OpenAI](https://platform.openai.com/docs/introduction)** >=1.65.1
  - LLM integration for query processing
  - Response generation

- **[Sentence Transformers](https://www.sbert.net/)** >=3.4.1
  - Text embedding generation
  - Used by ChromaDB for document vectorization

### Document Processing
- **[PyPDF2](https://pypdf2.readthedocs.io/en/latest/)** >=3.0.1
  - PDF document processing
  - Text extraction from PDF files

- **[python-docx](https://python-docx.readthedocs.io/en/latest/)** >=1.1.2
  - Word document processing
  - Text extraction from DOCX files

### Utilities
- **[python-dotenv](https://github.com/theskumar/python-dotenv)** >=1.0.1
  - Environment variable management
  - API key configuration

### Development Dependencies
- **pytest** >=8.3.5: Testing framework
- **ruff** >=0.9.9: Linting and formatting
