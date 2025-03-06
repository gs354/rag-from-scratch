import os
from pathlib import Path

import tomllib
from dotenv import load_dotenv

# Load environment variables for sensitive data
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Load TOML config
PACKAGE_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PACKAGE_ROOT / "config" / "config.toml"
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")

with open(CONFIG_PATH, "rb") as f:
    config = tomllib.load(f)

# OpenAI Configuration
OPENAI_MODEL = config["openai"]["model"]
OPENAI_TEMPERATURE = config["openai"]["temperature"]
OPENAI_MAX_TOKENS = config["openai"]["max_tokens"]

# Paths Configuration
RESULTS_DIR = Path(config["paths"]["results_dir"])
DOCS_DIR = Path(config["paths"]["docs_dir"])
CHROMA_DIR = Path(config["paths"]["chroma_dir"])

# ChromaDB Configuration
EMBEDDING_MODEL = config["chroma"]["embedding_model"]
COLLECTION_NAME = config["chroma"]["collection_name"]

# Logging Configuration
LOG_LEVEL = config["logging"]["level"]
LOG_FILE = config["logging"]["log_file"]

# Ensure required directories exist
RESULTS_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)
