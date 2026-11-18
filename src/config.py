from pathlib import Path

# Project root = one level above the src/ folder, no matter where you run the scripts from
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
DB_DIR = str(ROOT / "chroma_db")
COLLECTION_NAME = "documents"

# The SAME model must be used to ingest and to search
EMBED_MODEL = "nomic-embed-text"

# Friendly names for the files in data/
BOOK_TITLES = {
    "evangelho-espritismo": "O Evangelho segundo o Espiritismo",
    "livro-dos-mediuns": "O Livro dos Médiuns",
}

CHAT_MODEL = "llama3.2:3b"

TOP_K = 4
MAX_DISTANCE = 0.7