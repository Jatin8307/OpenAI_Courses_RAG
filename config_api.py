import os
from dotenv import load_dotenv

# Load .env
load_dotenv()

def get_openai_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing from .env")
    return key

# DB & data paths (adjust if you placed files into a subfolder)
DB_PATH = os.getenv("LOCAL_DB_PATH", "courses.db")
DATA_DIR = os.getenv("LOCAL_DATA_DIR", "data")
# FAISS_DIR = os.getenv("FAISS_DIR", "vectorstore/faiss_index")

# Local embedding model name (sentence-transformers)
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

