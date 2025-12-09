import os
from dotenv import load_dotenv

# Load .env
load_dotenv()

def get_openai_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing from .env")
    return key
