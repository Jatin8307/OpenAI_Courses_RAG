# embedder_local.py
from sentence_transformers import SentenceTransformer
import numpy as np
from config_api import LOCAL_EMBEDDING_MODEL

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
    return _model

def embed_text(text: str):
    model = get_model()
    vec = model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0].astype("float32")
    return vec
