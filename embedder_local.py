from langchain_huggingface import HuggingFaceEmbeddings
from config_api import LOCAL_EMBEDDING_MODEL

_embedding_model = None

def get_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embedding_model


def embed_text(text: str):
    model = get_model()
    vec = model.embed_query(text)   # returns numpy array (float32)
    return vec
