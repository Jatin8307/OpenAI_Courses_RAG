import os
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from langchain_core.documents import Document
from typing import List

load_dotenv()

MODEL_NAME = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "courses-index")

model = SentenceTransformer(MODEL_NAME)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

class PineconeRetrieverAdapter:
    def __init__(self, top_k: int = 20):
        self.top_k = top_k

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        LangChain expects a retriever with method get_relevant_documents(query)
        returning a list of langchain_core.documents.Document.
        """
        q_vec = model.encode(query).astype("float32").tolist()
        res = index.query(vector=q_vec, top_k=self.top_k, include_metadata=True)
        matches = res.get("matches", [])

        docs = []
        for m in matches:
            md = m.get("metadata", {}) or {}
            text = md.get("title", "") + "\n\n" + md.get("description", "")
            # attach source/id in metadata
            doc_meta = {
                "id": md.get("id"),
                "title": md.get("title"),
                "category": md.get("category"),
                "level": md.get("level"),
                "score": m.get("score")
            }
            docs.append(Document(page_content=text, metadata=doc_meta))
        return docs

# convenience function
def get_retriever(k: int = 20):
    return PineconeRetrieverAdapter(top_k=k)
