import sqlite3
import os
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from config_api import DB_PATH, FAISS_DIR, LOCAL_EMBEDDING_MODEL
import numpy as np

# Make sure directory exists
os.makedirs(FAISS_DIR, exist_ok=True)

def load_courses_from_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, title, description FROM courses")
    rows = cur.fetchall()
    conn.close()
    return rows

def build_faiss_index():
    # Load courses
    rows = load_courses_from_db()
    print(f"Loaded {len(rows)} courses from DB")

    # Convert to LangChain documents
    docs = []
    for cid, title, desc in rows:
        text = f"{title}\n{desc}"
        docs.append(
            Document(
                page_content=text,
                metadata={"id": cid, "title": title, "description": desc},
            )
        )

    # Embeddings model (NO torch)
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Build vector store
    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(docs, embedding)

    # Save
    vectorstore.save_local(FAISS_DIR)

    print("\n🎉 FAISS index created successfully!")
    print(f"Saved at: {FAISS_DIR}")

if __name__ == "__main__":
    build_faiss_index()
