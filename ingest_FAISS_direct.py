import os 
import sqlite3
import numpy as np # numpy module numerical operations ke liye use hota hai
import faiss # faiss module vector similarity search ke liye use hota hai
import pickle # yeh module python objects ko serialize aur deserialize karne ke liye use hota hai
from langchain.schema import Document # Document class ko import karta hai
from langchain.vectorstores import FAISS # FAISS vector store ko import karne ke liye
from config_api import DATA_DIR, FAISS_DIR, DB_PATH

os.makedirs(FAISS_DIR, exist_ok=True)

def load_embeddings():  # embeddings aur ids ko load kardega
    emb_path = os.path.join(DATA_DIR, "embeddings.npy")
    ids_path = os.path.join(DATA_DIR, "course_ids.npy")
    if not os.path.exists(emb_path) or not os.path.exists(ids_path):
        raise FileNotFoundError("Missing embeddings.npy or course_ids.npy in data/ - run build_local_embeddings.py first")
    vectors = np.load(emb_path).astype("float32")
    ids = np.load(ids_path).astype("int32")
    return ids, vectors

def load_metadata():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, title, description, category, level FROM courses ORDER BY id")
    rows = cur.fetchall()
    conn.close()

    docs = []
    for r in rows:
        cid, title, desc, cat, lvl = r
        meta = {"id": int(cid), "title": title, "category": cat, "level": lvl}
        # store textual content in page_content (title + description) so retrieval returns readable text
        docs.append(Document(page_content=(title + "\n\n" + (desc or "")), metadata=meta))
    return docs

def build_faiss_index():
    ids, vectors = load_embeddings()
    docs = load_metadata()

    if len(ids) != len(docs):
        raise ValueError("IDs count and metadata count mismatch")

    d = vectors.shape[1]
    print("Vector dimension:", d)

    # FlatL2 exact index
    index = faiss.IndexFlatL2(d)
    print("Adding vectors to FAISS index (exact)...")
    index.add(vectors)

    # LangChain FAISS wrapper expects an embeddings instance usually, but we can attach
    # documents and index directly. We'll create a FAISS store with None embeddings and supply index & docs.
    store = FAISS(embedding_function=None, index=index, docstore=None, index_to_docstore_id=None)
    # Use private members to set documents and metadata in LangChain's structure.
    # Simpler approach: use from_documents with an embeddings wrapper, but that re-embeds.
    # We'll save the index and save metadata separately (LangChain expects "index.faiss" + "docstore.pkl").

    # Save index to disk
    faiss.write_index(index, os.path.join(FAISS_DIR, "index.faiss"))
    # Save docs metadata as pickled list
    with open(os.path.join(FAISS_DIR, "docstore.pkl"), "wb") as f:
        pickle.dump(docs, f)

    print("FAISS index and metadata saved to:", FAISS_DIR)

if __name__ == "__main__":
    build_faiss_index()
