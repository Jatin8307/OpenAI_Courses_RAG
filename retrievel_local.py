import os
import sqlite3
import numpy as np
import faiss
import pickle
from typing import List, Dict, Any, Optional
from config_api import FAISS_DIR, DB_PATH
from embedder_local import embed_text

_INDEX = None
_DOCS = None

def _load_faiss_store():
    global _INDEX, _DOCS
    if _INDEX is not None:
        return _INDEX, _DOCS
    idx_path = os.path.join(FAISS_DIR, "index.faiss")
    doc_path = os.path.join(FAISS_DIR, "docstore.pkl")
    if not os.path.exists(idx_path) or not os.path.exists(doc_path):
        raise FileNotFoundError("FAISS index files not found. Run ingest_faiss_direct.py first.")
    _INDEX = faiss.read_index(idx_path)
    with open(doc_path, "rb") as f:
        _DOCS = pickle.load(f)
    return _INDEX, _DOCS

def sql_search(keywords: List[str], limit: int = 20):
    if not keywords:
        return []
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    conds = " OR ".join(["(title LIKE ? OR description LIKE ?)" for _ in keywords])
    params = []
    for k in keywords:
        kw = f"%{k}%"
        params.extend([kw, kw])
    query = f"SELECT id, title, description, category, level FROM courses WHERE {conds} LIMIT ?"
    cur.execute(query, params + [limit])
    rows = cur.fetchall()
    conn.close()
    results = [{"id": r[0], "title": r[1], "description": r[2], "category": r[3], "level": r[4]} for r in rows]
    return results

def vector_search_by_query(query: str, k: int = 40, metadata_filter: Optional[Dict[str,str]] = None):
    index, docs = _load_faiss_store()
    q_vec = embed_text(query)  # returns numpy float32
    q_vec = np.expand_dims(q_vec, axis=0)
    # perform exact L2 search (IndexFlatL2)
    D, I = index.search(q_vec, k)
    idxs = I[0].tolist()
    results = []
    for idx in idxs:
        if idx < 0 or idx >= len(docs):
            continue
        doc = docs[idx]
        meta = doc.metadata or {}
        if metadata_filter:
            ok = True
            for kf, v in metadata_filter.items():
                if meta.get(kf) != v:
                    ok = False
                    break
            if not ok:
                continue
        results.append({
            "id": meta.get("id"),
            "title": meta.get("title"),
            "description": doc.page_content,
            "category": meta.get("category"),
            "level": meta.get("level")
        })
    return results

def hybrid_retrieve(query: str, keywords=None, k=40, metadata_filter: Optional[Dict[str,str]] = None, sql_first=True):
    if keywords is None:
        keywords = [t.strip().lower() for t in query.split() if t.strip()]
    if sql_first:
        sql_res = sql_search(keywords, limit=10)
        if sql_res:
            return sql_res
    # vector fallback
    vec_res = vector_search_by_query(query, k=k, metadata_filter=metadata_filter)
    return vec_res
