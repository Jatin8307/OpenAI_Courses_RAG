import os
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from config_api import DATA_DIR, DB_PATH, LOCAL_EMBEDDING_MODEL

os.makedirs(DATA_DIR, exist_ok=True)  # ye ek nayi directory banata hai agar nahi hai to

def fetch_courses():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, title, description FROM courses ORDER BY id")
    rows = cur.fetchall()
    conn.close()
    return rows

def main():
    rows = fetch_courses()
    ids = [r[0] for r in rows]
    texts = [f"{r[1]} {r[2] or ''}" for r in rows]

    print(f"Computing embeddings for {len(texts)} courses using {LOCAL_EMBEDDING_MODEL} ...")
    model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
    vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    emb_path = os.path.join(DATA_DIR, "embeddings.npy")
    id_path = os.path.join(DATA_DIR, "course_ids.npy")
    np.save(emb_path, vectors.astype("float32"))
    np.save(id_path, np.array(ids, dtype="int32"))

    print("Saved embeddings:", emb_path)
    print("Saved ids:", id_path)

if __name__ == "__main__":
    main()
